# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import time

import torch
from torch import nn, optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config
import model
from dataset import CUDAPrefetcher, TrainImageDataset, TestImageDataset
from test import test
from utils import build_iqa_model, load_state_dict, make_directory, save_checkpoint, AverageMeter, ProgressMeter


def main():
    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0
    best_ssim = 0.0

    train_data_prefetcher, test_data_prefetcher = load_dataset(config.train_gt_images_dir,
                                                               config.train_gt_image_size,
                                                               config.test_gt_images_dir,
                                                               config.test_lr_images_dir,
                                                               config.upscale_factor,
                                                               config.batch_size,
                                                               config.num_workers,
                                                               config.device)
    print("Load all datasets successfully.")

    sr_model, ema_sr_model = build_model(config.model_arch_name, config.device)
    print(f"Build `{config.model_arch_name}` model successfully.")

    criterion = define_loss(config.device)
    print("Define all loss functions successfully.")

    optimizer = define_optimizer(sr_model)
    print("Define all optimizer functions successfully.")

    scheduler = define_scheduler(optimizer)
    print("Define all optimizer scheduler functions successfully.")

    # Create an IQA evaluation model
    psnr_model, ssim_model = build_iqa_model(config.upscale_factor, config.only_test_y_channel, config.device)

    print("Check whether to load pretrained model weights...")
    if config.pretrained_model_weights_path:
        sr_model = load_state_dict(sr_model, config.pretrained_model_weights_path)
        print(f"Loaded `{config.pretrained_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    print("Check whether the resume model is restored...")
    if config.resume_model_weights_path:
        sr_model, ema_sr_model, start_epoch, best_psnr, best_ssim, optimizer, scheduler = load_state_dict(
            sr_model,
            config.resume_model_weights_path,
            ema_sr_model,
            optimizer,
            scheduler,
            "resume")
        print("Loaded resume model weights.")
    else:
        print("Resume training model not found. Start training from scratch.")

    # Create a experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    make_directory(samples_dir)
    make_directory(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    for epoch in range(start_epoch, config.epochs):
        train(sr_model,
              ema_sr_model,
              train_data_prefetcher,
              criterion,
              optimizer,
              epoch,
              scaler,
              writer,
              config.device,
              config.train_print_frequency)
        psnr, ssim = test(sr_model,
                          test_data_prefetcher,
                          psnr_model,
                          ssim_model,
                          config.device,
                          config.test_print_frequency)

        # Write the evaluation results to the tensorboard
        writer.add_scalar(f"Test/PSNR", psnr, epoch + 1)
        writer.add_scalar(f"Test/SSIM", ssim, epoch + 1)

        print("\n")

        # Update LR
        scheduler.step()

        # Automatically save the model with the highest index
        is_best = psnr > best_psnr and ssim > best_ssim
        is_last = (epoch + 1) == config.epochs
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)
        save_checkpoint({"epoch": epoch + 1,
                         "best_psnr": best_psnr,
                         "best_ssim": best_ssim,
                         "state_dict": sr_model.state_dict(),
                         "ema_state_dict": ema_sr_model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict()},
                        f"epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "best.pth.tar",
                        "last.pth.tar",
                        is_best,
                        is_last)


def load_dataset(
        train_gt_images_dir: str,
        train_gt_image_size: int,
        test_gt_images_dir: str,
        test_lr_images_dir: str,
        upscale_factor: int,
        batch_size: int,
        num_workers: int,
        device: torch.device,
) -> [CUDAPrefetcher, CUDAPrefetcher]:
    # Load train, test and valid datasets
    train_datasets = TrainImageDataset(train_gt_images_dir, train_gt_image_size, upscale_factor)
    test_datasets = TestImageDataset(test_gt_images_dir, test_lr_images_dir)

    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=False)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, device)
    test_prefetcher = CUDAPrefetcher(test_dataloader, device)

    return train_prefetcher, test_prefetcher


def build_model(model_arch_name: str, device: torch.device) -> [nn.Module, nn.Module]:
    # Build model
    sr_model = model.__dict__[model_arch_name]()
    # Generate exponential average model, stabilize model training
    ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: (1 - config.model_ema_decay) * averaged_model_parameter + config.model_ema_decay * model_parameter
    ema_sr_model = AveragedModel(sr_model, avg_fn=ema_avg_fn)

    sr_model = sr_model.to(device=device)
    ema_sr_model = ema_sr_model.to(device=device)

    return sr_model, ema_sr_model


def define_loss(device) -> nn.L1Loss:
    criterion = nn.L1Loss().to(device=device)

    return criterion


def define_optimizer(sr_model: nn.Module) -> optim.Adam:
    optimizer = optim.Adam(sr_model.parameters(),
                           config.model_lr,
                           config.model_betas,
                           config.model_eps)

    return optimizer


def define_scheduler(optimizer) -> lr_scheduler.StepLR:
    scheduler = lr_scheduler.StepLR(optimizer,
                                    config.lr_scheduler_step_size,
                                    config.lr_scheduler_gamma)

    return scheduler


def train(
        sr_model: nn.Module,
        ema_sr_model: nn.Module,
        train_data_prefetcher: CUDAPrefetcher,
        criterion: nn.L1Loss,
        optimizer: optim.Adam,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter,
        device: torch.device = torch.device("cpu"),
        print_frequency: int = 1,
) -> None:
    # Calculate how many iterations there are under epoch
    batches = len(train_data_prefetcher)
    # Progress bar print information
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses], prefix=f"Epoch: [{epoch}]")

    # Put the generator in training mode
    sr_model.train()

    # Define loss function weights
    loss_weight = torch.Tensor(config.loss_weight).to(device=device)

    # Initialize data batches
    batch_index = 0
    # Set the dataset iterator pointer to 0
    train_data_prefetcher.reset()
    # Record the start time of training a batch
    end = time.time()
    # load the first batch of data
    batch_data = train_data_prefetcher.next()

    while batch_data is not None:
        gt = batch_data["gt"].to(config.device, non_blocking=True)
        lr = batch_data["lr"].to(config.device, non_blocking=True)

        # Record the data loading time for training a batch
        data_time.update(time.time() - end)

        # Initialize the generator gradient
        sr_model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            sr = sr_model(lr)
            loss = criterion(sr, gt)
            loss = torch.sum(torch.mul(loss_weight, loss))

        # Gradient zoom
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # Update generator weight
        scaler.step(optimizer)
        scaler.update()

        # update exponentially averaged model weights
        ema_sr_model.update_parameters(sr_model)

        # record the loss value
        losses.update(loss.item(), lr.size(0))

        # Record the total time of training a batch
        batch_time.update(time.time() - end)
        end = time.time()

        # Output training log information once
        if batch_index % print_frequency == 0:
            # Write training log information to tensorboard
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            progress.display(batch_index)

        # Preload the next batch of data
        batch_data = train_data_prefetcher.next()

        # Add 1 to the number of data batches
        batch_index += 1


if __name__ == "__main__":
    main()
