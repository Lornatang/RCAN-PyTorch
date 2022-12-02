# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
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
# ==============================================================================
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

import config
import model
from dataset import TestImageDataset, CUDAPrefetcher
from utils import build_iqa_model, load_state_dict, make_directory, AverageMeter, ProgressMeter


def load_dataset(test_gt_images_dir: str, test_lr_images_dir: str, device: torch.device) -> CUDAPrefetcher:
    test_datasets = TestImageDataset(test_gt_images_dir, test_lr_images_dir)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=False)
    test_data_prefetcher = CUDAPrefetcher(test_dataloader, device)

    return test_data_prefetcher


def build_model(model_arch_name: str, device: torch.device) -> nn.Module:
    # Build model
    sr_model = model.__dict__[model_arch_name]()
    sr_model = sr_model.to(device=device)
    # Set the model to evaluation mode
    sr_model.eval()

    return sr_model


def test(
        sr_model: nn.Module,
        test_data_prefetcher: CUDAPrefetcher,
        psnr_model: nn.Module,
        ssim_model: nn.Module,
        device: torch.device = torch.device("cpu"),
        print_frequency: int = 1,
) -> [float, float]:
    # The information printed by the progress bar
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimes = AverageMeter("SSIM", ":4.4f")
    progress = ProgressMeter(len(test_data_prefetcher), [batch_time, psnres, ssimes], prefix=f"Test: ")

    # Set the model as validation model
    sr_model.eval()

    # Initialize data batches
    batch_index = 0

    # Set the data set iterator pointer to 0 and load the first batch of data
    test_data_prefetcher.reset()
    batch_data = test_data_prefetcher.next()

    # Record the start time of verifying a batch
    end = time.time()

    while batch_data is not None:
        # Load batches of data
        gt = batch_data["gt"].to(device=device, non_blocking=True)
        lr = batch_data["lr"].to(device=device, non_blocking=True)

        # inference
        with torch.no_grad():
            sr = sr_model(lr)

        # Calculate the image IQA
        psnr = psnr_model(sr, gt)
        ssim = ssim_model(sr, gt)
        psnres.update(psnr.item(), lr.size(0))
        ssimes.update(ssim.item(), lr.size(0))

        # Record the total time to verify a batch
        batch_time.update(time.time() - end)
        end = time.time()

        # Output a verification log information
        if batch_index % print_frequency == 0:
            progress.display(batch_index + 1)

        # Preload the next batch of data
        batch_data = test_data_prefetcher.next()

        # Add 1 to the number of data batches
        batch_index += 1

    # Print the performance index of the model at the current epoch
    progress.display_summary()

    return psnres.avg, ssimes.avg


def main() -> None:
    test_data_prefetcher = load_dataset(config.test_gt_images_dir, config.test_lr_images_dir, config.device)
    sr_model = build_model(config.model_arch_name, config.device)
    psnr_model, ssim_model = build_iqa_model(config.upscale_factor, config.only_test_y_channel, config.device)

    # Load the super-resolution bsrgan_model weights
    sr_model = load_state_dict(sr_model, config.model_weights_path)

    # Create a folder of super-resolution experiment results
    make_directory(config.test_sr_images_dir)

    psnr, ssim = test(sr_model,
                      test_data_prefetcher,
                      psnr_model,
                      ssim_model,
                      config.device)

    print(f"PSNR: {psnr:.2f} dB"
          f"SSIM: {ssim:.4f} [u]")


if __name__ == "__main__":
    main()
