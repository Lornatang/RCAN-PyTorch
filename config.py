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
# ==============================================================================
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = True
# Model architecture name
model_arch_name = "rcan_x4"
# Upscale factor
upscale_factor = 4
# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = "RCAN_x4-DIV2K"

if mode == "train":
    train_gt_images_dir = f"./data/DIV2K/RCAN/train"

    test_gt_images_dir = f"./data/Set5/GTmod12"
    test_lr_images_dir = f"./data/Set5/LRbicx{upscale_factor}"

    train_gt_image_size = int(upscale_factor * 48)
    batch_size = 16
    num_workers = 4

    # Load the address of the pretrained model
    pretrained_model_weights_path = f""

    # Incremental training and migration training
    resume_model_weights_path = f""

    # Total num epochs
    epochs = 1000

    # Loss function weight
    loss_weight = [1.0]

    # Optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.99)
    model_eps = 1e-4  # Keep no nan
    model_weight_decay = 0.0

    # EMA parameter
    model_ema_decay = 0.999

    # StepLR scheduler parameter
    lr_scheduler_step_size = epochs // 5
    lr_scheduler_gamma = 0.5

    # How many iterations to print the training result
    train_print_frequency = 100
    test_print_frequency = 1

if mode == "test":
    test_gt_images_dir = f"./data/Set5/GTmod12"
    test_sr_images_dir = f"./results/test/{exp_name}"
    test_lr_images_dir = f"./data/Set5/LRbicx{upscale_factor}"

    model_weights_path = f"./results/pretrained_models/RCAN_x4-DIV2K-2dfffdd2.pth.tar"
