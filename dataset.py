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
"""Realize the function of dataset preparation."""
import os

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode as IMode

import imgproc

__all__ = ["ImageDataset"]


class ImageDataset(Dataset):
    """Customize the data set loading function and prepare low/high resolution image data in advance.

    Args:
        dataroot         (str): Training data set address
        image_size       (int): High resolution image size
        upscale_factor   (int): Image magnification
        mode             (str): Data set loading method, the training data set is for data enhancement,
                             and the verification data set is not for data enhancement

    """

    def __init__(self, dataroot: str, image_size: int, upscale_factor: int, mode: str) -> None:
        super(ImageDataset, self).__init__()
        self.image_file_names = [os.path.join(dataroot, x) for x in os.listdir(dataroot)]

        if mode == "train":
            self.hr_transforms = transforms.RandomCrop(image_size)
        elif mode == "valid":
            self.hr_transforms = transforms.CenterCrop(image_size)
        else:
            raise "Unsupported data processing model, please use `train` or `valid`."

        self.lr_transforms = transforms.Resize(image_size // upscale_factor, interpolation=IMode.BICUBIC)

    def __getitem__(self, batch_index: int) -> [Tensor, Tensor]:
        # Read a batch of image data
        image = Image.open(self.image_file_names[batch_index])

        # Data augment
        hr_image = self.hr_transforms(image)
        lr_image = self.lr_transforms(hr_image)

        lr_image, hr_image = imgproc.random_rotate(lr_image, hr_image, degrees=[0, 90, 180, 270])
        lr_image, hr_image = imgproc.random_horizontally_flip(lr_image, hr_image, p=0.5)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        lr_tensor = imgproc.image2tensor(lr_image, range_norm=False, half=False)
        hr_tensor = imgproc.image2tensor(hr_image, range_norm=False, half=False)

        return lr_tensor, hr_tensor

    def __len__(self) -> int:
        return len(self.image_file_names)
