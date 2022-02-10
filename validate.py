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
"""File description: Realize the verification function after model training."""
import os

import numpy as np
import torch
from PIL import Image
from natsort import natsorted

import config
import imgproc
from model import RCAN


def main() -> None:
    # Initialize the super-resolution model
    print("Build RCAN model...")
    model = RCAN(config.upscale_factor).to(config.device)
    print("Build RCAN model successfully.")

    # Load the super-resolution model weights
    print(f"Load SR model weights `{os.path.abspath(config.model_path)}`...")
    state_dict = torch.load(config.model_path, map_location=config.device)
    model.load_state_dict(state_dict)
    print(f"Load SR model weights `{os.path.abspath(config.model_path)}` successfully.")

    # Create a folder of super-resolution experiment results
    results_dir = os.path.join("results", "test", config.exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Start the verification mode of the model.
    model.eval()
    # Turn on half-precision inference.
    model.half()

    # Initialize the image evaluation index.
    total_psnr = 0.0

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(config.lr_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        lr_image_path = os.path.join(config.lr_dir, file_names[index])
        sr_image_path = os.path.join(config.sr_dir, file_names[index])
        hr_image_path = os.path.join(config.hr_dir, file_names[index])

        print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        lr_image = Image.open(lr_image_path).convert("RGB")
        hr_image = Image.open(hr_image_path).convert("RGB")

        # Extract RGB channel image data
        lr_tensor = imgproc.image2tensor(lr_image, range_norm=False, half=True).to(config.device).unsqueeze_(0)

        # Reconstruct the RGB channel image data.
        with torch.no_grad():
            sr_tensor = model(lr_tensor)

        # Save SR image
        sr_image = imgproc.tensor2image(sr_tensor, range_norm=False, half=True)
        sr_image = Image.fromarray(sr_image)
        sr_image.save(sr_image_path)

        # Cal PSNR for Y image data
        sr_image = np.array(sr_image).astype(np.float32)
        hr_image = np.array(hr_image).astype(np.float32)

        sr_ycbcr_image = imgproc.convert_rgb_to_ycbcr(sr_image)
        hr_ycbcr_image = imgproc.convert_rgb_to_ycbcr(hr_image)

        sr_y_image = sr_ycbcr_image[..., 0]
        hr_y_image = hr_ycbcr_image[..., 0]

        sr_y_image /= 255.
        hr_y_image /= 255.

        sr_y_tensor = torch.from_numpy(sr_y_image).unsqueeze(0)
        hr_y_tensor = torch.from_numpy(hr_y_image).unsqueeze(0)

        total_psnr += 10. * torch.log10(1. / torch.mean((sr_y_tensor - hr_y_tensor) ** 2))

    print(f"PSNR: {total_psnr / total_files:.2f} dB.\n")


if __name__ == "__main__":
    main()
