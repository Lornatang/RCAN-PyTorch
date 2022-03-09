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
import math

import torch
from torch import nn

__all__ = [
    "ChannelAttentionLayer", "ResidualChannelAttentionBlock", "ResidualGroup",
    "RCAN",
]


class ChannelAttentionLayer(nn.Module):
    """Attention Mechanism Module"""

    def __init__(self, channel: int, reduction: int):
        super(ChannelAttentionLayer, self).__init__()
        self.channel_attention_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, (1, 1), (1, 1), (0, 0)),
            nn.ReLU(True),
            nn.Conv2d(channel // reduction, channel, (1, 1), (1, 1), (0, 0)),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.channel_attention_layer(x)

        out = torch.mul(out, identity)

        return out


class ResidualChannelAttentionBlock(nn.Module):
    """Residual Channel Attention Block (RCAB)"""

    def __init__(self, channel: int, reduction: int):
        super(ResidualChannelAttentionBlock, self).__init__()
        self.residual_channel_attention_block = nn.Sequential(
            nn.Conv2d(channel, channel, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(channel, channel, (3, 3), (1, 1), (1, 1)),
            ChannelAttentionLayer(channel, reduction),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.residual_channel_attention_block(x)

        out = torch.add(out, identity)

        return out


class ResidualGroup(nn.Module):
    """Residual Group (RG)"""

    def __init__(self, channel: int, reduction: int):
        super(ResidualGroup, self).__init__()
        residual_group = []

        for _ in range(20):
            residual_group.append(ResidualChannelAttentionBlock(channel, reduction))
        residual_group.append(nn.Conv2d(channel, channel, (3, 3), (1, 1), (1, 1)))

        self.residual_group = nn.Sequential(*residual_group)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.residual_group(x)

        out = torch.add(out, identity)

        return out


class UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.upsample_block(x)

        return out


class RCAN(nn.Module):
    def __init__(self, upscale_factor: int):
        super(RCAN, self).__init__()
        # First layer
        self.conv1 = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))

        # Residual Group
        trunk = []
        for _ in range(10):
            trunk.append(ResidualGroup(64, 16))
        self.trunk = nn.Sequential(*trunk)

        # Second layer
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))

        # Upsampling layers
        upsampling = []
        if upscale_factor == 2 or upscale_factor == 4 or upscale_factor == 8:
            for _ in range(int(math.log(upscale_factor, 2))):
                upsampling.append(UpsampleBlock(64, 2))
        elif upscale_factor == 3:
            upsampling.append(UpsampleBlock(64, 3))
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1))

        self.register_buffer("mean", torch.Tensor([0.4488, 0.4371, 0.4040]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The images by subtracting the mean RGB value of the DIV2K dataset.
        out = x.sub_(self.mean).mul_(255.)

        out1 = self.conv1(out)
        out = self.trunk(out1)
        out = self.conv2(out)
        out = torch.add(out, out1)
        out = self.upsampling(out)
        out = self.conv3(out)

        out = out.div_(255.).add_(self.mean)

        return out
