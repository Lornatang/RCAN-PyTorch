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
from torch import nn, Tensor

__all__ = [
    "RCAN",
    "rcan_x2", "rcan_x3", "rcan_x4", "rcan_x8",
]


class RCAN(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: int,
            reduction: int,
            num_rcab: int,
            num_rg: int,
            upscale_factor: int,
            rgb_mean: tuple = None,
    ) -> None:
        super(RCAN, self).__init__()
        if rgb_mean is None:
            rgb_mean = [0.4488, 0.4371, 0.4040]

        # The first layer of convolutional layer
        self.conv1 = nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1))

        # Feature extraction backbone
        trunk = []
        for _ in range(num_rg):
            trunk.append(_ResidualGroup(channels, reduction, num_rcab))
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))

        # Upsampling convolutional layer.
        upsampling = []
        if upscale_factor == 2 or upscale_factor == 4 or upscale_factor == 8:
            for _ in range(int(math.log(upscale_factor, 2))):
                upsampling.append(_UpsampleBlock(channels, 2))
        elif upscale_factor == 3:
            upsampling.append(_UpsampleBlock(channels, 3))
        self.upsampling = nn.Sequential(*upsampling)

        # Output layer.
        self.conv3 = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

        self.register_buffer("mean", Tensor(rgb_mean).view(1, 3, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        x = x.sub_(self.mean).mul_(1.)

        conv1 = self.conv1(x)
        x = self.trunk(conv1)
        x = self.conv2(x)
        x = torch.add(x, conv1)
        x = self.upsampling(x)
        x = self.conv3(x)

        x = x.div_(1.).add_(self.mean)

        return x


class _ChannelAttentionLayer(nn.Module):
    def __init__(self, channel: int, reduction: int):
        super(_ChannelAttentionLayer, self).__init__()
        self.channel_attention_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, (1, 1), (1, 1), (0, 0)),
            nn.ReLU(True),
            nn.Conv2d(channel // reduction, channel, (1, 1), (1, 1), (0, 0)),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.channel_attention_layer(x)

        out = torch.mul(out, x)

        return out


class _ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, channel: int, reduction: int):
        super(_ResidualChannelAttentionBlock, self).__init__()
        self.residual_channel_attention_block = nn.Sequential(
            nn.Conv2d(channel, channel, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(channel, channel, (3, 3), (1, 1), (1, 1)),
            _ChannelAttentionLayer(channel, reduction),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.residual_channel_attention_block(x)

        out = torch.add(out, identity)

        return out


class _ResidualGroup(nn.Module):
    def __init__(self, channel: int, reduction: int, num_rcab: int):
        super(_ResidualGroup, self).__init__()
        residual_group = []

        for _ in range(num_rcab):
            residual_group.append(_ResidualChannelAttentionBlock(channel, reduction))
        residual_group.append(nn.Conv2d(channel, channel, (3, 3), (1, 1), (1, 1)))

        self.residual_group = nn.Sequential(*residual_group)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.residual_group(x)

        out = torch.add(out, identity)

        return out


class _UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(_UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample_block(x)

        return x


def rcan_x2(**kwargs) -> RCAN:
    model = RCAN(3, 3, 64, 16, 20, 10, 2, **kwargs)

    return model


def rcan_x3(**kwargs) -> RCAN:
    model = RCAN(3, 3, 64, 16, 20, 10, 3, **kwargs)

    return model


def rcan_x4(**kwargs) -> RCAN:
    model = RCAN(3, 3, 64, 16, 20, 10, 4, **kwargs)

    return model


def rcan_x8(**kwargs) -> RCAN:
    model = RCAN(3, 3, 64, 16, 20, 10, 8, **kwargs)

    return model
