# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. and EPFL. All Rights Reserved.
"""
Adapted from
VidTok: https://github.com/microsoft/VidTok/blob/d6ad92d98e8d67617a6ee8fa1ac2111d6bdaa739/vidtok/modules/model_3dcausal.py#L162
MagViT2: https://github.com/lucidrains/magvit2-pytorch/blob/a00519fa9b9ca58f783d0ab16c9e09579314c9cd/magvit2_pytorch/magvit2_pytorch.py#L892
"""

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


class CausalConv3d(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        pad_mode="constant",
        **kwargs,
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)
        dilation = kwargs.pop("dilation", 1)
        stride = kwargs.pop("stride", 1)
        dilation = cast_tuple(dilation, 3)
        stride = cast_tuple(stride, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        self.pad_mode = pad_mode
        time_pad = dilation[0] * (time_kernel_size - 1) + (1 - stride[0])
        height_pad = dilation[1] * (height_kernel_size - 1) + (1 - stride[1])
        width_pad = dilation[2] * (height_kernel_size - 1) + (1 - stride[2])

        self.time_pad = time_pad
        self.time_causal_padding = (
            width_pad // 2,
            width_pad - width_pad // 2,
            height_pad // 2,
            height_pad - height_pad // 2,
            time_pad,
            0,
        )

        self.conv = nn.Conv3d(
            chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs
        )

    def forward(self, x: torch.FloatTensor):
        """
        x: torhc.FloatTensor of shape (B, C, T, H, W)
        """
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else "constant"

        x = F.pad(x, self.time_causal_padding, mode=pad_mode)
        return self.conv(x)


if __name__ == "__main__":
    x = torch.rand(2, 3, 5, 32, 32)
    conv = CausalConv3d(3, 16, (2, 3, 3))
    print(x.shape, conv(x).shape)
