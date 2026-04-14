# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. and EPFL. All Rights Reserved.
from typing import List

import einops

import torch
import torch.nn as nn

__all__ = ["packed_proj"]


def packed_proj(x_list: List[torch.Tensor], proj: nn.Module) -> List[torch.Tensor]:
    x_packed, ps = einops.pack(x_list, "b * d")
    x_packed = proj(x_packed)
    return einops.unpack(x_packed, ps, "b * d")
