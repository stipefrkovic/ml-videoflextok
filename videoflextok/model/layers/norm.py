# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. and EPFL. All Rights Reserved.
import einops

import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Fp32LayerNorm", "Fp32SpatioTemporalLayerNorm"]


class Fp32LayerNorm(nn.LayerNorm):
    """Mixed precision friendly LayerNorm.
    From torchtune.modules.Fp32LayerNorm.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class Fp32SpatioTemporalLayerNorm(Fp32LayerNorm):
    def forward(self, input):
        input = einops.rearrange(input, "b c t h w -> b t h w c")
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        output = einops.rearrange(output, "b t h w c -> b c t h w")
        return output.type_as(input)
