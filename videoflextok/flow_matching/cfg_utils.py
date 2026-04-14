# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. and EPFL. All Rights Reserved.
from typing import Tuple

import torch


class CFGRescaler:
    def __call__(
        self, output_cfg: torch.Tensor, output_cond: torch.Tensor, output_uncond: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()


class CFGWithRescale(CFGRescaler):
    """
    From https://arxiv.org/abs/2305.08891
    """

    def __init__(self, rescale: float = 0.7, channel_wise: bool = True):
        super().__init__()
        self.rescale = rescale
        self.channel_wise = channel_wise

    def __call__(
        self, output_cfg: torch.Tensor, output_cond: torch.Tensor, output_uncond: torch.Tensor
    ) -> torch.Tensor:
        dims = [2, 3] if self.channel_wise else [1, 2, 3]
        std_cond = output_cond.std(dims, keepdim=True)
        std_cfg = output_cfg.std(dims, keepdim=True)
        factor = std_cond / std_cfg
        factor = self.rescale * factor + (1 - self.rescale)
        return output_cfg * factor


class DynamicCFGRescaler(CFGRescaler):
    def __init__(
        self, mimic_scale: float = 4.0, quantile: float = 0.997, channel_wise: bool = True
    ):
        super().__init__()
        self.mimic_scale = mimic_scale
        self.quantile = quantile
        self.channel_wise = channel_wise

    def __call__(
        self, output_cfg: torch.Tensor, output_cond: torch.Tensor, output_uncond: torch.Tensor
    ) -> torch.Tensor:
        dims = [2, 3] if self.channel_wise else [1, 2, 3]

        output_mimic = output_uncond + self.mimic_scale * (output_cond - output_uncond)

        mean_cfg = output_cond.mean(dims, keepdim=True)
        mean_mimic = output_mimic.mean(dims, keepdim=True)

        centered_cfg = output_cfg - mean_cfg
        centered_mimic = output_mimic - mean_mimic

        # Compute quantile
        B, C, H, W = centered_cfg.shape
        if self.channel_wise:
            centered_cfg_flat = centered_cfg.abs().view(B, C, -1)
            max_val_dynamic = torch.quantile(
                centered_cfg_flat.float(), q=self.quantile, dim=2, keepdim=True
            ).view(B, C, 1, 1)
            centered_mimic_flat = centered_mimic.abs().view(B, C, -1)
            max_val_mimic = centered_mimic_flat.max(dim=2, keepdim=True).values.view(B, C, 1, 1)
        else:
            centered_cfg_flat = centered_cfg.abs().view(B, -1)
            max_val_dynamic = torch.quantile(
                centered_cfg_flat.float(), q=self.quantile, dim=1, keepdim=True
            ).view(-1, 1, 1, 1)
            centered_mimic_flat = centered_mimic.abs().view(B, -1)
            max_val_mimic = centered_mimic_flat.max(dim=1, keepdim=True).values.view(-1, 1, 1, 1)
        max_val_dynamic = torch.maximum(max_val_dynamic, max_val_mimic)

        cfg_centered = centered_cfg / max_val_dynamic * max_val_mimic

        return cfg_centered + mean_cfg


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def project(
    v0: torch.Tensor,  # [B, C, H, W]
    v1: torch.Tensor,  # [B, C, H, W]
) -> Tuple[torch.Tensor, torch.Tensor]:
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def normalized_guidance(
    pred_cond: torch.Tensor,  # [B, C, H, W]
    pred_uncond: torch.Tensor,  # [B, C, H, W]
    guidance_scale: float,
    momentum_buffer: MomentumBuffer = None,
    eta: float = 0.0,
    norm_threshold: float = 2.5,
) -> torch.Tensor:
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    diff_parallel, diff_orthogonal = project(diff, pred_cond)
    normalized_update = diff_orthogonal + eta * diff_parallel
    pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
    return pred_guided


def classifier_free_guidance(
    pred_cond: torch.Tensor,  # [B, C, H, W]
    pred_uncond: torch.Tensor,  # [B, C, H, W]
    guidance_scale: float,
) -> torch.Tensor:
    return pred_uncond + guidance_scale * (pred_cond - pred_uncond)
