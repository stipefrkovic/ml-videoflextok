# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. and EPFL. All Rights Reserved.
from diffusers.schedulers.scheduling_utils import SchedulerMixin

import torch


def get_sigmas(noise_scheduler: SchedulerMixin, timesteps: torch.Tensor, n_dim: int = 4):
    """Computes the sigmas for the given timesteps.

    Args:
        noise_scheduler: Flow matching noise scheduler
        timesteps: Tensor of timesteps
        n_dim: Unsqueeze sigmas to this dimension
    """
    sigmas = noise_scheduler.sigmas.to(dtype=timesteps.dtype)
    schedule_timesteps = noise_scheduler.timesteps
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def resolution_dependent_sigma_shift(
    sigmas: torch.Tensor, resolutions: torch.Tensor, base_resolution: int = 256
):
    """Shift sigmas depending on the image resolution, accordingto Section 5.3.2. in the
    SD3 paper (https://arxiv.org/abs/2403.03206). Higher resolutions require more noising.

    Args:
        sigmas: Original sigmas
        resolutions: (B, 2) tensor of image heights and widths
        base_resolution: Base resolution to use for shifting the sigmas
    """
    n = base_resolution**2
    m = resolutions[:, 0] * resolutions[:, 1]
    alphas = (m / n).sqrt()
    return alphas * sigmas / (1 + (alphas - 1) * sigmas)
