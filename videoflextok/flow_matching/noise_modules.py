# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. and EPFL. All Rights Reserved.
import math
from collections.abc import Iterable
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

try:
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )
    from diffusers.training_utils import (
        compute_density_for_timestep_sampling,
        compute_loss_weighting_for_sd3,
    )
except:
    print(
        "Outdated diffusers. Install `pip install --upgrade git+https://github.com/huggingface/diffusers@a1d55e1`"
    )

from .utils import get_sigmas, resolution_dependent_sigma_shift

__all__ = ["MultiResFlowMatchingNoiseModule", "PrefixMinRFNoiseModule"]


def interpolate_param(
    value_start: Optional[float], value_end: Optional[float], global_step: int, warmup_steps: int
):
    """Linear interpolation between a given start and end value."""
    if value_start is None:
        return value_end
    assert global_step is not None
    assert global_step >= 0
    assert warmup_steps > 0
    alpha = global_step / warmup_steps if global_step < warmup_steps else 1.0
    return (1 - alpha) * value_start + alpha * value_end


class MultiResFlowMatchingNoiseModule(nn.Module):
    """Module that preprocesses a data_dict to create noised versions of clean images/latents.
    Adds the following entries into the data_dict that are needed for the decoder and flow loss:
    Noised images, timesteps, sigmas, flow loss weights.

    Args:
        clean_images_read_key: Dictionary key to read out clean images
        noised_images_write_key: Dictionary key to write noised images
        noise_write_key: Dictionary key to write the per-image noises
        timesteps_write_key: Dictionary key to write flow matching timesteps
        sigmas_write_key: Dictionary key to write flow matching sigmas
        weighting_write_key: Dictionary key to write flow matching weighting
        num_train_timesteps: Number of flow matching steps during training
        noise_shift: Resolution-independent noise shift. See Section 5.3.2. of the SD3 paper
        weighting_scheme: Flow matching weighting scheme. One of 'logit_normal', 'mode', 'sigma_sqrt', 'cosmap'.
        warmup_steps: Optionally warm up the noise schedule parameters for the given number of steps
        logit_mean_start: Warmup start value for logit_mean
        logit_mean: SD3 script's default is 0.0. Only used for `logit_normal` `weighting_scheme`
        logit_std_start: Warmup start value for logit_std
        logit_std: SD3 script's default is 1.0. Only used for `logit_normal` `weighting_scheme`
        mode_scale_start: Warmup start value for mode_scale
        mode_scale: SD3 script's default is 1.29. Only used for `mode` `weighting_scheme`
        base_resolution: Optional base resolution if noise should be shifted according to the
            resolution as in Section 5.3.2. of the SD3 paper
    """

    def __init__(
        self,
        clean_images_read_key: str,
        noised_images_write_key: str,
        timesteps_write_key: str,
        sigmas_write_key: str,
        weighting_write_key: str,
        num_train_timesteps: int,
        noise_shift: float,
        weighting_scheme: str,
        warmup_steps: int = None,
        logit_mean_start: Optional[float] = None,
        logit_mean: Optional[float] = None,
        logit_std_start: Optional[float] = None,
        logit_std: Optional[float] = None,
        mode_scale_start: Optional[float] = None,
        mode_scale: Optional[float] = None,
        base_resolution: Optional[int] = None,
        noise_read_key: Optional[str] = None,
        noise_write_key: str = "flow_noise",
        detach_noise: bool = False,
    ):
        super().__init__()
        self.clean_images_read_key = clean_images_read_key
        self.noised_images_write_key = noised_images_write_key
        self.noise_read_key = noise_read_key
        self.timesteps_write_key = timesteps_write_key
        self.sigmas_write_key = sigmas_write_key
        self.weighting_write_key = weighting_write_key
        self.noise_write_key = noise_write_key
        self.detach_noise = detach_noise

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=noise_shift,
        )

        self.weighting_scheme = weighting_scheme
        self.warmup_steps = warmup_steps
        self.logit_mean_start, self.logit_mean = logit_mean_start, logit_mean
        self.logit_std_start, self.logit_std = logit_std_start, logit_std
        self.mode_scale_start, self.mode_scale = mode_scale_start, mode_scale
        self.base_resolution = base_resolution

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any], global_step: int = None) -> Dict[str, Any]:
        clean_images = data_dict[self.clean_images_read_key]
        device = clean_images[0].device

        # Optionally warm up the noise schedule parameters
        if self.warmup_steps is None:
            logit_mean = self.logit_mean
            logit_std = self.logit_std
            mode_scale = self.mode_scale
        else:
            logit_mean = interpolate_param(
                self.logit_mean_start, self.logit_mean, global_step, self.warmup_steps
            )
            logit_std = interpolate_param(
                self.logit_std_start, self.logit_std, global_step, self.warmup_steps
            )
            mode_scale = interpolate_param(
                self.mode_scale_start, self.mode_scale, global_step, self.warmup_steps
            )

        if self.noise_read_key is not None:
            noises = data_dict[self.noise_read_key]
            if self.detach_noise:
                noises = [noise.clone().detach() for noise in noises]
        else:
            noises = [torch.randn_like(img) for img in clean_images]

        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=len(clean_images),
            logit_mean=logit_mean,
            logit_std=logit_std,
            mode_scale=mode_scale,
        )

        self.noise_scheduler.set_timesteps(
            self.noise_scheduler.config.num_train_timesteps, device=device
        )

        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices]

        sigmas = get_sigmas(self.noise_scheduler, timesteps, n_dim=1)

        # Optionally shift sigmas according to image resolution
        if self.base_resolution is not None:
            resolutions = torch.tensor(
                [[img.shape[-2], img.shape[-1]] for img in clean_images],
                device=device,
            )
            sigmas = resolution_dependent_sigma_shift(
                sigmas, resolutions, base_resolution=self.base_resolution
            )

        noised_images_list = [
            sigma * noise + (1.0 - sigma) * clean_img
            for sigma, noise, clean_img in zip(sigmas, noises, clean_images)
        ]

        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=self.weighting_scheme, sigmas=sigmas
        )

        data_dict[self.noised_images_write_key] = noised_images_list
        data_dict[self.noise_write_key] = noises
        data_dict[self.timesteps_write_key] = timesteps
        data_dict[self.sigmas_write_key] = sigmas
        data_dict[self.weighting_write_key] = weighting

        return data_dict


class MinRFNoiseModule(nn.Module):
    """
    Adapted from https://github.com/cloneofsimo/minRF
    """

    def __init__(
        self,
        clean_images_read_key: str,
        noised_images_write_key: str,
        timesteps_write_key: str,
        sigmas_write_key: str,
        ln: bool = True,  # log-normal
        stratisfied: bool = False,
        mode_scale: float = 0.0,  # Only applicable for ln=False
        noise_write_key: str = "flow_noise",
        noise_read_key: Optional[str] = None,
    ):
        super().__init__()
        self.clean_images_read_key = clean_images_read_key
        self.noised_images_write_key = noised_images_write_key
        self.timesteps_write_key = timesteps_write_key
        self.sigmas_write_key = sigmas_write_key
        self.noise_write_key = noise_write_key
        self.noise_read_key = noise_read_key

        self.ln = ln
        self.stratisfied = stratisfied
        self.mode_scale = mode_scale

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any], global_step: int = None) -> Dict[str, Any]:
        clean_images = data_dict[self.clean_images_read_key]
        device = clean_images[0].device
        batch_size = len(clean_images)

        if self.noise_read_key is not None:
            noises = data_dict[self.noise_read_key]
        else:
            noises = [torch.randn_like(img) for img in clean_images]

        if self.ln:
            if self.stratisfied:
                # stratified sampling of normals
                # first stratified sample from uniform
                quantiles = torch.linspace(0, 1, batch_size + 1).to(device)
                z = quantiles[:-1] + torch.rand((batch_size,)).to(device) / batch_size
                # now transform to normal
                z = torch.erfinv(2 * z - 1) * math.sqrt(2)
                sigmas = torch.sigmoid(z)
            else:
                nt = torch.randn((batch_size,)).to(device)
                sigmas = torch.sigmoid(nt)
        else:
            sigmas = torch.rand((batch_size,)).to(device)
            if self.mode_scale != 0.0:
                sigmas = (
                    1
                    - sigmas
                    - self.mode_scale * (torch.cos(math.pi * sigmas / 2) ** 2 - 1 + sigmas)
                )

        noised_images_list = [
            sigma * noise + (1.0 - sigma) * clean_img
            for sigma, noise, clean_img in zip(sigmas, noises, clean_images)
        ]

        data_dict[self.noised_images_write_key] = noised_images_list
        data_dict[self.noise_write_key] = noises
        data_dict[self.sigmas_write_key] = sigmas
        data_dict[self.timesteps_write_key] = sigmas

        return data_dict


class FullNoiseModule(nn.Module):
    """
    This module always samples full noise (or 0s if zero_noise=True).
    It is an utility class useful for implementing the mock flow architecture that resembles MSE decoder.
    User needs to make sure to set the MinRFLoss (or another flow loss) correctly.
    For MinRFLoss: no preconditioning + standard MSE objective.
    """

    def __init__(
        self,
        clean_images_read_key: str,
        noised_images_write_key: str,
        timesteps_write_key: str,
        sigmas_write_key: str,
        zero_noise: bool,  # to set noise to 0 for MSE decoder
        noise_write_key: str = "flow_noise",
        noise_read_key: Optional[str] = None,
    ):
        super().__init__()
        self.clean_images_read_key = clean_images_read_key
        self.noised_images_write_key = noised_images_write_key
        self.timesteps_write_key = timesteps_write_key
        self.sigmas_write_key = sigmas_write_key
        self.noise_write_key = noise_write_key
        self.noise_read_key = noise_read_key

        self.zero_noise = zero_noise

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any], global_step: int = None) -> Dict[str, Any]:
        clean_images = data_dict[self.clean_images_read_key]
        device = clean_images[0].device
        batch_size = len(clean_images)

        if self.noise_read_key is not None:
            # sanity check; we should not go in here
            raise NotImplementedError
            noises = data_dict[self.noise_read_key]
        else:
            if self.zero_noise:
                # this it to model MSE objective
                noises = [torch.zeros_like(img) for img in clean_images]
            else:
                noises = [torch.randn_like(img) for img in clean_images]

        # full noise
        sigmas = torch.ones((batch_size,)).to(device)

        # just to create copies
        noised_images_list = [sigma * noise for sigma, noise in zip(sigmas, noises)]

        data_dict[self.noised_images_write_key] = noised_images_list
        data_dict[self.noise_write_key] = noises
        data_dict[self.sigmas_write_key] = sigmas
        data_dict[self.timesteps_write_key] = sigmas

        return data_dict


class PrefixMinRFNoiseModule(nn.Module):
    """
    Temporal version of MinRFNoiseModule for video data.
    Adapted from https://github.com/cloneofsimo/minRF

    This module handles input in [T, *] format where T is the temporal dimension
    and * represents any number of spatial dimensions. It supports prefix preservation
    where the first prefix_len frames can be kept as original signal with probability
    keep_prefix_prob.

    Args:
        clean_images_read_key: Dictionary key to read clean images/videos
        noised_images_write_key: Dictionary key to write noised images/videos
        timesteps_write_key: Dictionary key to write timesteps
        sigmas_write_key: Dictionary key to write sigmas
        prefix_len: Number of frames at the beginning to potentially preserve.
            Can be an int (for backward compatibility) or a list of ints for multiple conditioning lengths.
        keep_prefix_prob: Probability of keeping the prefix frames as original signal.
            Can be a float (for backward compatibility) or a list of floats corresponding to each prefix_len.
            If prefix_len is a list, probabilities must sum to 1.0.
        ln: Whether to use log-normal distribution for sigma sampling
        stratisfied: Whether to use stratified sampling
        mode_scale: Scale parameter for mode-based sampling (only applicable for ln=False)
        noise_write_key: Dictionary key to write noise
        noise_read_key: Optional dictionary key to read pre-generated noise
        loss_mask_write_key: Dictionary key to write loss mask indicating which frames to include in loss computation
        prefix_len_write_key: Dictionary key to write the actual prefix length used for each sample
        cond_sigma_min: Minimum sigma value for conditioning frames noise (default: 0.0)
        cond_sigma_max: Maximum sigma value for conditioning frames noise (default: 0.02)
    """

    def __init__(
        self,
        clean_images_read_key: str,
        noised_images_write_key: str,
        timesteps_write_key: str,
        sigmas_write_key: str,
        prefix_len: Optional[Union[int, list[int]]] = None,
        keep_prefix_prob: Optional[Union[float, list[float]]] = None,
        ln: bool = True,  # log-normal
        stratisfied: bool = False,
        mode_scale: float = 0.0,  # Only applicable for ln=False
        noise_write_key: str = "flow_noise",
        noise_read_key: Optional[str] = None,
        loss_mask_write_key: str = "loss_mask",
        prefix_len_write_key: str = "prefix_len",
        cond_sigma_range: tuple[float, float] = (0.0, 0.02),
        cond_noise_prob: float = 0.5,
    ):
        super().__init__()
        self.clean_images_read_key = clean_images_read_key
        self.noised_images_write_key = noised_images_write_key
        self.timesteps_write_key = timesteps_write_key
        self.sigmas_write_key = sigmas_write_key
        self.noise_write_key = noise_write_key
        self.noise_read_key = noise_read_key
        self.loss_mask_write_key = loss_mask_write_key
        self.prefix_len_write_key = prefix_len_write_key

        # Handle both old (single value) and new (list) API
        if prefix_len is None or prefix_len == 0:
            self.prefix_lengths = []
            self.prefix_probs = []
        elif isinstance(prefix_len, int):
            # for backward compatibility with old configs
            assert prefix_len > 0, f"{prefix_len=}, you are probably doing something wrong..."
            self.prefix_lengths = [0, prefix_len]
            assert 0.0 <= keep_prefix_prob <= 1.0, f"{keep_prefix_prob=} must be <= 1.0 and >= 0.0"
            self.prefix_probs = (
                [1 - keep_prefix_prob, keep_prefix_prob] if prefix_len > 0 else [1.0]
            )
        elif isinstance(prefix_len, Iterable):
            self.prefix_lengths = [l for l in prefix_len]
            assert isinstance(
                keep_prefix_prob, Iterable
            ), "When prefix_len is a list, keep_prefix_prob must also be a list"
            self.prefix_probs = [p for p in keep_prefix_prob]
            # Validate that probabilities sum to 1.0 (with some tolerance)
            assert abs(sum(self.prefix_probs) - 1.0) < 1e-6, (
                f"keep_prefix_prob must sum to 1.0, got {self.prefix_probs=}. "
                f"prefix_len: {prefix_len}, keep_prefix_prob: {keep_prefix_prob}"
            )
        else:
            raise ValueError("prefix_len must be either an int or a list of ints if provided")

        assert len(self.prefix_lengths) == len(self.prefix_probs), (
            f"prefix_len and keep_prefix_prob must have the same length, "
            f"got {len(self.prefix_lengths)} and {len(self.prefix_probs)}"
        )

        self.ln = ln
        self.stratisfied = stratisfied
        self.mode_scale = mode_scale

        assert (
            len(cond_sigma_range) == 2
        ), f"cond_sigma_range must be a tuple of (min, max), got {cond_sigma_range}"
        assert (
            0.0 <= cond_sigma_range[0] <= cond_sigma_range[1] <= 1.0
        ), f"cond_sigma_range must be within [0.0, 1.0], got {cond_sigma_range}"
        self.cond_sigma_range = cond_sigma_range
        self.cond_noise_prob = cond_noise_prob

    @torch.compiler.disable
    def forward(self, data_dict: Dict[str, Any], global_step: int = None) -> Dict[str, Any]:
        clean_images = data_dict[self.clean_images_read_key]
        device = clean_images[0].device
        batch_size = len(clean_images)

        if self.noise_read_key is not None:
            noises = data_dict[self.noise_read_key]
        else:
            noises = [torch.randn_like(img) for img in clean_images]

        # Generate sigma values for each sample in the batch
        if self.ln:
            if self.stratisfied:
                # stratified sampling of normals
                # first stratified sample from uniform
                quantiles = torch.linspace(0, 1, batch_size + 1).to(device)
                z = quantiles[:-1] + torch.rand((batch_size,)).to(device) / batch_size
                # now transform to normal
                z = torch.erfinv(2 * z - 1) * math.sqrt(2)
                sigmas = torch.sigmoid(z)
            else:
                nt = torch.randn((batch_size,)).to(device)
                sigmas = torch.sigmoid(nt)
        else:
            sigmas = torch.rand((batch_size,)).to(device)
            if self.mode_scale != 0.0:
                sigmas = (
                    1
                    - sigmas
                    - self.mode_scale * (torch.cos(math.pi * sigmas / 2) ** 2 - 1 + sigmas)
                )

        noised_images_list = []
        loss_masks = []
        prefix_lens = []

        for sigma, noise, clean_img in zip(sigmas, noises, clean_images):
            # VAE latents come as [B=1, C, T, H, W] from VidTok, we need temporal dimension
            assert (
                clean_img.ndim == 5
            ), f"Expected VAE latents format [B, C, T, H, W], got shape {clean_img.shape}"
            assert (
                clean_img.shape[0] == 1
            ), f"Expected batch size 1 for VAE latents, got {clean_img.shape[0]}"
            T = clean_img.shape[2]  # Get temporal dimension

            # Apply noise to the entire temporal sequence
            noised_img = sigma * noise + (1.0 - sigma) * clean_img

            # Initialize loss mask (True = compute loss, False = skip loss)
            loss_mask = torch.ones(T, dtype=torch.bool, device=device)
            actual_prefix_len = 0

            # Sample a prefix length according to the specified probabilities
            if len(self.prefix_lengths) > 0:
                # Sample which prefix length to use based on probabilities (use numpy to avoid GPU overhead)
                prefix_idx = np.random.choice(len(self.prefix_probs), p=self.prefix_probs)
                actual_prefix_len = self.prefix_lengths[prefix_idx]

                if actual_prefix_len > 0:
                    # Ensure prefix length doesn't exceed sequence length
                    assert actual_prefix_len <= T, f"{actual_prefix_len=} > {T=}"

                    # Sample conditioning noise sigma from uniform distribution
                    if np.random.rand() < self.cond_noise_prob:
                        cond_sigma = (
                            np.random.rand() * (self.cond_sigma_range[1] - self.cond_sigma_range[0])
                            + self.cond_sigma_range[0]
                        )
                    else:
                        cond_sigma = 0.0

                    # Apply conditioning with sampled noise level along temporal dimension (dim=2)
                    noised_img[:, :, :actual_prefix_len] = (
                        cond_sigma * noise[:, :, :actual_prefix_len]
                        + (1.0 - cond_sigma) * clean_img[:, :, :actual_prefix_len]
                    )

                    # Mark conditioning frames as excluded from loss computation
                    loss_mask[:actual_prefix_len] = False

            noised_images_list.append(noised_img)
            loss_masks.append(loss_mask)
            prefix_lens.append(actual_prefix_len)

        data_dict[self.noised_images_write_key] = noised_images_list
        data_dict[self.noise_write_key] = noises
        data_dict[self.sigmas_write_key] = sigmas
        data_dict[self.timesteps_write_key] = sigmas
        data_dict[self.loss_mask_write_key] = loss_masks
        data_dict[self.prefix_len_write_key] = prefix_lens

        return data_dict
