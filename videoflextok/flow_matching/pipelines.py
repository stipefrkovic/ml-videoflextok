# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. and EPFL. All Rights Reserved.
import copy
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from diffusers.schedulers.scheduling_utils import SchedulerMixin

import torch

from tqdm import tqdm

from videoflextok.utils.misc import to_2tuple

from .cfg_utils import CFGRescaler, MomentumBuffer, classifier_free_guidance, normalized_guidance
from .utils import resolution_dependent_sigma_shift

__all__ = ["AnyResPipelineCond", "AnyResPipelineCondV2", "MinRFPipeline"]


class AnyResPipelineCond:
    """Pipeline for conditional image generation with FlowMatchEulerDiscreteScheduler scheduler.
    Expects same read/write keys as FlexibleRegisterDecoder.

    Args:
        model: The conditional flow matching model.
        scheduler: A FlowMatchEulerDiscreteScheduler scheduler
        base_resolution: Base resolution to shift noise
    """

    def __init__(
        self,
        model: torch.nn.Module,
        scheduler: SchedulerMixin,
        base_resolution: Optional[int] = None,
        noise_read_key: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.base_resolution = base_resolution
        self.noise_read_key = noise_read_key

    @torch.no_grad()
    def __call__(
        self,
        data_dict: Dict[str, Any],
        generator: Optional[torch.Generator] = None,
        timesteps: Optional[int] = None,
        image_sizes: Optional[Union[int, List[Tuple[int, int]]]] = None,
        verbose: bool = True,
        **ignore_kwargs,
    ) -> Dict[str, Any]:

        if image_sizes is None:
            image_sizes = data_dict[self.model.target_sizes_read_key]
        elif isinstance(image_sizes, int):
            batch_size = len(data_dict[self.model.latents_read_key])
            image_sizes = [to_2tuple(image_sizes) for _ in range(batch_size)]
        assert isinstance(image_sizes, list)
        batch_size = len(image_sizes)

        # Sample Gaussian noise to begin loop or read it from data_dict
        if self.noise_read_key is not None:
            images_list = data_dict[self.noise_read_key]
        else:
            images_list = [
                torch.randn(
                    (1, self.model.out_channels, h, w),
                    generator=generator,
                    device=self.model.device,
                )
                for h, w in image_sizes
            ]

        # Set step values
        timesteps_original = self.scheduler.config.num_train_timesteps
        timesteps = self.scheduler.config.num_train_timesteps if timesteps is None else timesteps
        self.scheduler.set_timesteps(timesteps)

        # Override sigmas if using resolution dependent shift
        sigmas_original = self.scheduler.sigmas.clone()
        if self.base_resolution is not None:
            # TODO: Make res sigma shift work with different image sizes
            # self.scheduler.sigmas = resolution_dependent_sigma_shift(
            #     self.scheduler.sigmas, torch.tensor(image_sizes), self.base_resolution
            # )
            raise NotImplementedError()

        if verbose:
            pbar = tqdm(total=len(self.scheduler.timesteps))

        for t in self.scheduler.timesteps:
            # 1. Predict noise model_output
            timesteps = t * torch.ones(batch_size, device=self.model.device)

            data_dict[self.model.timesteps_read_key] = timesteps
            data_dict[self.model.noised_images_read_key] = images_list

            data_dict = self.model(data_dict)

            model_output_list = data_dict[self.model.reconst_write_key]

            # 2. Compute previous image: x_t -> t_t-1
            with torch.amp.autocast("cuda", enabled=False):
                images_list_next = []
                for model_output, image in zip(model_output_list, images_list):
                    image_next = self.scheduler.step(
                        model_output.float(), t, image, generator=generator
                    ).prev_sample
                    images_list_next.append(image_next)
                    self.scheduler._step_index = None  # Stop scheduler from automatically incrementing idx every time step is run
                images_list = images_list_next

            if verbose:
                pbar.update()
        if verbose:
            pbar.close()

        self.scheduler.set_timesteps(timesteps_original)  # Reset timesteps to what it was before
        self.scheduler.sigmas = sigmas_original

        data_dict[self.model.reconst_write_key] = images_list
        return data_dict


class AnyResPipelineCondV2:
    """
    TODO(roman-bachmann): docstring
    """

    def __init__(
        self,
        model: torch.nn.Module,
        scheduler: SchedulerMixin,
        base_resolution: Optional[int] = None,
        noise_read_key: Optional[str] = None,
        target_sizes_read_key: Optional[str] = None,
        latents_read_key: Optional[str] = None,
        timesteps_read_key: Optional[str] = None,
        noised_images_read_key: Optional[str] = None,
        reconst_write_key: Optional[str] = None,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.base_resolution = base_resolution
        self.noise_read_key = noise_read_key
        self.target_sizes_read_key = target_sizes_read_key
        self.latents_read_key = latents_read_key
        self.timesteps_read_key = timesteps_read_key
        self.noised_images_read_key = noised_images_read_key
        self.reconst_write_key = reconst_write_key
        self.out_channels = out_channels

    @torch.no_grad()
    def __call__(
        self,
        data_dict: Dict[str, Any],
        generator: Optional[torch.Generator] = None,
        timesteps: Optional[int] = None,
        image_sizes: Optional[Union[int, List[Tuple[int, int]]]] = None,
        verbose: bool = True,
        guidance_scale: Union[float, Callable] = 1.0,
        cfg_rescaler: Optional[CFGRescaler] = None,
    ) -> Dict[str, Any]:

        do_cfg = callable(guidance_scale) or guidance_scale != 1.0

        if image_sizes is None:
            image_sizes = data_dict[self.target_sizes_read_key]
        elif isinstance(image_sizes, int):
            batch_size = len(data_dict[self.latents_read_key])
            image_sizes = [to_2tuple(image_sizes) for _ in range(batch_size)]
        assert isinstance(image_sizes, list)
        batch_size = len(image_sizes)

        # Sample Gaussian noise to begin loop or read it from data_dict
        if self.noise_read_key is not None:
            images_list = data_dict[self.noise_read_key]
        else:
            images_list = [
                torch.randn(
                    (1, self.out_channels, h, w),
                    generator=generator,
                    device=self.model.device,
                )
                for h, w in image_sizes
            ]

        # Set step values
        timesteps_original = self.scheduler.config.num_train_timesteps
        timesteps = self.scheduler.config.num_train_timesteps if timesteps is None else timesteps
        self.scheduler.set_timesteps(timesteps)

        # Override sigmas if using resolution dependent shift
        sigmas_original = self.scheduler.sigmas.clone()
        if self.base_resolution is not None:
            # TODO: Make res sigma shift work with different image sizes
            # self.scheduler.sigmas = resolution_dependent_sigma_shift(
            #     self.scheduler.sigmas, torch.tensor(image_sizes), self.base_resolution
            # )
            raise NotImplementedError()

        if verbose:
            pbar = tqdm(total=len(self.scheduler.timesteps))

        for t in self.scheduler.timesteps:
            # 1. Predict noise model_output
            timesteps = t * torch.ones(batch_size, device=self.model.device)

            data_dict[self.timesteps_read_key] = timesteps
            data_dict[self.noised_images_read_key] = images_list

            # 1.1 Conditional forward pass
            data_dict_cond = copy.deepcopy(data_dict)
            data_dict_cond = self.model(data_dict_cond)
            model_output_list = data_dict_cond[self.reconst_write_key]

            # 1.2 (Optional) unconditional forward pass
            if do_cfg:
                if callable(guidance_scale):
                    guidance_scale_value = guidance_scale(
                        t / self.scheduler.config.num_train_timesteps
                    )
                else:
                    guidance_scale_value = guidance_scale

                data_dict_uncond = copy.deepcopy(data_dict)
                data_dict_uncond["eval_dropout_mask"] = [True] * len(model_output_list)
                data_dict_uncond = self.model(data_dict_uncond)
                model_output_list_uncond = data_dict_uncond[self.reconst_write_key]

                model_output_list_cfg = []
                for output_cond, output_uncond in zip(model_output_list, model_output_list_uncond):
                    output_cfg = output_uncond + guidance_scale_value * (
                        output_cond - output_uncond
                    )
                    if cfg_rescaler is not None:
                        output_cfg = cfg_rescaler(output_cfg, output_cond, output_uncond)
                    model_output_list_cfg.append(output_cfg)

                model_output_list = model_output_list_cfg

            # 2. Compute previous image: x_t -> t_t-1
            with torch.amp.autocast("cuda", enabled=False):
                images_list_next = []
                for model_output, image in zip(model_output_list, images_list):
                    image_next = self.scheduler.step(
                        model_output.float(), t, image, generator=generator
                    ).prev_sample
                    images_list_next.append(image_next)
                    self.scheduler._step_index = None  # Stop scheduler from automatically incrementing idx every time step is run
                images_list = images_list_next

            if verbose:
                pbar.update()
        if verbose:
            pbar.close()

        self.scheduler.set_timesteps(timesteps_original)  # Reset timesteps to what it was before
        self.scheduler.sigmas = sigmas_original

        data_dict[self.reconst_write_key] = images_list
        return data_dict


class MinRFPipeline:
    """
    TODO(roman-bachmann): docstring
    """

    def __init__(
        self,
        model: torch.nn.Module,
        noise_read_key: Optional[str] = None,
        target_sizes_read_key: Optional[str] = None,
        latents_read_key: Optional[str] = None,
        timesteps_read_key: Optional[str] = None,
        noised_images_read_key: Optional[str] = None,
        reconst_write_key: Optional[str] = None,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.noise_read_key = noise_read_key
        self.target_sizes_read_key = target_sizes_read_key
        self.latents_read_key = latents_read_key
        self.timesteps_read_key = timesteps_read_key
        self.noised_images_read_key = noised_images_read_key
        self.reconst_write_key = reconst_write_key
        self.out_channels = out_channels

    @torch.no_grad()
    def __call__(
        self,
        data_dict: Dict[str, Any],
        generator: Optional[torch.Generator] = None,
        timesteps: int = 50,
        image_sizes: Optional[Union[int, List[Tuple[int, int]]]] = None,
        verbose: bool = True,
        guidance_scale: Union[float, Callable] = 1.0,
        cfg_rescaler: Optional[CFGRescaler] = None,
        perform_norm_guidance: bool = False,
    ) -> Dict[str, Any]:

        do_cfg = callable(guidance_scale) or guidance_scale != 1.0

        if image_sizes is None:
            image_sizes = data_dict[self.target_sizes_read_key]
        elif isinstance(image_sizes, int):
            batch_size = len(data_dict[self.latents_read_key])
            image_sizes = [to_2tuple(image_sizes) for _ in range(batch_size)]
        assert isinstance(image_sizes, list)
        batch_size = len(image_sizes)

        # Sample Gaussian noise to begin loop or read it from data_dict
        if self.noise_read_key is not None:
            images_list = data_dict[self.noise_read_key]
        else:
            images_list = [
                torch.randn(
                    (1, self.out_channels, h, w),
                    generator=generator,
                    device=self.model.device,
                )
                for h, w in image_sizes
            ]

        # Set step values
        dt = 1.0 / timesteps

        if perform_norm_guidance:
            momentum_buffers = [MomentumBuffer(-0.5)] * batch_size

        if verbose:
            pbar = tqdm(total=timesteps)

        for i in range(timesteps, 0, -1):
            t = i / timesteps
            timesteps_tensor = t * torch.ones(batch_size, device=self.model.device)

            data_dict[self.timesteps_read_key] = timesteps_tensor
            data_dict[self.noised_images_read_key] = images_list

            # 1.1 Conditional forward pass
            data_dict_cond = copy.deepcopy(data_dict)
            data_dict_cond = self.model(data_dict_cond)
            model_output_list = data_dict_cond[self.reconst_write_key]

            # 1.2 (Optional) unconditional forward pass
            if do_cfg:
                if callable(guidance_scale):
                    guidance_scale_value = guidance_scale(t)
                else:
                    guidance_scale_value = guidance_scale

                data_dict_uncond = copy.deepcopy(data_dict)
                data_dict_uncond["eval_dropout_mask"] = [True] * len(model_output_list)
                data_dict_uncond = self.model(data_dict_uncond)
                model_output_list_uncond = data_dict_uncond[self.reconst_write_key]

                model_output_list_cfg = []
                for j, (output_cond, output_uncond) in enumerate(
                    zip(model_output_list, model_output_list_uncond)
                ):
                    # TODO(roman-bachmann): Merge these with cfg_rescaler into some more general cfg modules
                    if not perform_norm_guidance:
                        output_cfg = classifier_free_guidance(
                            output_cond, output_uncond, guidance_scale_value
                        )
                    else:
                        output_cfg = normalized_guidance(
                            output_cond,
                            output_uncond,
                            guidance_scale_value,
                            momentum_buffers[j],
                            eta=0.0,
                            norm_threshold=2.5,
                        )
                    if cfg_rescaler is not None:
                        output_cfg = cfg_rescaler(output_cfg, output_cond, output_uncond)
                    model_output_list_cfg.append(output_cfg)

                model_output_list = model_output_list_cfg

            # 2. Compute previous image: x_t -> t_t-1
            with torch.amp.autocast("cuda", enabled=False):
                images_list_next = []
                for model_output, image in zip(model_output_list, images_list):
                    image_next = image - dt * model_output
                    images_list_next.append(image_next)
                images_list = images_list_next

            if verbose:
                pbar.update()
        if verbose:
            pbar.close()

        data_dict[self.reconst_write_key] = images_list
        return data_dict


class VideoMinRFPipeline:
    """
    Pipeline for video generation with optional conditioning prefix support.

    This pipeline supports conditioning on the first k frames of a video sequence,
    ensuring that these frames remain fixed throughout the denoising process.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        noise_read_key: Optional[str] = None,
        target_sizes_read_key: Optional[str] = None,
        latents_read_key: Optional[str] = None,
        timesteps_read_key: Optional[str] = None,
        noised_videos_read_key: Optional[str] = None,
        reconst_write_key: Optional[str] = None,
        out_channels: Optional[int] = None,
        zero_noise: bool = False,  # to replicate 1-step MSE decoder
        cond_vae_latents_key: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.noise_read_key = noise_read_key
        self.target_sizes_read_key = target_sizes_read_key
        self.latents_read_key = latents_read_key
        self.timesteps_read_key = timesteps_read_key
        self.noised_videos_read_key = noised_videos_read_key
        self.reconst_write_key = reconst_write_key
        self.out_channels = out_channels
        self.cond_vae_latents_key = cond_vae_latents_key

        self.zero_noise = zero_noise

    def _apply_vae_latents_conditioning(
        self,
        videos_list: List[torch.FloatTensor],
        condition_vae_latents_list: List[torch.FloatTensor],
    ) -> List[torch.FloatTensor]:
        """
        Replace the first k frames of each video with conditioning VAE latents.

        Args:
            videos_list: List of video tensors to modify
            condition_vae_latents_list: List of conditioning VAE latent tensors

        Returns:
            List of videos with first k frames replaced by conditioning VAE latents
        """
        conditioned_videos = []
        for video, cond_latents in zip(videos_list, condition_vae_latents_list):
            # video: (1, out_channels, t, h, w)
            # cond_latents: (1, out_channels, k, h, w)
            k = cond_latents.shape[2]

            # Create a copy to avoid modifying the original
            conditioned_video = video.clone()
            # Replace first k frames with conditioning VAE latents
            conditioned_video[:, :, :k] = cond_latents
            conditioned_videos.append(conditioned_video)

        return conditioned_videos

    def _apply_noise_vae_latents_conditioning(
        self,
        videos_list: List[torch.FloatTensor],
        condition_vae_latents_list: List[torch.FloatTensor],
        noises_list: List[torch.FloatTensor],
        timestep: float,
    ) -> List[torch.FloatTensor]:
        """
        Replace the first k frames of each video with the noised version of the conditioning VAE latents.

        Args:
            videos_list: List of video tensors to modify
            condition_vae_latents_list: List of conditioning VAE latent tensors
            noise: List of noise tensors to apply to the conditioning latents
            timestep: Current timestep for noise scaling

        Returns:
            List of videos with first k frames replaced by noised conditioning VAE latents
        """
        print("====> Applying NOISED VAE latents conditioning at timestep:", timestep)
        conditioned_videos = []
        for video, cond_latents, noise_tensor in zip(
            videos_list, condition_vae_latents_list, noises_list
        ):
            # video: (1, out_channels, t, h, w)
            # cond_latents: (1, out_channels, k, h, w)
            k = cond_latents.shape[2]

            # Create a copy to avoid modifying the original
            conditioned_video = video.clone()
            # Compute noised latents
            sigma_t = timestep  # Assuming timestep is already scaled appropriately
            # Replace first k frames with noised VAE latents
            conditioned_video[:, :, :k] = (1 - sigma_t) * cond_latents + sigma_t * noise_tensor[
                :, :, :k
            ]
            conditioned_videos.append(conditioned_video)

        return conditioned_videos

    @torch.no_grad()
    def __call__(
        self,
        data_dict: Dict[str, Any],
        generator: Optional[torch.Generator] = None,
        timesteps: int = 50,
        video_sizes: Optional[List[Tuple[int, int, int]]] = None,
        noise_list: Optional[List[torch.Tensor]] = None,
        verbose: bool = True,
        guidance_scale: Union[float, Callable] = 1.0,
        cfg_rescaler: Optional[CFGRescaler] = None,
        perform_norm_guidance: bool = False,
        momentum: float = 0.0,
        eta: float = 0.0,
        norm_threshold: float = 0.6,
        cond_type: Literal["clean", "sdedit"] = "clean",
    ) -> Dict[str, Any]:
        """
        Generate videos with optional conditioning prefix support.

        Args:
            data_dict: Dictionary containing input data. If cond_vae_latents_key is provided,
                      data_dict[cond_vae_latents_key] should contain conditioning frames.
            generator: Random number generator for reproducibility
            timesteps: Number of denoising timesteps
            video_sizes: Target video sizes as (t, h, w) tuples
            verbose: Whether to show progress bar
            guidance_scale: Classifier-free guidance scale (1.0 = no guidance)
            cfg_rescaler: Optional CFG rescaler
            perform_norm_guidance: Whether to use normalized guidance
            momentum: Momentum parameter for normalized guidance
            eta: ETA parameter for normalized guidance
            norm_threshold: Norm threshold for normalized guidance
            cond_type: Type of conditioning to apply ("clean" or "sdedit")

        Data dict conditioning:
            If self.cond_vae_latents_key is provided, data_dict[self.cond_vae_latents_key] should contain
            conditioning frames in one of these formats:
            - Single tensor: (batch_size, out_channels, k, h, w)
            - List of tensors: [(1, out_channels, k, h, w), ...]

            Sanity checks performed:
            - Spatial dimensions (h, w) must match video_sizes
            - k (number of conditioning frames) must not exceed total time t
            - Channel dimensions must match pipeline configuration

        Returns:
            Dictionary with generated videos where first k frames are exactly the conditioning frames
        """

        do_cfg = callable(guidance_scale) or guidance_scale != 1.0

        if video_sizes is None:
            video_sizes = data_dict[self.target_sizes_read_key]
        elif isinstance(video_sizes, int):
            batch_size = len(data_dict[self.latents_read_key])
            video_sizes = [to_2tuple(video_sizes) for _ in range(batch_size)]
        elif isinstance(video_sizes, (list, tuple)) and (
            not isinstance(video_sizes[0], (list, tuple))
        ):
            batch_size = len(data_dict[self.latents_read_key])
            video_sizes = [video_sizes[:] for _ in range(batch_size)]
        assert isinstance(video_sizes, list)
        batch_size = len(video_sizes)

        # Sample Gaussian noise to begin loop or read it from data_dict
        if self.noise_read_key is not None:
            videos_list = data_dict[self.noise_read_key]
        else:
            if self.zero_noise:
                videos_list = [
                    torch.zeros(
                        (1, self.out_channels, t, h, w),
                        device=self.model.device,
                    )
                    for t, h, w in video_sizes
                ]
            elif noise_list is not None:
                videos_list = [noise_list[i].to(self.model.device) for i in range(batch_size)]
                assert len(videos_list) == batch_size
            else:
                videos_list = [
                    torch.randn(
                        (1, self.out_channels, t, h, w),
                        generator=generator,
                        device=self.model.device,
                    )
                    for t, h, w in video_sizes
                ]
        noise_list = copy.deepcopy(videos_list)

        # Get conditioning VAE latents if provided
        condition_vae_latents_list = None

        # Handle VAE latents conditioning
        if self.cond_vae_latents_key is not None and self.cond_vae_latents_key in data_dict:
            condition_vae_latents_list = data_dict[self.cond_vae_latents_key]

        # Set step values
        dt = 1.0 / timesteps

        if perform_norm_guidance:
            momentum_buffers = [MomentumBuffer(momentum)] * batch_size

        if verbose:
            pbar = tqdm(total=timesteps)

        for i in range(timesteps, 0, -1):
            t = i / timesteps
            timesteps_tensor = t * torch.ones(batch_size, device=self.model.device)

            data_dict[self.timesteps_read_key] = timesteps_tensor

            # Apply VAE latents conditioning before model forward pass
            if condition_vae_latents_list is not None:
                if cond_type == "clean":
                    videos_list = self._apply_vae_latents_conditioning(
                        videos_list, condition_vae_latents_list
                    )
                elif cond_type == "sdedit":
                    videos_list = self._apply_noise_vae_latents_conditioning(
                        videos_list,
                        condition_vae_latents_list,
                        noise_list,
                        timestep=t,
                    )
                else:
                    raise ValueError(f"Unknown cond_type: {cond_type}")

            data_dict[self.noised_videos_read_key] = videos_list

            # 1.1 Conditional forward pass
            data_dict_cond = copy.deepcopy(data_dict)
            data_dict_cond = self.model(data_dict_cond)
            model_output_list = data_dict_cond[self.reconst_write_key]

            # 1.2 (Optional) unconditional forward pass
            if do_cfg:
                if callable(guidance_scale):
                    guidance_scale_value = guidance_scale(t)
                else:
                    guidance_scale_value = guidance_scale

                data_dict_uncond = copy.deepcopy(data_dict)
                data_dict_uncond["eval_dropout_mask"] = [True] * len(model_output_list)
                # Note: data_dict_uncond already has conditioned videos_list from above
                data_dict_uncond = self.model(data_dict_uncond)
                model_output_list_uncond = data_dict_uncond[self.reconst_write_key]

                model_output_list_cfg = []
                for j, (output_cond, output_uncond) in enumerate(
                    zip(model_output_list, model_output_list_uncond)
                ):
                    # TODO(roman-bachmann): Merge these with cfg_rescaler into some more general cfg modules
                    if not perform_norm_guidance:
                        output_cfg = classifier_free_guidance(
                            output_cond, output_uncond, guidance_scale_value
                        )
                    else:
                        output_cfg = normalized_guidance(
                            output_cond,
                            output_uncond,
                            guidance_scale_value,
                            momentum_buffers[j],
                            eta=eta,
                            norm_threshold=norm_threshold,
                        )
                    if cfg_rescaler is not None:
                        output_cfg = cfg_rescaler(output_cfg, output_cond, output_uncond)
                    model_output_list_cfg.append(output_cfg)

                model_output_list = model_output_list_cfg

            # 2. Compute previous video: x_t -> t_t-1
            with torch.amp.autocast("cuda", enabled=False):
                videos_list_next = []
                for model_output, video in zip(model_output_list, videos_list):
                    video_next = video - dt * model_output
                    videos_list_next.append(video_next)
                videos_list = videos_list_next

            if verbose:
                pbar.update()

        if verbose:
            pbar.close()

        # Ensure final output contains exact conditioning VAE latents
        if condition_vae_latents_list is not None:
            if cond_type == "clean":
                videos_list = self._apply_vae_latents_conditioning(
                    videos_list, condition_vae_latents_list
                )
            elif cond_type == "sdedit":
                pass
            else:
                raise ValueError(f"Unknown cond_type: {cond_type}")

        data_dict[self.reconst_write_key] = videos_list
        return data_dict
