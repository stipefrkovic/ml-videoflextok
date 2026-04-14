# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. and EPFL. All Rights Reserved.
import copy
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Union

import einops
from huggingface_hub import PyTorchModelHubMixin

import torch
from torch import nn

from hydra.utils import instantiate

from .utils.checkpoint import _sanitize_hydra_config


class VideoFlexTok(nn.Module):
    """Unified resampler that combines VAE, encoder, decoder, and bottleneck regularizer.
    The VAE encodes videos into latents on which the resampler operates. The resampler
    consists of an encoder and a decoder with a discrete or continuous regularizer as the
    bottleneck.

    This model supports both standard reconstruction and flow matching-based generation.
    When flow_matching_noise_module and pipeline are provided, the decoder operates as a
    flow matching model during training and inference.

    Args:
        vae: Video VAE that contains an encode and decode function.
        encoder: Resampler encoder model
        decoder: Resampler decoder model
        regularizer: Bottleneck regularizer between the encoder and decoder, FSQ in this case.
        flow_matching_noise_module: The flow matching noise module that adds noise to the latents during training.
        pipeline: Optional. The flow matching pipeline module that performs the decoder denoising
            during inference.
    """

    def __init__(
        self,
        vae: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        regularizer: nn.Module,
        flow_matching_noise_module: Optional[nn.Module] = None,
        pipeline: partial = None,
    ):
        super().__init__()
        self.vae = vae
        self.encoder = encoder
        self.decoder = decoder
        self.regularizer = regularizer
        self.flow_matching_noise_module = flow_matching_noise_module
        self.pipeline = pipeline(model=self.decoder)
        self.cond_read_key = "cond_frames"
        self.cond_vae_latents_key = "cond_vae_latents"

        self.token_write_key = self.regularizer.tokens_write_key
        self.quants_write_key = self.regularizer.quants_write_key
        self.image_write_key = self.vae.images_reconst_write_key

    def init_weights_muP(self):
        self.encoder.init_weights_muP()
        self.decoder.init_weights_muP()

    @property
    def temporal_downsample_factor(self) -> int:
        return self.vae.temporal_downsample_factor

    @property
    def is_causal(self) -> int:
        return self.vae.is_causal

    def prepare_data_dict_for_detokenize(
        self,
        token_ids_list: list[torch.Tensor],
        num_keep_tokens_list: Optional[list[int]] = None,
    ) -> Dict[str, Any]:
        """Prepare a datadict for decoding token ids to videos.

        Args:
            token_ids_list: List of token id sequences.
            num_keep_tokens_list: Optional list of number of tokens to keep from the sequence.

        Raises:
            NotImplementedError: Unsupported decoder type encountered.

        Returns:
            Data dict for decoding into videos.
        """

        token_ids = torch.cat(token_ids_list, dim=0)
        quant = self.regularizer.indices_to_embedding(token_ids)
        quant_list = torch.split(quant, 1, dim=0)

        tmp_data_dict = {self.quants_write_key: quant_list}

        if num_keep_tokens_list is not None:
            dec_nested_dropout = self.decoder.module_dict["dec_nested_dropout"]
            tmp_data_dict[dec_nested_dropout.eval_keep_k_read_key] = num_keep_tokens_list

        return tmp_data_dict

    def tokenize(self, data_dict: dict[str, Any]) -> list[torch.Tensor]:
        """Tokenize the input.

        Args:
            data_dict: Input data containing the following keys:

                - ``'{self.vae_read_key}'`` - list of RGB videos of shape ``[1, C, T, H, W]``.
                and optionally:
                - ``'{self.vae_latents_write_key}'`` - list VAE latents of shape ``[1, c, t, h, w]``.

        Returns:
            List of video token ids. Each of shape like [1, L].
        """
        data_dict = self.encode(data_dict)
        token_ids_list = data_dict[self.token_write_key]
        return token_ids_list

    def detokenize(
        self,
        token_ids_list: list[torch.Tensor],
        vae_video_sizes: list[tuple[int, int, int]] = [(5, 16, 16)],
        num_keep_tokens_list: Optional[list[int]] = None,
        conditioning_frames: Optional[Union[torch.FloatTensor, List[torch.FloatTensor]]] = None,
        **kwargs,
    ) -> list[torch.Tensor]:
        """Decode videos from token ids with optional conditioning frames.

        Args:
            token_ids_list: List of video tokens ids. Each of shape like [1, *].
            vae_video_sizes: The vae grid sizes to decode, list of (t, h, w) tuples.
            num_keep_tokens_list: Optional list of number of tokens to keep from the sequence.
            conditioning_frames: Optional conditioning frames for first k frames of each video.
                               Can be either:
                               - Single tensor: (batch_size, out_channels, k, h, w)
                               - List of tensors: [(1, out_channels, k, h, w), ...]
                               Only used if the resampler supports conditioning (has cond_read_key).

        Returns:
            List of decoded RGB videos. Each of [1, C, T, H, W].
            If conditioning frames were provided, the first k frames will be exactly those frames.
        """
        tmp_data_dict = self.prepare_data_dict_for_detokenize(token_ids_list, num_keep_tokens_list)

        # Add conditioning frames to data_dict if provided and supported
        if conditioning_frames is not None:
            # The decode method will handle encoding the conditioning frames through VAE if needed
            tmp_data_dict[self.cond_read_key] = conditioning_frames

        tmp_data_dict = self.decode(data_dict=tmp_data_dict, video_sizes=vae_video_sizes, **kwargs)

        return tmp_data_dict[self.image_write_key]

    def encode(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        with torch.no_grad():
            data_dict = self.vae.encode(data_dict)
        data_dict = self.encoder(data_dict)
        data_dict = self.regularizer(data_dict)
        return data_dict

    def decode(
        self,
        data_dict: Dict[str, Any],
        **pipeline_kwargs,
    ) -> Dict[str, Any]:
        if self.cond_read_key is not None and self.cond_read_key in data_dict:
            conditioning_frames = data_dict[self.cond_read_key]

            # If VAE latents are not already in data_dict, encode the conditioning frames
            if self.cond_vae_latents_key not in data_dict:
                # Prepare a temporary data_dict for VAE encoding
                if isinstance(conditioning_frames, list):
                    # For list of tensors
                    vae_input_dict = {self.vae.images_read_key: conditioning_frames}
                else:
                    # For single tensor, convert to list of tensors
                    vae_input_dict = {
                        self.vae.images_read_key: list(torch.split(conditioning_frames, 1, dim=0))
                    }

                # Encode conditioning frames through VAE
                with torch.no_grad():
                    vae_output_dict = self.vae.encode(vae_input_dict)

                # Store VAE latents of conditioning frames
                data_dict[self.cond_vae_latents_key] = vae_output_dict[
                    self.vae.vae_latents_write_key
                ]

                if "dec_prefix_masking" in self.decoder.module_dict:
                    # add prefix masking lengths to drop the conditioning frames' registers
                    dec_prefix_masking = self.decoder.module_dict["dec_prefix_masking"]
                    # VAE latents shape: [B, C, T, H, W]
                    cond_latent_frames = data_dict[self.cond_vae_latents_key][0].shape[2]
                    data_dict[dec_prefix_masking.eval_prefix_len_read_key] = [
                        cond_latent_frames for _ in range(len(conditioning_frames))
                    ]

        data_dict = self.pipeline(
            data_dict,
            **pipeline_kwargs,
        )
        with torch.no_grad():
            data_dict = self.vae.decode(data_dict)
        return data_dict

    def autoencode(
        self,
        data_dict: Dict[str, Any],
        **pipeline_kwargs,
    ) -> Dict[str, Any]:
        # Save conditioning frames before encode
        cond_frames = None
        if self.cond_read_key is not None and self.cond_read_key in data_dict:
            cond_frames = data_dict[self.cond_read_key]

        # First encode the data
        data_dict = self.encode(data_dict)  # VAE encode | Encoder | Regularizer

        # Restore conditioning frames for decode
        if cond_frames is not None:
            data_dict[self.cond_read_key] = cond_frames

        # Decode with conditioning
        data_dict = self.decode(
            data_dict,
            **pipeline_kwargs,
        )
        return data_dict

    def forward(
        self, data_dict: Dict[str, Any], global_step: Optional[int] = None
    ) -> Dict[str, Any]:
        data_dict = self.encode(data_dict)  # VAE encode | Encoder | Regularizer
        # Adds noised video, timesteps, sigmas, weights.
        data_dict = self.flow_matching_noise_module(data_dict, global_step)
        data_dict = self.decoder(data_dict)  # Decoder
        return data_dict


class VideoChunkTokenizerWrapper(nn.Module):
    """
    Wrapper that chunks videos into temporal segments, and applies video tokenization to each chunk.

    During encoding, videos are split into chunks with overlap_size_frames overlap, which are then
    tokenized using the underlying video_tokenizer. The resulting token sequences from each chunk
    are concatenated, removing overlap_size_tokens tokens from the overlapping regions.

    Example:
        For a video of 33 frames, chunk_size=17, overlap_size_frames=1
        Chunks:
            Chunk 1: frames 0:17
            Chunk 2: frames 16:33
        Overlap: frame 16
        Tokenization:
            Tokens 1: tokens for frames 0:17
            Tokens 2: tokens for frames 16:33
        Final tokens: Tokens 1 + Tokens 2[1:]

    During decoding, the first chunk is decoded normally. Subsequent chunks are decoded with
    conditioning on the last overlap_size_frames frames from the previously _decoded_ chunk to ensure
    temporal consistency. Overlapping frames are removed from the final output to avoid duplication.

    Notice that during decoding, the conditioning is done on already decoded frames, not on the
    original input frames.

    Example:
        Decoding Chunk 1: frames 0:17
        Decoding Chunk 2: frames 16:33 with conditioning on _decoded_ frames 16:17
        Final video: frames 0:32 (removing duplicate frame 16)

    Args:
        video_tokenizer: The underlying video tokenizer to apply to chunks of shape [C, Tc, H, W].
        chunk_size: Size of temporal chunks to split videos into, in frames.
        num_frames: Total number of frames in the input videos.
        overlap_size_frames: Number of overlapping frames between consecutive chunks during encoding/decoding.
        overlap_size_tokens: Number of overlapping tokens to remove from the token sequences during encoding.
        vae_video_sizes: The vae grid sizes to decode, tuple of (t, h, w). 
    """

    def __init__(
        self,
        video_tokenizer: VideoFlexTok,
        chunk_size: int,
        overlap_size_frames: int,
        vae_video_sizes: tuple[int, int, int],
    ):
        super().__init__()
        self.video_tokenizer = video_tokenizer
        self.chunk_size = chunk_size
        self.vae_video_sizes = vae_video_sizes

        self._chunk_size_tokens = (
            chunk_size - int(self.video_tokenizer.is_causal)
        ) // self.video_tokenizer.temporal_downsample_factor + int(self.video_tokenizer.is_causal)

        self.overlap_size_frames = overlap_size_frames
        self.overlap_size_tokens = 0

        if overlap_size_frames > 0:
            assert (
                overlap_size_frames - self.is_causal
            ) % self.video_tokenizer.temporal_downsample_factor == 0, f"overlap_size_frames {overlap_size_frames} incompatible with temporal_downsample_factor {self.video_tokenizer.temporal_downsample_factor} and is_causal {self.is_causal}."

            assert (
                self.video_tokenizer.cond_read_key is not None
            ), "video_tokenizer must support conditioning (cond_read_key not None) when using overlap_size_frames > 0."

            self.overlap_size_tokens = (
                self.overlap_size_frames - self.is_causal
            ) // self.temporal_downsample_factor + self.is_causal

        # Extract keys from video_tokenizer
        self.images_read_key = self.video_tokenizer.vae.images_read_key
        self.token_write_key = self.video_tokenizer.token_write_key
        self.images_reconst_write_key = self.video_tokenizer.image_write_key

        assert (
            self.overlap_size_frames < self.chunk_size
        ), "overlap_size_frames must be less than chunk_size"

        # Calculate stride for overlapping chunks
        self.stride = (
            self.chunk_size - self.overlap_size_frames
            if self.overlap_size_frames > 0
            else self.chunk_size
        )

    @property
    def temporal_downsample_factor(self) -> int:
        """Temporal downsample factor of the underlying video tokenizer."""
        return self.video_tokenizer.temporal_downsample_factor

    @property
    def is_causal(self) -> bool:
        """Whether the underlying video tokenizer is causal."""
        return self.video_tokenizer.is_causal

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        return self.encode(data_dict)

    def tokenize(self, videos: torch.Tensor) -> list[torch.Tensor]:
        """Tokenize the input videos.

        Args:
            videos: Input videos of shape [B, C, T, H, W].

        Returns:
            List of video token ids. Each of shape like [1, t, k, ...].
        """
        data_dict = {self.images_read_key: list(videos.split(1, dim=0))}
        data_dict = self.encode(data_dict)
        token_ids_list = data_dict[self.token_write_key]
        return token_ids_list

    def detokenize(
        self,
        token_ids_list: list[torch.Tensor],
        num_keep_tokens_list: Optional[list[int]] = None,
        conditioning_frames: Optional[Union[torch.FloatTensor, List[torch.FloatTensor]]] = None,
        **kwargs,
    ) -> list[torch.Tensor]:
        """Decode videos from token ids.

        Args:
            token_ids_list: List of video tokens ids. Each of shape like [1, T, ...].
            conditioning_frames: Optional conditioning frames for the first chunk.
                               Can be either a single tensor (B, C, k, H, W) or list of tensors [(1, C, k, H, W), ...].
            **kwargs: Additional arguments passed to decode.

        Returns:
            List of decoded RGB videos. Each of [1, C, T, H, W].
        """
        # If vae_video_sizes not provided, calculate based on chunk_size
        tmp_data_dict = {self.token_write_key: token_ids_list, "eval_keep_k": num_keep_tokens_list}

        # Add conditioning frames to data_dict if provided
        if conditioning_frames is not None:
            if self.video_tokenizer.cond_read_key is None:
                raise ValueError(
                    "video_tokenizer does not support conditioning (cond_read_key is None). "
                    "Cannot use conditioning_frames."
                )
            tmp_data_dict[self.video_tokenizer.cond_read_key] = conditioning_frames

        tmp_data_dict = self.decode(tmp_data_dict, video_sizes=self.vae_video_sizes, **kwargs)
        return tmp_data_dict[self.images_reconst_write_key]

    def encode(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize videos in data_dict by chunking them into temporal segments.

        data_dict: Dict[str, Any]
            Input data containing the following keys:
                - ``self.images_read_key`` - list of RGB videos of shape ``[1, C, T, H, W]``.
        Returns:
            Dict[str, Any]:
                Updated data_dict with token ids under ``self.token_write_key`` - list of token tensors.
        """
        # Get input videos - list of [1, C, T, H, W]
        videos_list = data_dict[self.images_read_key]

        # Check that all videos have the same number of frames
        assert all(
            videos.shape[2] == videos_list[0].shape[2] for videos in videos_list
        ), "All videos must have the same number of frames"

        # Concatenate all videos into a single batch: [B, C, T, H, W]
        videos = torch.cat(videos_list, dim=0)
        B, C, T, H, W = videos.shape

        if self.overlap_size_frames > 0:
            _num_chunks = (T - self.chunk_size) // self.stride + 1
            # Extract overlapping chunks for all videos at once
            chunks = []
            for i in range(_num_chunks):
                start_frame = i * self.stride
                end_frame = start_frame + self.chunk_size
                chunk = videos[:, :, start_frame:end_frame, :, :]  # [B, C, chunk_size, H, W]
                chunks.append(chunk)

            # Stack chunks: [num_chunks, B, C, chunk_size, H, W]
            video_chunks = torch.stack(chunks, dim=0)
            # Reshape to [B * num_chunks, C, chunk_size, H, W]
            video_chunks: torch.Tensor = einops.rearrange(
                video_chunks, "nc b c tc h w -> (b nc) c tc h w"
            )
        else:
            _num_chunks = T // self.chunk_size
            # Non-overlapping: reshape videos
            # [B, C, (num_chunks * chunk_size), H, W] -> [B * num_chunks, C, chunk_size, H, W]
            video_chunks = einops.rearrange(
                videos,
                "b c (nc tc) h w -> (b nc) c tc h w",
                nc=_num_chunks,
                tc=self.chunk_size,
            )

        # Convert to list of [1, C, chunk_size, H, W] for video_tokenizer
        chunk_list = video_chunks.split(1, dim=0)
        chunk_data_dict = {self.images_read_key: chunk_list}

        # Tokenize all chunks in one call
        tokenized_dict = self.video_tokenizer.encode(chunk_data_dict)
        chunk_tokens_list = tokenized_dict[self.token_write_key]  # List of [1, ...] tokens

        # Stack tokens: each is [1, tc, ...] -> [B * num_chunks, tc, ...]
        # where ... can be (h, w) for spatial tokens or (L,) for flattened tokens
        chunk_tokens = torch.cat(chunk_tokens_list, dim=0)

        # Reshape to separate batch and chunk dimensions: [B, num_chunks, tc, ...]
        chunk_tokens: torch.Tensor = einops.rearrange(
            chunk_tokens, "(b nc) ... -> b nc ...", b=B, nc=_num_chunks
        )

        # Remove overlapping tokens from chunks 1 onwards if applicable
        if self.overlap_size_frames > 0 and self.overlap_size_tokens > 0:
            # Remove first overlap_size_tokens from chunks 1 onwards: [B, num_chunks, tc, ...]
            processed_chunks = []
            for chunk_idx in range(_num_chunks):
                chunk = chunk_tokens[:, chunk_idx]  # [B, tc, ...]
                if chunk_idx > 0:
                    # Remove first overlap_size_tokens from this chunk
                    chunk = chunk[:, self.overlap_size_tokens :]  # [B, tc - overlap, ...]
                processed_chunks.append(chunk)

            # Concatenate along temporal dimension: [B, total_tc, ...]
            final_tokens = torch.cat(processed_chunks, dim=1)
        else:
            # No overlap: concatenate chunks along temporal dimension
            # [B, num_chunks, tc, ...] -> [B, num_chunks*tc, ...]
            final_tokens: torch.Tensor = einops.rearrange(
                chunk_tokens, "b nc tc ... -> b (nc tc) ..."
            )

        # Convert to list of [1, total_tc, ...]
        all_tokenized_chunks = final_tokens.split(1, dim=0)

        # Update data_dict with tokenized results
        data_dict[self.token_write_key] = all_tokenized_chunks

        return data_dict

    def decode(self, data_dict: Dict[str, Any], video_sizes=None, **kwargs) -> Dict[str, Any]:
        """
        Decode videos from token ids in data_dict by chunking them into temporal segments.

        data_dict: Dict[str, Any]
            Input data containing the following keys:
                - ``self.token_write_key`` - list of token ids of shape ``[1, total_tc, h, w]`` for 3D tokenizer.
                - (Optional) ``self.video_tokenizer.cond_read_key`` - conditioning frames for the first chunk.

        Returns:
            Dict[str, Any]:
                Updated data_dict with decoded videos under ``self.images_reconst_write_key`` - list of videos.
        """

        # Get token ids list - list of [1, total_tc, h, w]
        token_ids_list = data_dict[self.token_write_key]

        assert all(
            token_ids.shape[1] == token_ids_list[0].shape[1] for token_ids in token_ids_list
        ), "All token ids must have the same temporal length"

        # Stack all videos' tokens: [B, total_tc, ...]
        all_tokens = torch.cat(token_ids_list, dim=0)
        B, total_tc = all_tokens.shape[:2]

        # Extract external conditioning frames if provided
        external_conditioning = None
        if (
            self.video_tokenizer.cond_read_key is not None
            and self.video_tokenizer.cond_read_key in data_dict
        ):
            external_conditioning = data_dict[self.video_tokenizer.cond_read_key]

        noise_list = kwargs.pop("noise_list", None)

        assert (
            total_tc >= self._chunk_size_tokens
        ), f"Total tokens {total_tc} less than chunk size tokens {self._chunk_size_tokens}"

        tokens_per_chunk_after_removal = self._chunk_size_tokens - self.overlap_size_tokens
        _num_chunks = (total_tc - self._chunk_size_tokens) // tokens_per_chunk_after_removal + 1

        # Split tokens back into chunks for all videos: list of [B, tc_tokens, ...]
        all_token_chunks = []
        start_idx = 0
        for chunk_idx in range(_num_chunks):
            if chunk_idx == 0:
                chunk_size = self._chunk_size_tokens
            else:
                chunk_size = tokens_per_chunk_after_removal
            end_idx = start_idx + chunk_size
            chunk_tokens = all_tokens[:, start_idx:end_idx]  # [B, tc_tokens, ...]
            all_token_chunks.append(chunk_tokens)
            start_idx = end_idx

        # Decode chunks sequentially with conditional frame support
        all_decoded_chunks = []  # Will contain [B, C, chunk_frames, H, W] tensors

        # Prepare conditioning frames for first chunk if provided
        first_chunk_cond_frames = None
        if external_conditioning is not None:
            # Use external conditioning frames for the first chunk
            # Convert to list format if needed
            if isinstance(external_conditioning, list):
                first_chunk_cond_frames = external_conditioning
            else:
                # Single tensor: [B, C, k, H, W] -> List of [1, C, k, H, W]
                first_chunk_cond_frames = [frames.unsqueeze(0) for frames in external_conditioning]

        for chunk_idx in range(_num_chunks):
            if chunk_idx == 0:
                # First chunk: use external conditioning if provided
                chunk_tokens = all_token_chunks[chunk_idx]  # [B, tc_toks, ...]

                if first_chunk_cond_frames is not None:
                    # Encode conditioning frames to get the corresponding token prefix
                    cond_data_dict = {self.images_read_key: first_chunk_cond_frames}
                    cond_encoded = self.video_tokenizer.encode(cond_data_dict)
                    cond_tokens_list = cond_encoded[
                        self.token_write_key
                    ]  # List of [1, tc_cond, ...]
                    cond_tokens = torch.cat(cond_tokens_list, dim=0)  # [B, tc_cond, ...]

                    # Concatenate conditioning tokens with chunk tokens to get full chunk
                    full_chunk_tokens = torch.cat(
                        [cond_tokens, chunk_tokens[:, cond_tokens.shape[1] :]], dim=1
                    )  # [B, tc_toks, ...]
                    chunk_token_list = full_chunk_tokens.split(
                        1, dim=0
                    )  # List of [1, tc_toks, ...]

                    decoded_videos = self.video_tokenizer.detokenize(
                        token_ids_list=chunk_token_list,
                        vae_video_sizes=video_sizes,
                        num_keep_tokens_list=data_dict.get("eval_keep_k", None),
                        conditioning_frames=first_chunk_cond_frames,
                        noise_list=copy.deepcopy(noise_list),
                        **kwargs,
                    )
                else:
                    # No conditioning: decode chunk tokens directly
                    chunk_token_list = [
                        tokens.unsqueeze(0) for tokens in chunk_tokens
                    ]  # List of [1, tc_toks, ...]

                    decoded_videos = self.video_tokenizer.detokenize(
                        token_ids_list=chunk_token_list,
                        vae_video_sizes=video_sizes,
                        num_keep_tokens_list=data_dict.get("eval_keep_k", None),
                        conditioning_frames=None,
                        noise_list=copy.deepcopy(noise_list),
                        **kwargs,
                    )

                # Stack into [B, C, tc, H, W]
                decoded_chunk = torch.cat(decoded_videos, dim=0)
            else:
                if self.overlap_size_frames > 0:
                    # Subsequent chunks: use overlap frames from previous decoded chunk as condition
                    prev_decoded = all_decoded_chunks[chunk_idx - 1]  # [B, C, tc, H, W]
                    overlap_frames = prev_decoded[
                        :, :, -self.overlap_size_frames :
                    ]  # [B, C, overlap_frames, H, W]

                    # Encode overlap frames to get the corresponding token prefix
                    # This is needed because we removed these tokens during encoding
                    # and the decoded frames may differ from the original input frames
                    overlap_frames_list = [
                        frames.unsqueeze(0) for frames in overlap_frames
                    ]  # List of [1, C, overlap_frames, H, W]
                    overlap_data_dict = {self.images_read_key: overlap_frames_list}
                    overlap_encoded = self.video_tokenizer.encode(overlap_data_dict)
                    overlap_tokens_list = overlap_encoded[
                        self.token_write_key
                    ]  # List of [1, tc_overlap, ...]
                    overlap_tokens = torch.cat(overlap_tokens_list, dim=0)  # [B, tc_overlap, ...]

                    # Get the tokens for the current chunk
                    chunk_tokens = all_token_chunks[chunk_idx]  # [B, tc_toks - t_overlap, ...]

                    # Concatenate overlap tokens with chunk tokens to get the full chunk for decoding
                    full_chunk_tokens = torch.cat(
                        [overlap_tokens, chunk_tokens], dim=1
                    )  # [B, tc_toks, ...]
                    chunk_token_list = full_chunk_tokens.split(
                        1, dim=0
                    )  # List of [1, tc_toks, ...]

                    decoded_videos = self.video_tokenizer.detokenize(
                        token_ids_list=chunk_token_list,
                        vae_video_sizes=video_sizes,
                        num_keep_tokens_list=data_dict.get("eval_keep_k", None),
                        conditioning_frames=overlap_frames_list,
                        **kwargs,
                    )
                else:
                    # No overlap: decode chunk tokens directly
                    chunk_tokens = all_token_chunks[chunk_idx]  # [B, tc_toks, ...]
                    chunk_token_list = [
                        tokens.unsqueeze(0) for tokens in chunk_tokens
                    ]  # List of [1, tc_toks, ...]

                    decoded_videos = self.video_tokenizer.detokenize(
                        token_ids_list=chunk_token_list,
                        vae_video_sizes=video_sizes,
                        num_keep_tokens_list=data_dict.get("eval_keep_k", None),
                        conditioning_frames=None,
                        noise_list=copy.deepcopy(noise_list),
                        **kwargs,
                    )
                # Stack into [B, C, tc_frames, H, W]
                decoded_chunk = torch.cat(decoded_videos, dim=0)

                # Remove the conditioning frames from the result
                decoded_chunk = decoded_chunk[:, :, self.overlap_size_frames :]

            all_decoded_chunks.append(decoded_chunk)

        # Concatenate all decoded chunks along the temporal dimension: [B, C, total_frames, H, W]
        final_videos = torch.cat(all_decoded_chunks, dim=2)

        # Convert back to list of [1, C, total_frames, H, W]
        all_decoded_videos = final_videos.split(1, dim=0)

        # Update data_dict with decoded results
        data_dict[self.images_reconst_write_key] = all_decoded_videos

        return data_dict

    def autoencode(self, data_dict: Dict[str, Any], video_sizes=None, **kwargs) -> Dict[str, Any]:
        """
        Autoencode by using encode() to tokenize and decode() to reconstruct.
        This ensures consistent overlap handling across all methods.

        Args:
            data_dict: Input data containing RGB videos under ``self.images_read_key``.
            video_sizes: Ignored. Video sizes are calculated internally based on chunk_size.
            **kwargs: Additional arguments passed to decode.

        Returns:
            Dict[str, Any]: Updated data_dict with reconstructed videos.
        """
        # Step 1: Tokenize using encode() method (handles all overlap logic)
        data_dict = self.encode(data_dict)

        # Step 2: Decode using decode() method (handles sequential overlap reconstruction)
        data_dict = self.decode(data_dict, video_sizes=video_sizes, **kwargs)

        return data_dict


class VideoFlexTokFromHub(VideoChunkTokenizerWrapper, PyTorchModelHubMixin):
    """Wrapper around VideoChunkTokenizerWrapper and VideoFlexTok for easy loading with Huggingface Hub.

    Args:
        config (dict): Dictionary containing the model configuration,
            used for loading from Huggingface Hub.
    """

    def __init__(self, config: dict):

        config = copy.deepcopy(config)
        # Sanitize config before handing it off to hydra.utils.instantiate()
        _sanitize_hydra_config(config)

        self.video_preprocess_args = config.get("video_preprocess_args", {})

        super().__init__(
            video_tokenizer=instantiate(config["video_flex_tok"]),
            chunk_size=config.get("chunk_size", 17),
            overlap_size_frames=config.get("overlap_size_frames", 1),
            vae_video_sizes=config.get("vae_video_sizes", (5, 32, 32)),
        )
