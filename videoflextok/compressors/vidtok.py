# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. and EPFL. All Rights Reserved.
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

from einops import rearrange

import torch

from omegaconf import OmegaConf

from videoflextok.vidtok.modules.util import instantiate_from_config

from .base import AbstractVideoAE

VidTokVAEModelName = Literal["vidtok_kl_causal_488_16chn",]

VIDTOK_COMPRESSION_FACTORS: Dict[VidTokVAEModelName, Tuple[int, int, int]] = {
    "vidtok_kl_causal_488_16chn": (4, 8, 8),
}

_VIDTOK_CONFIG_DIR = Path(__file__).resolve().parent.parent / "vidtok" / "configs"


def _get_packaged_config_path(tokenizer_id: str) -> Path:
    cfg_path = _VIDTOK_CONFIG_DIR / f"{tokenizer_id}.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Could not find packaged VidTok config for {tokenizer_id}: {cfg_path}"
        )
    return cfg_path


def load_model_from_config(
    config_path: Path, ignore_keys: Optional[list[str]] = None, verbose: bool = False
):
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(config, dict) or "model" not in config:
        raise ValueError(f"Invalid VidTok config at {config_path}: expected top-level `model` key.")

    model_cfg = config["model"]
    model_params = model_cfg.setdefault("params", {})
    model_params["ignore_keys"] = ignore_keys or []
    model_params["verbose"] = verbose
    return instantiate_from_config(model_cfg)


class VidTokVAE(AbstractVideoAE):
    def __init__(
        self,
        images_read_key: str,
        vae_latents_read_key: str,
        vae_latents_write_key: str,
        images_reconst_write_key: str,
        tokenizer_id: VidTokVAEModelName,
        compile: bool = False,
        chunk_size: Optional[int] = None,
        force_vae_encode: bool = True,
    ):
        super().__init__(
            images_read_key, vae_latents_read_key, vae_latents_write_key, images_reconst_write_key
        )

        self.force_vae_encode = force_vae_encode
        self.chunk_size = chunk_size
        self._is_causal = "noncausal" not in tokenizer_id

        vc, hc, wc = VIDTOK_COMPRESSION_FACTORS[tokenizer_id]
        assert hc == wc
        self._spatial_downsample_factor = hc
        self._temporal_downsample_factor = vc

        cfg_path = _get_packaged_config_path(tokenizer_id)
        self.vidtok = load_model_from_config(cfg_path)

        if compile and hasattr(torch, "compile"):
            self.vidtok.encoder = torch.compile(self.vidtok.encoder)
            self.vidtok.decoder = torch.compile(self.vidtok.decoder)

        self.freeze()

    def freeze(self) -> "VidTokVAE":
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def _encode(self, input_video: torch.Tensor) -> torch.Tensor:
        """Encode [B, C, T, H, W] videos into VAE latents."""
        batch_size, _, num_frames = input_video.shape[:3]
        chunk_size = self.chunk_size or num_frames

        input_video = rearrange(input_video, "b c (nc tc) h w -> (b nc) c tc h w", tc=chunk_size)

        tokens = self.vidtok.encode(input_video)

        if self.chunk_size is not None:
            tokens = rearrange(tokens, "(b nc) ... -> b nc ...", b=batch_size)

        return tokens

    def _decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode VAE latents of shape [B, C, (Nc,) T, H, W] into [B, 3, T, H, W]."""
        batch_size = tokens.shape[0]

        if self.chunk_size is not None:
            tokens = rearrange(tokens, "b c nc t ... -> (b nc) c t ...")

        rec_video = self.vidtok.decode(tokens)
        rec_video = rearrange(rec_video, "(b nc) c t ... -> b c (nc t) ...", b=batch_size)
        return rec_video

    def encode(self, data_dict):
        if self.vae_latents_write_key in data_dict and not self.force_vae_encode:
            return data_dict

        videos = data_dict[self.images_read_key]

        is_list = False
        if isinstance(videos, (list, tuple)):
            videos = torch.cat(videos, dim=0)
            is_list = True

        latents = self._encode(videos)

        if is_list:
            latents = list(latents.split(1, dim=0))

        data_dict[self.vae_latents_write_key] = latents
        return data_dict

    def decode(self, data_dict, **ignore_kwargs):
        latents = data_dict[self.vae_latents_read_key]

        is_list = False
        if isinstance(latents, (list, tuple)):
            latents = torch.cat(latents, dim=0)
            is_list = True

        videos = self._decode(latents)

        if is_list:
            videos = list(videos.split(1, dim=0))

        data_dict[self.images_reconst_write_key] = videos
        return data_dict

    @property
    def downsample_factor(self) -> int:
        return self._spatial_downsample_factor

    @property
    def temporal_downsample_factor(self) -> int:
        return self._temporal_downsample_factor

    @property
    def is_causal(self) -> bool:
        return self._is_causal

    def init_weights_muP(self):
        pass

    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        return self.autoencode(data_dict)
