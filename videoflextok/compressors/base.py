# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. and EPFL. All Rights Reserved.
from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn


class AbstractAE(nn.Module, ABC):
    def __init__(
        self,
        images_read_key: str,
        vae_latents_read_key: str,
        vae_latents_write_key: str,
        images_reconst_write_key: str,
    ):
        super().__init__()
        self.images_read_key = images_read_key
        self.vae_latents_read_key = vae_latents_read_key
        self.vae_latents_write_key = vae_latents_write_key
        self.images_reconst_write_key = images_reconst_write_key

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def device_type(self) -> str:
        return self.device.type

    @property
    @abstractmethod
    def downsample_factor(self) -> int:
        pass

    @abstractmethod
    def init_weights_muP(self):
        pass

    @abstractmethod
    def encode(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def decode(self, data_dict: Dict[str, Any], **ignore_kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def forward(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def autoencode(self, data_dict: Dict[str, Any], **ignore_kwargs) -> Dict[str, Any]:
        data_dict = self.encode(data_dict)
        data_dict[self.vae_latents_read_key] = data_dict[self.vae_latents_write_key]
        return self.decode(data_dict)


class AbstractVideoAE(AbstractAE):
    @property
    @abstractmethod
    def temporal_downsample_factor(self) -> int:
        pass

    @property
    @abstractmethod
    def is_causal(self) -> bool:
        pass
