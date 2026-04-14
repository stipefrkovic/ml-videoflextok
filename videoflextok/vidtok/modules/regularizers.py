# This file includes code adapted from microsoft/VidTok (MIT License).
# Copyright (c) Microsoft Corporation.
# Source: https://github.com/microsoft/VidTok
# Modifications: Copyright (c) 2026 Apple Inc. and EPFL. All Rights Reserved.
# Full license text: see ACKNOWLEDGEMENTS.md (VidTok).

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .distributions import DiagonalGaussianDistribution


class AbstractRegularizer(nn.Module, ABC):
    @abstractmethod
    def forward(
        self, z: torch.Tensor, n_steps: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    @abstractmethod
    def get_trainable_parameters(self) -> Any:
        raise NotImplementedError()


class DiagonalGaussianRegularizer(AbstractRegularizer):
    def __init__(self, sample: bool = True):
        super().__init__()
        self.sample = sample

    def get_trainable_parameters(self) -> Any:
        yield from ()

    def forward(
        self, z: torch.Tensor, n_steps: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del n_steps
        posterior = DiagonalGaussianDistribution(z)
        latents = posterior.sample() if self.sample else posterior.mode()
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        return latents, {"kl_loss": kl_loss}
