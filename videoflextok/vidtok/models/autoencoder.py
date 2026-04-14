# This file includes code adapted from microsoft/VidTok (MIT License).
# Copyright (c) Microsoft Corporation.
# Source: https://github.com/microsoft/VidTok
# Modifications: Copyright (c) 2026 Apple Inc. and EPFL. All Rights Reserved.
# Full license text: see ACKNOWLEDGEMENTS.md (VidTok).
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from videoflextok.vidtok.modules.util import instantiate_from_config


class AutoencodingEngine(nn.Module):
    """Inference-only VidTok autoencoder.

    This intentionally excludes training-time components like losses, optimizers,
    EMA, and Lightning plumbing.
    """

    def __init__(
        self,
        *,
        encoder_config: Dict[str, Any],
        decoder_config: Dict[str, Any],
        regularizer_config: Dict[str, Any],
        compile_model: bool = False,
        **ignore_kwargs,
    ):
        super().__init__()

        if compile_model and hasattr(torch, "compile"):
            compile_fn = torch.compile
        else:
            compile_fn = lambda module: module

        self.encoder = compile_fn(instantiate_from_config(encoder_config))
        self.decoder = compile_fn(instantiate_from_config(decoder_config))
        self.regularization = instantiate_from_config(regularizer_config)

    def encode(self, x: Any, return_reg_log: bool = False) -> Any:
        z = self.encoder(x)
        z, reg_log = self.regularization(z, n_steps=0)
        if return_reg_log:
            return z, reg_log
        return z

    def decode(self, z: Any) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: Any) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        z, reg_log = self.encode(x, return_reg_log=True)
        dec = self.decode(z)
        return z, dec, reg_log
