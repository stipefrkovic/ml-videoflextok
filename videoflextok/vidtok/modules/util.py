# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. and EPFL. All Rights Reserved.
import importlib
from collections.abc import Mapping
from typing import Any

# Manual list of allowed targets for VidTok config instantiation. This protects
# against importing arbitrary modules from untrusted configs.
ALLOWED_TARGETS = {
    "videoflextok.vidtok.models.autoencoder.AutoencodingEngine",
    "videoflextok.vidtok.modules.model_3dcausal.EncoderCausal3D",
    "videoflextok.vidtok.modules.model_3dcausal.EncoderCausal3DPadding",
    "videoflextok.vidtok.modules.model_3dcausal.DecoderCausal3D",
    "videoflextok.vidtok.modules.model_3dcausal.DecoderCausal3DPadding",
    "videoflextok.vidtok.modules.regularizers.DiagonalGaussianRegularizer",
}

TARGET_KEYS = ("target", "_target_")


def _validate_target(target: Any) -> str:
    if not isinstance(target, str) or target not in ALLOWED_TARGETS:
        raise ValueError(f"Potentially unsafe target in VidTok config: {target!r}")
    return target


def _sanitize_target_config(cfg: Any) -> None:
    if isinstance(cfg, Mapping):
        for key, val in cfg.items():
            if key in TARGET_KEYS:
                _validate_target(val)
            else:
                _sanitize_target_config(val)
    elif isinstance(cfg, (list, tuple)):
        for item in cfg:
            _sanitize_target_config(item)


def _extract_params(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if "params" in config:
        params = config["params"]
        if not isinstance(params, Mapping):
            raise TypeError("Expected `params` to be a mapping.")
        return params
    return {k: v for k, v in config.items() if k not in TARGET_KEYS}


def get_obj_from_str(string: str, reload: bool = False, invalidate_cache: bool = True) -> Any:
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_obj = importlib.import_module(module)
        importlib.reload(module_obj)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: Mapping[str, Any]) -> Any:
    _sanitize_target_config(config)
    target_key = "target" if "target" in config else "_target_" if "_target_" in config else None
    if target_key is None:
        raise KeyError("Expected key `target` or `_target_` to instantiate.")
    target = _validate_target(config[target_key])
    return get_obj_from_str(target)(**dict(_extract_params(config)))
