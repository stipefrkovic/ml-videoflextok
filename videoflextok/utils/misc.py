# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. and EPFL. All Rights Reserved.
import collections.abc
import hashlib
from contextlib import nullcontext
from itertools import repeat
from typing import List, Optional, Union

import einops

import torch

import torchvision.transforms.functional as TF

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def detect_bf16_support():
    """
    Checks if the current GPU supports BF16 precision.

    Returns:
        bool: True if BF16 is supported, False otherwise.
    """
    if torch.cuda.is_available():
        # For NVIDIA GPUs, BF16 support typically requires compute capability 8.0 or higher.
        cc_major, _ = torch.cuda.get_device_capability(0)
        return cc_major >= 8
    return False


def get_bf16_context(enable_bf16: bool = detect_bf16_support(), device_type: str = "cuda"):
    """
    Returns an autocast context that uses BF16 precision if enable_bf16 is True,
    otherwise returns a no-op context.
    """
    if enable_bf16:
        # When BF16 is enabled, we use torch.cuda.amp.autocast.
        return torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=True)
    else:
        # Otherwise, we use a no-op context.
        return nullcontext()


def str_to_dtype(dtype_str: Optional[str]):
    if dtype_str is None:
        return None
    elif dtype_str in ["float16", "fp16"]:
        return torch.float16
    elif dtype_str in ["bfloat16", "bf16"]:
        return torch.bfloat16
    elif dtype_str in ["float32", "fp32"]:
        return torch.float32
    else:
        raise ValueError(f"Invalid dtype string representation: {dtype_str}")


def get_autocast_context(x: Union[torch.Tensor, List[torch.Tensor]], dtype_override: torch.dtype):
    device_type = x[0].device.type
    if dtype_override is None:
        auto_cast_context = nullcontext()
    else:
        auto_cast_context = torch.amp.autocast(
            device_type, dtype=dtype_override, enabled=dtype_override != torch.float32
        )
    return auto_cast_context


def denormalize(img, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    """
    Denormalizes an image.

    Args:
        img (torch.Tensor): Image to denormalize.
        mean (tuple): Mean to use for denormalization.
        std (tuple): Standard deviation to use for denormalization.
    """
    return TF.normalize(
        img.clone(), mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
    )


def denormalize_video(video, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    """
    Denormalizes videos.

    Args:
        video (torch.Tensor): Video to denormalize.
        mean (tuple): Mean to use for denormalization.
        std (tuple): Standard deviation to use for denormalization.
    """
    if len(video.shape) == 4:
        # single video, use denormalize
        return denormalize(video, mean=mean, std=std)

    B = video.shape[0]

    # pack frames into the batch dimension
    img = einops.rearrange(video, "b c t h w -> (b t) c h w")

    # denormalize each frame
    img_norm = denormalize(img, mean=mean, std=std)

    # unpack videos
    norm_video = einops.rearrange(img_norm, "(b t) c h w -> b c t h w", b=B)

    return norm_video


def generate_uint15_hash(seed_str):
    """Generates a hash of the seed string as an unsigned int15 integer"""
    return int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest(), 16) % (2**15)


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
