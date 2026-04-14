# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. and EPFL. All Rights Reserved.
import decord
import math
from typing import Optional

import numpy as np
import torch

import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo

# Default transform settings
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

# defatult parameters for the d18-d28 model
SIZE = 256
FRAMES_CHUNK_SIZE = 17
OVERLAP_SIZE_FRAMES = 1


def read_mp4(
    file: str,
    fps: Optional[int] = None,
    num_frames: Optional[int] = None,
    chunk_size: int = FRAMES_CHUNK_SIZE, 
    size: int = SIZE,
    overlap_size: int = OVERLAP_SIZE_FRAMES
) -> torch.Tensor:
    """Read and sample frames from an MP4 file.

    Args:
        file: Path to the MP4 file
        fps: Frames per second to sample (default: None)
        num_frames: Number of frames to sample (default: None)
        chunk_size: Number of frames per chunk (default: 17)
        size: Size to resize the frames (default: 256)
        overlap_size: Number of overlapping frames between chunks (default: 1)

    Returns:
        Tensor of shape (C, T, H, W) with values in [-1, 1] if transform is applied,
        otherwise in [0, 1]
    """
    assert num_frames is None or fps is None, "Only one of num_frames or fps should be specified, but got both"        

    vr = decord.VideoReader(file, ctx=decord.cpu(0))
    total_frames = len(vr)
    
    # Calculate num_frames from fps if needed
    if num_frames is None and fps is not None:
        video_fps = vr.get_avg_fps()
        # Duration in seconds
        duration = total_frames / video_fps
        # Desired number of frames at target fps
        desired_frames = int(duration * fps)
        # Round up to valid value: 1 + K * (chunk_size - overlap_size) for some integer K >= 0
        chunk_stride = chunk_size - overlap_size
        K = max(0, math.ceil((desired_frames - 1) / chunk_stride))
        num_frames = 1 + K * chunk_stride

    assert (num_frames - 1) % (
        chunk_size - overlap_size
    ) == 0, f"num_frames should be of the form 1 + k * (chunk_size - overlap_size) for some integer k >= 0, but got {num_frames} with chunk_size={chunk_size} and overlap_size={overlap_size}"
    
    # Sample frames uniformly
    idx = np.linspace(0, total_frames - 1, num_frames, dtype=np.int32)

    # Get frames as numpy array (T, H, W, C)
    frames = vr.get_batch(idx)

    # Convert to torch tensor if needed
    if isinstance(frames, torch.Tensor):
        frames = frames.permute(3, 0, 1, 2).contiguous()
    else:
        # NumPy array
        frames = frames.asnumpy()
        frames = torch.from_numpy(frames).permute(3, 0, 1, 2).contiguous()

    # Normalize to [0, 1]
    frames = frames.float() / 255.0

    # Apply transform if requested
    transform = transforms.Compose(
        [
            # Resize
            transforms.Resize(
                size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            # Clamp to [0, 1]
            transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
            # CenterCrop
            transforms.CenterCrop(size),
            # [0, 1] -> [-1, 1]
            NormalizeVideo(
                mean=MEAN,
                std=STD,
            ),
        ]
    )
    frames = transform(frames)

    del vr
    return frames


def denormalize(video, mean=MEAN, std=STD):
    """
    Denormalizes an image.

    Args:
        video (torch.Tensor): Video tensor to denormalize of [C, T, H, W]
        mean (tuple): Mean to use for denormalization.
        std (tuple): Standard deviation to use for denormalization.
    """
    return TF.normalize(
        video.clone().permute(1, 0, 2, 3),
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std],
    ).permute(
        1, 0, 2, 3
    )  # (C, T, H, W)
