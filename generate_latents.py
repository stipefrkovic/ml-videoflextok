import json
import time
from pathlib import Path

import decord
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo

from videoflextok.wrappers import VideoFlexTokFromHub
from videoflextok.utils.demo import MEAN, STD


def read_mp4_first_n(file: str, num_frames: int, size: int = 256, **_unused) -> torch.Tensor:
    """Read the first `num_frames` frames of an mp4 (no uniform sub-sampling)."""
    vr = decord.VideoReader(file, ctx=decord.cpu(0))
    idx = np.arange(min(num_frames, len(vr)), dtype=np.int32)
    frames = vr.get_batch(idx)
    if isinstance(frames, torch.Tensor):
        frames = frames.permute(3, 0, 1, 2).contiguous()
    else:
        frames = torch.from_numpy(frames.asnumpy()).permute(3, 0, 1, 2).contiguous()
    frames = frames.float() / 255.0
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        transforms.CenterCrop(size),
        NormalizeVideo(mean=MEAN, std=STD),
    ])
    return transform(frames)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_ROOT = Path(
    "/gpfs/home3/scur0531/izzi/genieRedux/data_generation/datasets/random_game_video"
)
LATENTS_ROOT = Path(
    "/gpfs/home3/scur0531/stipe/random_game_video_latents"
)
model = VideoFlexTokFromHub.from_pretrained('EPFL-VILAB/videoflextok_d18_d28').eval().to(device)
print(f"Model on {device}")

# Smallest valid grid length: 1 + k * stride with k=1.
stride = model.stride
k = 1
NUM_FRAMES = 1 + k * stride
print(f"stride={stride} -> NUM_FRAMES={NUM_FRAMES}")

# Frame -> latent mapping (causal magvit-style).
chunk_size = model.chunk_size
overlap_frames = model.overlap_size_frames
tdf = model.video_tokenizer.temporal_downsample_factor
is_causal = getattr(model.video_tokenizer, "is_causal", False)


def latent_groups(frame_offset: int, n_chunk_frames: int):
    """Return list of input-frame index lists, one per output latent frame."""
    if is_causal:
        # 1 + k*tdf input frames -> 1 + k latent frames
        groups = [[frame_offset]]  # first latent: single frame
        body = n_chunk_frames - 1
        for j in range(body // tdf):
            start = frame_offset + 1 + j * tdf
            groups.append(list(range(start, start + tdf)))
        return groups
    # non-causal: pure tdf-pooling
    return [
        list(range(frame_offset + j * tdf, frame_offset + (j + 1) * tdf))
        for j in range(n_chunk_frames // tdf)
    ]


print(f"\nFrame -> latent grouping (chunk_size={chunk_size}, stride={stride}, tdf={tdf}, causal={is_causal}):")
chunk_starts = [c * stride for c in range(k)]
latent_to_frames: list[list[int]] = []
for ci, cstart in enumerate(chunk_starts):
    groups = latent_groups(cstart, chunk_size)
    print(f"  chunk {ci} (frames {cstart}..{cstart + chunk_size - 1}):")
    for g in groups:
        print(f"    latent {len(latent_to_frames)}: frames {g}")
        latent_to_frames.append(g)
print(f"Total latent frames: {len(latent_to_frames)}\n")

mp4_paths = sorted(DATASET_ROOT.rglob("frames.mp4"))
print(f"Found {len(mp4_paths)} videos under {DATASET_ROOT}")

for mp4_path in mp4_paths:
    rel = mp4_path.relative_to(DATASET_ROOT).parent  # mirrors session dir
    out_dir = LATENTS_ROOT / rel
    out_path = out_dir / "latents.pt"

    t0 = time.perf_counter()
    # Read NUM_FRAMES - 1 real frames, then prepend a duplicate of frame 0
    # so latent 0 encodes a clean no-op baseline.
    raw = read_mp4_first_n(
        str(mp4_path),
        num_frames=NUM_FRAMES - 1,
        **model.video_preprocess_args,
    )  # (C, T-1, H, W)
    video_tensor = torch.cat([raw[:, :1], raw], dim=1).to(device)  # (C, T, H, W)

    actions_path = mp4_path.parent / "actions.json"
    with open(actions_path) as f:
        raw_actions = json.load(f)["actions"][:NUM_FRAMES - 1]
    actions = [raw_actions[0]] + raw_actions  # duplicate first action to match

    print(
        f"[{rel}] read {video_tensor.shape}, {len(actions)} actions in "
        f"{time.perf_counter() - t0:.2f}s -> saving to {out_path}"
    )
    for li, frames in enumerate(latent_to_frames):
        latent_actions = [actions[fi]["action"] for fi in frames if fi < len(actions)]
        print(f"  latent {li} (frames {frames}): actions {latent_actions}")

    with torch.no_grad():
        tokens_list = model.tokenize(video_tensor[None])
    print(f"  tokens_list: {len(tokens_list)} seq, shapes {[t.shape for t in tokens_list]}")
    print(f"  derived latent frames: {len(latent_to_frames)}")

    # Keep only latents 1 (no-op) and 2 (first action) — the no-action/action pair.
    # Subsequent latents are causally conditioned on these and are dropped.
    KEEP = [1, 2]
    kept_tokens = [t[:, KEEP] for t in tokens_list]  # (B, 2, D)
    kept_frames = [latent_to_frames[i] for i in KEEP]
    kept_actions = [[actions[fi] for fi in g] for g in kept_frames]
    print(
        f"  saving latents {KEEP}: shapes {[t.shape for t in kept_tokens]}, "
        f"frames {kept_frames}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "tokens": [t.cpu() for t in kept_tokens],
            "actions": kept_actions,
            "latent_to_frames": kept_frames,
            "kept_latents": KEEP,
            "num_frames": NUM_FRAMES,
        },
        out_path,
    )
