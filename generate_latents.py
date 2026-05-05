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


def read_mp4_first_n(file: str, num_frames: int, size: int = 256, start: int = 0, **_unused) -> torch.Tensor:
    """Read frames `[start..start+num_frames)` of an mp4 (no uniform sub-sampling)."""
    vr = decord.VideoReader(file, ctx=decord.cpu(0))
    end = min(start + num_frames, len(vr))
    idx = np.arange(start, end, dtype=np.int32)
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
NUM_ACT_EXTRACT_PERVID = 8           # number of no-op/action pairs per video
WINDOW_STRIDE_FRAMES = 8             # raw-frame step between successive windows (one action cycle)
print(f"stride={stride} -> NUM_FRAMES={NUM_FRAMES}, pairs/video={NUM_ACT_EXTRACT_PERVID}")

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

min_frames = min(len(decord.VideoReader(str(p), ctx=decord.cpu(0))) for p in mp4_paths)
max_pairs = (min_frames - (NUM_FRAMES - 1)) // WINDOW_STRIDE_FRAMES + 1
print(f"Shortest video: {min_frames} frames -> max pairs/video = {max_pairs}")
if NUM_ACT_EXTRACT_PERVID > max_pairs:
    print(
        f"WARNING: NUM_ACT_EXTRACT_PERVID={NUM_ACT_EXTRACT_PERVID} exceeds max ({max_pairs}); "
        f"shorter videos will be skipped."
    )

for mp4_path in mp4_paths:
    rel = mp4_path.relative_to(DATASET_ROOT).parent  # mirrors session dir
    out_dir = LATENTS_ROOT / rel
    out_path = out_dir / "latents.pt"

    t0 = time.perf_counter()
    # Read all raw frames needed across all sliding windows in one shot.
    raw_per_window = NUM_FRAMES - 1
    total_raw = (NUM_ACT_EXTRACT_PERVID - 1) * WINDOW_STRIDE_FRAMES + raw_per_window
    raw = read_mp4_first_n(
        str(mp4_path),
        num_frames=total_raw,
        **model.video_preprocess_args,
    )  # (C, total_raw, H, W)

    actions_path = mp4_path.parent / "actions.json"
    with open(actions_path) as f:
        raw_actions = json.load(f)["actions"][:total_raw]

    if raw.shape[1] < total_raw or len(raw_actions) < total_raw:
        print(f"  skipping {rel}: only {raw.shape[1]} frames / {len(raw_actions)} actions, need {total_raw}")
        continue

    # Process N windows one at a time (batch=1) to keep memory low.
    KEEP = [1, 2]
    kept_frames = [latent_to_frames[i] for i in KEEP]
    per_window_tokens: list[list[torch.Tensor]] = []   # [N][num_seq] of (1, 2, D)
    per_window_actions: list[list[list]] = []          # (N, 2, 4)

    print(f"[{rel}] {NUM_ACT_EXTRACT_PERVID} windows, read in {time.perf_counter() - t0:.2f}s -> saving to {out_path}")

    for i in range(NUM_ACT_EXTRACT_PERVID):
        s = i * WINDOW_STRIDE_FRAMES
        wraw = raw[:, s:s + raw_per_window]                                  # (C, 16, H, W)
        wvid = torch.cat([wraw[:, :1], wraw], dim=1).to(device)              # (C, 17, H, W)
        wacts = raw_actions[s:s + raw_per_window]
        wacts_dup = [wacts[0]] + wacts                                       # 17 actions

        with torch.no_grad():
            tokens_list = model.tokenize(wvid[None])                         # each (1, 5, D)
        per_window_tokens.append([t[:, KEEP].cpu() for t in tokens_list])
        per_window_actions.append([[wacts_dup[fi] for fi in g] for g in kept_frames])

    # Stack across windows: list-per-seq of (N, 2, D)
    n_seq = len(per_window_tokens[0])
    kept_tokens = [
        torch.cat([per_window_tokens[i][s] for i in range(NUM_ACT_EXTRACT_PERVID)], dim=0)
        for s in range(n_seq)
    ]
    kept_actions = per_window_actions
    print(
        f"  saving latents {KEEP}: shapes {[t.shape for t in kept_tokens]}, "
        f"frames {kept_frames}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "tokens": kept_tokens,  # already on CPU
            "actions": kept_actions,
            "latent_to_frames": kept_frames,
            "kept_latents": KEEP,
            "num_frames": NUM_FRAMES,
            "window_stride_frames": WINDOW_STRIDE_FRAMES,
            "n_windows": NUM_ACT_EXTRACT_PERVID,
        },
        out_path,
    )
