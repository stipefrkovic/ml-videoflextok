import time

import decord
import imageio.v3 as iio
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo

from videoflextok.wrappers import VideoFlexTokFromHub
from videoflextok.utils.demo import MEAN, STD, denormalize
from videoflextok.utils.misc import get_bf16_context, detect_bf16_support

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_grad_enabled(False)


VIDEO_PATH = (
    "/gpfs/home3/scur0531/izzi/genieRedux/data_generation/datasets/random_game_video/"
    "retro_act_v0.0.0/retro_8eyes-nes_v0.0.0/000000/000000/frames.mp4"

)
K = 1  # smallest valid clip: NUM_FRAMES = 1 + K * stride


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
enable_bf16 = detect_bf16_support()
print(f"Device: {device}, BF16: {enable_bf16}")
model = VideoFlexTokFromHub.from_pretrained('EPFL-VILAB/videoflextok_d18_d28').eval().to(device)

stride = model.stride
NUM_FRAMES = 1 + K * stride
raw_per_window = NUM_FRAMES - 1 if K == 1 else NUM_FRAMES

if hasattr(model.video_tokenizer, 'is_causal'):
    print(f"is_causal: {model.video_tokenizer.is_causal}")
if hasattr(model.video_tokenizer, 'temporal_downsample_factor'):
    print(f"temporal_downsample_factor: {model.video_tokenizer.temporal_downsample_factor}")
print(f"chunk_size: {model.chunk_size}")
print(f"overlap_size_frames: {model.overlap_size_frames}")
print(f"overlap_size_tokens: {model.overlap_size_tokens}")
print(f"chunk_size_tokens: {model._chunk_size_tokens}")
print(f"stride: {stride}")
print(f"video_preprocess_args: {model.video_preprocess_args}")
print(f"K={K} -> NUM_FRAMES={NUM_FRAMES}, reading {raw_per_window} consecutive raw frames")

raw = read_mp4_first_n(
    VIDEO_PATH,
    num_frames=raw_per_window,
    **model.video_preprocess_args,
)  # (C, raw_per_window, H, W)
print(f"Raw frames shape: {raw.shape}")

# For K=1, prepend a duplicate of the first frame so the causal model gets
# 1 + K*stride input frames; for K>1, the consecutive read is already aligned.
clip = torch.cat([raw[:, :1], raw], dim=1) if K == 1 else raw
clip = clip.to(device)
print(f"Model input shape: {clip.shape}")

t0 = time.perf_counter()
with get_bf16_context(enable_bf16):
    tokens_list = model.tokenize(clip[None])
print(f"Tokenize: {time.perf_counter() - t0:.2f}s — {len(tokens_list)} sequences, first shape: {tokens_list[0].shape}")

decoder_generation_kwargs = dict(
    timesteps=40,
    guidance_scale=25,
    perform_norm_guidance=True,
    generator=torch.Generator(device).manual_seed(42),
    eta=0.,
    momentum=0.,
    norm_threshold=0.6,
    verbose=False,
)

t0 = time.perf_counter()
with get_bf16_context(enable_bf16):
    reconst = model.detokenize(tokens_list, **decoder_generation_kwargs)
print(f"Detokenize: {time.perf_counter() - t0:.2f}s — reconstruction shape: {reconst[0].shape}")

video = denormalize(reconst[0].squeeze(0).cpu().float()).clamp(0, 1)  # (C, T, H, W) in [0, 1]
frames = (video.permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)  # (T, H, W, 3)
iio.imwrite("output.mp4", frames, fps=8, plugin="FFMPEG", codec="libx264", pixelformat="yuv420p")
print("Saved output.mp4")
