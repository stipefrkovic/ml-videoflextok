import time
import torch
import imageio.v3 as iio
import numpy as np
from videoflextok.wrappers import VideoFlexTokFromHub
from videoflextok.utils.demo import read_mp4, denormalize

model = VideoFlexTokFromHub.from_pretrained('EPFL-VILAB/videoflextok_d18_d28').eval()

video_tensor = read_mp4(
    "./data/video_examples/frames.mp4",
    num_frames=33,
    **model.video_preprocess_args,
)  # (C, T, H, W)
print(f"Video tensor shape: {video_tensor.shape}")
if hasattr(model.video_tokenizer, 'is_causal'):
    print(f"is_causal: {model.video_tokenizer.is_causal}")
if hasattr(model.video_tokenizer, 'temporal_downsample_factor'):
    print(f"temporal_downsample_factor: {model.video_tokenizer.temporal_downsample_factor}")

for name, module in model.named_modules():
    if hasattr(module, 'n_eval'):
        print(f"{name}: n_min={module.n_min}, n_max={module.n_max}, n_eval={module.n_eval}, size_sampling_mode={module.size_sampling_mode}, valid_sizes={module.valid_sizes}")

print(f"chunk_size: {model.chunk_size}")
print(f"overlap_size_frames: {model.overlap_size_frames}")
print(f"overlap_size_tokens: {model.overlap_size_tokens}")
print(f"chunk_size_tokens: {model._chunk_size_tokens}")
print(f"stride: {model.stride}")
print(f"video_preprocess_args: {model.video_preprocess_args}")

t0 = time.perf_counter()
tokens_list = model.tokenize(video_tensor[None])
print(f"Tokenize: {time.perf_counter() - t0:.2f}s — {len(tokens_list)} sequences, first shape: {tokens_list[0].shape}")

t0 = time.perf_counter()
reconst = model.detokenize(
    tokens_list,
    timesteps=30,
    guidance_scale=20.,
    perform_norm_guidance=True,
)
print(f"Detokenize: {time.perf_counter() - t0:.2f}s — reconstruction shape: {reconst[0].shape}")

video = denormalize(reconst[0].squeeze(0).cpu().float())  # (C, T, H, W) in [0, 1]
frames = (video.permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)  # (T, H, W, 3)
iio.imwrite("output.mp4", frames, fps=8, plugin="FFMPEG", codec="libx264", pixelformat="yuv420p")
print("Saved output.mp4")
