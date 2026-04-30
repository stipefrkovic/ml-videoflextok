import torch
import torchvision.io
from videoflextok.wrappers import VideoFlexTokFromHub
from videoflextok.utils.demo import read_mp4

model = VideoFlexTokFromHub.from_pretrained('EPFL-VILAB/videoflextok_d18_d28').eval()

video_tensor = read_mp4(
    "./data/video_examples/porsche.mp4",
    fps=8,
    **model.video_preprocess_args,
)  # (C, T, H, W)
print(f"Video tensor shape: {video_tensor.shape}")

tokens_list = model.tokenize(video_tensor[None])
print(f"Tokens: {len(tokens_list)} sequences, first shape: {tokens_list[0].shape}")

torch.save(tokens_list, "tokens.pt")
print("Saved tokens.pt")

# reconst = model.detokenize(
#     tokens_list,
#     timesteps=30,
#     guidance_scale=20.,
#     perform_norm_guidance=True,
# )
# print(f"Reconstruction shape: {reconst[0].shape}")

# # reconst[0]: [1, 3, T, H, W] in [-1, 1] -> (T, H, W, 3) uint8
# video = reconst[0].squeeze(0)          # (3, T, H, W)
# video = (video.clamp(-1, 1) + 1) / 2  # [0, 1]
# video = (video * 255).byte()           # [0, 255]
# video = video.permute(1, 2, 3, 0)      # (T, H, W, 3)
# torchvision.io.write_video("output.mp4", video.cpu(), fps=8)
# print("Saved output.mp4")
