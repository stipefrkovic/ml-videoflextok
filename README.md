# VideoFlexTok: Flexible-Length Coarse-to-Fine Video Tokenization

[`Website`](https://videoflextok.epfl.ch) | [`arXiv`](https://arxiv.org/abs/PLACEHOLDER) | [`BibTeX`](#citation)

Official inference implementation for:

[**VideoFlexTok: Flexible-Length Coarse-to-Fine Video Tokenization**](), arXiv 2026
*[Andrei Atanov](https://andrewatanov.github.io)\*, [Jesse Allardice](https://github.com/JesseAllardice)\*, [Roman Bachmann](https://roman-bachmann.github.io/), [Oğuzhan Fatih Kar](https://ofkar.github.io/), [R Devon Hjelm](https://rdevon.github.io/), [David Griffiths](https://www.dgriffiths.uk/), [Peter Fu](https://scholar.google.com/citations?user=zAgqOn8AAAAJ&hl=en&oi=ao), [Afshin Dehghan](https://scholar.google.com/citations?user=wcX-UW4AAAAJ), [Amir Zamir](https://vilab.epfl.ch/zamir/)*


![VideoFlexTok main figure](./assets/videoflextok_pull.gif)


## Table of contents
- [VideoFlexTok: Flexible-Length Coarse-to-Fine Video Tokenization](#videoflextok-flexible-length-coarse-to-fine-video-tokenization)
  - [Table of contents](#table-of-contents)
  - [Usage](#usage)
    - [Installation](#installation)
    - [Getting started](#getting-started)
  - [VideoFlexTok tokenizers](#videoflextok-tokenizers)
    - [Inference example](#inference-example)
  - [License](#license)
  - [Citation](#citation)


## Usage

### Installation
1. Clone this repository and navigate to the root directory:
```bash
git clone https://github.com/apple/ml-videoflextok
cd ml-videoflextok
```

2. Create a new conda environment, then install the package and its dependencies:
```bash
conda create -n videoflextok python=3.10 -y
source activate videoflextok
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Verify that CUDA is available in PyTorch by running the following in a Python shell:
```bash
# Run in Python shell
import torch
print(torch.cuda.is_available())  # Should return True
```
If CUDA is not available, consider re-installing PyTorch following the [official installation instructions](https://pytorch.org/get-started/locally/).

4. (Optional) Expose the new conda environment as a kernel to Jupyter notebooks:
```bash
pip install ipykernel
python -m ipykernel install --user --name videoflextok --display-name "VideoFlexTok (videoflextok)"
```


### Getting started
We recommend checking the Jupyter notebook in [notebooks/videoflextok_inference.ipynb](notebooks/videoflextok_inference.ipynb) to get started with the VideoFlexTok tokenizer.


## VideoFlexTok tokenizers

This repository provides the inference code implementation for the following VideoFlexTok models released by `EPFL-VILAB`: 

- `EPFL-VILAB/videoflextok_d18_d28` -- larger model with 28 decoder layers, working with 256x256 resolution
- `EPFL-VILAB/videoflextok_d18_d18_k600` -- smaller model with 18 decoder layers, working with 128x128 resolution, trained on the Kinetics-600 dataset,

see the [project page](https://videoflextok.epfl.ch) and [huggingface.co/EPFL-VILAB](https://huggingface.co/EPFL-VILAB) for more details on the checkpoints.

### Inference example

Example usage, loading a `VideoFlexTok d18-d28` model directly from HuggingFace Hub:
```python
from videoflextok.wrappers import VideoFlexTokFromHub
model = VideoFlexTokFromHub.from_pretrained('EPFL-VILAB/videoflextok_d18_d28').eval()
```

The model can also be loaded by downloading the safetensors checkpoint manually and loading it using our helper functions:
```python
from hydra.utils import instantiate
from videoflextok.utils.checkpoint import load_safetensors

ckpt, config = load_safetensors('/path/to/model.safetensors')
model = instantiate(config).eval()
model.load_state_dict(ckpt)
```

After loading a VideoFlexTok model, image batches can be encoded using:
```python
from videoflextok.utils.demo import read_mp4
# Load example video into a float tensor of shape (3, T, 256, 256), normalized to [-1,1]
# it will sample frame at approx. 8 FPS, ensuring T = 1 + K * (chunk_size - overlap_size) for some integer K >= 1,
# which is required for the chunking mechanism in VideoFlexTok
video_tensor = read_mp4("./data/video_examples/red_ball.mp4", fps=8, **model.video_preprocess_args)  # (C, T, H, W)

# Encode into a list of discrete token sequences, where each sequence is of shape [1, t, 256]
# this will automatically apply the encoder in the sliding window fashion, and concatenate the resulting tokens along the sequence dimension
tokens_list = model.tokenize(video_tensor[None])
```

The list of token sequences can be truncated in a nested fashion:
```python
k_keep = 64 # For example, only keep the first 64 out of 256 tokens for each timestep
tokens_list = [t[..., :k_keep] for t in tokens_list]
```

To decode the tokens with VideoFlexTok's rectified flow decoder, call:
```python
# tokens_list is a list of [1, t, l] discrete token sequences, with l <= 256
# reconst is a list of RGB videos of shape [1, 3, T, 256, 256] tensor, normalized to [-1,1]
reconst = model.detokenize(
    tokens_list,
    timesteps=30, # Number of denoising steps
    guidance_scale=20., # Classifier-free guidance scale (15-30 typically works well)
    perform_norm_guidance=True, # See https://arxiv.org/abs/2410.02416
)
```

## License
The code in this repository is released under the license as found in the [LICENSE](LICENSE) file.

## Citation
If you find this repository helpful, please consider citing our work:
```
@article{videoflextok,
    title={{VideoFlexTok}: Flexible-Length Coarse-to-Fine Video Tokenization},
    author={Andrei Atanov and Jesse Allardice and Roman Bachmann and O{\u{g}}uzhan Fatih Kar and Peter Fu and David Griffiths and Devon Hjelm and Afshin Dehghan and Amir Zamir},
    journal={arXiv 2026},
    year={2026},
}
```
