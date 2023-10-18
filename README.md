# Latent Consistency Models

Official Repository of the paper: *[Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference](https://arxiv.org/abs/2310.04378)*.

Project Page: https://latent-consistency-models.github.io

## Demos & Models Released
Ours Hugging Face Demo and Model are released ! Also be integrated in 🧨 Diffusers library. 

Hugging Face Demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/SimianLuo/Latent_Consistency_Model)  (It is truly amazing!)

LCM Model Download: https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7

<p align="center">
    <img src="teaser.png">
</p>

By distilling classifier-free guidance into the model's input, LCM can generate high-quality images in very short inference time. We compare the inference time at the setting of 768 x 768 resolution, CFG scale w=8, batchsize=4, using a A800 GPU. 

<p align="center">
    <img src="speed_fid.png">
</p>

## Usage

You can try out Latency Consistency Models directly on:
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/SimianLuo/Latent_Consistency_Model)

To run the model yourself, you can leverage the 🧨 Diffusers library:
1. Install the library:
```
pip install diffusers transformers accelerate
```

2. Run the model:
```py
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", custom_pipeline="latent_consistency_txt2img", custom_revision="main")

# To save GPU memory, torch.float16 can be used, but it may compromise image quality.
pipe.to(torch_device="cuda", torch_dtype=torch.float32)

prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

# Can be set to 1~50 steps. LCM support fast inference even <= 4 steps. Recommend: 1~8 steps.
num_inference_steps = 4 

images = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=8.0, lcm_origin_steps=50, output_type="pil").images
```

## BibTeX

```bibtex
@misc{luo2023latent,
      title={Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference}, 
      author={Simian Luo and Yiqin Tan and Longbo Huang and Jian Li and Hang Zhao},
      year={2023},
      eprint={2310.04378},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
