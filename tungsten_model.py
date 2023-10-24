"""
Tungsten model definition
Reference: https://github.com/tungsten-ai/tungstenkit

Before start building a model, download weights & pipeline definition:
$ git lfs install
$ git clone https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7
"""

import os
import random
import sys
from typing import List

import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from safetensors.torch import load_file
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from tungstenkit import BaseIO, Field, Image, Option, define_model

MODEL_DIR = "LCM_Dreamshaper_v7"
sys.path.append("LCM_Dreamshaper_v7")


from lcm_pipeline import LatentConsistencyModelPipeline
from lcm_scheduler import LCMScheduler


class Input(BaseIO):
    prompt: str = Field(description="Input prompt")
    image_dimensions: str = Option(
        default="768x768",
        description="Pixel dimensions of output image (width x height)",
        choices=["512x512", "512x768", "768x512", "768x768"],
    )
    num_output_images: int = Option(
        description="Number of output images",
        le=4,
        ge=1,
        default=1,
    )
    seed: int = Option(
        description="Random seed. Set as -1 to randomize the seed",
        default=-1,
        ge=-1,
        le=4294967293,
    )
    num_inference_steps: int = Option(
        description="Number of denoising steps", ge=1, le=50, default=4
    )
    guidence_scale: float = Option(
        description="Scale for classifier-free guidance", ge=1, le=20, default=8
    )


class Output(BaseIO):
    images: List[Image]


@define_model(
    input=Input,
    output=Output,
    gpu=True,
    system_packages=[
        "libgl1-mesa-glx",
        "libglib2.0-0",
    ],
    python_packages=[
        "torch",
        "torchvision",
        "accelerate",
        "diffusers==0.21.4",
        "transformers==4.34.1",
        "opencv-python",
    ],
    batch_size=1,
)
class LCMModel:
    def setup(self):
        """Load model"""
        # Initalize Diffusers Model:
        vae = AutoencoderKL.from_pretrained(MODEL_DIR, subfolder="vae")
        text_encoder = CLIPTextModel.from_pretrained(
            MODEL_DIR, subfolder="text_encoder"
        )
        tokenizer = CLIPTokenizer.from_pretrained(MODEL_DIR, subfolder="tokenizer")
        unet = UNet2DConditionModel.from_pretrained(
            MODEL_DIR,
            subfolder="unet",
            device_map=None,
            low_cpu_mem_usage=False,
            local_files_only=True,
        )
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            MODEL_DIR, subfolder="safety_checker"
        )
        feature_extractor = CLIPImageProcessor.from_pretrained(
            MODEL_DIR, subfolder="feature_extractor"
        )

        # Initalize Scheduler:du
        scheduler = LCMScheduler(
            beta_start=0.00085,
            beta_end=0.0120,
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
        )

        # Replace the unet with LCM:
        lcm_unet_ckpt = os.path.join(MODEL_DIR, "LCM_Dreamshaper_v7_4k.safetensors")
        ckpt = load_file(lcm_unet_ckpt)
        unet.load_state_dict(ckpt, strict=False)

        # LCM Pipeline:
        self.pipe = LatentConsistencyModelPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.pipe.to("cuda")

    def predict(self, inputs: List[Input]):
        """Run batch prediction"""
        input = inputs[0]  # batch_size == 1

        if input.seed == -1:
            input.seed = random.randrange(4294967294)
            print(f"Using seed {input.seed}\n")

        torch.random.manual_seed(input.seed)

        width, height = input.image_dimensions.split("x")
        width, height = int(width), int(height)

        output_pil_images = self.pipe(
            prompt=input.prompt,
            width=width,
            height=height,
            guidance_scale=input.guidence_scale,
            num_inference_steps=input.num_inference_steps,
            num_images_per_prompt=input.num_output_images,
            lcm_origin_steps=50,
            output_type="pil",
        ).images

        return [
            Output(
                images=[Image.from_pil_image(pil_img) for pil_img in output_pil_images]
            )
        ]
