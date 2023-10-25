# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import torch
from diffusers import DiffusionPipeline
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        # # Official LCM Pipeline supported now.
        # self.pipe = DiffusionPipeline.from_pretrained(
        #     "SimianLuo/LCM_Dreamshaper_v7",
        #     cache_dir="model_cache",
        #     local_files_only=True,
        # )

        # Want to use older ones, need to add "revision="fb9c5d1"
        self.pipe = DiffusionPipeline.from_pretrained(
            "SimianLuo/LCM_Dreamshaper_v7",
            custom_pipeline="latent_consistency_txt2img",
            custom_revision="main",
            revision="fb9c5d1",
            cache_dir="model_cache",
            local_files_only=True,
        )
        self.pipe.to(torch_device="cuda", torch_dtype=torch.float32)

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        ),
        width: int = Input(
            description="Width of output image. Lower the setting if out of memory.",
            default=768,
        ),
        height: int = Input(
            description="Height of output image. Lower the setting if out of memory.",
            default=768,
        ),
        num_images: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps. Can be set to 1~50 steps. LCM support fast inference even <= 4 steps. Recommend: 1~8 steps.",
            ge=1,
            le=50,
            default=8,
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=8.0
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> list[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        torch.manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=num_images,
            lcm_origin_steps=50,
            output_type="pil",
        ).images

        output_paths = []
        for i, sample in enumerate(result):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
