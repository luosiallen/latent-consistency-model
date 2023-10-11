# Latent Consistency Models

Official Repository of the paper: *[Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference](https://arxiv.org/abs/2310.04378)*.

Project Page: https://latent-consistency-models.github.io


<p align="center">
    <img src="teaser.png">
</p>

By distilling classifier-free guidance into the model's input, LCM can generate high-quality images in very short inference time. We compare the inference time at the setting of 768 x 768 resolution, CFG scale w=8, batchsize=4, using a A800 GPU. 

<p align="center">
    <img src="speed_fid.png">
</p>

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
