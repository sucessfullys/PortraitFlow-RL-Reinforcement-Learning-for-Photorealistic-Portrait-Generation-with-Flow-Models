import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.grpo import flux2_klein_base_4b
from flow_grpo.diffusers_patch.flux2_pipeline_with_logprob import (
    pipeline_with_logprob,
)
from flow_grpo.flux2_utils import Flux2Policy, build_flux2_pipeline


config = flux2_klein_base_4b()
device = torch.device("cuda")
pipe = build_flux2_pipeline(config, device)
policy = Flux2Policy(
    pipe.dit,
    embedded_guidance=config.sample.embedded_guidance,
    checkpointing=False,
).eval()

for noise_level in [0.0, 0.6, 0.8]:
    images, _, _ = pipeline_with_logprob(
        pipe=pipe,
        policy=policy,
        prompt="A studio photograph of a red ceramic teapot.",
        height=512,
        width=512,
        num_inference_steps=10,
        group_size=1,
        cfg_scale=(
            config.sample.guidance_scale if config.train.cfg else 1.0
        ),
        noise_level=noise_level,
        sde_type=config.sample.sde_type,
        cps_logprob_type=config.sample.cps_logprob_type,
        sde_window=(2, 4),
        base_seed=42,
    )
    images[0].save(f"flux2_sde_noise_{noise_level:.1f}.png")
