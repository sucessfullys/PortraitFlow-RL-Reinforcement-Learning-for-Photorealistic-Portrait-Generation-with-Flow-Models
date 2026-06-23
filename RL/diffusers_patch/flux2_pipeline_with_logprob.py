"""DiffSynth FLUX.2 rollout adapter for Flow-GRPO-Fast."""

from dataclasses import dataclass

import torch
from einops import rearrange

from .flux2_sde_with_logprob import ode_step, sde_step_with_logprob


@dataclass
class Flux2Transition:
    latent: torch.Tensor
    next_latent: torch.Tensor
    old_log_prob: torch.Tensor
    timestep: float
    sigma: float
    sigma_next: float


def prepare_image_ids(height, width, device):
    ids = torch.cartesian_prod(
        torch.arange(1),
        torch.arange(height // 16),
        torch.arange(width // 16),
        torch.arange(1),
    )
    return ids.unsqueeze(0).to(device)


def repeat_batch(tensor, batch_size):
    if tensor is None:
        return None
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] != 1:
        raise ValueError(
            "Expected prompt conditioning batch size 1 or "
            f"{batch_size}, got {tensor.shape[0]}."
        )
    repeats = (batch_size,) + (1,) * (tensor.ndim - 1)
    return tensor.repeat(repeats)


@torch.no_grad()
def encode_prompt(pipe, prompt):
    from diffsynth.pipelines.flux2_image import (
        Flux2Unit_PromptEmbedder,
        Flux2Unit_Qwen3PromptEmbedder,
    )

    if pipe.text_encoder_qwen3 is not None:
        unit = Flux2Unit_Qwen3PromptEmbedder()
        return unit.encode_prompt(
            pipe.text_encoder_qwen3,
            pipe.tokenizer,
            prompt,
            dtype=pipe.torch_dtype,
            device=pipe.device,
        )
    unit = Flux2Unit_PromptEmbedder()
    return unit.encode_prompt(
        pipe.text_encoder,
        pipe.tokenizer,
        prompt,
        dtype=pipe.torch_dtype,
        device=pipe.device,
    )


@torch.no_grad()
def decode_latents(pipe, latents, height, width):
    latent_grid = rearrange(
        latents.to(pipe.torch_dtype),
        "B (H W) C -> B C H W",
        H=height // 16,
        W=width // 16,
    )
    decoded = pipe.vae.decode(latent_grid)
    return [
        pipe.vae_output_to_image(item, pattern="C H W")
        for item in decoded
    ]


@torch.no_grad()
def pipeline_with_logprob(
    pipe,
    policy,
    prompt,
    height,
    width,
    num_inference_steps,
    group_size,
    cfg_scale,
    noise_level,
    sde_type,
    cps_logprob_type,
    sde_window,
    base_seed,
):
    """Collect a prompt group and save only stochastic-window transitions."""
    prompt_embeds, text_ids = encode_prompt(pipe, prompt)
    negative_prompt_embeds, negative_text_ids = encode_prompt(pipe, "")
    image_ids = prepare_image_ids(height, width, pipe.device)
    pipe.scheduler.set_timesteps(
        num_inference_steps,
        dynamic_shift_len=(height // 16) * (width // 16),
    )
    sigmas = pipe.scheduler.sigmas.float().tolist()
    timesteps = pipe.scheduler.timesteps.float().tolist()
    window_start, window_end = sde_window

    # Flow-GRPO-Fast shares one deterministic ODE prefix, then branches the
    # group only where stochastic transitions are trained.
    generator = torch.Generator(device="cpu").manual_seed(base_seed)
    prefix_latent = torch.randn(
        (1, (height // 16) * (width // 16), 128),
        generator=generator,
        dtype=torch.float32,
    ).to(device=pipe.device, dtype=pipe.torch_dtype)
    for index in range(window_start):
        sigma = sigmas[index]
        sigma_next = sigmas[index + 1]
        prediction = policy(
            prefix_latent,
            timesteps[index],
            prompt_embeds,
            text_ids,
            image_ids,
            negative_prompt_embeds,
            negative_text_ids,
            cfg_scale,
        )
        prefix_latent = ode_step(
            prediction, prefix_latent, sigma, sigma_next
        )

    group_prompt_embeds = repeat_batch(prompt_embeds, group_size)
    group_text_ids = repeat_batch(text_ids, group_size)
    group_negative_prompt_embeds = repeat_batch(
        negative_prompt_embeds, group_size
    )
    group_negative_text_ids = repeat_batch(negative_text_ids, group_size)
    group_image_ids = repeat_batch(image_ids, group_size)

    latent = prefix_latent.expand(group_size, *prefix_latent.shape[1:]).clone()
    trajectories = [[] for _ in range(group_size)]
    for index in range(window_start, len(timesteps)):
        sigma = sigmas[index]
        timestep = timesteps[index]
        sigma_next = sigmas[index + 1] if index + 1 < len(sigmas) else 0.0
        prediction = policy(
            latent,
            timestep,
            group_prompt_embeds,
            group_text_ids,
            group_image_ids,
            group_negative_prompt_embeds,
            group_negative_text_ids,
            cfg_scale,
        )
        if window_start <= index < window_end:
            next_latent, old_log_prob, _, _ = sde_step_with_logprob(
                prediction,
                latent,
                sigma,
                sigma_next,
                noise_level=noise_level,
                sde_type=sde_type,
                cps_logprob_type=cps_logprob_type,
            )
            for group_index in range(group_size):
                trajectories[group_index].append(
                    Flux2Transition(
                        latent=latent[group_index : group_index + 1]
                        .detach()
                        .cpu(),
                        next_latent=next_latent[group_index : group_index + 1]
                        .detach()
                        .cpu(),
                        old_log_prob=old_log_prob[
                            group_index : group_index + 1
                        ]
                        .detach()
                        .cpu(),
                        timestep=timestep,
                        sigma=sigma,
                        sigma_next=sigma_next,
                    )
                )
            latent = next_latent.to(pipe.torch_dtype)
        else:
            latent = ode_step(prediction, latent, sigma, sigma_next)

    images = decode_latents(pipe, latent, height, width)

    conditioning = {
        "prompt_embeds": prompt_embeds.detach(),
        "text_ids": text_ids.detach(),
        "negative_prompt_embeds": negative_prompt_embeds.detach(),
        "negative_text_ids": negative_text_ids.detach(),
        "image_ids": image_ids,
    }
    return images, trajectories, conditioning
