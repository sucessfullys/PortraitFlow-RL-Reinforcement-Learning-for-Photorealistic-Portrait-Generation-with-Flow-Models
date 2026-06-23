"""Flow-GRPO-Fast training for DiffSynth-Studio FLUX.2 Klein."""

from collections import defaultdict
import contextlib
import datetime
import json
import math
import os
from pathlib import Path
import random
import time

from absl import app, flags
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from ml_collections import config_flags
import numpy as np
from PIL import Image
from safetensors.torch import load_file, save_file
import torch
from torch.utils.data import DataLoader, DistributedSampler

import flow_grpo.rewards
from flow_grpo.diffusers_patch.flux2_pipeline_with_logprob import (
    pipeline_with_logprob,
)
from flow_grpo.diffusers_patch.flux2_sde_with_logprob import (
    sde_step_with_logprob,
)
from flow_grpo.flux2_utils import (
    Flux2Policy,
    MetadataPromptDataset,
    build_flux2_pipeline,
    lora_state_dict,
)
from flow_grpo.ema import EMAModuleWrapper


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", "config/grpo.py:flux2_klein_base_4b", "Training config."
)


def image_list_to_tensor(images):
    arrays = [np.asarray(image.convert("RGB"), dtype=np.float32) for image in images]
    tensor = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)
    return tensor / 255.0


def repeat_batch(tensor, batch_size):
    if tensor is None:
        return None
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] != 1:
        raise ValueError(
            "Expected conditioning batch size 1 or "
            f"{batch_size}, got {tensor.shape[0]}."
        )
    repeats = (batch_size,) + (1,) * (tensor.ndim - 1)
    return tensor.repeat(repeats)


def build_lr_scheduler(optimizer, max_steps, warmup_steps, min_lr_ratio):
    def multiplier(step):
        if warmup_steps and step < warmup_steps:
            return max(1e-8, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        progress = min(max(progress, 0), 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def mix_seed(*values):
    seed = 0x9E3779B97F4A7C15
    for value in values:
        item = int(value) & 0xFFFFFFFFFFFFFFFF
        seed ^= item + 0x9E3779B97F4A7C15 + ((seed << 6) & 0xFFFFFFFFFFFFFFFF) + (seed >> 2)
        seed &= 0xFFFFFFFFFFFFFFFF
    return seed % (2**63 - 1)


def save_checkpoint(
    accelerator,
    policy,
    optimizer,
    scheduler,
    ema,
    trainable_parameters,
    config,
    global_step,
):
    checkpoint = Path(config.save_dir) / f"checkpoint-{global_step}"
    if accelerator.is_main_process:
        checkpoint.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(policy)
        if config.use_lora:
            save_file(
                lora_state_dict(unwrapped),
                checkpoint / "lora.safetensors",
            )
            if config.train.ema and ema is not None:
                ema.copy_ema_to(trainable_parameters, store_temp=True)
                save_file(
                    lora_state_dict(unwrapped),
                    checkpoint / "lora_ema.safetensors",
                )
                ema.copy_temp_to(trainable_parameters)
        else:
            save_file(
                {
                    key: value.detach().cpu().contiguous()
                    for key, value in unwrapped.dit.state_dict().items()
                },
                checkpoint / "student.safetensors",
            )
            if config.train.ema and ema is not None:
                ema.copy_ema_to(trainable_parameters, store_temp=True)
                save_file(
                    {
                        key: value.detach().cpu().contiguous()
                        for key, value in unwrapped.dit.state_dict().items()
                    },
                    checkpoint / "student_ema.safetensors",
                )
                ema.copy_temp_to(trainable_parameters)
        torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
        torch.save(scheduler.state_dict(), checkpoint / "scheduler.pt")
        if config.train.ema and ema is not None:
            torch.save(ema.state_dict(), checkpoint / "ema.pt")
        with (checkpoint / "trainer_state.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump({"global_step": global_step}, file)
        with (checkpoint / "config.json").open("w", encoding="utf-8") as file:
            json.dump(config.to_dict(), file, indent=2)
    accelerator.wait_for_everyone()


def load_checkpoint(
    policy,
    optimizer,
    scheduler,
    ema,
    trainable_parameters,
    checkpoint,
):
    checkpoint = Path(checkpoint)
    lora_path = checkpoint / "lora.safetensors"
    student_path = checkpoint / "student.safetensors"
    if lora_path.is_file():
        policy.dit.load_state_dict(load_file(str(lora_path)), strict=False)
    elif student_path.is_file():
        policy.dit.load_state_dict(load_file(str(student_path)), strict=True)
    else:
        raise FileNotFoundError(f"No model weights found in {checkpoint}.")
    optimizer.load_state_dict(
        torch.load(
            checkpoint / "optimizer.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    scheduler.load_state_dict(
        torch.load(
            checkpoint / "scheduler.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    ema_path = checkpoint / "ema.pt"
    if ema is not None and ema_path.is_file():
        ema.load_state_dict(
            torch.load(
                ema_path,
                map_location="cpu",
                weights_only=False,
            )
        )
    elif ema is not None:
        ema.ema_parameters = [
            parameter.clone().detach().to(ema.device)
            for parameter in trainable_parameters
        ]
    with (checkpoint / "trainer_state.json").open(
        "r", encoding="utf-8"
    ) as file:
        return int(json.load(file)["global_step"])


def save_sample_group(images, prompt, rewards, config, global_step):
    sample_dir = Path(config.save_dir) / "samples" / f"step-{global_step}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images):
        image.save(sample_dir / f"{index:02d}.jpg", quality=95)
    (sample_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    with (sample_dir / "rewards.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                key: torch.as_tensor(value).float().cpu().tolist()
                for key, value in rewards.items()
            },
            file,
            indent=2,
        )


def reference_policy_context(accelerator, policy, config):
    if config.train.beta <= 0:
        return contextlib.nullcontext()
    if not config.use_lora:
        raise NotImplementedError(
            "FLUX.2 reference-policy KL currently uses LoRA "
            "disable_adapter(), so set config.use_lora=True or "
            "config.train.beta=0 for full-parameter GRPO."
        )
    dit = accelerator.unwrap_model(policy).dit
    disable_adapter = getattr(dit, "disable_adapter", None)
    if disable_adapter is None:
        raise AttributeError(
            "The FLUX.2 DIT does not expose disable_adapter(); reference KL "
            "requires a PEFT/LoRA-wrapped transformer."
        )
    return disable_adapter()


def main(_):
    config = FLAGS.config
    if config.train.beta > 0 and not config.use_lora:
        raise NotImplementedError(
            "FLUX.2 reference-policy KL currently uses LoRA "
            "disable_adapter(), so set config.use_lora=True or "
            "config.train.beta=0 for full-parameter GRPO."
        )
    if config.sample.train_batch_size != 1 or config.train.batch_size != 1:
        raise NotImplementedError(
            "FLUX.2 GRPO currently supports one prompt per process per "
            "rollout. Keep config.sample.train_batch_size=1 and "
            "config.train.batch_size=1; use config.sample.num_image_per_prompt "
            "to control the GRPO group size."
        )
    if not config.run_name:
        config.run_name = "flux2_" + datetime.datetime.now().strftime(
            "%Y.%m.%d_%H.%M.%S"
        )
    os.makedirs(config.save_dir, exist_ok=True)

    project_config = ProjectConfiguration(
        project_dir=config.save_dir,
        automatic_checkpoint_naming=False,
        total_limit=config.num_checkpoint_limit,
    )
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        project_config=project_config,
    )
    set_seed(config.seed, device_specific=True)
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    pipe = build_flux2_pipeline(config, accelerator.device)
    policy = Flux2Policy(
        pipe.dit,
        embedded_guidance=config.sample.embedded_guidance,
        checkpointing=config.activation_checkpointing,
    )
    trainable_parameters = [
        parameter for parameter in policy.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )
    scheduler = build_lr_scheduler(
        optimizer,
        config.max_steps,
        config.train.lr_warmup_steps,
        config.train.min_lr_ratio,
    )

    dataset = MetadataPromptDataset(
        config.dataset, config.dataset_prompt_column
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        seed=config.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=config.dataset_num_workers,
        collate_fn=lambda batch: batch[0],
    )
    policy, optimizer = accelerator.prepare(policy, optimizer)
    pipe.dit = accelerator.unwrap_model(policy).dit
    trainable_parameters = [
        parameter
        for parameter in accelerator.unwrap_model(policy).parameters()
        if parameter.requires_grad
    ]
    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(
            trainable_parameters,
            decay=config.train.ema_decay,
            update_step_interval=config.train.ema_update_step_interval,
            device=accelerator.device,
        )

    global_step = 0
    if config.resume_from:
        global_step = load_checkpoint(
            accelerator.unwrap_model(policy),
            optimizer,
            scheduler,
            ema,
            trainable_parameters,
            config.resume_from,
        )

    reward_fn = flow_grpo.rewards.multi_score(
        accelerator.device, config.reward_fn
    )
    autocast = (
        accelerator.autocast
        if accelerator.device.type == "cuda"
        else contextlib.nullcontext
    )

    if accelerator.is_main_process:
        trainable_count = sum(item.numel() for item in trainable_parameters)
        print(config)
        print(
            "FLUX.2 Flow-GRPO: "
            f"processes={accelerator.num_processes}, "
            f"group_size={config.sample.num_image_per_prompt}, "
            f"trainable_params={trainable_count:,}, "
            f"resolution={config.resolution}, "
            f"sampling_steps={config.sample.num_steps}, "
            f"ema={config.train.ema}"
        )

    epoch = 0
    while global_step < config.max_steps:
        sampler.set_epoch(epoch)
        for batch_index, (prompt, metadata) in enumerate(dataloader):
            if global_step >= config.max_steps:
                break
            if batch_index >= config.sample.num_batches_per_epoch:
                break
            batch_start_time = time.perf_counter()

            window_seed = mix_seed(
                config.seed,
                global_step,
                epoch,
                batch_index,
                accelerator.process_index,
            )
            window_rng = random.Random(window_seed)
            start_min, start_max = config.sample.sde_window_range
            window_start = window_rng.randint(start_min, start_max)
            window_end = window_start + config.sample.sde_window_size
            if window_end > config.sample.num_steps - 1:
                raise ValueError(
                    "SDE window must exclude the final singular transition."
                )

            policy.eval()
            with autocast():
                images, trajectories, conditioning = pipeline_with_logprob(
                    pipe=pipe,
                    policy=policy,
                    prompt=prompt,
                    height=config.resolution,
                    width=config.resolution,
                    num_inference_steps=config.sample.num_steps,
                    group_size=config.sample.num_image_per_prompt,
                    cfg_scale=(
                        config.sample.guidance_scale
                        if config.train.cfg
                        else 1.0
                    ),
                    noise_level=config.sample.noise_level,
                    sde_type=config.sample.sde_type,
                    cps_logprob_type=config.sample.cps_logprob_type,
                    sde_window=(window_start, window_end),
                    base_seed=mix_seed(
                        config.seed
                        + 17,
                        global_step,
                        epoch,
                        batch_index,
                        accelerator.process_index,
                    ),
                )
            rollout_seconds = time.perf_counter() - batch_start_time

            reward_images = image_list_to_tensor(images)
            reward_details, _ = reward_fn(
                reward_images,
                [prompt] * len(images),
                [metadata] * len(images),
            )
            reward_seconds = (
                time.perf_counter() - batch_start_time - rollout_seconds
            )
            rewards = torch.as_tensor(
                reward_details["avg"],
                device=accelerator.device,
                dtype=torch.float32,
            )
            reward_std = rewards.std(unbiased=False)
            if reward_std < config.train.min_reward_std:
                if accelerator.is_main_process:
                    print(
                        f"step={global_step} skipped_zero_reward_std=1 "
                        f"reward_std={reward_std.item():.3e} "
                        f"min_reward_std={config.train.min_reward_std:.3e} "
                        f"rollout_s={rollout_seconds:.2f} "
                        f"reward_s={reward_seconds:.2f} "
                        f"prompt={prompt[:120]!r}",
                        flush=True,
                    )
                continue
            advantages = (rewards - rewards.mean()) / (reward_std + 1e-6)
            advantages = advantages.clamp(
                -config.train.adv_clip_max,
                config.train.adv_clip_max,
            )

            policy.train()
            transition_count = sum(len(item) for item in trajectories)
            transition_steps = list(zip(*trajectories))
            if transition_steps and transition_count != (
                len(trajectories) * len(transition_steps)
            ):
                raise ValueError(
                    "All rollout trajectories must have the same number of "
                    "transitions for batched replay."
                )
            replay_batch_size = len(trajectories)
            replay_prompt_embeds = repeat_batch(
                conditioning["prompt_embeds"], replay_batch_size
            )
            replay_text_ids = repeat_batch(
                conditioning["text_ids"], replay_batch_size
            )
            replay_image_ids = repeat_batch(
                conditioning["image_ids"], replay_batch_size
            )
            replay_negative_prompt_embeds = repeat_batch(
                conditioning["negative_prompt_embeds"], replay_batch_size
            )
            replay_negative_text_ids = repeat_batch(
                conditioning["negative_text_ids"], replay_batch_size
            )
            for inner_epoch in range(config.train.num_inner_epochs):
                train_start_time = time.perf_counter()
                stats = defaultdict(float)
                max_ratio_error = 0.0
                with accelerator.accumulate(policy):
                    for transitions in transition_steps:
                        latent = torch.cat(
                            [
                                transition.latent
                                for transition in transitions
                            ],
                            dim=0,
                        ).to(accelerator.device, dtype=pipe.torch_dtype)
                        next_latent = torch.cat(
                            [
                                transition.next_latent
                                for transition in transitions
                            ],
                            dim=0,
                        ).to(accelerator.device, dtype=torch.float32)
                        old_log_prob = torch.cat(
                            [
                                transition.old_log_prob
                                for transition in transitions
                            ],
                            dim=0,
                        ).to(accelerator.device)
                        transition = transitions[0]
                        with autocast():
                            prediction = policy(
                                latent,
                                transition.timestep,
                                replay_prompt_embeds,
                                replay_text_ids,
                                replay_image_ids,
                                replay_negative_prompt_embeds,
                                replay_negative_text_ids,
                                (
                                    config.sample.guidance_scale
                                    if config.train.cfg
                                    else 1.0
                                ),
                            )
                        (
                            _,
                            log_prob,
                            prev_sample_mean,
                            transition_std,
                        ) = sde_step_with_logprob(
                            prediction,
                            latent,
                            transition.sigma,
                            transition.sigma_next,
                            noise_level=config.sample.noise_level,
                            sde_type=config.sample.sde_type,
                            cps_logprob_type=(
                                config.sample.cps_logprob_type
                            ),
                            prev_sample=next_latent,
                        )
                        kl_loss = torch.zeros_like(log_prob)
                        if config.train.beta > 0:
                            with torch.no_grad():
                                with reference_policy_context(
                                    accelerator, policy, config
                                ):
                                    with autocast():
                                        ref_prediction = policy(
                                            latent,
                                            transition.timestep,
                                            replay_prompt_embeds,
                                            replay_text_ids,
                                            replay_image_ids,
                                            replay_negative_prompt_embeds,
                                            replay_negative_text_ids,
                                            (
                                                config.sample.guidance_scale
                                                if config.train.cfg
                                                else 1.0
                                            ),
                                        )
                                (
                                    _,
                                    _,
                                    prev_sample_mean_ref,
                                    _,
                                ) = sde_step_with_logprob(
                                    ref_prediction,
                                    latent,
                                    transition.sigma,
                                    transition.sigma_next,
                                    noise_level=(
                                        config.sample.noise_level
                                    ),
                                    sde_type=config.sample.sde_type,
                                    cps_logprob_type=(
                                        config.sample.cps_logprob_type
                                    ),
                                    prev_sample=next_latent,
                                )
                            kl_loss = (
                                (
                                    prev_sample_mean
                                    - prev_sample_mean_ref
                                )
                                .square()
                                / (
                                    2
                                    * transition_std.square().clamp_min(
                                        1e-12
                                    )
                                )
                            ).mean(
                                dim=tuple(
                                    range(1, prev_sample_mean.ndim)
                                )
                            )
                        log_ratio = log_prob - old_log_prob
                        ratio = torch.exp(log_ratio.clamp(-20, 20))
                        max_ratio_error = max(
                            max_ratio_error,
                            float(torch.abs(ratio.detach() - 1).max()),
                        )
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * ratio.clamp(
                            1 - config.train.clip_range,
                            1 + config.train.clip_range,
                        )
                        policy_loss = torch.maximum(
                            unclipped_loss, clipped_loss
                        )
                        approx_kl = 0.5 * log_ratio.square()
                        loss = (
                            policy_loss + config.train.beta * kl_loss
                        ).mean() / max(1, len(transition_steps))
                        accelerator.backward(loss)

                        stats["policy_loss"] += float(
                            policy_loss.detach().sum()
                        )
                        stats["approx_kl"] += float(
                            approx_kl.detach().sum()
                        )
                        stats["kl_loss"] += float(
                            kl_loss.detach().sum()
                        )
                        stats["ratio"] += float(ratio.detach().sum())
                        stats["clipfrac"] += float(
                            (
                                torch.abs(ratio - 1)
                                > config.train.clip_range
                            )
                            .float()
                            .detach()
                            .sum()
                        )

                    if (
                        inner_epoch == 0
                        and config.train.verify_on_policy
                        and max_ratio_error
                        > config.train.on_policy_ratio_tolerance
                    ):
                        raise RuntimeError(
                            "On-policy consistency check failed before the "
                            "optimizer update: max |ratio-1|="
                            f"{max_ratio_error:.6g}, tolerance="
                            f"{config.train.on_policy_ratio_tolerance:.6g}. "
                            "Rollout and replay model paths are inconsistent."
                        )

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            trainable_parameters,
                            config.train.max_grad_norm,
                        )
                    optimizer.step()
                    if accelerator.sync_gradients:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    did_optimizer_step = accelerator.sync_gradients
                    if did_optimizer_step:
                        global_step += 1
                        if ema is not None:
                            ema.step(trainable_parameters, global_step)
                        denominator = max(1, transition_count)
                        if (
                            accelerator.is_main_process
                            and global_step % config.log_every == 0
                        ):
                            reward_text = " ".join(
                                f"reward_{key}="
                                f"{torch.as_tensor(value).float().mean().item():.4f}"
                                for key, value in reward_details.items()
                            )
                            print(
                                f"step={global_step} "
                                f"inner_epoch={inner_epoch} "
                                f"policy_loss="
                                f"{stats['policy_loss']/denominator:.6f} "
                                f"approx_kl="
                                f"{stats['approx_kl']/denominator:.6f} "
                                f"kl_loss="
                                f"{stats['kl_loss']/denominator:.6f} "
                                f"ratio={stats['ratio']/denominator:.4f} "
                                f"ratio_error_max={max_ratio_error:.3e} "
                                f"clipfrac="
                                f"{stats['clipfrac']/denominator:.4f} "
                                f"reward_std={reward_std.item():.4f} "
                                f"lr={scheduler.get_last_lr()[0]:.3e} "
                                f"sde_type={config.sample.sde_type} "
                                f"cps_logprob="
                                f"{config.sample.cps_logprob_type} "
                                f"window={window_start}:{window_end} "
                                f"{reward_text}",
                                flush=True,
                            )
                        if (
                            accelerator.is_main_process
                            and config.save_samples_steps > 0
                            and global_step % config.save_samples_steps == 0
                        ):
                            save_sample_group(
                                images,
                                prompt,
                                reward_details,
                                config,
                                global_step,
                            )
                        if (
                            config.checkpointing_steps > 0
                            and global_step % config.checkpointing_steps == 0
                        ):
                            save_checkpoint(
                                accelerator,
                                policy,
                                optimizer,
                                scheduler,
                                ema,
                                trainable_parameters,
                                config,
                                global_step,
                            )
                if accelerator.is_main_process:
                    train_seconds = time.perf_counter() - train_start_time
                    total_seconds = time.perf_counter() - batch_start_time
                    print(
                        f"rollout_batch epoch={epoch} "
                        f"batch_index={batch_index} "
                        f"global_step={global_step} "
                        f"optimizer_step={int(did_optimizer_step)} "
                        f"rollout_s={rollout_seconds:.2f} "
                        f"reward_s={reward_seconds:.2f} "
                        f"train_s={train_seconds:.2f} "
                        f"total_s={total_seconds:.2f} "
                        f"reward_std={reward_std.item():.4f} "
                        f"window={window_start}:{window_end}",
                        flush=True,
                    )
                if global_step >= config.max_steps:
                    break
        epoch += 1

    if (
        config.checkpointing_steps <= 0
        or global_step % config.checkpointing_steps != 0
    ):
        save_checkpoint(
            accelerator,
            policy,
            optimizer,
            scheduler,
            ema,
            trainable_parameters,
            config,
            global_step,
        )
    accelerator.end_training()


if __name__ == "__main__":
    app.run(main)
