"""SDE transition and log probability for DiffSynth FLUX.2 schedules."""

import math

import torch


def sde_step_with_logprob(
    model_output,
    sample,
    sigma,
    sigma_next,
    noise_level=0.8,
    sde_type="cps",
    cps_logprob_type="repo",
    prev_sample=None,
    generator=None,
):
    model_output = model_output.float()
    sample = sample.float()
    sigma = torch.as_tensor(sigma, device=sample.device, dtype=torch.float32)
    sigma_next = torch.as_tensor(
        sigma_next, device=sample.device, dtype=torch.float32
    )
    dt = sigma_next - sigma
    if not bool(dt < 0):
        raise ValueError(f"FLUX.2 sigmas must decrease, got dt={dt.item()}.")

    if sde_type == "sde":
        sigma_denom = torch.where(sigma >= 1.0, sigma_next, sigma)
        sigma_denom = sigma_denom.clamp(min=1e-6, max=1 - 1e-6)
        std_dev = torch.sqrt(sigma / (1 - sigma_denom)) * noise_level
        prev_sample_mean = (
            sample * (1 + std_dev.square() / (2 * sigma) * dt)
            + model_output
            * (1 + std_dev.square() * (1 - sigma) / (2 * sigma))
            * dt
        )
        transition_std = (std_dev * torch.sqrt(-dt)).clamp_min(1e-6)
        log_prob_constants = True
    elif sde_type == "cps":
        std_dev = sigma_next * math.sin(noise_level * math.pi / 2)
        pred_original_sample = sample - sigma * model_output
        noise_estimate = sample + model_output * (1 - sigma)
        clean_coeff = 1 - sigma_next
        noise_coeff = torch.sqrt(
            (sigma_next.square() - std_dev.square()).clamp_min(0.0)
        )
        prev_sample_mean = (
            pred_original_sample * clean_coeff
            + noise_estimate * noise_coeff
        )
        transition_std = torch.as_tensor(
            std_dev, device=sample.device, dtype=torch.float32
        ).clamp_min(1e-6)
        log_prob_constants = False
    else:
        raise ValueError(f"Unknown FLUX.2 sde_type: {sde_type!r}.")

    if prev_sample is None:
        noise = torch.randn(
            prev_sample_mean.shape,
            generator=generator,
            device=prev_sample_mean.device,
            dtype=prev_sample_mean.dtype,
        )
        prev_sample = prev_sample_mean + transition_std * noise
    else:
        prev_sample = prev_sample.float()

    if log_prob_constants:
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean).square())
            / (2 * transition_std.square())
            - torch.log(transition_std)
            - 0.5 * math.log(2 * math.pi)
        )
    else:
        centered_square = (prev_sample.detach() - prev_sample_mean).square()
        if cps_logprob_type == "repo":
            # Match the original Flow-GRPO CPS implementation. This omits the
            # Gaussian variance scale and is kept as the default for parity.
            log_prob = -centered_square
        elif cps_logprob_type == "gaussian":
            # Strict Gaussian transition log-prob up to additive constants.
            # The -log(std) and -0.5*log(2*pi) terms cancel in the same-step
            # importance ratio, but the variance scale does not.
            log_prob = -centered_square / (
                2 * transition_std.square().clamp_min(1e-12)
            )
        else:
            raise ValueError(
                "Unknown FLUX.2 CPS log-prob mode: "
                f"{cps_logprob_type!r}. Expected 'repo' or 'gaussian'."
            )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return prev_sample, log_prob, prev_sample_mean, transition_std


def ode_step(model_output, sample, sigma, sigma_next):
    return sample + model_output * (float(sigma_next) - float(sigma))
