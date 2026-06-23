import torch

from flow_grpo.diffusers_patch.flux2_sde_with_logprob import (
    sde_step_with_logprob,
)


def _assert_on_policy_ratio_is_one(sde_type):
    torch.manual_seed(0)
    sample = torch.randn(2, 16, 8)
    velocity = torch.randn_like(sample)
    next_sample, old_log_prob, _, _ = sde_step_with_logprob(
        velocity,
        sample,
        0.8,
        0.7,
        noise_level=0.8,
        sde_type=sde_type,
    )
    _, new_log_prob, _, _ = sde_step_with_logprob(
        velocity,
        sample,
        0.8,
        0.7,
        noise_level=0.8,
        sde_type=sde_type,
        prev_sample=next_sample,
    )
    torch.testing.assert_close(old_log_prob, new_log_prob)
    torch.testing.assert_close(
        torch.exp(new_log_prob - old_log_prob),
        torch.ones_like(old_log_prob),
    )


def test_flux2_sde_on_policy_ratio_is_one():
    _assert_on_policy_ratio_is_one("sde")


def test_flux2_cps_on_policy_ratio_is_one():
    _assert_on_policy_ratio_is_one("cps")


if __name__ == "__main__":
    test_flux2_sde_on_policy_ratio_is_one()
    test_flux2_cps_on_policy_ratio_is_one()
    print("FLUX.2 SDE/CPS replay tests passed.")
