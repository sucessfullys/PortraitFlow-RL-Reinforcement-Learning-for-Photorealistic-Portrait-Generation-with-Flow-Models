import torch

from flow_grpo.flux2_utils import Flux2Policy


class FakeDiT(torch.nn.Module):
    def forward(
        self,
        hidden_states,
        timestep,
        guidance,
        encoder_hidden_states,
        **kwargs,
    ):
        condition = encoder_hidden_states.mean().to(hidden_states.dtype)
        return hidden_states + condition


def test_flux2_cfg_matches_diffusers_formula():
    policy = Flux2Policy(FakeDiT(), embedded_guidance=4.0)
    latents = torch.zeros(1, 2, 3)
    positive = torch.full((1, 1, 2), 2.0)
    negative = torch.full((1, 1, 2), 0.5)
    ids = torch.zeros(1, 1, 4)
    output = policy(
        latents,
        500,
        positive,
        ids,
        ids,
        negative,
        ids,
        4.0,
    )
    expected = torch.full_like(latents, 0.5 + 4.0 * (2.0 - 0.5))
    torch.testing.assert_close(output, expected)


if __name__ == "__main__":
    test_flux2_cfg_matches_diffusers_formula()
    print("FLUX.2 CFG policy test passed.")

