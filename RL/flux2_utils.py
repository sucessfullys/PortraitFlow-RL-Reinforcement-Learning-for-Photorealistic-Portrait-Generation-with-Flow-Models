import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import load_file
from torch.utils.data import Dataset


FLUX2_KLEIN_LORA_TARGETS = (
    "to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,"
    "linear_in,linear_out,to_qkv_mlp_proj,"
    + ",".join(
        f"single_transformer_blocks.{index}.attn.to_out"
        for index in range(20)
    )
)


class MetadataPromptDataset(Dataset):
    def __init__(self, path, prompt_column="caption"):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix == ".jsonl":
            prompts = []
            metadata = []
            with path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, 1):
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    prompt = item.get(prompt_column)
                    if not isinstance(prompt, str) or not prompt.strip():
                        raise ValueError(
                            f"Missing {prompt_column!r} at "
                            f"{path}:{line_number}."
                        )
                    prompts.append(prompt.strip())
                    metadata.append(item)
        else:
            with path.open("r", encoding="utf-8") as file:
                prompts = [line.strip() for line in file if line.strip()]
            metadata = [{} for _ in prompts]
        if not prompts:
            raise ValueError(f"No prompts found in {path}.")
        self.prompts = prompts
        self.metadata = metadata

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        return self.prompts[index], self.metadata[index]


class Flux2Policy(torch.nn.Module):
    def __init__(self, dit, embedded_guidance=4.0, checkpointing=True):
        super().__init__()
        self.dit = dit
        self.embedded_guidance = embedded_guidance
        self.checkpointing = checkpointing

    def _predict(self, latents, timestep, prompt_embeds, text_ids, image_ids):
        batch = latents.shape[0]
        timestep = torch.as_tensor(
            timestep, device=latents.device, dtype=latents.dtype
        ).reshape(-1)
        if timestep.numel() == 1:
            timestep = timestep.expand(batch)
        guidance = torch.full(
            (batch,),
            self.embedded_guidance,
            device=latents.device,
            dtype=latents.dtype,
        )
        return self.dit(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=image_ids,
            use_gradient_checkpointing=(
                self.checkpointing and torch.is_grad_enabled()
            ),
            use_gradient_checkpointing_offload=False,
        )

    def forward(
        self,
        latents,
        timestep,
        prompt_embeds,
        text_ids,
        image_ids,
        negative_prompt_embeds=None,
        negative_text_ids=None,
        cfg_scale=1.0,
    ):
        positive = self._predict(
            latents, timestep, prompt_embeds, text_ids, image_ids
        )
        if cfg_scale == 1:
            return positive
        negative = self._predict(
            latents,
            timestep,
            negative_prompt_embeds,
            negative_text_ids,
            image_ids,
        )
        return negative + cfg_scale * (positive - negative)


def build_flux2_pipeline(config, device):
    diffsynth_root = Path(config.pretrained.diffsynth_root)
    sys.path.insert(0, str(diffsynth_root))
    from diffsynth.core import ModelConfig
    from diffsynth.pipelines.flux2_image import Flux2ImagePipeline

    base = Path(config.pretrained.model)
    text_encoder = sorted(
        str(path) for path in (base / "text_encoder").glob("*.safetensors")
    )
    transformer = sorted(
        str(path) for path in (base / "transformer").glob("*.safetensors")
    )
    if not text_encoder or not transformer:
        raise FileNotFoundError(f"Incomplete FLUX.2 model at {base}.")
    transformer_path = (
        transformer[0] if len(transformer) == 1 else transformer
    )
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=text_encoder),
            ModelConfig(path=transformer_path),
            ModelConfig(
                path=str(
                    base / "vae" / "diffusion_pytorch_model.safetensors"
                )
            ),
        ],
        tokenizer_config=ModelConfig(path=str(base / "tokenizer")),
    )
    pipe.freeze_except([])

    init_dit_path = config.pretrained.get("init_dit_path", None)
    if init_dit_path:
        init_dit_path = Path(init_dit_path)
        if not init_dit_path.is_file():
            raise FileNotFoundError(init_dit_path)
        state_dict = load_file(str(init_dit_path), device="cpu")
        missing, unexpected = pipe.dit.load_state_dict(
            state_dict, strict=False
        )
        if missing or unexpected:
            raise RuntimeError(
                "Failed to initialize FLUX.2 DIT from "
                f"{init_dit_path}: missing={missing}, unexpected={unexpected}"
            )

    if config.use_lora:
        targets = [
            item
            for item in config.train.lora_target_modules.split(",")
            if item
        ]
        pipe.dit = inject_adapter_in_model(
            LoraConfig(
                r=config.train.lora_rank,
                lora_alpha=config.train.lora_alpha,
                target_modules=targets,
            ),
            pipe.dit,
        )
        for name, parameter in pipe.dit.named_parameters():
            parameter.requires_grad = "lora_" in name
            if parameter.requires_grad:
                parameter.data = parameter.data.to(torch.bfloat16)
    else:
        pipe.dit.requires_grad_(True)
    return pipe


def lora_state_dict(policy):
    return {
        key: value.detach().cpu().contiguous()
        for key, value in policy.dit.state_dict().items()
        if "lora_A" in key or "lora_B" in key
    }
