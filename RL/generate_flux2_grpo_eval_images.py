import argparse
import io
import json
import os
import sys
import tarfile
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image, ImageOps
from safetensors.torch import load_file
from torch.distributed.elastic.multiprocessing.errors import record


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
DEFAULT_COMPONENTS_MODEL = (
    "/mnt/image-edit/datasets/duanyufa/"
    "FLUX.2-klein-base-4B"
)
DEFAULT_METADATA = (
    "/mnt/image-edit/datasets/duanyufa/"
    "DiffSynth-Studio/data/test_clean.jsonl"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate evaluation images from a FLUX.2 Flow-GRPO LoRA checkpoint."
        )
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help=(
            "Checkpoint directory such as checkpoint-300, or a direct "
            "lora/lora_ema .safetensors path."
        ),
    )
    parser.add_argument(
        "--lora_name",
        default="lora_ema.safetensors",
        choices=["lora_ema.safetensors", "lora.safetensors"],
        help="Used only when --checkpoint is a directory.",
    )
    parser.add_argument(
        "--base_model",
        default=None,
        help="Overrides pretrained.model from checkpoint config.json.",
    )
    parser.add_argument(
        "--components_model",
        default=None,
        help=(
            "Source for text encoder, VAE, and tokenizer. Defaults to "
            "--base_model or the local Klein Base 4B path."
        ),
    )
    parser.add_argument(
        "--diffsynth_root",
        default=None,
        help="Overrides pretrained.diffsynth_root from checkpoint config.json.",
    )
    parser.add_argument(
        "--init_dit_path",
        default=None,
        help=(
            "Optional full DiT initialization. Defaults to "
            "pretrained.init_dit_path from checkpoint config.json."
        ),
    )
    parser.add_argument("--metadata_path", default=DEFAULT_METADATA)
    parser.add_argument("--reference_dir", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_column", default="image")
    parser.add_argument("--caption_column", default="caption")
    parser.add_argument("--tar_column", default="tar_file")
    parser.add_argument("--tar_cache_size", type=int, default=8)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Zero generates every metadata record.",
    )
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=1024 * 1024)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--embedded_guidance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rand_device",
        choices=["cpu", "cuda"],
        default="cuda",
    )
    parser.add_argument("--lora_scale", type=float, default=1.0)
    return parser.parse_args()


def distributed_context():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FLUX.2 inference.")
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(hours=6),
        )
    rank = dist.get_rank() if dist.is_initialized() else 0
    return rank, local_rank, world_size


def barrier():
    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])


def read_metadata(path, max_samples):
    records = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
                if max_samples > 0 and len(records) >= max_samples:
                    break
    if not records:
        raise ValueError(f"No records found in {path}.")
    return records


def resolve_checkpoint(checkpoint, lora_name):
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        config_path = checkpoint / "config.json"
        lora_path = checkpoint / lora_name
    else:
        config_path = checkpoint.parent / "config.json"
        lora_path = checkpoint
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint config: {config_path}")
    if not lora_path.is_file():
        raise FileNotFoundError(f"Missing LoRA checkpoint: {lora_path}")
    return config_path, lora_path


def read_checkpoint_config(config_path):
    with Path(config_path).open("r", encoding="utf-8") as file:
        return json.load(file)


def model_weight_files(directory):
    files = sorted(str(path) for path in Path(directory).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors weights found under {directory}")
    return files[0] if len(files) == 1 else files


def validate_args(args, config, lora_path):
    pretrained = config.get("pretrained", {})
    base_model = Path(args.base_model or pretrained.get("model", ""))
    components_model = Path(
        args.components_model
        or args.base_model
        or pretrained.get("model", "")
        or DEFAULT_COMPONENTS_MODEL
    )
    diffsynth_root = Path(
        args.diffsynth_root or pretrained.get("diffsynth_root", "")
    )
    init_dit_path = args.init_dit_path
    if init_dit_path is None:
        init_dit_path = pretrained.get("init_dit_path") or ""

    required = [
        base_model / "transformer",
        components_model / "text_encoder",
        components_model / "vae" / "diffusion_pytorch_model.safetensors",
        components_model / "tokenizer",
        Path(args.metadata_path),
        lora_path,
    ]
    if diffsynth_root:
        required.append(diffsynth_root)
    if init_dit_path:
        required.append(Path(init_dit_path))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))
    if (args.height is None) != (args.width is None):
        raise ValueError("height and width must either both be set or both unset.")
    if args.height is not None and (args.height % 16 or args.width % 16):
        raise ValueError("height and width must be divisible by 16.")
    if args.max_pixels < 16 * 16:
        raise ValueError("max_pixels must be at least 256.")
    if args.reference_dir and not Path(args.reference_dir).is_dir():
        raise FileNotFoundError(
            f"Reference directory does not exist: {args.reference_dir}"
        )
    return base_model, components_model, diffsynth_root, init_dit_path


def load_pipeline(args, config, lora_path, device):
    base_model, components_model, diffsynth_root, init_dit_path = validate_args(
        args,
        config,
        lora_path,
    )
    if diffsynth_root:
        sys.path.insert(0, str(diffsynth_root))
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from diffsynth.core import ModelConfig
    from diffsynth.pipelines.flux2_image import Flux2ImagePipeline
    from peft import LoraConfig, inject_adapter_in_model

    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=model_weight_files(components_model / "text_encoder")),
            ModelConfig(path=model_weight_files(base_model / "transformer")),
            ModelConfig(
                path=str(
                    components_model / "vae" / "diffusion_pytorch_model.safetensors"
                )
            ),
        ],
        tokenizer_config=ModelConfig(path=str(components_model / "tokenizer")),
    )
    pipe.freeze_except([])

    if init_dit_path:
        state_dict = load_file(str(init_dit_path), device="cpu")
        missing, unexpected = pipe.dit.load_state_dict(state_dict, strict=False)
        del state_dict
        if unexpected:
            raise RuntimeError(
                f"Unexpected keys while loading init_dit_path: {unexpected[:20]}"
            )
        if missing:
            print(
                f"Loaded init_dit_path with {len(missing)} missing keys "
                "(usually LoRA keys before injection).",
                flush=True,
            )

    train_config = config.get("train", {})
    target_modules = [
        item for item in train_config.get("lora_target_modules", "").split(",") if item
    ]
    if not target_modules:
        raise ValueError("No lora_target_modules found in checkpoint config.json.")
    pipe.dit = inject_adapter_in_model(
        LoraConfig(
            r=int(train_config.get("lora_rank", 32)),
            lora_alpha=int(train_config.get("lora_alpha", 32)),
            target_modules=target_modules,
        ),
        pipe.dit,
    )
    lora_state = load_file(str(lora_path), device="cpu")
    if args.lora_scale != 1.0:
        lora_state = {
            key: value * args.lora_scale
            for key, value in lora_state.items()
        }
    missing, unexpected = pipe.dit.load_state_dict(lora_state, strict=False)
    del lora_state
    if unexpected:
        raise RuntimeError(
            f"Unexpected LoRA keys in {lora_path}: {unexpected[:20]}"
        )
    loaded_lora_keys = [
        key
        for key in pipe.dit.state_dict().keys()
        if "lora_A" in key or "lora_B" in key
    ]
    if not loaded_lora_keys:
        raise RuntimeError("No LoRA layers were injected.")
    pipe.dit.eval()
    return pipe


class TarReaderCache:
    def __init__(self, max_open):
        self.max_open = max_open
        self.archives = OrderedDict()

    def get(self, path):
        path = str(path)
        archive = self.archives.pop(path, None)
        if archive is None:
            archive = tarfile.open(path, mode="r:*")
        self.archives[path] = archive
        while len(self.archives) > self.max_open:
            _, oldest = self.archives.popitem(last=False)
            oldest.close()
        return archive

    def close(self):
        for archive in self.archives.values():
            archive.close()
        self.archives.clear()


class TargetSizeReader:
    def __init__(self, args):
        from diffsynth.core.data.operators import ImageCropAndResize

        self.args = args
        self.reference_dir = Path(args.reference_dir) if args.reference_dir else None
        self.tar_cache = TarReaderCache(args.tar_cache_size)
        self.image_processor = ImageCropAndResize(
            height=args.height,
            width=args.width,
            max_pixels=args.max_pixels,
            height_division_factor=16,
            width_division_factor=16,
        )

    def _reference_path(self, index):
        matches = [
            path
            for path in self.reference_dir.glob(f"{index:08d}.*")
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one reference image for index {index}, got {matches}"
            )
        return matches[0]

    def get(self, index, record):
        if self.args.height is not None:
            return self.args.height, self.args.width
        if self.reference_dir is not None:
            with Image.open(self._reference_path(index)) as image:
                image = ImageOps.exif_transpose(image)
                return self.image_processor.get_height_width(image)
        archive = self.tar_cache.get(record[self.args.tar_column])
        member = archive.extractfile(record[self.args.image_column])
        if member is None:
            raise FileNotFoundError(
                f"{record[self.args.image_column]} not found in "
                f"{record[self.args.tar_column]}"
            )
        with Image.open(io.BytesIO(member.read())) as image:
            image = ImageOps.exif_transpose(image)
            return self.image_processor.get_height_width(image)

    def close(self):
        self.tar_cache.close()


def valid_cached_image(path, width, height):
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            return image.size == (width, height)
    except OSError:
        return False


@torch.inference_mode()
def generate_images(pipe, records, output_dir, rank, world_size, args):
    output_dir.mkdir(parents=True, exist_ok=True)
    size_reader = TargetSizeReader(args)
    try:
        for index in range(rank, len(records), world_size):
            record = records[index]
            height, width = size_reader.get(index, record)
            output_path = output_dir / f"{index:08d}.png"
            if valid_cached_image(output_path, width, height):
                continue
            try:
                image = pipe(
                    prompt=str(record[args.caption_column]),
                    negative_prompt="",
                    cfg_scale=args.cfg_scale,
                    embedded_guidance=args.embedded_guidance,
                    height=height,
                    width=width,
                    seed=args.seed + index,
                    rand_device=args.rand_device,
                    num_inference_steps=args.num_inference_steps,
                )
                temporary_path = output_path.with_suffix(".tmp.png")
                image.save(temporary_path)
                temporary_path.replace(output_path)
            except Exception as error:
                raise RuntimeError(
                    f"Generation failed on rank={rank}, index={index}, "
                    f"size={width}x{height}, seed={args.seed + index}"
                ) from error
            print(
                f"[rank {rank}] {index + 1}/{len(records)} "
                f"({width}x{height}) -> {output_path}",
                flush=True,
            )
    finally:
        size_reader.close()


def validate_outputs(output_dir, num_samples):
    missing = []
    for index in range(num_samples):
        path = output_dir / f"{index:08d}.png"
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(index)
    if missing:
        raise RuntimeError(
            f"{len(missing)} outputs are missing; first indices: {missing[:10]}"
        )


@record
def main():
    args = parse_args()
    config_path, lora_path = resolve_checkpoint(args.checkpoint, args.lora_name)
    config = read_checkpoint_config(config_path)
    rank, local_rank, world_size = distributed_context()
    records = read_metadata(args.metadata_path, args.max_samples)
    output_dir = Path(args.output_dir)

    if rank == 0:
        resolution = (
            f"{args.width}x{args.height}"
            if args.height is not None
            else f"GT aspect ratio with max_pixels={args.max_pixels}"
        )
        init_dit = args.init_dit_path or config.get("pretrained", {}).get(
            "init_dit_path"
        )
        print(
            f"Generating {len(records)} samples with {world_size} process(es).\n"
            f"Checkpoint: {Path(args.checkpoint)}\n"
            f"LoRA: {lora_path}\n"
            f"Init DiT: {init_dit}\n"
            f"Output: {output_dir}\n"
            f"Inference: {resolution}, steps={args.num_inference_steps}, "
            f"cfg={args.cfg_scale}, embedded_guidance={args.embedded_guidance}, "
            f"seed={args.seed}+index"
        )

    pipe = load_pipeline(args, config, lora_path, torch.device("cuda", local_rank))
    generate_images(pipe, records, output_dir, rank, world_size, args)
    barrier()

    if rank == 0:
        validate_outputs(output_dir, len(records))
        manifest = {
            "num_samples": len(records),
            "metadata_path": str(Path(args.metadata_path).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "lora_path": str(lora_path.resolve()),
            "config_path": str(config_path.resolve()),
            "output_dir": str(output_dir.resolve()),
            "reference_dir": (
                str(Path(args.reference_dir).resolve())
                if args.reference_dir
                else None
            ),
            "height": args.height,
            "width": args.width,
            "resolution_mode": (
                "fixed" if args.height is not None else "gt_aspect_ratio"
            ),
            "max_pixels": args.max_pixels,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "embedded_guidance": args.embedded_guidance,
            "seed": args.seed,
            "lora_scale": args.lora_scale,
        }
        manifest_path = output_dir / "generation_config.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Generation complete. Manifest saved to {manifest_path}")

    del pipe
    torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
