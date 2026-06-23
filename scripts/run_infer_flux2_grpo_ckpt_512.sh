#!/usr/bin/env bash
set -euo pipefail

cd /mnt/image-edit/datasets/duanyufa/flow_grpo

/mnt/image-edit/datasets/duanyufa/conda_envs/flow_grpo/bin/torchrun \
  --standalone \
  --nproc_per_node=8 \
  /mnt/image-edit/datasets/duanyufa/flow_grpo/flow_grpo/generate_flux2_grpo_eval_images.py \
  --checkpoint /mnt/image-edit/datasets/duanyufa/flow_grpo/outputs/flux2_klein_base_4b_fullft1500_pickscore_aesthetic003_lora_group16_cfg4_step30_window3/checkpoint-24 \
  --metadata_path /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/data/test_clean.jsonl \
  --output_dir /mnt/image-edit/datasets/duanyufa/flow_grpo/outputs/flux2_klein_base_4b_fullft1500_pickscore_aesthetic003_lora_group16_cfg4_step30_window3/fid_evaluation/checkpoint-24-1 \
  --max_samples 2000 \
  --height 512 \
  --width 512 \
  --num_inference_steps 50 \
  --cfg_scale 4 \
  --embedded_guidance 1 \
  --seed 0
