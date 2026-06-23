#!/usr/bin/env bash
set -euo pipefail

cd /mnt/image-edit/datasets/duanyufa/flow_grpo

MODE="${1:-train}"
if [[ $# -gt 0 ]]; then
  shift
fi

for arg in "$@"; do
  case "${arg}" in
    --config.save_dir=*)
      OUTPUT_DIR="${arg#--config.save_dir=}"
      ;;
  esac
done

if [[ "${MODE}" == "smoke" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-/mnt/image-edit/datasets/duanyufa/flow_grpo/outputs/flow_grpo_flux2_klein_base_4b_smoke}"
  mkdir -p "${OUTPUT_DIR}"
  accelerate launch \
    --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_processes 1 \
    --main_process_port 29507 \
    scripts/train_flux2.py \
    --config config/grpo.py:flux2_klein_base_4b_smoke \
    --config.save_dir="${OUTPUT_DIR}" \
    "$@"
  exit 0
fi

if [[ "${MODE}" == "pickscore" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-/mnt/image-edit/datasets/duanyufa/flow_grpo/outputs/flow_grpo_flux2_klein_base_4b_pickscore_lora}"
  mkdir -p "${OUTPUT_DIR}"
  accelerate launch \
    --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_processes 8 \
    --main_process_port 29507 \
    scripts/train_flux2.py \
    --config config/grpo.py:flux2_klein_base_4b_pickscore \
    --config.save_dir="${OUTPUT_DIR}" \
    "$@" \
    2>&1 | tee -a "${OUTPUT_DIR}/train.log"
  exit 0
fi

if [[ "${MODE}" == "full" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-/mnt/image-edit/datasets/duanyufa/flow_grpo/outputs/flow_grpo_flux2_klein_base_4b_full}"
  mkdir -p "${OUTPUT_DIR}"
  accelerate launch \
    --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_processes 8 \
    --main_process_port 29507 \
    scripts/train_flux2.py \
    --config config/grpo.py:flux2_klein_base_4b_full \
    --config.save_dir="${OUTPUT_DIR}" \
    "$@" \
    2>&1 | tee -a "${OUTPUT_DIR}/train.log"
  exit 0
fi

if [[ "${MODE}" == "pickscore_full" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-/mnt/image-edit/datasets/duanyufa/flow_grpo/outputs/flow_grpo_flux2_klein_base_4b_pickscore_full}"
  mkdir -p "${OUTPUT_DIR}"
  accelerate launch \
    --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_processes 8 \
    --main_process_port 29507 \
    scripts/train_flux2.py \
    --config config/grpo.py:flux2_klein_base_4b_pickscore_full \
    --config.save_dir="${OUTPUT_DIR}" \
    "$@" \
    2>&1 | tee -a "${OUTPUT_DIR}/train.log"
  exit 0
fi

if [[ "${MODE}" == "pickscore_group16_full" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-/mnt/image-edit/datasets/duanyufa/flow_grpo/outputs/flow_grpo_flux2_klein_base_4b_pickscore_group16_full}"
  mkdir -p "${OUTPUT_DIR}"
  accelerate launch \
    --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_processes 8 \
    --main_process_port 29507 \
    scripts/train_flux2.py \
    --config config/grpo.py:flux2_klein_base_4b_pickscore_group16_full \
    --config.save_dir="${OUTPUT_DIR}" \
    "$@" \
    2>&1 | tee -a "${OUTPUT_DIR}/train.log"
  exit 0
fi

if [[ "${MODE}" == "pickscore_group16" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-/mnt/image-edit/datasets/duanyufa/flow_grpo/outputs/flow_grpo_flux2_klein_base_4b_pickscore_group16_lora}"
  mkdir -p "${OUTPUT_DIR}"
  accelerate launch \
    --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_processes 8 \
    --main_process_port 29507 \
    scripts/train_flux2.py \
    --config config/grpo.py:flux2_klein_base_4b_pickscore_group16 \
    --config.save_dir="${OUTPUT_DIR}" \
    "$@" \
    2>&1 | tee -a "${OUTPUT_DIR}/train.log"
  exit 0
fi

if [[ "${MODE}" == "pickscore_group8" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-/mnt/image-edit/datasets/duanyufa/flow_grpo/outputs/flow_grpo_flux2_klein_base_4b_pickscore_group8_lora}"
  mkdir -p "${OUTPUT_DIR}"
  accelerate launch \
    --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_processes 8 \
    --main_process_port 29507 \
    scripts/train_flux2.py \
    --config config/grpo.py:flux2_klein_base_4b_pickscore_group8 \
    --config.save_dir="${OUTPUT_DIR}" \
    "$@" \
    2>&1 | tee -a "${OUTPUT_DIR}/train.log"
  exit 0
fi

if [[ "${MODE}" == "lora" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-/mnt/image-edit/datasets/duanyufa/flow_grpo/outputs/flow_grpo_flux2_klein_base_4b_lora}"
  mkdir -p "${OUTPUT_DIR}"
  accelerate launch \
    --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_processes 8 \
    --main_process_port 29507 \
    scripts/train_flux2.py \
    --config config/grpo.py:flux2_klein_base_4b_lora \
    --config.save_dir="${OUTPUT_DIR}" \
    "$@" \
    2>&1 | tee -a "${OUTPUT_DIR}/train.log"
  exit 0
fi

if [[ "${MODE}" == "pickscore_lora" ]]; then
  OUTPUT_DIR="${OUTPUT_DIR:-/mnt/image-edit/datasets/duanyufa/flow_grpo/outputs/flow_grpo_flux2_klein_base_4b_pickscore_lora}"
  mkdir -p "${OUTPUT_DIR}"
  accelerate launch \
    --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_processes 8 \
    --main_process_port 29507 \
    scripts/train_flux2.py \
    --config config/grpo.py:flux2_klein_base_4b_pickscore_lora \
    --config.save_dir="${OUTPUT_DIR}" \
    "$@" \
    2>&1 | tee -a "${OUTPUT_DIR}/train.log"
  exit 0
fi

OUTPUT_DIR="${OUTPUT_DIR:-/mnt/image-edit/datasets/duanyufa/flow_grpo/outputs/flow_grpo_flux2_klein_base_4b_lora}"
mkdir -p "${OUTPUT_DIR}"
accelerate launch \
  --config_file scripts/accelerate_configs/multi_gpu.yaml \
  --num_processes 8 \
  --main_process_port 29507 \
  scripts/train_flux2.py \
  --config config/grpo.py:flux2_klein_base_4b \
  --config.save_dir="${OUTPUT_DIR}" \
  "$@" \
  2>&1 | tee -a "${OUTPUT_DIR}/train.log"
