#!/usr/bin/env bash
set -euo pipefail

cd /mnt/image-edit/datasets/duanyufa/flow_grpo

# =========================
# 常用实验参数
# =========================

# 实验输出目录：日志、checkpoint、sample 都会写到这里。
OUTPUT_DIR="/mnt/image-edit/datasets/duanyufa/flow_grpo/outputs/flux2_klein_base_4b_fullft1500_pickscore_aesthetic003_lora_group16_cfg4_step150_lr_3e4_window3"

# GRPO 起始 DiT 权重：这里从 full finetune checkpoint-1500 开始训 LoRA。
INIT_DIT_PATH="/mnt/image-edit/datasets/duanyufa/outputs/flux2_klein_base_4b_full_finetune/checkpoint-1500/student.safetensors"

# 训练数据 jsonl，以及读取 prompt 的字段名。
DATASET="/mnt/zixuan_workspace/caption_scripts/vllm_caption_gemma/caption_splits/all_id_1person_caption_end_analysis/sample_250k.jsonl"
DATASET_PROMPT_COLUMN="caption"

# 总训练步数：optimizer step 数，不是 epoch。
MAX_STEPS=150

# reward 组合：avg = PICK_SCORE_WEIGHT * pickscore + AESTHETIC_WEIGHT * aesthetic。
# aesthetic 分数尺度通常比 pickscore 更敏感，建议从 0.03 或 0.05 这种小权重开始。
PICKSCORE_WEIGHT=1.0
AESTHETIC_WEIGHT=0.03

# 保存间隔：每隔多少 step 保存 checkpoint / sample。
CHECKPOINTING_STEPS=10
SAVE_SAMPLES_STEPS=5

# 断点续训：空字符串表示新训；填 checkpoint 目录会恢复 optimizer/scheduler/EMA。
RESUME_FROM=""
# 例子：
# RESUME_FROM="${OUTPUT_DIR}/checkpoint-50"


# =========================
# 训练规模和采样参数
# =========================

# 使用 GPU 数；单机 8 卡就保持 8。
NUM_PROCESSES=8
MAIN_PROCESS_PORT=29507

# 固定训练分辨率。
RESOLUTION=512

# 训练 rollout 推理步数、保存 sample 的推理步数。
NUM_STEPS=50
EVAL_NUM_STEPS=50

# 只在 denoise 轨迹中的一小段窗口上 replay/反传。
# NUM_STEPS=50 时，(0, 25) 表示随机抽前半段中的窗口；WINDOW_SIZE=3 表示每次优化连续 3 个 transition。
SDE_WINDOW_SIZE=3
SDE_WINDOW_START=0
SDE_WINDOW_END=25

# 外部 CFG 设置：
# 这里不是 FLUX.2 Klein 的 embedded_guidance；Klein 的 embedded_guidance 基本不改变 transformer 输出。
# TRAIN_CFG=true 时，代码会用正 prompt 和空 negative prompt 各 forward 一次，做传统 CFG。
# rollout 和 replay 都走同一个 cfg_scale，所以不会因为 CFG 本身导致 ratio/on-policy 不一致。
# 如果 TRAIN_CFG=false，即使 GUIDANCE_SCALE=4，训练代码里也会强制按 cfg=1 训练。
TRAIN_CFG=true
GUIDANCE_SCALE=4.0
EVAL_GUIDANCE_SCALE=4.0

# GRPO group size：每个 prompt 生成多少张图做组内偏好比较。
NUM_IMAGE_PER_PROMPT=16

# 每个 epoch 采多少个 prompt batch。
# 8 卡、batch=1、group=16 时，64 表示每个 epoch 约 32 个 prompt group，
# 对应 gradient_accumulation_steps=32，每个 epoch 更新 2 次。
NUM_BATCHES_PER_EPOCH=64

# 每卡每次处理的 prompt 数；FLUX.2 这里建议保持 1。
TRAIN_BATCH_SIZE=1


# =========================
# 优化器和 GRPO 参数
# =========================

# 学习率。你之前 1e-5 比较稳；想更猛再试 2e-5 或 5e-5。
LEARNING_RATE=3.0e-4

# warmup 步数。150 step 且 lr=3e-4 时，20 步 warmup 更稳。
LR_WARMUP_STEPS=20

# 梯度累积步数。一般设成 NUM_BATCHES_PER_EPOCH / 2。
GRADIENT_ACCUMULATION_STEPS=32

# 同一批 rollout 数据重复训练几遍；1 最 on-policy。
NUM_INNER_EPOCHS=1

# PPO/GRPO ratio clip，越小越保守。
CLIP_RANGE=0.001

# advantage 裁剪，防止单个样本 reward 差异过大。
ADV_CLIP_MAX=5

# reference KL 权重；0 表示不加 reference KL。
BETA=0.0


# =========================
# LoRA / EMA
# =========================

# LoRA 容量。rank/alpha 通常保持相同。
LORA_RANK=32
LORA_ALPHA=32

# 保存 EMA LoRA；推理测试一般优先用 lora_ema.safetensors。
EMA=true
EMA_DECAY=0.99


mkdir -p "${OUTPUT_DIR}"

CONFIG_FILE="${OUTPUT_DIR}/generated_grpo_config.py"
cat > "${CONFIG_FILE}" <<PY
import sys

sys.path.insert(0, "/mnt/image-edit/datasets/duanyufa/flow_grpo")

from config import grpo as base_grpo


def get_config(*unused_args):
    config = base_grpo.flux2_klein_base_4b_pickscore_aesthetic_group16()
    config.reward_fn = {
        "pickscore": ${PICKSCORE_WEIGHT},
        "aesthetic": ${AESTHETIC_WEIGHT},
    }
    return config
PY

args=(
  accelerate
  launch
  --config_file
  scripts/accelerate_configs/multi_gpu.yaml
  --num_processes
  "${NUM_PROCESSES}"
  --main_process_port
  "${MAIN_PROCESS_PORT}"
  scripts/train_flux2.py
  --config
  "${CONFIG_FILE}"
  "--config.pretrained.init_dit_path=${INIT_DIT_PATH}"
  "--config.dataset=${DATASET}"
  "--config.dataset_prompt_column=${DATASET_PROMPT_COLUMN}"
  "--config.save_dir=${OUTPUT_DIR}"
  "--config.max_steps=${MAX_STEPS}"
  "--config.checkpointing_steps=${CHECKPOINTING_STEPS}"
  "--config.save_samples_steps=${SAVE_SAMPLES_STEPS}"
  "--config.log_every=1"
  "--config.resolution=${RESOLUTION}"
  "--config.sample.num_steps=${NUM_STEPS}"
  "--config.sample.eval_num_steps=${EVAL_NUM_STEPS}"
  "--config.sample.sde_window_size=${SDE_WINDOW_SIZE}"
  "--config.sample.sde_window_range=(${SDE_WINDOW_START}, ${SDE_WINDOW_END})"
  "--config.sample.guidance_scale=${GUIDANCE_SCALE}"
  "--config.sample.eval_guidance_scale=${EVAL_GUIDANCE_SCALE}"
  "--config.sample.num_image_per_prompt=${NUM_IMAGE_PER_PROMPT}"
  "--config.sample.num_batches_per_epoch=${NUM_BATCHES_PER_EPOCH}"
  "--config.sample.train_batch_size=${TRAIN_BATCH_SIZE}"
  "--config.train.batch_size=${TRAIN_BATCH_SIZE}"
  "--config.train.cfg=${TRAIN_CFG}"
  "--config.train.learning_rate=${LEARNING_RATE}"
  "--config.train.lr_warmup_steps=${LR_WARMUP_STEPS}"
  "--config.train.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
  "--config.train.num_inner_epochs=${NUM_INNER_EPOCHS}"
  "--config.train.clip_range=${CLIP_RANGE}"
  "--config.train.adv_clip_max=${ADV_CLIP_MAX}"
  "--config.train.beta=${BETA}"
  "--config.train.lora_rank=${LORA_RANK}"
  "--config.train.lora_alpha=${LORA_ALPHA}"
  "--config.train.ema=${EMA}"
  "--config.train.ema_decay=${EMA_DECAY}"
)

if [[ -n "${RESUME_FROM}" ]]; then
  args+=("--config.resume_from=${RESUME_FROM}")
fi

"${args[@]}" "$@" 2>&1 | tee -a "${OUTPUT_DIR}/train.log"
