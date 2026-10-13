#!/usr/bin/env bash
# 训练启动脚本
# 用法: bash scripts/train.sh [extra args...]
set -euo pipefail
cd "$(dirname "$0")/.."

python train.py \
    --model_config configs/model_config.yaml \
    --data_config  configs/data.yaml \
    --output_dir   ./outputs/tipsv2-20261013 \
    --freeze_mode  none \
    --batch_size   12 \
    --accumulate   4 \
    --epochs       50 \
    --num_workers  6 \
    --patience     8 \
    --seed         31554656 \
    "$@"
