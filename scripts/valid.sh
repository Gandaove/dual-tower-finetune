#!/usr/bin/env bash
# 验证脚本
# 用法: bash scripts/valid.sh --ckpt_path ./outputs/exp1/best.safetensors
set -euo pipefail
cd "$(dirname "$0")/.."

python valid.py \
    --ckpt_path    outputs/bioclip2-20261106/merged/bioclip2-20261109.safetensors \
    --output_dir   outputs/bioclip2-20261106/valid1 \
    --num_workers 4 \
    --save_confusion_matrix
    "$@"
