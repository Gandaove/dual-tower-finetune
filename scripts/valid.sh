#!/usr/bin/env bash
# 验证脚本
# 用法: bash scripts/valid.sh --ckpt_path ./outputs/exp1/best.safetensors
set -euo pipefail
cd "$(dirname "$0")/.."

python valid.py \
    --ckpt_path    outputs/fgclip2-20261019/merged/fgclip2-20261019.safetensors \
    --output_dir   outputs/fgclip2-20261019/valid1/ \
    --num_workers 4 \
    --save_confusion_matrix
    "$@"
