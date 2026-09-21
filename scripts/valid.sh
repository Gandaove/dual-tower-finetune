#!/usr/bin/env bash
# 验证脚本
# 用法: bash scripts/valid.sh --ckpt_path ./outputs/exp1/best.safetensors
set -euo pipefail
cd "$(dirname "$0")/.."

python valid.py \
    --ckpt_path    outputs/tipsv2-20260914/merged/tipsv2-20260915.safetensors \
    --output_dir   outputs/tipsv2-20260914/valid1/ \
    --num_workers 4 \
    "$@"
