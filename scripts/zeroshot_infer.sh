#!/usr/bin/env bash
# Zeroshot 推理脚本
# 用法: bash scripts/infer.sh ...
set -euo pipefail
cd "$(dirname "$0")/.."

python infer.py \
    --ckpt_path    outputs/fgclip-20260910/best.safetensors \
    --images       tests/test-batch1/ \
    --output       outputs/tests/batch1-results/ \
    --save_sheet \
    --topk         1 \
    --embed_img 
    "$@"