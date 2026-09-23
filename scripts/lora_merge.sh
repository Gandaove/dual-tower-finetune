#!/usr/bin/env bash
# 合并lora权重
set -euo pipefail
cd "$(dirname "$0")/.."

python -m models.merger \
    --checkpoint_path outputs/fgclip2-20260923/best-lora.safetensors \
    --output_path outputs/fgclip2-20260923/merge \
    --save_processor
