#!/usr/bin/env bash
# 合并lora权重
set -euo pipefail
cd "$(dirname "$0")/.."

python -m models.merger \
    --checkpoint_path outputs/siglip2-20261030/best-lora.safetensors \
    --output_path outputs/siglip2-20261030/merged \
    --save_processor
