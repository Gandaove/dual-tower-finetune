#!/usr/bin/env bash
# 导出脚本
# 用法: bash scripts/export.sh ...
set -euo pipefail
cd "$(dirname "$0")/.."

python export.py \
    --ckpt_path outputs/tipsv2-20260914/merged/tipsv2-20260920.safetensors \
    --output_dir outputs/tipsv2-20260914/trt \
    --target_format engine \
    --precision bf16 \
    --max_batch_size 32 \
    --text close-set \
    --calib_dir /home/syk/pest_cls_dataset/calib_imgs/ \
    --calib_num 128
