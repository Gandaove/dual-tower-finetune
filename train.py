"""训练入口。

示例:
  python train.py --model_config configs/model_config.yaml \
      --data_config configs/data.yaml --output_dir ./outputs/exp1 \
      --freeze_mode text --batch_size 64 --accumulate 2 --epochs 20 \
      --num_workers 8 --patience 3
"""
from __future__ import annotations

from utils.trainer import Trainer
from utils.config import get_args


if __name__ == "__main__":
    cfg = get_args()
    Trainer(cfg).run()
