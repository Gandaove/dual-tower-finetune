"""训练编排器 Trainer。

封装 PyTorch Lightning Trainer, 负责:
  - 目录/快照初始化;
  - 通过 EpochEvalCallback(位于 utils.callbacks)在每轮结束后跑完整 Evaluator 评估, 挑选 best 指标;
  - 自定义 safetensors 检查点 (best/last/interval);
  - 早停、Tensorboard logger、merger 导出。
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import TQDMProgressBar, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

from dataset.dataset import build_dual_tower_datamodule, DualTowerDataModule
from models.dual_tower_multimodal import BaseDualTowerModel, build_model as build_dual_tower_model
from utils.callbacks import EpochEvalCallback, MergeCallback, MetricsCSVCallback
from utils.config import Config, dataclass_to_dict, resolve_precision, write_set_config
from utils.evaluator import Evaluator
from utils.logger import get_logger

_log = get_logger("trainer")


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.out = Path(cfg.train.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.best_metric = -1e9 if cfg.model.eval.save_metric_mode == "max" else 1e9
        self.best_epoch = -1
        self.best_source = "regular"  # 记录 best 来自 "regular" 还是 "ema"
        self.start_epoch = 0
        self.model: Optional[BaseDualTowerModel] = None
        self.dm: Optional[DualTowerDataModule] = None
        self.metrics_csv = None
        self.trainer: Optional[pl.Trainer] = None

    def setup(self):
        cfg = self.cfg
        if self.trainer is None or self.trainer.is_global_zero:
            write_set_config(self.out, cfg.model, "set_model_config.json")
            write_set_config(self.out, cfg.data, "set_data_config.json")

        self.dm = build_dual_tower_datamodule(cfg.data, cfg.train)
        self.dm.setup("fit")

        self.model = build_dual_tower_model(cfg, classes=self.dm.taxonomy.cnames, transform=self.dm.transform)

        try:
            self.steps_per_epoch = len(self.dm.train_dataloader())
        except Exception:
            self.steps_per_epoch = 2000
        self.model._steps_per_epoch = self.steps_per_epoch

        if cfg.train.ckpt_path and Path(cfg.train.ckpt_path).is_file():
            _log.info(f"[Trainer] 读取预训练/断点权重: {cfg.train.ckpt_path}")
            meta = self.model.load_safetensors(cfg.train.ckpt_path, load_opt=False)
            if cfg.train.resume:
                self.start_epoch = int(meta.get("epochs", 0)) + 1
                try:
                    self.best_metric = float(meta.get("save_metric", self.best_metric))
                except Exception:
                    pass

        # CSV 记录字段 (覆盖 regular 与 ema 双轨指标)
        ks = cfg.model.eval.retrieval_topk or [1]
        base_keys = [f"i2t_R@{k}" for k in ks] + [f"t2i_R@{k}" for k in ks] + ["zeroshot_f1", "zeroshot_acc"]
        reg_fields = [f"val/{k}" for k in base_keys]
        ema_fields = [f"val_ema/{k}" for k in base_keys]
        fields = ["epoch", "lr", "train_loss", "best_source"] + reg_fields + ema_fields

        self.metrics_csv = MetricsCSVCallback(str(self.out / "training-metrics.csv"), fields)
        self.eval_cb = EpochEvalCallback(self)

        cbs = [
            TQDMProgressBar(),
            LearningRateMonitor(logging_interval="step"),
            self.metrics_csv,
            self.eval_cb,
            MergeCallback(str(self.out / "merged")),
        ]

        self.tb_logger = pl.loggers.TensorBoardLogger(save_dir=str(self.out), name="", version="", default_hp_metric=False)
        use_gpu = torch.cuda.is_available() and cfg.train.devices != "-1"
        devices = _parse_devices(cfg.train.devices)
        strategy = cfg.train.strategy
        if strategy == "ddp" or (strategy == "auto" and isinstance(devices, list) and len(devices) > 1):
            strategy = DDPStrategy(
                timeout=timedelta(seconds=7200),        # 2小时超时保护，防止单卡验证时其余卡崩溃
                find_unused_parameters=False,
            )
        elif strategy == "auto":
            strategy = "auto"

        self.trainer = pl.Trainer(
            max_epochs=cfg.train.epochs,
            accelerator="gpu" if use_gpu else "cpu",
            devices=devices if devices is not None else "auto",
            strategy=strategy,
            precision=resolve_precision(cfg.model, cfg.train.precision),
            accumulate_grad_batches=cfg.train.accumulate,
            gradient_clip_val=cfg.model.grad_clip,
            callbacks=cbs,
            logger=self.tb_logger,
            enable_progress_bar=True,
            log_every_n_steps=20,
            limit_val_batches=0,
            use_distributed_sampler=False,
        )

    def _evaluate_dual(self) -> Tuple[Dict[str, float], Optional[Dict[str, float]], Optional[Tuple], Optional[Tuple]]:
        evaluator = Evaluator(
            self.model,
            self.cfg.model,
            taxonomy=self.dm.taxonomy,
            device=self.model.device
        )
        val_dl = self.dm.val_dataloader()
        reg_metrics, cached_proto = evaluator.evaluate(val_dl, desc="[Eval: Regular]")
        reg_cm = evaluator.last_confusion

        # 2. 上下文借调 EMA 评估
        ema_metrics, ema_cm = None, None
        if self.model.ema_shadow is not None:
            with self.model.use_ema():
                ema_metrics, _ = evaluator.evaluate(val_dl, desc="[Eval: EMA]")
                ema_cm = evaluator.last_confusion

        return reg_metrics, ema_metrics, reg_cm, ema_cm

    def _current_lr(self) -> float:
        if self.model and self.model._optimizer:
            return max(float(pg["lr"]) for pg in self.model._optimizer.param_groups)
        return 0.0

    def _train_loss_avg(self) -> float:
        if self.model and self.model._log_total > 0:
            avg = self.model._log_train_loss / self.model._log_total
            self.model._log_train_loss = 0.0
            self.model._log_total = 0.0
            return float(avg)
        return 0.0

    def _train_config_snapshot(self) -> dict:
        t = self.cfg.train
        return {
            "output_dir": t.output_dir,
            "freeze_mode": t.freeze_mode,
            "batch_size": t.batch_size,
            "accumulate": t.accumulate,
            "epochs": t.epochs,
            "num_workers": t.num_workers,
            "patience": t.patience,
            "resume": t.resume,
            "ckpt_path": t.ckpt_path,
            "seed": t.seed,
            "devices": t.devices,
            "strategy": t.strategy,
            "model_config": t.model_config,
            "data_config": t.data_config,
        }

    def run(self):
        self.setup()
        self.trainer.fit(self.model, datamodule=self.dm, ckpt_path=None)
        _log.info(
            f"[Trainer] 训练结束. best_metric={self.best_metric:.4f} "
            f"(来自 {self.best_source}) @ epoch {self.best_epoch}"
        )
        return self.out


def _parse_devices(s: str) -> Optional[Union[List[int], int]]:
    if not s or s in ("auto", "-1"):
        return None
    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    try:
        return int(s)
    except ValueError:
        return None


if __name__ == "__main__":
    from utils.config import get_args
    cfg = get_args()
    Trainer(cfg).run()
    