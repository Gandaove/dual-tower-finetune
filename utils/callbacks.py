"""PyTorch Lightning Callbacks(全部集中于此)。

Skill §4.5 / 用户要求: 所有 callback 都应放在 callbacks.py。
  - MergeCallback: 训练结束时合并 LoRA 并导出 HF 权重目录;
  - EMACallback: 验证/测试前切换 EMA 权重, 结束后恢复;
  - MetricsCSVCallback: 每个 epoch 记录 training-metrics.csv;
  - EpochEvalCallback: 每轮结束后跑完整 Evaluator, 挑选 best 指标, 保存
        best/last/interval 检查点, 并在连续未提升时早停(早停/Checkpoints/Merge 三件套)。

注意: EpochEvalCallback 通过 owner(Trainer) 获取评估/保存能力, 二者为单向依赖
(callbacks 不反向 import trainer), 保持解耦。
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional
import torch
import torch.distributed as dist
import pytorch_lightning as pl

from utils.logger import get_logger
from utils.config import write_set_config

_log = get_logger("callbacks")


class MergeCallback(pl.Callback):
    def __init__(self, export_dir: str):
        super().__init__()
        self.export_dir = Path(export_dir)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not pl_module.model_cfg.use_lora:
            return  # 全量微调不合并

        from models.merger import MergeConfig, merge_lora_weights
        out_dir = Path(pl_module.train_cfg.output_dir)
        lora_ckpt = out_dir / "best-lora.safetensors"
        if not lora_ckpt.is_file():
            _log.warning(f"Not detected best-lora.safetensors, check training metrics & settings!")
            return

        m_cfg = MergeConfig(
            checkpoint_path=str(lora_ckpt),
            output_path=str(self.export_dir),
            save_processor=False,
        )
        merge_lora_weights(m_cfg)


class MetricsCSVCallback(pl.Callback):
    def __init__(self, csv_path: str, fields: List[str]):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.fields = fields
        self.rows = []

    def append(self, row: dict, is_global_zero: bool):
        if not is_global_zero:
            return
        self.rows.append(row)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fields)
            if write_header:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in self.fields})

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        _log.info(f"[Metrics] 训练历史记录写入完成: {self.csv_path} (共 {len(self.rows)} 轮)")


class EpochEvalCallback(pl.Callback):
    """每轮训练结束: 双轨 (Regular & EMA) 完整评估 -> 择优保存 best / last / interval -> 早停"""

    def __init__(self, trainer_owner):
        super().__init__()
        self.owner = trainer_owner

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        owner = self.owner
        actual_epoch = owner.start_epoch + trainer.current_epoch
        close_epoch = owner.cfg.data.augment.close_augment_epoch
        if actual_epoch >= close_epoch and hasattr(owner.dm, "set_augment_status"):
            owner.dm.set_augment_status(False)
            _log.info(f"[Augment] 当前 epoch {actual_epoch} >= {close_epoch}，关闭数据增强")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        owner = self.owner
        cfg = owner.cfg
        actual_epoch = owner.start_epoch + trainer.current_epoch
        is_ddp = dist.is_available() and dist.is_initialized()

        if trainer.is_global_zero:
            metric_name = cfg.model.eval.save_metric
            mode = cfg.model.eval.save_metric_mode
            tc = owner._train_config_snapshot()

            try:
                reg_metrics, ema_metrics, reg_cm, ema_cm = owner._evaluate_dual()
            except Exception as e:
                _log.error(f"[Eval] epoch {actual_epoch} 双轨评估异常: {e}")
                reg_metrics, ema_metrics, reg_cm, ema_cm = {}, None, None, None

            reg_score = reg_metrics.get(metric_name, 0.0)
            ema_score = ema_metrics.get(metric_name, None) if ema_metrics else None

            winner_source, winner_score, winner_cm = "regular", reg_score, reg_cm
            if ema_score is not None:
                if (mode == "max" and ema_score > reg_score) or (mode == "min" and ema_score < reg_score):
                    winner_source, winner_score, winner_cm = "ema", ema_score, ema_cm

            current_lr = owner._current_lr()
            avg_train_loss = owner._train_loss_avg()
            row = {
                "epoch": actual_epoch,
                "lr": current_lr,
                "train_loss": avg_train_loss,
                "best_source": winner_source,
            }
            for k, v in reg_metrics.items():
                row[f"val/{k}"] = round(float(v), 4) if isinstance(v, (int, float)) else v
            if ema_metrics:
                for k, v in ema_metrics.items():
                    row[f"val_ema/{k}"] = round(float(v), 4) if isinstance(v, (int, float)) else v
            owner.metrics_csv.append(row, is_global_zero=True)

            if trainer.logger is not None:
                tb_scalars = {"epoch/train_loss": avg_train_loss, "epoch/lr": current_lr}
                for k, v in reg_metrics.items():
                    if isinstance(v, (int, float)): tb_scalars[f"val/{k}"] = float(v)
                if ema_metrics:
                    for k, v in ema_metrics.items():
                        if isinstance(v, (int, float)): tb_scalars[f"val_ema/{k}"] = float(v)
                trainer.logger.log_metrics(tb_scalars, step=actual_epoch)

            suffix = owner.model.ckpt_suffix
            last_path = str(owner.out / f"last{suffix}.safetensors")
            best_path = str(owner.out / f"best{suffix}.safetensors")

            owner.model.save_safetensors(last_path, epoch=actual_epoch, metric=reg_score, save_optimizer=True, train_config=tc)

            is_best = (winner_score > owner.best_metric) if mode == "max" else (winner_score < owner.best_metric)
            if is_best:
                owner.best_metric = winner_score
                owner.best_epoch = actual_epoch
                owner.best_source = winner_source
                if winner_source == "ema":
                    with owner.model.use_ema():
                        owner.model.save_safetensors(best_path, epoch=actual_epoch, metric=winner_score, save_optimizer=False, train_config=tc)
                else:
                    owner.model.save_safetensors(best_path, epoch=actual_epoch, metric=winner_score, save_optimizer=False, train_config=tc)

                if winner_cm is not None:
                    from utils.evaluator import Evaluator
                    Evaluator.save_confusion_png(winner_cm, str(owner.out / "best-confusion-matrix.png"), epoch=actual_epoch)

            si = cfg.train.save_interval
            if si and (actual_epoch + 1) % si == 0:
                owner.model.save_safetensors(
                    str(owner.out / f"checkpoint-{actual_epoch}.safetensors"),
                    epoch=actual_epoch, metric=reg_score, save_optimizer=True, train_config=tc
                )

            # 早停判定
            if cfg.train.patience:
                gap = actual_epoch - max(owner.best_epoch, owner.start_epoch)
                if gap >= cfg.train.patience:
                    _log.info(f"[EarlyStopping] 连续 {gap} 轮未提升，终止训练")
                    trainer.should_stop = True

        # ---------------- 同步 Early-Stopping 标志位 ---------------- #
        if is_ddp:
            stop_flag = torch.tensor(1 if trainer.should_stop else 0, device=pl_module.device, dtype=torch.int32)
            dist.broadcast(stop_flag, src=0)
            trainer.should_stop = bool(stop_flag.item() == 1)
            trainer.strategy.barrier()
        