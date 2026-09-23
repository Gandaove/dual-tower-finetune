"""双塔零样本图像分类与跨模态检索离线验证

根据检查点所在目录自动加载模型与数据配置快照，支持动态覆盖路径、
多级批大小推断、LoRA 权重状态自适应校验以及混淆矩阵和指标导出。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import safetensors
import peft

from dataset.dataset import build_dual_tower_datamodule
from models.dual_tower_multimodal import build_model as build_dual_tower_model
from utils.config import Config, DataConfig, ModelConfig, TrainConfig
from utils.evaluator import Evaluator
from utils.logger import get_logger

_log = get_logger("valid")


def _resolve_device(devices_arg: str) -> torch.device:
    if devices_arg == "-1" or not torch.cuda.is_available():
        return torch.device("cpu")
    if devices_arg == "auto" or not devices_arg:
        return torch.device("cuda:0")
    first_dev = devices_arg.split(",")[0].strip()
    return torch.device(f"cuda:{first_dev}")


class StandaloneValidator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.ckpt_path = Path(args.ckpt_path).resolve()
        if not self.ckpt_path.is_file():
            raise FileNotFoundError(f"未找到指定的权重检查点文件: {self.ckpt_path}")

        self.ckpt_dir = self.ckpt_path.parent
        self.output_dir = Path(args.output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = _resolve_device(args.devices)

        # 拦截未合并的 LoRA 检查点
        self._verify_merged_checkpoint()
        self.model_cfg, self.data_cfg = self._load_configs()
        self.model_cfg.use_lora = False

        self.batch_size = (
            args.batch_size
            if (args.batch_size and args.batch_size > 0)
            else (self.model_cfg.eval.batch_size or 32)
        )
        self.train_cfg = TrainConfig(
            output_dir=str(self.output_dir),
            batch_size=self.batch_size,
            num_workers=args.num_workers,
            devices=args.devices,
            freeze_mode="both",
        )
        self.model_cfg.use_lora = False

    def _verify_merged_checkpoint(self):
        """检查权重命名与底层参数，防止未合并模型进入验证。"""
        if "-lora" in self.ckpt_path.name.lower():
            raise RuntimeError(
                f"输入权重 '{self.ckpt_path.name}' 包含 '-lora' 后缀，属于未合并的 LoRA 增量检查点。"
                "离线验证仅支持全量权重，请先调用 models/merger.py 完成合并。"
            )

        with safetensors.safe_open(str(self.ckpt_path), framework="pt") as f:
            meta = f.metadata() or {}
            keys = f.keys()
            if meta.get("use_lora") == "True" or any("lora_" in k for k in keys):
                raise RuntimeError(
                    f"权重文件 '{self.ckpt_path.name}' 内检测到 LoRA 增量参数或元数据。"
                    "请先调用 models/merger.py 完成合并后再进行验证。"
                )

    def _load_configs(self) -> Tuple[ModelConfig, DataConfig]:
        model_json = self.ckpt_dir / "set_model_config.json"
        if not model_json.is_file():
            raise FileNotFoundError(f"未在检查点同级目录下找到模型配置快照: {model_json}")

        with open(model_json, "r", encoding="utf-8") as f:
            model_cfg = ModelConfig.from_dict(json.load(f))
        _log.info(f"[Config] 载入检查点模型快照: {model_json.name}")

        if self.args.data_config:
            d_path = Path(self.args.data_config).resolve()
            if not d_path.is_file():
                raise FileNotFoundError(f"指定的 data_config 不存在: {d_path}")
            from utils.config import _load_yaml
            data_cfg = DataConfig.from_dict(_load_yaml(str(d_path)))
            _log.info(f"[Config] 使用指定的外部数据配置: {d_path.name}")
        else:
            data_json = self.ckpt_dir / "set_data_config.json"
            if not data_json.is_file():
                raise FileNotFoundError(f"未在检查点同级目录下找到数据配置快照: {data_json}")
            with open(data_json, "r", encoding="utf-8") as f:
                data_cfg = DataConfig.from_dict(json.load(f))
            _log.info(f"[Config] 载入检查点数据快照: {data_json.name}")

        return model_cfg, data_cfg

    def run(self) -> Dict[str, float]:
        dm = build_dual_tower_datamodule(self.data_cfg, self.train_cfg)
        dm.setup("validate")

        cfg = Config(model=self.model_cfg, data=self.data_cfg, train=self.train_cfg)
        model = build_dual_tower_model(cfg, classes=dm.taxonomy.cnames, transform=dm.transform)

        # 防御性断言：确保无任何 LoRA/PEFT 包装类漏网
        for module in model.modules():
            if isinstance(module, peft.PeftModel):
                raise RuntimeError("模型检测到未解耦的 PeftModel 实例，请检查配置并确认模型已合并。")

        _log.info(f"[Run] 读取检查点权重: {self.ckpt_path.name}")
        model.load_safetensors(str(self.ckpt_path), load_opt=False)
        model = model.to(self.device)
        model.eval()

        _log.info(f"[Run] 启动 Evaluator 评估流程 (Batch Size: {self.batch_size})...")
        evaluator = Evaluator(
            model=model,
            model_cfg=self.model_cfg,
            taxonomy=dm.taxonomy,
            device=self.device,
        )

        metrics, _ = evaluator.evaluate(
            dataloader=dm.val_dataloader(),
            desc=f"[Validating: {self.ckpt_path.stem}]",
        )

        _log.info("==================== 验证指标结果 ====================")
        for k, v in metrics.items():
            _log.info(f"{k:25s}: {v:.4f}" if isinstance(v, float) else f"{k:25s}: {v}")
        _log.info("=====================================================")

        metrics_file = self.output_dir / "valid_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        _log.info(f"[Export] 评估指标已保存: {metrics_file}")

        if evaluator.last_confusion is not None:
            cm_path = self.output_dir / "confusion_matrix.png"
            Evaluator.save_confusion_png(
                evaluator.last_confusion, 
                str(cm_path),
                best_f1=metrics.get("zeroshot_f1", None)
            )
            _log.info(f"[Export] 混淆矩阵已保存")

        return metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Dual-Tower Standalone Validation Engine")
    p.add_argument("--ckpt_path", required=True, help="已合并的全量 safetensors 权重路径")
    p.add_argument("--data_config", default=None, help="覆盖用的外部 data.yaml 路径 (缺省读取检查点同级快照)")
    p.add_argument("--output_dir", default="./outputs/valid", help="指标及混淆矩阵输出目录")
    p.add_argument("--batch_size", type=int, default=None, help="批大小 (缺省回退快照 eval.batch_size，再回退 32)")
    p.add_argument("--num_workers", type=int, default=8, help="DataLoader 进程数")
    p.add_argument("--devices", default="auto", help="计算设备: 'auto', '0', '1', '-1'(cpu)")
    return p


def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)
    validator = StandaloneValidator(args)
    validator.run()


if __name__ == "__main__":
    main()
