'''Merge LoRA'''

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict
import safetensors.torch as st
import torch
from torch import nn
from peft import PeftModel

from models.dual_tower_multimodal import build_dual_tower_model
from utils.config import Config, DataConfig, ModelConfig, TrainConfig, write_set_config
from utils.logger import get_logger

_log = get_logger("merger")


@dataclass
class MergeConfig:
    checkpoint_path: str
    output_path: str                              # 支持指定 .safetensors 单文件或目标目录
    base_model_path: Optional[str] = None         # 覆盖 pretrained_path，缺省回退快照
    save_processor: bool = True


def _unload_peft_towers(arch: str, model_core: nn.Module) -> None:
    """
    根据具体架构显式提取双塔，执行 LoRA 融合并强制回填原生模型对象，
    彻底剥离 PeftModel 包装，确保 state_dict 键名恢复官方原生命名规范。
    """
    if arch in ("siglip2", "fgclip2"):
        if isinstance(model_core.vision_model, PeftModel):
            model_core.vision_model = model_core.vision_model.merge_and_unload()
        if isinstance(model_core.text_model, PeftModel):
            model_core.text_model = model_core.text_model.merge_and_unload()
    elif arch == "tipsv2":
        if isinstance(model_core.vision_encoder, PeftModel):
            model_core.vision_encoder = model_core.vision_encoder.merge_and_unload()
        if isinstance(model_core.text_encoder, PeftModel):
            model_core.text_encoder = model_core.text_encoder.merge_and_unload()
    else:
        raise ValueError(f"未支持的架构分发: {arch}")


def _extract_loss_scalars(model_module) -> Dict[str, torch.Tensor]:
    """提取训练收敛的 scale 与 bias 标量，优先从 loss_fn，兜底回退至主干"""
    scalars = {}
    loss_fn = getattr(model_module, "loss_fn", None)
    target = getattr(loss_fn, "global_loss", loss_fn) if loss_fn else None

    # 提取 logit_scale
    if target and hasattr(target, "logit_scale"):
        scalars["logit_scale"] = target.logit_scale.detach().cpu()
    elif hasattr(model_module.model, "logit_scale"):
        scalars["logit_scale"] = model_module.model.logit_scale.detach().cpu()

    # 提取 logit_bias
    if target and hasattr(target, "logit_bias"):
        scalars["logit_bias"] = target.logit_bias.detach().cpu()
    elif hasattr(model_module.model, "logit_bias"):
        scalars["logit_bias"] = model_module.model.logit_bias.detach().cpu()

    return scalars


def merge_lora_weights(cfg: MergeConfig) -> Path:
    ckpt_file = Path(cfg.checkpoint_path).resolve()
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"检查点权重未找到: {ckpt_file}")

    ckpt_dir = ckpt_file.parent
    model_json = ckpt_dir / "set_model_config.json"
    data_json = ckpt_dir / "set_data_config.json"

    if not model_json.is_file() or not data_json.is_file():
        raise FileNotFoundError("缺少配置快照: set_model_config.json 或 set_data_config.json")

    with open(model_json, "r", encoding="utf-8") as f:
        model_cfg = ModelConfig.from_dict(json.load(f))
    with open(data_json, "r", encoding="utf-8") as f:
        data_cfg = DataConfig.from_dict(json.load(f))

    target = Path(cfg.output_path)
    if target.suffix.lower() == ".safetensors":
        export_file = target
        export_dir = target.parent
    else:
        export_dir = target
        date_str = dt.datetime.now().strftime("%Y%m%d")
        export_file = export_dir / f"{model_cfg.arch.lower()}-{date_str}.safetensors"

    export_dir.mkdir(parents=True, exist_ok=True)

    if cfg.base_model_path:
        model_cfg.pretrained_path = cfg.base_model_path

    model_cfg.use_lora = True
    full_cfg = Config(model=model_cfg, data=data_cfg, train=TrainConfig())
    model_module = build_dual_tower_model(full_cfg)

    _log.info(f"[Merge] 读取增量检查点: {ckpt_file.name}")
    model_module.load_safetensors(str(ckpt_file), load_opt=False)

    _unload_peft_towers(model_cfg.arch.lower(), model_module.model)

    merged_sd = {
        k.replace("_orig_mod.", ""): v.detach().cpu().contiguous()
        for k, v in model_module.model.state_dict().items()
    }

    scalars = _extract_loss_scalars(model_module)
    for k, v in scalars.items():
        v_contig = v.contiguous()
        if hasattr(model_module.model, k):
            p = getattr(model_module.model, k)
            p.data.copy_(v_contig.view_as(p).to(p.device))

        merged_sd[k] = v_contig

    meta = {
        "model": "dual_tower_merged",
        "arch": model_cfg.arch.lower(),
        "use_lora": "false",
        "time": dt.datetime.now().isoformat(),
    }

    st.save_file(merged_sd, str(export_file), metadata=meta)
    _log.info(f"[Merge] 融合权重完成 -> {export_file}")

    model_module.model.config.save_pretrained(str(export_dir))

    # 输出已关闭 use_lora 的配置快照，便于 export.py 与 valid.py 直接加载
    model_cfg.use_lora = False
    write_set_config(export_dir, model_cfg, "set_model_config.json")
    write_set_config(export_dir, data_cfg, "set_data_config.json")

    if cfg.save_processor and model_module.processor is not None:
        model_module.processor.save_pretrained(str(export_dir))

    return export_file


def main():
    p = argparse.ArgumentParser("Standalone LoRA Merger")
    p.add_argument("--checkpoint_path", required=True, help="*-lora.safetensors 权重路径")
    p.add_argument("--output_path", required=True, help="输出目标路径，支持目录或具体 .safetensors 文件名")
    p.add_argument("--base_model_path", default=None, help="基座预训练路径 (默认从配置快照 pretrained_path 读取)")
    p.add_argument("--save_processor", action='store_true', help='保存分词器, 预处理器')
    args = p.parse_args()

    cfg = MergeConfig(
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        base_model_path=args.base_model_path,
        save_processor=args.save_processor
    )
    merge_lora_weights(cfg)


if __name__ == "__main__":
    main()
