"""参数配置系统。

对外主要入口:
  - get_args(argv=None) -> Config        # 解析 argv / sys.argv
  - Config.from_yaml(...)                # 仅从 yaml 构造 (供测试/复用)
  - Config.to_dict()                     # 序列化快照 (set_*_config.json)
  - resolve_precision(model_cfg, cli)    # amp 字符串 -> PL precision 字符串
"""
from __future__ import annotations

import argparse
import copy
import json
import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, List, Optional, Union, get_args as t_get_args, get_origin

import yaml


# --------------------------------------------------------------------------- #
# 基础配置片段 (字段定义保持完全一致)
# --------------------------------------------------------------------------- #
@dataclass
class OptimizerConfig:
    enable: str = "AdamW"                       # ["AdamW", "MuSGD"]
    weight_decay: float = 0.001
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    layer_decay: float = 0.85                   # <=0 或 None 关闭 LLRD
    head_lr_scale: float = 1.0                 # logit 头 / 投影层 lr 放大
    musgd: dict = field(default_factory=lambda: {
        "use_muon": True, "muon_weight": 0.5, "sgd_weight": 0.5,
        "momentum": 0.95, "nesterov": True, "ns_steps": 5,
    })


@dataclass
class SchedulerConfig:
    enable: str = "cosine"                       # ["linear","cosine","onecycle","multistep"]
    warmup_epochs: float = 1.0
    warmup_start_factor: float = 0.01
    interval: str = "step"                       # ["step","epoch"]
    multistep: dict = field(default_factory=lambda: {"milestones": [10, 16], "gamma": 0.1})
    onecycle: dict = field(default_factory=lambda: {
        "pct_start": 0.1, "div_factor": 25.0, "final_div_factor": 10000.0,
    })


@dataclass
class LoRAConfig:
    type: str = "dora"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"                           # ["none","all","lora_only"]
    apply_to: List[str] = field(default_factory=lambda: ["vision"])
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "out_proj"])


@dataclass
class LossConfig:
    name: str = "auto"                           # ["auto","siglip","tipsv2","infonce"]
    tips_global: str = "sigmoid"                 # ["infonce", "sigmoid"]
    learnable_logit: bool = True
    max_scale: float = 100.0
    label_constrain: bool = False
    init_scale: Optional[Union[str, float]] = "auto"
    init_bias: Optional[Union[str, float]] = "auto"


@dataclass
class EvalConfig:
    save_metric: str = "zeroshot_f1"
    save_metric_mode: str = "max"                # ["max","min"]
    retrieval_topk: List[int] = field(default_factory=lambda: [1, 5, 10])
    zeroshot: bool = True
    prompt_template: str = "a photo of {}."
    batch_size: Optional[int] = None


@dataclass
class ModelConfig:
    arch: str = "siglip2"                        # ["siglip2","tipsv2"]
    pretrained_path: str = "./huggingface_models/siglip2-base-patch16-224"
    trust_remote_code: bool = False
    max_text_length: int = 64
    text_padding: Union[str, bool] = False       # [max_length, longest, False]
    lr: float = 1e-6
    min_lr: float = 1e-9
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    amp: str = "bf16"                            # ["bf16","fp16", false->""]
    dropout: float = 0.01
    ema: float = 0.999
    grad_clip: Optional[float] = 1.0
    label_smoothing: float = 0.0
    compile: bool = False
    loss: LossConfig = field(default_factory=LossConfig)
    use_lora: bool = True
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @staticmethod
    def from_dict(d: dict) -> "ModelConfig":
        return _build(ModelConfig, copy.deepcopy(d))


@dataclass
class AugResize:
    method: str = "letterbox"
    size: int = 256
    interp: str = "bilinear"
    pad_value: int = 114


@dataclass
class AugCrop:
    enabled: bool = False
    method: str = "center"
    size: int = 256


@dataclass
class AugRotate:
    limit: List[int] = field(default_factory=lambda: [-10, 10])
    prob: float = 0.2


@dataclass
class AugAffine:
    percent: float = 0.12
    scale: List[float] = field(default_factory=lambda: [0.9, 1.1])
    shear: int = 10
    prob: float = 0.4


@dataclass
class AugPerspective:
    scale: List[float] = field(default_factory=lambda: [0.005, 0.01])
    prob: float = 0.15


@dataclass
class AugHSV:
    h: float = 0.03
    s: float = 0.5
    v: float = 0.45
    prob: float = 0.6


@dataclass
class AugGaussBlur:
    blur: List[int] = field(default_factory=lambda: [3, 5])
    sigma: List[float] = field(default_factory=lambda: [0.3, 1.0])
    prob: float = 0.2


@dataclass
class AugErase:
    prob: float = 0.4
    range: List[float] = field(default_factory=lambda: [0.05, 0.12])
    max_holes: int = 8


@dataclass
class AugNormalize:
    mean: List[float] = field(default_factory=lambda: [0, 0, 0])
    std: List[float] = field(default_factory=lambda: [1, 1, 1])


@dataclass
class AugmentConfig:
    enabled: bool = True
    close_augment_epoch: int = 170
    resize: AugResize = field(default_factory=AugResize)
    crop: AugCrop = field(default_factory=AugCrop)
    flip: float = 0.5
    vflip: float = 0.0
    rotate: AugRotate = field(default_factory=AugRotate)
    affine: AugAffine = field(default_factory=AugAffine)
    perspective: AugPerspective = field(default_factory=AugPerspective)
    hsv: AugHSV = field(default_factory=AugHSV)
    gauss_blur: AugGaussBlur = field(default_factory=AugGaussBlur)
    erase: AugErase = field(default_factory=AugErase)
    normalize: AugNormalize = field(default_factory=AugNormalize)


@dataclass
class DataConfig:
    root: str = "/home/syk/pest_cls_dataset"
    images: str = "captions/multimodal.lmdb"
    train: str = "captions/train_captions.json"
    val: str = "captions/val_captions.json"
    class_map_csv: str = "captions/pest_filted.csv"
    caption_sample: str = "random"
    val_caption_sample: str = "first"
    augment: AugmentConfig = field(default_factory=AugmentConfig)

    @staticmethod
    def from_dict(d: dict) -> "DataConfig":
        return _build(DataConfig, copy.deepcopy(d))


@dataclass
class TrainConfig:
    """CLI 运行期参数 (不会出现在 yaml 内)。"""
    model_config: str = "configs/model_config.yaml"
    data_config: str = "configs/data.yaml"
    output_dir: str = "./outputs/exp1"
    freeze_mode: str = "none"                    # ["vision","text","both","none"]
    batch_size: int = 64
    accumulate: int = 1
    epochs: int = 20
    num_workers: int = 8
    patience: int = 3
    resume: bool = False
    ckpt_path: Optional[str] = None
    save_interval: int = 0                       # >0 时额外保存间隔 epoch 的 checkpoint
    seed: int = 42
    devices: str = "auto"                        # ["auto","1","0,1", "-1"(cpu)]
    strategy: str = "auto"                       # ["auto","ddp","deepspeed",...]
    precision: Optional[str] = None               # None -> 跟随 model_config.amp


@dataclass
class Config:
    """合并后的全局配置树。"""
    model: ModelConfig
    data: DataConfig
    train: TrainConfig = field(default_factory=TrainConfig)

    @staticmethod
    def from_yaml(model_yaml: str, data_yaml: str) -> "Config":
        mcfg = ModelConfig.from_dict(_load_yaml(model_yaml))
        dcfg = DataConfig.from_dict(_load_yaml(data_yaml))
        return Config(model=mcfg, data=dcfg)

    def to_dict(self) -> dict:
        return dataclass_to_dict(self)


# --------------------------------------------------------------------------- #
# CLI 解析
# --------------------------------------------------------------------------- #
def get_args(argv: Optional[List[str]] = None) -> Config:
    p = argparse.ArgumentParser("Dual-Tower Fine-Tuning Framework", add_help=True)
    p.add_argument("--model_config", default="configs/model_config.yaml")
    p.add_argument("--data_config", default="configs/data.yaml")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--freeze_mode", default="none", choices=["vision", "text", "both", "none"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--accumulate", type=int, default=1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--ckpt_path", default=None)
    p.add_argument("--save_interval", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--devices", default="auto")
    p.add_argument("--strategy", default="auto")
    p.add_argument("--precision", default=None)
    args = p.parse_args(argv)

    cfg = Config.from_yaml(args.model_config, args.data_config)

    cfg.train = TrainConfig(
        model_config=str(Path(args.model_config).resolve()) if Path(args.model_config).is_file() else args.model_config,
        data_config=str(Path(args.data_config).resolve()) if Path(args.data_config).is_file() else args.data_config,
        output_dir=str(Path(args.output_dir).resolve()),
        freeze_mode=args.freeze_mode,
        batch_size=args.batch_size,
        accumulate=args.accumulate,
        epochs=args.epochs,
        num_workers=args.num_workers,
        patience=args.patience,
        resume=args.resume,
        ckpt_path=args.ckpt_path,
        save_interval=args.save_interval,
        seed=args.seed,
        devices=args.devices,
        strategy=args.strategy,
        precision=args.precision,
    )

    cfg.model.pretrained_path = _resolve_pretrained(cfg.model.pretrained_path)
    return cfg


# --------------------------------------------------------------------------- #
# 精度映射
# --------------------------------------------------------------------------- #
def resolve_precision(model_cfg: ModelConfig, cli_precision: Optional[str] = None) -> str:
    p = cli_precision or model_cfg.amp
    if not p or p is False or str(p).lower() in ("false", "none", ""):
        return "32-true"
    p = str(p).lower()
    if p.startswith("bf16"):
        return "bf16-mixed"
    if p.startswith("fp16") or p == "16":
        return "16-mixed"
    return "32-true"


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #
def _load_yaml(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_set_config(out_dir: Path, dataclass_cfg: Any, file_name: str):
    with open(out_dir / file_name, "w", encoding="utf-8") as f:
        json.dump(dataclass_to_dict(dataclass_cfg), f, ensure_ascii=False, indent=2)


def _resolve_pretrained(path: str) -> str:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return str(p.resolve())

    # 锚定项目根目录 (utils 上一级) 的 huggingface_models
    repo_root = Path(__file__).resolve().parent.parent
    cand = repo_root / "huggingface_models" / p.name
    if cand.exists():
        return str(cand.resolve())

    # 再次尝试相对 repo_root 路径
    direct_cand = repo_root / path
    if direct_cand.exists():
        return str(direct_cand.resolve())

    return path


def _unwrap_optional(tp: Any) -> Any:
    """解包 Optional[T] 为真实类型 T"""
    origin = get_origin(tp)
    if origin is Union:
        args = [a for a in t_get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def _build(cls, d: Union[dict, Any]):
    if not is_dataclass(cls) or not isinstance(d, dict):
        return d
    from typing import get_type_hints
    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        raw_type = hints.get(f.name, f.type)
        ftype = _unwrap_optional(raw_type)

        if f.name not in d or d[f.name] is None:
            if f.name in d and d[f.name] is None:
                kwargs[f.name] = None
            elif f.default_factory is not dataclasses.MISSING:
                kwargs[f.name] = f.default_factory()
            elif f.default is not dataclasses.MISSING:
                kwargs[f.name] = f.default
            else:
                kwargs[f.name] = None
            continue

        val = d[f.name]
        if is_dataclass(ftype) and isinstance(val, dict):
            kwargs[f.name] = _build(ftype, val)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


def dataclass_to_dict(obj) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: dataclass_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    return obj


if __name__ == "__main__":
    c = get_args([
        "--model_config", "configs/model_config.yaml",
        "--data_config", "configs/data.yaml",
        "--output_dir", "./outputs/selfcheck",
        "--freeze_mode", "text",
        "--batch_size", "32",
    ])
    print(c.model.arch, c.train.freeze_mode, c.train.batch_size)
    print("OK")
