from __future__ import annotations

import math
import re
from typing import Dict, List, Tuple

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler, MultiStepLR, OneCycleLR


# =========================================================================== #
# 1. Newton-Schulz 正交化
# =========================================================================== #
def ns_orthogonalize(matrix: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz 五阶迭代，返回近似半正交矩阵 (支持任意 2D 矩形矩阵)"""
    if matrix.ndim != 2:
        return matrix
    a, b, c = 3.4445, -4.7750, 2.0315
    orig_type = matrix.dtype
    x = matrix.to(torch.float32)
    x = x / (x.norm() + eps)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.t()
    for _ in range(steps):
        A = x @ x.t()
        B = b * A + c * (A @ A)
        x = a * x + B @ x
    if transposed:
        x = x.t()
    return x.to(orig_type)


# =========================================================================== #
# 2. 参数分组 (LLRD)
# =========================================================================== #

def is_decay_excluded(name: str) -> bool:
    """对齐3类模型所有 1D 归一化、偏置与尺度因子"""
    name_lower = name.lower()
    return (
        name.endswith("bias")
        or "norm" in name_lower           # norm, norm1, norm2, LayerNorm, post_layernorm, final_layer_norm
        or ".ln_" in name_lower           # TIPSv2: resblocks.ln_1, resblocks.ln_2
        or "ln_final" in name_lower       # TIPSv2: text_encoder.ln_final
        or ".ls1" in name_lower           # TIPSv2: LayerScale 1
        or ".ls2" in name_lower           # TIPSv2: LayerScale 2
        or "logit_scale" in name
        or "logit_bias" in name
        or "position_embedding" in name_lower
        or "pos_embedder" in name_lower
    )


def _is_head_param(name: str) -> bool:
    """匹配 Logit 标量以及3类模型的所有池化头与文本头"""
    return (
        "logit_scale" in name
        or "logit_bias" in name
        or "dense_feature_head" in name   # FG-CLIP2
        or "longtext_head" in name        # FG-CLIP2
        or "boxtext_head" in name         # FG-CLIP2
        or ".head." in name               # SigLIP/FG-CLIP2: vision_model.head, text_model.head
        or name.endswith(".head.weight")
        or name.endswith(".head.bias")
    )


def _extract_vision_depth(name: str, n_blocks: int) -> int:
    """解析视觉塔层深 (TIPSv2: blocks.X; SigLIP/FG-CLIP2: encoder.layers.X)"""
    m = re.search(r"\.(?:blocks|layers)\.(\d+)\.", name)
    if m:
        return min(int(m.group(1)), n_blocks - 1)
    
    # 输入端 Patch Embedding 赋予第 0 层学习率
    if any(k in name for k in ("patch_embed", "embeddings")):
        return 0

    return n_blocks - 1


def _extract_text_depth(name: str, n_blocks: int) -> int:
    """解析文本塔层深 (TIPSv2: transformer.resblocks.X; SigLIP/FG-CLIP2: encoder.layers.X)"""
    m = re.search(r"\.(?:resblocks|layers)\.(\d+)\.", name)
    if m:
        return min(int(m.group(1)), n_blocks - 1)

    # 输入端 Token/Pos Embedding 赋予第 0 层学习率
    if any(k in name for k in ("token_embedding", "pos_embedder", "embeddings")):
        return 0

    # 顶层 Norm (ln_final / final_layer_norm) 归入末层 (n_blocks - 1)
    return n_blocks - 1


def group_parameters(
    model: torch.nn.Module,
    model_cfg,
    n_v_blocks: int = 12,
    n_t_blocks: int = 12,
) -> List[Dict]:
    head_lr_scale = model_cfg.optimizer.head_lr_scale
    layer_decay = model_cfg.optimizer.layer_decay
    base_lr = model_cfg.lr
    weight_decay = model_cfg.optimizer.weight_decay
    use_llrd = bool(layer_decay is not None and 0.0 < layer_decay < 1.0)

    buckets: Dict[Tuple[float, float], List[torch.nn.Parameter]] = {}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        clean_name = name.replace("_orig_mod.", "")
        decay = 0.0 if is_decay_excluded(clean_name) else weight_decay

        # 1. 优先匹配各类头算子与 Logit 标量
        if _is_head_param(clean_name):
            lr = base_lr * head_lr_scale

        # 2. LoRA 增量参数保持 base_lr，不参与主干深度衰减
        elif (
            "lora_" in clean_name
            or "position_embedding" in clean_name       # 暂定位置编码按base_lr设置
            or "pos_embedder" in clean_name
        ):
            lr = base_lr

        # 3. 主干 Trunk 按模型所属塔拆分，独立执行 LLRD
        elif use_llrd:
            if "vision_model" in clean_name or "vision_encoder" in clean_name:
                depth = _extract_vision_depth(clean_name, n_v_blocks)
                lr = base_lr * (layer_decay ** (n_v_blocks - 1 - depth))
            elif "text_model" in clean_name or "text_encoder" in clean_name:
                depth = _extract_text_depth(clean_name, n_t_blocks)
                lr = base_lr * (layer_decay ** (n_t_blocks - 1 - depth))
            else:
                lr = base_lr
        else:
            lr = base_lr

        key = (round(lr, 12), round(decay, 8))
        buckets.setdefault(key, []).append(p)

    param_groups = []
    for (lr, decay), params in buckets.items():
        param_groups.append({
            "params": params,
            "lr": lr,
            "weight_decay": decay,
        })
    return param_groups


# =========================================================================== #
# 3. MuSGD 优化器
# =========================================================================== #
class MuSGD(Optimizer):
    """Muon + SGD 动量双分支融合优化器"""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        muon_weight: float = 0.5,
        sgd_weight: float = 0.5,
        ns_steps: int = 5,
        use_muon: bool = True,
        maximize: bool = False,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            muon_weight=muon_weight,
            sgd_weight=sgd_weight,
            ns_steps=ns_steps,
            use_muon=use_muon,
            maximize=maximize,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            muon_w = group["muon_weight"]
            sgd_w = group["sgd_weight"]
            ns_steps = group["ns_steps"]
            use_muon = group["use_muon"]
            sign = -1.0 if not group["maximize"] else 1.0

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("MuSGD does not support sparse gradients.")

                # 安全解耦权重衰减 (避免原地破坏计算图)
                if wd != 0.0:
                    p.add_(p, alpha=sign * lr * wd)

                state = self.state[p]

                # 2D 矩阵走 Muon 分支
                if use_muon and p.ndim == 2:
                    update = ns_orthogonalize(grad.detach(), ns_steps)
                    scale = max(1.0, (p.size(0) / p.size(1)) ** 0.5)
                    update = update * scale

                    if "muon_buffer" not in state:
                        buf = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["muon_buffer"] = buf
                    else:
                        buf = state["muon_buffer"]

                    buf.mul_(momentum).add_(update)
                    d_p = update.add(buf, alpha=momentum) if nesterov else buf
                    p.add_(d_p, alpha=sign * muon_w * lr)

                # 非 2D 参数走 SGD 分支
                else:
                    if "sgd_buffer" not in state:
                        buf = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["sgd_buffer"] = buf
                    else:
                        buf = state["sgd_buffer"]

                    buf.mul_(momentum).add_(grad)
                    d_p = grad.add(buf, alpha=momentum) if nesterov else buf
                    p.add_(d_p, alpha=sign * sgd_w * lr)

        return loss


# =========================================================================== #
# 4. 学习率调度器
# =========================================================================== #
class WarmupLRScheduler(LRScheduler):
    """Warmup + 余弦/线性退火调度器"""

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int,
        base_lr: float,
        min_lr: float = 0.0,
        mode: str = "cosine",
        warmup_start_factor: float = 0.01,
        last_epoch: int = -1,
    ):
        if total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        self.total_steps = total_steps
        self.warmup_steps = max(1, warmup_steps)
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.mode = mode
        self.warmup_start_factor = warmup_start_factor
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        step = self.last_epoch
        if step < self.warmup_steps:
            factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * (
                step / max(1, self.warmup_steps)
            )
            return [base * factor for base in self.base_lrs]

        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))

        if self.mode == "cosine":
            ratio = 0.5 * (1.0 + math.cos(math.pi * progress))
            coef = (self.min_lr / self.base_lr) + (1.0 - self.min_lr / self.base_lr) * ratio
        else:  # linear
            coef = 1.0 - (1.0 - self.min_lr / self.base_lr) * progress

        return [base * coef for base in self.base_lrs]


# =========================================================================== #
# 5. 工厂函数
# =========================================================================== #
def build_optimizer(model: torch.nn.Module, model_cfg, n_v_blocks: int = 12, n_t_blocks: int = 12) -> Tuple[Optimizer, dict]:
    oc = model_cfg.optimizer
    param_groups = group_parameters(model, model_cfg, n_v_blocks=n_v_blocks, n_t_blocks=n_t_blocks)
    extra = {"type": oc.enable}

    if oc.enable == "MuSGD":
        mcfg = oc.musgd
        opt = MuSGD(
            param_groups,
            lr=model_cfg.lr,
            momentum=mcfg.get("momentum", 0.95),
            nesterov=mcfg.get("nesterov", True),
            weight_decay=oc.weight_decay,
            muon_weight=mcfg.get("muon_weight", 0.5),
            sgd_weight=mcfg.get("sgd_weight", 0.5),
            ns_steps=mcfg.get("ns_steps", 5),
            use_muon=mcfg.get("use_muon", True),
        )
        return opt, extra

    # 默认 AdamW
    opt = AdamW(param_groups, betas=tuple(oc.betas), eps=oc.eps, weight_decay=oc.weight_decay)
    return opt, extra


def build_scheduler(
    optimizer: Optimizer,
    model_cfg,
    total_epochs: int,
    steps_per_epoch: int,
    interval: str = "step",
) -> Tuple[LRScheduler, str]:
    sc = model_cfg.scheduler
    base_lr = model_cfg.lr
    min_lr = model_cfg.min_lr

    if interval == "step":
        total_steps = max(1, total_epochs * steps_per_epoch)
        warmup_steps = max(1, int(round(sc.warmup_epochs * steps_per_epoch)))
    else:
        total_steps = max(1, total_epochs)
        warmup_steps = max(1, int(round(sc.warmup_epochs)))

    if sc.enable == "onecycle":
        max_lrs = [pg["lr"] for pg in optimizer.param_groups]
        sched = OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            total_steps=total_steps,
            pct_start=sc.onecycle["pct_start"],
            div_factor=sc.onecycle["div_factor"],
            final_div_factor=sc.onecycle["final_div_factor"],
        )
        return sched, interval

    if sc.enable == "multistep":
        ms = sc.multistep
        milestones = [int(m * (steps_per_epoch if interval == "step" else 1)) for m in ms["milestones"]]
        sched = MultiStepLR(optimizer, milestones=milestones, gamma=ms["gamma"])
        return sched, interval

    sched = WarmupLRScheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        base_lr=base_lr,
        min_lr=min_lr,
        mode=sc.enable,
        warmup_start_factor=sc.warmup_start_factor,
    )
    return sched, interval
