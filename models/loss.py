"""损失函数集合。

设计原则: 模型 forward 输出的是 原始归一化前的嵌入 (image_emb, text_emb);
温度/偏置为可学习标量, 由本模块持有并在 forward 内对 logits 施加。
这样冻结文本塔时, logit_scale/bias 仍可被训练, 满足 Skill §2.1 工程注意。
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union


class SiglipSigmoidLoss(nn.Module):
    """逐对 Sigmoid 二分类对比损失 (采用数值稳定的 BCEWithLogits 实现)。

    对齐 SigLIP 标准论文与 Google 原生实现：
    - 正样本目标为 1，负样本目标为 0
    - logit_scale 施加上限截断保护，防止混合精度下溢或溢出
    """

    def __init__(
        self, 
        learnable_logit: bool = True, 
        init_scale: Union[str, float, None] = "auto",
        init_bias: Union[str, float, None] = "auto",
        max_scale: float = 100.0,
        label_constrain: bool = False,
    ):
        super().__init__()
        self.learnable = learnable_logit
        self.cfg_scale = init_scale
        self.cfg_bias = init_bias

        default_log_scale = math.log(10.0)
        default_bias = -10.0
        if isinstance(init_scale, (int, float)):
            default_log_scale = math.log(float(init_scale))
        if isinstance(init_bias, (int, float)):
            default_bias = float(init_bias)

        if self.learnable:
            self.logit_scale = nn.Parameter(torch.tensor(default_log_scale, dtype=torch.float32))
            self.logit_bias = nn.Parameter(torch.tensor(default_bias, dtype=torch.float32))
        else:
            self.register_buffer("logit_scale", torch.tensor(default_log_scale, dtype=torch.float32))
            self.register_buffer("logit_bias", torch.tensor(default_bias, dtype=torch.float32))

        self.max_scale = math.log(max_scale)
        self.label_constrain = label_constrain

    def sync_from_model_params(self, model: nn.Module):
        '''从权重同步logit_scale, logit_bias'''
        with torch.no_grad():
            if str(self.cfg_scale).lower() in ("auto", "none"):
                if hasattr(model, "logit_scale") and isinstance(model.logit_scale, (nn.Parameter, torch.Tensor)):
                    self.logit_scale.fill_(model.logit_scale.detach().cpu().item())
            if str(self.cfg_bias).lower() in ("auto", "none"):
                if hasattr(model, "logit_bias") and isinstance(model.logit_bias, (nn.Parameter, torch.Tensor)):
                    self.logit_bias.fill_(model.logit_bias.detach().cpu().item())

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor, labels: Optional[List[str]] = None, **kwargs) -> torch.Tensor:
        image_emb = F.normalize(image_emb.float(), dim=-1)
        text_emb = F.normalize(text_emb.float(), dim=-1)

        # 约束 scale 范围，避免梯度爆炸
        scale = torch.clamp(self.logit_scale, max=self.max_scale).exp()
        logits = scale * (image_emb @ text_emb.t()) + self.logit_bias  # [B, B]

        b = logits.size(0)
        if labels is not None and self.label_constrain:
            lbl_arr = np.array(labels)
            target_matrix = torch.from_numpy(lbl_arr[:, None] == lbl_arr[None, :]).to(
                device=logits.device, dtype=logits.dtype
            )
        else:
            target_matrix = torch.eye(b, device=logits.device, dtype=logits.dtype)
        loss = F.binary_cross_entropy_with_logits(logits, target_matrix, reduction="sum") / b
        return loss


class InfoNCELoss(nn.Module):
    """对称 InfoNCE (带温度截断保护)。"""

    def __init__(
        self, 
        learnable_logit: bool = True, 
        init_scale: Union[str, float, None] = "auto",
        label_smoothing: float = 0.0, 
        max_scale: float = 100.0
    ):
        super().__init__()
        self.learnable = learnable_logit
        default_log_scale = math.log(10.0)
        if isinstance(init_scale, (int, float)):
            default_log_scale = math.log(float(init_scale))

        if self.learnable:
            self.logit_scale = nn.Parameter(torch.tensor(default_log_scale, dtype=torch.float32))
        else:
            self.register_buffer("logit_scale", torch.tensor(default_log_scale, dtype=torch.float32))

        self.label_smoothing = label_smoothing
        self.max_scale = math.log(max_scale)

    def sync_from_model_params(self, model: nn.Module):
        '''从权重同步logit_scale'''
        with torch.no_grad():
            if str(self.cfg_scale).lower() in ("auto", "none"):
                if hasattr(model, "logit_scale") and isinstance(model.logit_scale, (nn.Parameter, torch.Tensor)):
                    self.logit_scale.fill_(model.logit_scale.detach().cpu().item())

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor, **kwargs) -> torch.Tensor:
        image_emb = F.normalize(image_emb.float(), dim=-1)
        text_emb = F.normalize(text_emb.float(), dim=-1)

        scale = torch.clamp(self.logit_scale, max=self.max_scale).exp()
        logits = scale * (image_emb @ text_emb.t())  # [B, B]

        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        loss_t = F.cross_entropy(logits.t(), labels, label_smoothing=self.label_smoothing)
        return 0.5 * (loss_i + loss_t)


class TIPsv2Loss(nn.Module):
    """TIPsv2 损失 (补齐 Mask 屏蔽与局部协同防崩机制)"""

    def __init__(
        self,
        learnable_logit: bool = True,
        global_name: str = "siglip",
        label_smoothing: float = 0.0,
        init_scale: Union[str, float, None] = "auto",
        init_bias: Union[str, float, None] = "auto",
        max_scale: float = 100.0,
        label_constrain: bool = False,
    ):
        super().__init__()
        if global_name == "infonce":
            self.global_loss = InfoNCELoss(
                learnable_logit,
                init_scale=init_scale,
                label_smoothing=label_smoothing, 
                max_scale=max_scale,
            )
        elif global_name == "sigmoid":
            self.global_loss = SiglipSigmoidLoss(
                learnable_logit, 
                init_scale=init_scale,
                init_bias=init_bias,
                max_scale=max_scale,
                label_constrain=label_constrain,
            )
        else:
            raise ValueError(f"Not support {global_name} of TIPsv2 Loss choice!")

    def forward(
        self,
        image_emb: torch.Tensor,
        text_emb: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        loss = self.global_loss(image_emb, text_emb, **kwargs)

        return loss


def build_loss(model_cfg) -> nn.Module:
    lc = model_cfg.loss
    name = lc.name
    if name == "auto":
        name = "tipsv2" if model_cfg.arch == "tipsv2" else "siglip"

    if name == "siglip":
        return SiglipSigmoidLoss(
            learnable_logit=lc.learnable_logit,
            init_scale=lc.init_scale,
            init_bias=lc.init_bias,
            max_scale=lc.max_scale,
            label_constrain=lc.label_constrain
        )
    if name == "infonce":
        return InfoNCELoss(
            learnable_logit=lc.learnable_logit, 
            init_scale=lc.init_scale,
            label_smoothing=model_cfg.label_smoothing, 
            max_scale=lc.max_scale
        )
    if name == "tipsv2":
        return TIPsv2Loss(
            learnable_logit=lc.learnable_logit,
            global_name=lc.tips_global,
            label_smoothing=model_cfg.label_smoothing,
            init_scale=lc.init_scale,
            init_bias=lc.init_bias,
            max_scale=lc.max_scale,
            label_constrain=lc.label_constrain
        )
    raise ValueError(f"Unknown loss configuration: {name}")
