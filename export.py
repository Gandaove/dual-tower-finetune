"""双塔多模态模型部署导出工具 (支持闭集分类与解耦双塔导出)"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import safetensors
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from polygraphy.backend.trt import (
    CreateConfig,
    Profile,
    engine_from_network,
    network_from_onnx_path,
    save_engine as polygraphy_save_engine,
)

from dataset.augment import DualTowerTransforms
from models.dual_tower_multimodal import build_dual_tower_model
from utils.config import Config, DataConfig, ModelConfig, TrainConfig, TaxonomyClasses
from utils.logger import get_logger

_log = get_logger("export")
EXPORT_DUMMY_BATCH = 2  # 防止 Dynamo 符号系统产生 batch=1 特化


class ModelPatcherRegistry:
    """各模型专用 Dynamo 兼容补丁注册中心"""
    _registry: Dict[str, Callable[..., None]] = {}

    @classmethod
    def register(cls, arch: str):
        def decorator(fn: Callable[..., None]):
            cls._registry[arch.lower().strip()] = fn
            return fn
        return decorator

    @classmethod
    def patch(cls, arch: str, model: nn.Module, **kwargs) -> None:
        key = arch.lower().strip()
        patcher = cls._registry.get(key)
        if patcher is not None:
            patcher(model, **kwargs)
        else:
            _log.debug(f"[Export Patch] 架构 '{arch}' 无须执行专用 Patch")


@ModelPatcherRegistry.register("tipsv2")
def patch_tipsv2(model: nn.Module, **_):
    """替换 TIPSv2 视觉塔 MemEffAttention 及文本塔残差块原生 Attention 实现"""
    vision_encoder = getattr(model.model, "vision_encoder", None)
    if vision_encoder and hasattr(vision_encoder, "interpolate_antialias"): # 兼容尺寸
        vision_encoder.interpolate_antialias = False
        _log.info("[Export Patch] 已强制关闭 TIPSv2 的 interpolate_antialias 属性")

    if vision_encoder and hasattr(vision_encoder, "blocks"):
        for block in vision_encoder.blocks:
            if hasattr(block, "attn"):
                m = block.attn

                def _forward_vision(x: torch.Tensor, m_mod=m) -> torch.Tensor:
                    B, N, C = x.shape
                    num_heads = getattr(m_mod, "num_heads", 12)
                    head_dim = C // num_heads
                    qkv = m_mod.qkv(x).reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]

                    scale = 1.0 / math.sqrt(head_dim)
                    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                    weights = m_mod.attn_drop(F.softmax(scores, dim=-1))

                    out = torch.matmul(weights, v).transpose(1, 2).reshape(B, N, C)
                    return m_mod.proj_drop(m_mod.proj(out))

                m.forward = _forward_vision
        _log.info(f"[Export Patch] 已挂载 TIPSv2 视觉塔 {len(vision_encoder.blocks)} 个注意力原生实现")

    text_encoder = getattr(model.model, "text_encoder", None)
    if text_encoder and hasattr(text_encoder, "transformer") and hasattr(text_encoder.transformer, "resblocks"):
        for block in text_encoder.transformer.resblocks:
            if hasattr(block, "attn"):
                b, a = block, block.attn
                num_heads = a.num_heads
                embed_dim = a.embed_dim
                head_dim = embed_dim // num_heads

                def _attention_call(x: torch.Tensor, mask: Optional[torch.Tensor] = None, b_mod=b, a_mod=a, *_, **__) -> torch.Tensor:
                    L = x.shape[0]
                    x_b = x.transpose(0, 1)

                    qkv = F.linear(x_b, a_mod.in_proj_weight, a_mod.in_proj_bias)
                    q, k, v = qkv.chunk(3, dim=-1)
                    q = q.view(-1, L, num_heads, head_dim).transpose(1, 2)
                    k = k.view(-1, L, num_heads, head_dim).transpose(1, 2)
                    v = v.view(-1, L, num_heads, head_dim).transpose(1, 2)

                    scale = 1.0 / math.sqrt(head_dim)
                    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

                    if getattr(b_mod, "attn_mask", None) is not None:
                        c_mask = b_mod.attn_mask[:L, :L] if b_mod.attn_mask.dim() == 2 else b_mod.attn_mask
                        scores = scores + c_mask.to(dtype=scores.dtype, device=scores.device)

                    if mask is not None:
                        kp_mask = mask
                        if kp_mask.dim() == 2:
                            if kp_mask.size(0) == L and kp_mask.size(1) != L:
                                kp_mask = kp_mask.transpose(0, 1)
                            kp_mask = kp_mask.unsqueeze(1).unsqueeze(2)

                        mask_val = -1e4
                        if kp_mask.dtype == torch.bool:
                            scores = scores.masked_fill(~kp_mask, mask_val)
                        elif kp_mask.is_floating_point():
                            scores = scores.masked_fill(kp_mask < 0.5, mask_val)
                        else:
                            scores = scores.masked_fill(kp_mask > 0, mask_val)

                    weights = F.softmax(scores, dim=-1)
                    out = torch.matmul(weights, v).transpose(1, 2).contiguous().view(-1, L, embed_dim)
                    return a_mod.out_proj(out).transpose(0, 1)

                block.attention = _attention_call
        _log.info(f"[Export Patch] 已挂载 TIPSv2 文本塔 {len(text_encoder.transformer.resblocks)} 个注意力原生实现")


@ModelPatcherRegistry.register("fgclip2")
def patch_fgclip2(model: nn.Module, img_size: int = 224, **_):
    """固化 FG-CLIP2 视觉位置编码为常量 Buffer，并重构多头池化注意力前向"""
    m = getattr(model, "model", model)
    vision_model = getattr(m, "vision_model", None)
    if not (vision_model and hasattr(vision_model, "embeddings")):
        _log.warning("[Export Patch] 未检测到 FG-CLIP2 vision_model.embeddings，跳过")
        return

    embeddings = vision_model.embeddings
    patch_size = getattr(embeddings, "patch_size", 16)
    h_p = img_size // patch_size
    w_p = img_size // patch_size
    target_patches = h_p * w_p
    device = embeddings.position_embedding.weight.device

    grid_size = embeddings.position_embedding_size
    embed_dim = embeddings.embed_dim
    pos_2d = embeddings.position_embedding.weight.reshape(grid_size, grid_size, embed_dim)
    concrete_shapes = torch.tensor([[h_p, w_p]], dtype=torch.long, device=device)

    with torch.no_grad():
        cached_pos = embeddings.resize_positional_embeddings(
            positional_embeddings=pos_2d,
            spatial_shapes=concrete_shapes,
            max_length=target_patches,
        )

    embeddings.register_buffer("static_pos_embeddings", cached_pos.contiguous(), persistent=False)
    embeddings.resize_positional_embeddings = lambda *args, **kwargs: embeddings.static_pos_embeddings
    _log.info(f"[Export Patch] FG-CLIP2 视觉位置编码已固化为常量 Buffer: {cached_pos.shape}")

    if hasattr(vision_model, "head"):
        h = vision_model.head
        attn = h.attention
        num_heads = attn.num_heads
        head_dim = embed_dim // num_heads

        w_q, w_k, w_v = attn.in_proj_weight.chunk(3, dim=0)
        b_q, b_k, b_v = (
            attn.in_proj_bias.chunk(3, dim=0)
            if attn.in_proj_bias is not None
            else (None, None, None)
        )

        def _forward_pooling(
            hidden_state: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            *_,
            **__,
        ) -> torch.Tensor:
            B, N = hidden_state.shape[0], hidden_state.shape[1]
            q_in = h.probe.expand(B, -1, -1)

            q = F.linear(q_in, w_q, b_q).view(B, 1, num_heads, head_dim).transpose(1, 2)
            k = F.linear(hidden_state, w_k, b_k).view(B, N, num_heads, head_dim).transpose(1, 2)
            v = F.linear(hidden_state, w_v, b_v).view(B, N, num_heads, head_dim).transpose(1, 2)

            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            weights = F.softmax(scores, dim=-1)
            out = torch.matmul(weights, v).transpose(1, 2).contiguous().view(B, 1, embed_dim)
            out = attn.out_proj(out)

            residual = out
            out = h.layernorm(out)
            out = residual + h.mlp(out)
            return out.squeeze(1)

        h.forward = _forward_pooling
        _log.info("[Export Patch] 已挂载 FG-CLIP2 视觉池化头原生动态 Attention 实现")


@ModelPatcherRegistry.register("siglip2")
@ModelPatcherRegistry.register("siglip")
def patch_siglip(model: nn.Module, **_):
    """
    为 SigLIP / SigLIP 2 适配导出补丁：
    1. 重写视觉池化头 SiglipMultiheadAttentionPoolingHead，消除 probe.repeat 导致的批次特化
    2. 校验文本塔位置编码与最大长度一致性，确保大于 64 时的符号安全
    """
    m = getattr(model, "model", model)
    vision_model = getattr(m, "vision_model", None)
    text_model = getattr(m, "text_model", None)

    v_head = vision_model.head
    v_head_attn = getattr(v_head, "attention", None)
        
    # 仅针对使用 MultiheadAttention 的官方池化头进行原生重写
    if v_head_attn is not None and isinstance(v_head_attn, nn.MultiheadAttention):
        embed_dim = v_head_attn.embed_dim
        num_heads = v_head_attn.num_heads
        head_dim = embed_dim // num_heads

        w_q, w_k, w_v = v_head_attn.in_proj_weight.chunk(3, dim=0)
        b_q, b_k, b_v = (
            v_head_attn.in_proj_bias.chunk(3, dim=0)
            if v_head_attn.in_proj_bias is not None
            else (None, None, None)
        )

        def _forward_siglip_pooling(
            hidden_state: torch.Tensor,
            *args,
            **kwargs,
        ) -> torch.Tensor:
            # hidden_state: [B, N, C]
            B, N = hidden_state.shape[0], hidden_state.shape[1]

            # 使用 expand 替代 repeat，彻底避免对动态批次符号执行 int() 解包
            q_in = v_head.probe.expand(B, -1, -1)  # [B, 1, C]

            q = F.linear(q_in, w_q, b_q).view(B, 1, num_heads, head_dim).transpose(1, 2)
            k = F.linear(hidden_state, w_k, b_k).view(B, N, num_heads, head_dim).transpose(1, 2)
            v = F.linear(hidden_state, w_v, b_v).view(B, N, num_heads, head_dim).transpose(1, 2)

            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, 1, N]
            weights = F.softmax(scores, dim=-1)
            
            out = torch.matmul(weights, v).transpose(1, 2).contiguous().view(B, 1, embed_dim)
            out = v_head_attn.out_proj(out)  # [B, 1, C]

            # 残差与 MLP 处理
            residual = out
            out = v_head.layernorm(out)
            out = residual + v_head.mlp(out)
            return out.squeeze(1)  # 输出严格保证 [B, C]

        v_head.forward = _forward_siglip_pooling
        _log.info("[Export Patch] 已成功挂载 SigLIP 视觉池化头动态 Batch 原生实现")

    # ---------------- 2. 校验长文本位置编码一致性 ---------------- #
    embed = text_model.embeddings
    pos_embed = getattr(embed, "position_embedding", None)
    if pos_embed is not None:
        max_pos = pos_embed.weight.shape[0]
        # 确保 position_ids buffer 长度与插值后的权重完全同步
        if not hasattr(embed, "position_ids") or embed.position_ids.shape[1] != max_pos:
            new_pos_ids = torch.arange(max_pos, device=pos_embed.weight.device).unsqueeze(0)
            embed.register_buffer("position_ids", new_pos_ids, persistent=False)
            _log.info(f"[Export Patch] 同步修正 SigLIP 文本位置索引 Buffer 长度至 {max_pos}")


class BaseDeployWrapper(nn.Module):
    """部署封装基类"""
    @staticmethod
    def _safe_l2_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """进行安全 L2 归一化"""
        denom = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + eps)
        return x / denom


class VisionDeployWrapper(BaseDeployWrapper):
    """视觉塔部署封装：支持纯向量输出或固化分类矩阵乘"""

    def __init__(
        self,
        base_model: nn.Module,
        class_prototypes: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        bias: Optional[float] = None,
        activation: str = "none",
    ):
        super().__init__()
        self.base_model = base_model
        self.is_classifier = class_prototypes is not None
        self.activation = activation

        if self.is_classifier:
            num_classes, dim = class_prototypes.shape
            fused_weight = class_prototypes.float() * float(scale)
            self.classifier = nn.Linear(dim, num_classes, bias=(bias is not None))
            self.classifier.weight.data.copy_(fused_weight)
            if bias is not None:
                self.classifier.bias.data.fill_(float(bias))
        else:
            self.classifier = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.base_model.get_vision_embedding(images, to_float32=False, return_dense=False)
        feats_norm = self._safe_l2_norm(feats.float())

        if not self.is_classifier:
            return feats_norm

        logits = self.classifier(feats_norm)
        if self.activation == "sigmoid":
            return torch.sigmoid(logits)
        elif self.activation == "softmax":
            return F.softmax(logits, dim=-1)
        return logits


class TextDeployWrapper(BaseDeployWrapper):
    """文本塔部署封装：统一暴露 input_ids 与 attention_mask 接口"""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.arch = getattr(base_model, "arch", "").lower()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        text_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "padding_mask": (1.0 - attention_mask.float()),
        }
        feats = self.base_model.get_text_embedding(text_inputs, to_float32=False, return_dense=False)
        return self._safe_l2_norm(feats.float())


class TRTCompiler:
    """TensorRT 引擎编译与 ModelOpt PTQ 量化执行器"""

    @staticmethod
    def apply_modelopt_ptq(
        onnx_path: Path,
        quant_mode: Optional[str],
        calib_data: Optional[Dict[str, np.ndarray]],
    ) -> Path:
        if not quant_mode:
            return onnx_path
        if not calib_data:
            raise RuntimeError(f"开启 {quant_mode} 量化必须提供有效的校准数据")

        from modelopt.onnx.quantization import quantize as moq_quantize

        out_path = onnx_path.with_suffix(f".{quant_mode}.onnx")
        exclude_ops = ["LayerNormalization", "ReduceMean", "Div", "Softmax", "Erf", "Sigmoid"]
        _log.info(f"[PTQ] 执行 ModelOpt {quant_mode.upper()} 离线量化 -> {out_path.name}")
        moq_quantize(
            str(onnx_path),
            quantize_mode=quant_mode,
            calibration_data=calib_data,
            calibration_method="entropy",
            output_path=str(out_path),
            op_types_to_exclude=exclude_ops,
        )
        return out_path

    @staticmethod
    def build_engine(
        onnx_path: Path,
        engine_path: Path,
        profile_dict: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]],
        precision: str = "fp16",
        quant_mode: Optional[str] = None,
    ) -> None:
        _log.info(f"[TRT] 编译 TensorRT Engine -> {engine_path.name}")
        profile = Profile()
        for name, (min_s, opt_s, max_s) in profile_dict.items():
            profile.add(name, min=min_s, opt=opt_s, max=max_s)

        use_fp16 = precision == "fp16" or quant_mode in ("fp8", "fp4")
        use_bf16 = precision == "bf16"
        use_fp8 = quant_mode == "fp8"

        config = CreateConfig(
            tf32=True,
            fp16=use_fp16,
            bf16=use_bf16,
            fp8=use_fp8,
            profiles=[profile],
            precision_constraints="obey",
        )
        loader = network_from_onnx_path(str(onnx_path))
        engine = engine_from_network(loader, config=config)
        polygraphy_save_engine(engine, path=str(engine_path))


class DualTowerExporter:
    def __init__(
        self,
        ckpt_path: Union[str, Path],
        output_dir: Union[str, Path],
        target_format: str = "engine",
        quant_mode: Optional[str] = None,
        precision: str = "fp16",
        max_batch_size: int = 32,
        calib_dir: Optional[Union[str, Path]] = None,
        calib_num: int = 128,
        text_arg: Optional[str] = None,
    ):
        self.ckpt_path = Path(ckpt_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.target_format = target_format.lower()
        self.quant_mode = quant_mode.lower() if quant_mode else None
        self.precision = precision.lower()
        self.max_batch_size = max_batch_size
        self.calib_dir = Path(calib_dir).resolve() if calib_dir else None
        self.calib_num = calib_num
        self.text_arg = text_arg.strip() if text_arg else None

        if not self.ckpt_path.is_file():
            raise FileNotFoundError(f"权重文件不存在: {self.ckpt_path}")
        if self.quant_mode not in (None, "fp8", "fp4"):
            raise ValueError(f"量化仅支持 'fp8' 或 'fp4'，传入非法值: {self.quant_mode}")
        if self.text_arg is not None:
            if self.text_arg not in ("1", "close-set") and not self.text_arg.endswith(".txt"):
                raise ValueError(
                    f"非法 --text 参数: '{self.text_arg}'。仅允许传入 '1', 'close-set' 或以 '.txt' 结尾的类列表文件路径。"
                )

        self.ckpt_dir = self.ckpt_path.parent
        self.model_cfg, self.data_cfg = self._load_configs()
        self.img_size = self.data_cfg.augment.resize.size
        self.max_text_length = self.model_cfg.max_text_length
        self.transforms = DualTowerTransforms(self.data_cfg)

    def _load_configs(self) -> Tuple[ModelConfig, DataConfig]:
        model_json = self.ckpt_dir / "set_model_config.json"
        data_json = self.ckpt_dir / "set_data_config.json"
        if not (model_json.is_file() and data_json.is_file()):
            raise FileNotFoundError(f"配置快照缺失: {model_json} 或 {data_json}")

        with open(model_json, "r", encoding="utf-8") as f:
            m_cfg = ModelConfig.from_dict(json.load(f))
        with open(data_json, "r", encoding="utf-8") as f:
            d_cfg = DataConfig.from_dict(json.load(f))
        return m_cfg, d_cfg

    def _verify_no_lora_weights(self) -> None:
        """严格校验权重文件，禁止未合并的 LoRA 检查点进入导出管线。"""
        if "-lora" in self.ckpt_path.name.lower():
            raise RuntimeError(
                f"检测到输入权重 '{self.ckpt_path.name}' 包含 '-lora' 后缀，属于未合并的 LoRA 增量检查点。"
                "导出流程仅支持完整合并后的全量权重，请先调用 models/merger.py 完成融合后再执行导出。"
            )

        with safetensors.safe_open(str(self.ckpt_path), framework="pt") as f:
            meta = f.metadata() or {}
            if str(meta.get("use_lora", "")).lower() == "true":
                raise RuntimeError(
                    f"权重文件 '{self.ckpt_path.name}' 元数据标记 use_lora=True。"
                    "导出仅支持合并后的完整权重，请先调用 models/merger.py 完成融合后再执行导出。"
                )
            if any("lora_" in k for k in f.keys()):
                raise RuntimeError(
                    f"权重文件 '{self.ckpt_path.name}' 内部检测到 'lora_' 增量参数键。"
                    "导出仅支持合并后的完整权重，请先调用 models/merger.py 完成融合后再执行导出。"
                )

    def _load_model(self) -> nn.Module:
        import peft

        self._verify_no_lora_weights()
        self.model_cfg.use_lora = False                 # 确保构造底层模型时不注入任何 PEFT 包装
        cfg = Config(model=self.model_cfg, data=self.data_cfg, train=TrainConfig(freeze_mode="both"))
        model = build_dual_tower_model(cfg, transform=self.transforms)
        _log.info(f"[Model] 载入全量权重: {self.ckpt_path.name}")
        model.load_safetensors(str(self.ckpt_path), load_opt=False)

        for mod in model.modules():
            if isinstance(mod, peft.PeftModel):
                raise RuntimeError("模型中存在未解耦的 PeftModel 实例，请检查权重完整性。")

        model.eval()
        ModelPatcherRegistry.patch(self.model_cfg.arch, model, img_size=self.img_size)
        return model

    def _resolve_candidate_texts(self) -> List[str]:
        """严格解析闭集分类候选文本，保证类别唯一序与 DataModule 完全一致。"""
        if self.text_arg in ("1", "close-set"):
            csv_path = Path(self.data_cfg.class_map_csv)
            if not csv_path.is_absolute():
                csv_path = Path(self.data_cfg.root) / csv_path
            if not csv_path.is_file():
                raise FileNotFoundError(f"未找到物种映射表 class_map_csv: {csv_path}")

            taxonomy = TaxonomyClasses.from_csv(csv_path)
            tmpl = self.model_cfg.eval.prompt_template or "{}"
            candidate_texts = [tmpl.format(latin) for latin in taxonomy.latins]
            return candidate_texts

        if self.text_arg.endswith(".txt"):
            txt_path = Path(self.text_arg)
            if not txt_path.is_file():
                raise FileNotFoundError(f"未找到指定的候选文本文件: {self.text_arg}")
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            if not lines:
                raise ValueError(f"候选文本文件内容为空: {txt_path}")
            return lines

        raise ValueError(f"非法 --text 参数: '{self.text_arg}'")

    def _encode_text_prototypes(self, model: nn.Module, texts: List[str]) -> torch.Tensor:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        feats = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i : i + batch_size]
                tok = model.tokenize_text(chunk)
                tok = {k: v.to(device) for k, v in tok.items()}
                emb = model.get_embeddings(texts=tok, to_float32=True)
                feats.append(BaseDeployWrapper._safe_l2_norm(emb).cpu())
        return torch.cat(feats, dim=0)

    def _prepare_vision_calib_data(self, input_name: str) -> Optional[Dict[str, np.ndarray]]:
        if not self.quant_mode:
            return None
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        search_dir = self.calib_dir if self.calib_dir and self.calib_dir.is_dir() else Path(self.data_cfg.root)
        calib_images = sorted([p for p in search_dir.rglob("*") if p.suffix.lower() in valid_exts])[: self.calib_num]
        if not calib_images:
            raise RuntimeError("未检索到用于视觉 PTQ 量化的校准图像")

        batch_tensors = []
        for p in calib_images:
            try:
                with Image.open(p) as img:
                    batch_tensors.append(self.transforms(np.asarray(img.convert("RGB")), train=False))
            except Exception as e:
                _log.warning(f"校准样本读取跳过: {p} ({e})")
        return {input_name: torch.stack(batch_tensors, dim=0).numpy()}

    def _prepare_text_calib_data(self, model: nn.Module) -> Optional[Dict[str, np.ndarray]]:
        """为文本塔 PTQ 量化准备真实词元校准集"""
        if not self.quant_mode:
            return None

        csv_path = Path(self.data_cfg.class_map_csv)
        if not csv_path.is_absolute():
            csv_path = Path(self.data_cfg.root) / csv_path

        sample_texts: List[str] = []
        if csv_path.is_file():
            df = pd.read_csv(csv_path)
            tmpl = self.model_cfg.eval.prompt_template or "{}"
            if "pest_latin_name" in df.columns:
                sample_texts = [
                    tmpl.format(str(r["pest_latin_name"]).strip())
                    for _, r in df.iterrows()
                    if str(r["pest_latin_name"]).strip().lower() != "nan"
                ]

        if not sample_texts:
            sample_texts = ["a photo of target object."] * self.calib_num

        if len(sample_texts) < self.calib_num:
            repeat_factor = (self.calib_num // len(sample_texts)) + 1
            sample_texts = (sample_texts * repeat_factor)[: self.calib_num]
        else:
            sample_texts = sample_texts[: self.calib_num]

        tok = model.tokenize_text(sample_texts)
        return {
            "input_ids": tok["input_ids"].cpu().numpy(),
            "attention_mask": tok["attention_mask"].cpu().numpy(),
        }

    def _extract_loss_scalars(self, model: nn.Module) -> Tuple[float, Optional[float], str]:
        loss_cfg = self.model_cfg.loss
        loss_name = loss_cfg.name if loss_cfg.name != "auto" else ("tipsv2" if self.model_cfg.arch == "tipsv2" else "siglip")
        act_type = "softmax" if (loss_name == "infonce" or (loss_name == "tipsv2" and loss_cfg.tips_global == "infonce")) else "sigmoid"

        loss_mod = getattr(model, "loss_fn", None)
        target = getattr(loss_mod, "global_loss", loss_mod) if loss_mod else None

        raw_scale = 10.0
        raw_bias = None

        if target and hasattr(target, "logit_scale"):
            raw_scale = math.exp(min(target.logit_scale.detach().cpu().item(), target.max_scale))
        elif hasattr(model.model, "logit_scale"):
            raw_scale = math.exp(model.model.logit_scale.detach().cpu().item())

        if act_type == "sigmoid":
            if target and hasattr(target, "logit_bias"):
                raw_bias = target.logit_bias.detach().cpu().item()
            elif hasattr(model.model, "logit_bias"):
                raw_bias = model.model.logit_bias.detach().cpu().item()

        return raw_scale, raw_bias, act_type

    def _export_closed_set_classifier(self, model: nn.Module) -> None:
        candidate_texts = self._resolve_candidate_texts()
        _log.info(f"[Export] 闭集分类模式: 固化文本候选数={len(candidate_texts)}")
        prototypes = self._encode_text_prototypes(model, candidate_texts)
        scale, bias, act_type = self._extract_loss_scalars(model)

        deploy_vision = VisionDeployWrapper(
            model,
            class_prototypes=prototypes,
            scale=scale,
            bias=bias,
            activation=act_type,
        ).cpu()

        onnx_path = self.output_dir / "classifier.onnx"
        dummy_img = torch.randn(EXPORT_DUMMY_BATCH, 3, self.img_size, self.img_size, dtype=torch.float32)
        torch.onnx.export(
            deploy_vision,
            dummy_img,
            str(onnx_path),
            do_constant_folding=True,
            input_names=["images"],
            output_names=["probs"],
            dynamic_axes={"images": {0: "batch_size"}, "probs": {0: "batch_size"}},
        )

        calib_data = self._prepare_vision_calib_data("images")
        processed_onnx = TRTCompiler.apply_modelopt_ptq(onnx_path, self.quant_mode, calib_data)

        if self.target_format == "engine":
            engine_path = self.output_dir / "classifier.engine"
            profiles = {
                "images": (
                    (1, 3, self.img_size, self.img_size),
                    (max(1, self.max_batch_size // 2), 3, self.img_size, self.img_size),
                    (self.max_batch_size, 3, self.img_size, self.img_size),
                )
            }
            TRTCompiler.build_engine(processed_onnx, engine_path, profiles, self.precision, self.quant_mode)
            _log.info(f"[Export] 已成功导出: {engine_path.name}")

    def _export_decoupled_towers(self, model: nn.Module) -> None:
        _log.info("[Export] 解耦模式: 独立导出视觉与文本模型...")

        # 1. 导出 Vision 塔
        deploy_vision = VisionDeployWrapper(model, class_prototypes=None).cpu()
        v_onnx_path = self.output_dir / "vision.onnx"
        dummy_img = torch.randn(EXPORT_DUMMY_BATCH, 3, self.img_size, self.img_size, dtype=torch.float32)
        torch.onnx.export(
            deploy_vision,
            dummy_img,
            str(v_onnx_path),
            do_constant_folding=True,
            input_names=["images"],
            output_names=["image_embeddings"],
            dynamic_axes={"images": {0: "batch_size"}, "image_embeddings": {0: "batch_size"}},
        )

        v_calib = self._prepare_vision_calib_data("images")
        v_processed_onnx = TRTCompiler.apply_modelopt_ptq(v_onnx_path, self.quant_mode, v_calib)

        if self.target_format == "engine":
            v_engine_path = self.output_dir / "vision.engine"
            v_profiles = {
                "images": (
                    (1, 3, self.img_size, self.img_size),
                    (max(1, self.max_batch_size // 2), 3, self.img_size, self.img_size),
                    (self.max_batch_size, 3, self.img_size, self.img_size),
                )
            }
            TRTCompiler.build_engine(v_processed_onnx, v_engine_path, v_profiles, self.precision, self.quant_mode)
            _log.info(f"[Export] 已成功导出: {v_engine_path.name}")

        # 2. 导出 Text 塔 (始终保持原浮点精度，不参与离线视觉量化)
        deploy_text = TextDeployWrapper(model).cpu()
        t_onnx_path = self.output_dir / "text.onnx"
        dummy_ids = torch.zeros(EXPORT_DUMMY_BATCH, self.max_text_length, dtype=torch.long)
        dummy_mask = torch.ones(EXPORT_DUMMY_BATCH, self.max_text_length, dtype=torch.long)
        torch.onnx.export(
            deploy_text,
            (dummy_ids, dummy_mask),
            str(t_onnx_path),
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["text_embeddings"],
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "text_embeddings": {0: "batch_size"},
            },
        )
        # TODO: [实验性特性] 解耦文本塔量化支持，放开与视觉塔一致的 quant_mode (FP8 / FP4) 试验
        t_calib = self._prepare_text_calib_data(model)
        t_processed_onnx = TRTCompiler.apply_modelopt_ptq(t_onnx_path, self.quant_mode, t_calib)
        if self.target_format == "engine":
            t_engine_path = self.output_dir / "text.engine"
            t_profiles = {
                "input_ids": (
                    (1, self.max_text_length),
                    (max(1, self.max_batch_size // 2), self.max_text_length),
                    (self.max_batch_size, self.max_text_length),
                ),
                "attention_mask": (
                    (1, self.max_text_length),
                    (max(1, self.max_batch_size // 2), self.max_text_length),
                    (self.max_batch_size, self.max_text_length),
                ),
            }
            TRTCompiler.build_engine(t_processed_onnx, t_engine_path, t_profiles, self.precision, quant_mode=self.quant_mode)
            _log.info(f"[Export] 已成功导出: {t_engine_path.name}")

    def export(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        model = self._load_model()

        if self.text_arg is not None:
            self._export_closed_set_classifier(model)
        else:
            self._export_decoupled_towers(model)


def build_export_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Dual-Tower Unified ModelOpt & TensorRT Exporter")
    p.add_argument("--ckpt_path", required=True, help="已完成合并的全量 safetensors 检查点路径")
    p.add_argument("--output_dir", required=True, help="导出目标目录")
    p.add_argument("--target_format", default="engine", choices=["onnx", "engine"], help="目标格式")
    p.add_argument(
        "--text",
        default=None,
        help="Classifier: 'close-set'(或'1')使用默认配置类目; txt文件路径为自定义类目; None为导出双塔",
    )
    p.add_argument("--quant_mode", default=None, choices=["fp8", "fp4"], help="视觉塔 PTQ 量化模式")
    p.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"], help="高精度回退格式")
    p.add_argument("--max_batch_size", type=int, default=32, help="动态 Profile 最大批次")
    p.add_argument("--calib_dir", default=None, help="PTQ 校准图像目录")
    p.add_argument("--calib_num", type=int, default=128, help="校准图像数量")
    return p


def main(argv: Optional[List[str]] = None):
    args = build_export_parser().parse_args(argv)
    exporter = DualTowerExporter(
        ckpt_path=args.ckpt_path,
        output_dir=args.output_dir,
        target_format=args.target_format,
        quant_mode=args.quant_mode,
        precision=args.precision,
        max_batch_size=args.max_batch_size,
        calib_dir=args.calib_dir,
        calib_num=args.calib_num,
        text_arg=args.text,
    )
    exporter.export()


if __name__ == "__main__":
    main()
