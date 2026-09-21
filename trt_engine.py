"""双塔多模态 TensorRT 推理架构"""
from __future__ import annotations

import abc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from utils.logger import get_logger
from utils.engine_base import BaseTRTEngine, ImageProcessor, trt_stream_guard

_logger = get_logger("dual_tower_classifier")


class BaseEngineClassifier(BaseTRTEngine, abc.ABC):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.conf_thresh = self.cfg.get("conf_thresh", 0.8)
        self.max_batch_size = self.cfg.get("batch_size", 16)
        self.input_name = next(iter(self.input_meta.keys()))
        self.output_name = next(iter(self.output_meta.keys()))

        raw_size = self.cfg.get("input_size", 224)
        target_size = (
            (raw_size, raw_size) if isinstance(raw_size, int) else tuple(raw_size)
        )

        self.image_processor = ImageProcessor.from_config(
            self.cfg.get("preprocess", {}),
            target_size=target_size,
            device=self.device,
            default_resize="letterbox",
            default_mean=None,
            default_std=None,
        )

    def preprocess(self, imgs: List[np.ndarray]) -> torch.Tensor:
        target_dtype = self.input_meta[self.input_name].dtype
        batch_tensor, _ = self.image_processor.preprocess(
            imgs, target_dtype=target_dtype
        )
        return batch_tensor

    @torch.no_grad()
    @trt_stream_guard
    def classify(self, imgs: List[np.ndarray]) -> Tuple[List[int], List[float]]:
        if not imgs or len(imgs) == 0:
            return [], []

        batch_size = len(imgs)
        if batch_size > self.max_batch_size:
            self.logger.warning(
                f"Batch ({batch_size}) 超出配置上限 ({self.max_batch_size})"
            )

        input_tensor = self.preprocess(imgs)
        outputs = self._infer_core({self.input_name: input_tensor})
        return self.postprocess(outputs[self.output_name])

    @abc.abstractmethod
    def postprocess(self, output_tensor: torch.Tensor) -> Tuple[List[int], List[float]]:
        raise NotImplementedError


class TowerClassifier(BaseEngineClassifier):
    """闭集固化分类器：绑定 classifier.engine，依赖固定 classes 映射"""

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        # 闭集固化分类器的权重矩阵与类目数强绑定，保留静态 classes 映射
        self.classes: Dict[int, str] = self.cfg.get("classes", {})
        if not self.classes:
            self.logger.warning("未检测到 classes 配置，后处理将仅输出索引 ID")

    def preprocess(self, imgs: List[np.ndarray]) -> torch.Tensor:
        if not isinstance(imgs, list) or (imgs and not isinstance(imgs[0], np.ndarray)):
            raise TypeError(f"输入数据必须为 List[np.ndarray]，检测到非法类型: {type(imgs[0]) if imgs else type(imgs)}")
        return super().preprocess(imgs)

    def postprocess(self, output_tensor: torch.Tensor) -> Tuple[List[int], List[float]]:
        # output_tensor: [B, Num_Classes]，图内已包含 Scale/Bias 与 Sigmoid/Softmax 激活
        max_conf, max_idx = torch.max(output_tensor, dim=1)
        max_idx[max_conf == 0.0] = -1
        return max_idx.cpu().tolist(), max_conf.cpu().tolist()



class TextTRTEngine(BaseTRTEngine):
    """专用文本塔 TRT 引擎：支持 input_ids 与 attention_mask 多输入并行前向"""
    def __init__(self, cfg: Dict, stream: torch.cuda.Stream):
        self.external_stream = stream
        super().__init__(cfg)
        # 强制复用外部流，杜绝多引擎并发数据竞争
        self.stream = self.external_stream

    def encode_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        feed_dict = {
            "input_ids": input_ids.contiguous(),
            "attention_mask": attention_mask.contiguous(),
        }
        outputs = self._infer_core(feed_dict)
        output_name = next(iter(self.output_meta.keys()))
        return outputs[output_name]


class TRTZeroShotClassifier(BaseEngineClassifier):
    """解耦动态 Zero-Shot 分类器：无固定 classes，支持候选词动态实时编译"""

    def __init__(self, cfg: Dict):
        super().__init__(cfg)

        text_engine_path = self.cfg.get("text_engine_path")
        if not text_engine_path or not Path(text_engine_path).exists():
            raise FileNotFoundError(f"未配置或不存在 text_engine_path: {text_engine_path}")

        text_cfg = {
            "model_path": text_engine_path,
            "device": str(self.device),
            "log_name": f"zeroshot-text",
        }
        self.text_engine = TextTRTEngine(text_cfg, stream=self.stream)

        # 初始化分词器
        tokenizer_path = self.cfg.get("tokenizer_path")
        if not tokenizer_path:
            raise ValueError("Zero-Shot 模式必须配置 tokenizer_path")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        self.max_text_len: int = self.cfg.get("max_text_length", 128)
        # 余弦相似度截断阈值，默认 0.25
        self.conf_thresh: float = self.cfg.get("conf_thresh", 0.25)

        # 动态文本原型状态缓存 (彻底移除 self.classes)
        self._cached_texts: Optional[List[str]] = None
        self.candidate_prototypes: Optional[torch.Tensor] = None

        # 若配置中带有默认候选词，则预热编码一次
        if "candidate_texts" in self.cfg:
            self.update_candidates(self.cfg["candidate_texts"])

    def preprocess(self, imgs: List[np.ndarray]) -> torch.Tensor:
        if not isinstance(imgs, list) or (imgs and not isinstance(imgs[0], np.ndarray)):
            raise TypeError(f"输入数据必须为 List[np.ndarray]，检测到非法类型: {type(imgs[0]) if imgs else type(imgs)}")
        return super().preprocess(imgs)

    @torch.no_grad()
    @trt_stream_guard
    def update_candidates(self, candidate_texts: List[str]) -> None:
        """比对候选文本集合，仅在词表变更时编译文本塔特征"""
        if not candidate_texts:
            raise ValueError("候选文本列表 candidate_texts 不能为空")

        # 检查候选词内容与顺序是否与缓存一致，一致则跳过文本编码
        if self._cached_texts == candidate_texts and self.candidate_prototypes is not None:
            return

        tok = self.tokenizer(
            candidate_texts,
            padding="max_length",
            max_length=self.max_text_len,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tok["input_ids"].to(self.device)
        attention_mask = tok["attention_mask"].to(self.device)

        # 文本塔推理并归一化
        text_embs = self.text_engine.encode_tokens(input_ids, attention_mask)
        self.candidate_prototypes = F.normalize(text_embs.float(), dim=-1)
        self._cached_texts = list(candidate_texts)
        self.logger.info(f"检测到候选词变更，已实时更新文本特征矩阵，当前候选数: {len(candidate_texts)}")

    @torch.no_grad()
    @trt_stream_guard
    def classify(
        self, 
        imgs: List[np.ndarray], 
        candidate_texts: Optional[List[str]] = None
    ) -> Tuple[List[int], List[float]]:
        """
        统一推理入口：
        入参严格保持 List[np.ndarray]，可选动态传入当前帧的 candidate_texts。
        返回的索引列表对应当前 candidate_texts 的下标。
        """
        if candidate_texts is not None:
            self.update_candidates(candidate_texts)

        if self.candidate_prototypes is None:
            raise RuntimeError("尚未设置候选文本，请在调用 classify 时传入 candidate_texts 或调用 update_candidates()")

        return super().classify(imgs)

    def postprocess(self, output_tensor: torch.Tensor) -> Tuple[List[int], List[float]]:
        # output_tensor: vision.engine 输出的 image_embeddings [B, Dim]
        img_feats = F.normalize(output_tensor.float(), dim=-1)
        
        # 矩阵乘计算标准余弦相似度 [B, Num_Candidates]
        similarity = img_feats @ self.candidate_prototypes.t()

        confs, preds = torch.max(similarity, dim=-1)
        pred_ids = preds.cpu().tolist()
        conf_vals = confs.cpu().tolist()

        # 低于阈值的判定为 -1（不属于给定候选集中的任何一类）
        filtered_ids = [
            pid if conf >= self.conf_thresh else -1
            for pid, conf in zip(pred_ids, conf_vals)
        ]
        return filtered_ids, conf_vals
