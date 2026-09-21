"""评估器 Evaluator。

在每轮训练后被 EpochEvalCallback 调用, 也可由 valid.py 直接调用。
支持:
  - 图文检索: Image→Text / Text→Image 的 Recall@K;
  - 零样本闭集分类: 以 prompt_template 生成各类文本原型, 取相似度最大者, 计算 F1 / Acc;
  - 混淆矩阵导出(PNG)
"""
from __future__ import annotations

import matplotlib
import contextlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from utils.config import ModelConfig
from utils.logger import get_logger, progress

_log = get_logger("eval")


class Evaluator:
    def __init__(
        self,
        model,
        model_cfg: ModelConfig,
        cname_classes: List[str],
        latin_classes: List[str],
        device: torch.device, 
        max_retrieval_samples: int = 5000,

    ):
        self.model = model
        self.model_cfg = model_cfg
        self.eval_cfg = model_cfg.eval
        self.cname_classes = list(cname_classes)
        self.latin_classes = list(latin_classes)
        self.last_confusion: Optional[Tuple[np.ndarray, List[str]]] = None
        self.device = device
        self.max_retrieval_samples = max_retrieval_samples
        self.autocast_dtype = self._resolve_amp_dtype(model_cfg.amp)
        amp_enabled = self.autocast_dtype is not None and device.type == "cuda"
        # 挂载 autocast
        self.autocast_ctx = torch.amp.autocast("cuda", dtype=self.autocast_dtype) if amp_enabled else contextlib.nullcontext()

    @staticmethod
    def _resolve_amp_dtype(amp_str: Union[str, bool]) -> Optional[torch.dtype]:
        if not amp_str or str(amp_str).lower() in ("false", "none", "fp32"):
            return None
        s = str(amp_str).lower()
        if "bf16" in s:
            return torch.bfloat16
        if "16" in s:
            return torch.float16
        return None

    @torch.no_grad()
    def _class_prototypes(self, device: torch.device, batch_size: int = 64) -> Optional[torch.Tensor]:
        """构建归一化后的零样本类别原型 [C, D]"""
        if not self.latin_classes:
            return None

        tmpl = self.eval_cfg.prompt_template
        prompts = [tmpl.format(latin) for latin in self.latin_classes]
        feats = []

        with self.autocast_ctx:
            for i in range(0, len(prompts), batch_size):
                chunk = prompts[i : i + batch_size]
                txt_tok = self.model.tokenize_text(chunk)
                txt_tok = {k: v.to(device) for k, v in txt_tok.items()}
                emb = self.model.get_embeddings(texts=txt_tok).float()
                feats.append(F.normalize(emb, dim=-1))

        return torch.cat(feats, dim=0)  # [C, D]

    @torch.no_grad()
    def _chunked_recall(
        self, 
        queries: torch.Tensor, 
        targets: torch.Tensor, 
        prefix: str, 
        chunk_size: int = 256
    ) -> Dict[str, float]:
        """分块计算 Top-K 召回率，显存与内存开销恒定，彻底弃用全局 argsort"""
        n = queries.size(0)
        ks = [k for k in self.eval_cfg.retrieval_topk if k <= n] or [1]
        max_k = max(ks)
        device = queries.device

        correct_counts = {k: 0 for k in ks}
        targets_t = targets.t()  # [D, N]

        for i in range(0, n, chunk_size):
            q_chunk = queries[i : i + chunk_size]  # [B, D]
            sim_chunk = q_chunk @ targets_t        # [B, N]
            
            # 使用 topk 替代 argsort，空间复杂度从 O(N^2) 骤降至 O(B * K)
            _, topk_indices = sim_chunk.topk(max_k, dim=1, largest=True, sorted=True)
            
            # 当前分块各行的真实正样本索引即为全局行号
            gt_indices = torch.arange(i, min(i + chunk_size, n), device=device).unsqueeze(1)
            hits = (topk_indices == gt_indices)  # [B, max_k]

            for k in ks:
                correct_counts[k] += hits[:, :k].any(dim=1).sum().item()

        return {f"{prefix}_R@{k}": round(correct_counts[k] / n, 4) for k in ks}

    @torch.no_grad()
    def evaluate(
        self, 
        dataloader, 
        desc: str = "Evaluating",
        cached_proto: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, float], Optional[torch.Tensor]]:
        eval_mod = self.model
        eval_mod.eval()

        proto = cached_proto
        if self.eval_cfg.zeroshot and self.cname_classes and proto is None:
            proto = self._class_prototypes(self.device)

        cname_to_idx = {name: i for i, name in enumerate(self.cname_classes)}
        y_true_list: List[int] = []
        y_pred_list: List[int] = []

        retrieval_imgs = []
        retrieval_txts = []
        collected_retrieval = 0
        total_val_loss = 0.0
        val_batches = 0

        with self.autocast_ctx:
            for batch in progress(dataloader, desc=desc, leave=False):
                images = batch["image"].to(self.device)
                txt_tok = eval_mod.tokenize_text(batch["text"])
                txt_tok = {k: v.to(self.device) for k, v in txt_tok.items()}

                ie, te = eval_mod.get_embeddings(images={"pixel_values": images}, texts=txt_tok)
                ie_norm = F.normalize(ie.float(), dim=-1)
                te_norm = F.normalize(te.float(), dim=-1)

                batch_loss = eval_mod.loss_fn(ie, te, labels=batch["label"])
                total_val_loss += batch_loss.detach().float().item()
                val_batches += 1

                if proto is not None:
                    logits = ie_norm @ proto.t()  # [B, C]
                    preds = logits.argmax(dim=-1).cpu().numpy().tolist()
                    y_pred_list.extend(preds)
                    y_true_list.extend([cname_to_idx.get(lbl) for lbl in batch["label"]])

                if collected_retrieval < self.max_retrieval_samples:
                    remain = self.max_retrieval_samples - collected_retrieval
                    retrieval_imgs.append(ie_norm[:remain].cpu())
                    retrieval_txts.append(te_norm[:remain].cpu())
                    collected_retrieval += min(len(images), remain)

        metrics: Dict[str, float] = {}
        metrics["loss"] = round(total_val_loss / max(1, val_batches), 4)

        # 计算零样本分类宏平均与混淆矩阵 (纯标量计算，内存消耗忽略不计)
        if y_true_list:
            y_true = np.array(y_true_list, dtype=np.int32)
            y_pred = np.array(y_pred_list, dtype=np.int32)
            valid_mask = y_true >= 0

            if valid_mask.any():
                yt_val = y_true[valid_mask]
                yp_val = y_pred[valid_mask]
                metrics["zeroshot_f1"] = float(f1_score(yt_val, yp_val, average="macro", zero_division=0))
                metrics["zeroshot_acc"] = float(accuracy_score(yt_val, yp_val))
                cm = confusion_matrix(yt_val, yp_val, labels=list(range(len(self.cname_classes))))
                self.last_confusion = (cm, self.cname_classes)

        # 安全分块计算跨模态检索指标
        if retrieval_imgs:
            r_img = torch.cat(retrieval_imgs, dim=0).to(self.device)
            r_txt = torch.cat(retrieval_txts, dim=0).to(self.device)
            metrics.update(self._chunked_recall(r_img, r_txt, prefix="i2t"))
            metrics.update(self._chunked_recall(r_txt, r_img, prefix="t2i"))

        return metrics, proto

    @staticmethod
    def save_confusion_png(cm_tuple: Tuple[np.ndarray, List[str]], path: str, epoch: Optional[int] = None):
        cm, classes = cm_tuple
        n_classes = len(classes)
        
        # 1. 规整画布尺寸与自适应字号
        cell_size = max(0.4, 12.0 / max(n_classes, 1))
        fig_dim = max(7.0, n_classes * cell_size)
        fig, ax = plt.subplots(figsize=(fig_dim, fig_dim), dpi=150)

        # 2. 计算按行归一化的召回率百分比 (避免除以 0 产生 NaN)
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(all="ignore"):
            cm_norm = np.nan_to_num(cm.astype(float) / np.where(row_sums == 0, 1, row_sums))

        # 3. 渲染热力图与颜色条
        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks(np.linspace(0, 1, 6))
        cbar.set_ticklabels([f"{int(x * 100)}%" for x in np.linspace(0, 1, 6)])

        # 4. 显式网格线 (对齐单元格边界)
        ax.set_xticks(np.arange(n_classes + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_classes + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="#D0D0D0", linestyle="-", linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)

        # 5. 坐标轴与文字 45 度旋转对齐
        label_fontsize = max(5, min(9, int(150 / n_classes)))
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(classes, rotation=45, ha="right", rotation_mode="anchor", fontsize=label_fontsize)
        ax.set_yticklabels(classes, fontsize=label_fontsize)
        ax.set_xlabel("Predicted Label", fontsize=label_fontsize + 2, fontweight="bold", labelpad=8)
        ax.set_ylabel("True Label", fontsize=label_fontsize + 2, fontweight="bold", labelpad=8)

        # 6. 格内填充百分比数值 (高亮度底色自动反白)
        text_fontsize = max(4, min(8, int(100 / n_classes)))
        thresh = 0.5
        for i in range(n_classes):
            for j in range(n_classes):
                ratio = cm_norm[i, j]
                text_color = "white" if ratio > thresh else "black"
                # 仅标注非零预测，避免视觉混乱；对角线强制显示
                if ratio > 0.0001 or i == j:
                    ax.text(
                        j, i, f"{ratio * 100:.1f}%",
                        ha="center", va="center",
                        color=text_color, fontsize=text_fontsize,
                    )

        # 7. 标题设定
        title_str = f"Confusion Matrix (Epoch {epoch})" if epoch is not None else "Confusion Matrix"
        ax.set_title(title_str, fontsize=label_fontsize + 4, fontweight="bold", pad=12)

        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        _log.info(f"[Eval] 混淆矩阵热力图已保存 -> {path}")
