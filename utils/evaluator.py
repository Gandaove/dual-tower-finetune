"""评估器 Evaluator。

在每轮训练后被 EpochEvalCallback 调用, 也可由 valid.py 直接调用。
支持:
  - 图文检索: Image→Text / Text→Image 的 Recall@K;
  - 零样本闭集分类: 以 prompt_template 生成各类文本原型, 取相似度最大者, 计算 F1 / Acc;
  - 混淆矩阵导出(PNG)
"""
from __future__ import annotations

import json
import matplotlib
import contextlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from utils.config import ModelConfig, TaxonomyClasses
from utils.logger import get_logger, progress

_log = get_logger("eval")


class Evaluator:
    def __init__(
        self,
        model,
        model_cfg: ModelConfig,
        taxonomy: TaxonomyClasses,
        device: torch.device, 
        max_retrieval_samples: int = 5000,

    ):
        self.model = model
        self.model_cfg = model_cfg
        self.eval_cfg = model_cfg.eval
        self.taxonomy = taxonomy
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
        if not self.taxonomy.latins:
            return None

        tmpl = self.eval_cfg.prompt_template
        prompts = [tmpl.format(latin) for latin in self.taxonomy.latins]
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
        if self.eval_cfg.zeroshot and self.taxonomy.cnames and proto is None:
            proto = self._class_prototypes(self.device)

        alias_to_idx = self.taxonomy.alias_to_idx
        y_true_list: List[int] = []
        y_pred_list: List[int] = []

        retrieval_imgs = []
        retrieval_txts = []
        collected_retrieval = 0
        total_val_loss = 0.0
        val_batches = 0

        with self.autocast_ctx:
            for batch in progress(dataloader, desc=desc, leave=True):
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
                    y_true_list.extend([alias_to_idx.get(lbl, -1) for lbl in batch["label"]])       # 兼容负样本 -1

                if collected_retrieval < self.max_retrieval_samples:
                    remain = self.max_retrieval_samples - collected_retrieval
                    retrieval_imgs.append(ie_norm[:remain].cpu())
                    retrieval_txts.append(te_norm[:remain].cpu())
                    collected_retrieval += min(len(images), remain)

        self.y_true = y_true_list
        self.y_pred = y_pred_list
        metrics: Dict[str, float] = {}
        metrics["loss"] = round(total_val_loss / max(1, val_batches), 4)

        # 计算零样本分类宏平均与混淆矩阵 (纯标量计算，内存消耗忽略不计)
        if y_true_list:
            y_true = np.array(y_true_list, dtype=np.int32)
            y_pred = np.array(y_pred_list, dtype=np.int32)

            pos_labels = self.taxonomy.pos_indices
            pos_cnames = self.taxonomy.pos_cnames

            metrics["zeroshot_f1"] = float(f1_score(
                y_true, y_pred, labels=pos_labels, average="macro", zero_division=0
            ))

            pos_mask = np.isin(y_true, pos_labels)
            if pos_mask.any():
                metrics["zeroshot_acc"] = float(accuracy_score(y_true[pos_mask], y_pred[pos_mask]))

            cm = confusion_matrix(y_true, y_pred, labels=pos_labels)
            true_pos_counts = np.array([(y_true == idx).sum() for idx in pos_labels], dtype=np.int64)
            self.last_confusion = (cm, pos_cnames, true_pos_counts)

        # 分块计算跨模态检索指标
        if retrieval_imgs:
            r_img = torch.cat(retrieval_imgs, dim=0).to(self.device)
            r_txt = torch.cat(retrieval_txts, dim=0).to(self.device)
            metrics.update(self._chunked_recall(r_img, r_txt, prefix="i2t"))
            metrics.update(self._chunked_recall(r_txt, r_img, prefix="t2i"))

        return metrics, proto

    @staticmethod
    def save_confusion_png(
        cm_tuple: Tuple[np.ndarray, List[str]], 
        path: str, 
        epoch: Optional[int] = None,
        best_f1: Optional[float] = None
    ):
        cm, classes, true_counts = cm_tuple
        n_classes = len(classes)

        cell_size = max(0.4, 12.0 / max(n_classes, 1))
        fig_dim = max(7.0, n_classes * cell_size)
        fig, ax = plt.subplots(figsize=(fig_dim, fig_dim), dpi=150)

        # 使用该正类真实总样本数做行归一化，正类被分流至负类时，对角线百分比直接体现扣分
        denom = np.where(true_counts[:, None] == 0, 1, true_counts[:, None])
        with np.errstate(all="ignore"):
            cm_norm = np.nan_to_num(cm.astype(float) / denom)

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks(np.linspace(0, 1, 6))
        cbar.set_ticklabels([f"{int(x * 100)}%" for x in np.linspace(0, 1, 6)])

        ax.set_xticks(np.arange(n_classes + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_classes + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="#D0D0D0", linestyle="-", linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)

        label_fontsize = max(5, min(9, int(150 / n_classes)))
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels(classes, rotation=45, ha="right", rotation_mode="anchor", fontsize=label_fontsize)
        ax.set_yticklabels(classes, fontsize=label_fontsize)
        ax.set_xlabel("Predicted Label (Positives Only)", fontsize=label_fontsize + 2, fontweight="bold", labelpad=8)
        ax.set_ylabel("True Label (Positives Only)", fontsize=label_fontsize + 2, fontweight="bold", labelpad=8)

        text_fontsize = max(4, min(8, int(100 / n_classes)))
        for i in range(n_classes):
            for j in range(n_classes):
                ratio = cm_norm[i, j]
                text_color = "white" if ratio > 0.5 else "black"
                if ratio > 0.0001 or i == j:
                    ax.text(j, i, f"{ratio * 100:.1f}%", ha="center", va="center", color=text_color, fontsize=text_fontsize)

        title_str = f"Confusion Matrix (Epoch {epoch})" if epoch is not None else "Confusion Matrix"
        ax.set_title(title_str, fontsize=label_fontsize + 4, fontweight="bold", pad=12)

        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        
        if epoch is not None and best_f1 is not None:
            _log.info(f"[Eval] 混淆矩阵已更新保存 | Epoch: {epoch} | Best-F1: {best_f1:.4f}")
        elif best_f1 is not None:
            _log.info(f"[Eval] 混淆矩阵已更新保存 | Best-F1: {best_f1:.4f}")
        elif epoch is not None:
            _log.info(f"[Eval] 混淆矩阵已更新保存 | Epoch: {epoch}")
        else:
            _log.info(f"[Eval] 混淆矩阵已更新保存")

    @torch.no_grad()
    def export_full_confusion_artifacts(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        proto: torch.Tensor,
        out_json_path: Path,
        out_png_path: Path,
        alpha: float = 0.7,
        min_thresh: float = 0.01,
    ) -> np.ndarray:
        """计算全量混淆强度矩阵，导出基于 alias 的结构化 JSON 并绘制全局热力图。"""
        aliases = self.taxonomy.cnames
        n = len(aliases)

        # 经验视觉错判率计算
        cm_raw = confusion_matrix(y_true, y_pred, labels=list(range(n)))
        row_sums = np.bincount(y_true[y_true >= 0], minlength=n)[:n]
        denom = np.where(row_sums[:, None] == 0, 1, row_sums[:, None])
        p_vis = cm_raw.astype(np.float32) / denom

        # 原型余弦相似度计算
        proto_norm = F.normalize(proto.float(), dim=-1)
        sim_matrix = (proto_norm @ proto_norm.t()).cpu().numpy()
        s_proto = np.clip(sim_matrix, 0.0, 1.0).astype(np.float32)

        # 加权融合并剔除混淆
        mat = alpha * p_vis + (1.0 - alpha) * s_proto
        np.fill_diagonal(mat, 0.0)

        # 导出基于 alias 的结构化 JSON (过滤低于 min_thresh 的噪点)
        confusion_dict: Dict[str, Dict[str, float]] = {}
        for i, src_alias in enumerate(aliases):
            row = mat[i]
            valid_targets = {}
            for j, dst_alias in enumerate(aliases):
                val = float(row[j])
                if val >= min_thresh and i != j:
                    valid_targets[dst_alias] = round(val, 4)
            if valid_targets:
                # 按混淆强度从大到小排序保存
                sorted_targets = dict(sorted(valid_targets.items(), key=lambda x: -x[1]))
                confusion_dict[src_alias] = sorted_targets

        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(confusion_dict, f, ensure_ascii=False, indent=2)
        _log.info(f"[Eval] 结构化混淆强度字典已保存 -> {out_json_path.name}")

        # 渲染全量混淆强度热力图 (支持大类别数缩放与正负类分割)
        self._plot_full_matrix(mat, aliases, out_png_path)
        return mat

    def _plot_full_matrix(self, mat: np.ndarray, aliases: List[str], save_path: Path):
        n = len(aliases)
        # 自适应画布尺寸：保证高密度类别展示下不拥挤
        dim = max(10.0, min(36.0, n * 0.12))
        fig, ax = plt.subplots(figsize=(dim, dim), dpi=180)

        im = ax.imshow(mat, interpolation="nearest", cmap="plasma", vmin=0.0, vmax=1.0)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Confusion Intensity", rotation=270, labelpad=15, fontsize=11)

        # 轴刻度控制：类别数大于 80 时稀疏标注，避免字体完全重叠
        if n <= 80:
            ax.set_xticks(np.arange(n))
            ax.set_yticks(np.arange(n))
            font_sz = max(4, min(8, int(200 / n)))
            ax.set_xticklabels(aliases, rotation=90, ha="center", fontsize=font_sz)
            ax.set_yticklabels(aliases, fontsize=font_sz)
        else:
            tick_step = max(1, n // 50)
            ticks = np.arange(0, n, tick_step)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels([aliases[i] for i in ticks], rotation=90, fontsize=7)
            ax.set_yticklabels([aliases[i] for i in ticks], fontsize=7)

        ax.set_xlabel("Predicted Competitor Class", fontsize=12, fontweight="bold", labelpad=10)
        ax.set_ylabel("True Target Class", fontsize=12, fontweight="bold", labelpad=10)
        ax.set_title(f"Full Global Confusion Intensity Matrix (Total Classes: {n})", fontsize=14, pad=12)

        fig.tight_layout()
        fig.savefig(str(save_path), bbox_inches="tight")
        plt.close(fig)
        _log.info(f"[Eval] 全量混淆强度热力图已保存 -> {save_path.name}")
