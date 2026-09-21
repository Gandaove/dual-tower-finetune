"""双塔零样本图像分类与向量抽取推理引擎。"""
from __future__ import annotations

import argparse
import json
import time
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Iterable, Iterator

import numpy as np
import pandas as pd
import safetensors
import torch
import torch.nn.functional as F
from PIL import Image

from dataset.augment import DualTowerTransforms
from models.dual_tower_multimodal import build_dual_tower_model
from utils.config import Config, DataConfig, ModelConfig, TrainConfig
from utils.logger import get_logger, progress

_log = get_logger("infer")

try:
    import orjson
    def _serialize_record(record: Dict[str, Any]) -> bytes:
        return orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE)
except ImportError:
    import json
    def _serialize_record(record: Dict[str, Any]) -> bytes:
        return (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")


class FastJSONLWriter:
    """带系统调用缓冲区的流式 JSONL 二进制写入器。"""

    def __init__(self, file_path: Path, buffer_size: int = 128 * 1024):
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = io.open(self.file_path, "wb", buffering=buffer_size)

    def write_batch(self, records: Iterable[Dict[str, Any]]):
        self._fp.writelines(_serialize_record(r) for r in records)

    def close(self):
        if not self._fp.closed:
            self._fp.flush()
            self._fp.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ZeroShotImageClassifier:
    """闭集零样本图像分类推理与表征提取器。"""

    def __init__(
        self,
        ckpt_path: Union[str, Path],
        device: str = "auto",
        dtype: Optional[str] = None,
    ):
        self.ckpt_path = Path(ckpt_path).resolve()
        if not self.ckpt_path.is_file():
            raise FileNotFoundError(f"权重文件不存在: {self.ckpt_path}")

        self.ckpt_dir = self.ckpt_path.parent
        self.model_cfg, self.data_cfg = self._load_configs()
        self._check_lora_checkpoint()
        self.device = self._resolve_device(device)

        cfg = Config(model=self.model_cfg, data=self.data_cfg, train=TrainConfig())
        self.transforms = DualTowerTransforms(self.data_cfg)
        self.model = build_dual_tower_model(cfg, transform=self.transforms)

        _log.info(f"[Init] 加载权重: {self.ckpt_path.name}")
        self.model.load_safetensors(str(self.ckpt_path), load_opt=False)
        self.model.to(device=self.device)

        if dtype is not None:
            target_dtype = self._resolve_dtype(dtype)
            self.model.to(dtype=target_dtype)
            _log.info(f"[Init] 显式转换模型精度至: {dtype}")

        self.model.eval()

        self.cand_classes: List[str] = []
        self.cand_texts: List[str] = []
        self.txt_prototypes: Optional[torch.Tensor] = None

    def _load_configs(self) -> Tuple[ModelConfig, DataConfig]:
        model_json = self.ckpt_dir / "set_model_config.json"
        data_json = self.ckpt_dir / "set_data_config.json"

        if not model_json.is_file():
            raise FileNotFoundError(f"未在权重同级目录下找到模型配置: {model_json}")
        if not data_json.is_file():
            raise FileNotFoundError(f"未在权重同级目录下找到数据配置: {data_json}")

        with open(model_json, "r", encoding="utf-8") as f:
            m_cfg = ModelConfig.from_dict(json.load(f))
        with open(data_json, "r", encoding="utf-8") as f:
            d_cfg = DataConfig.from_dict(json.load(f))
        return m_cfg, d_cfg

    def _check_lora_checkpoint(self):
        """扫描检查点元数据与键名，检测未合并的 LoRA 结构。"""
        with safetensors.safe_open(str(self.ckpt_path), framework="pt") as f:
            meta = f.metadata() or {}
            keys = f.keys()
            is_lora = (
                # meta.get("use_lora") == "True"
                any("lora_" in k for k in keys)
                or self.ckpt_path.name.endswith("-lora.safetensors")
            )
            if is_lora:
                _log.warning(
                    f"[Check] 警告: 权重 '{self.ckpt_path.name}' 仍为未合并的 LoRA 增量形态！"
                    "推理正在携带动态 LoRA 结构运行。如需最高吞吐部署，请先调用 merger.py 进行融合导出。"
                )

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "-1" or not torch.cuda.is_available():
            return torch.device("cpu")
        if device_str == "auto":
            return torch.device("cuda:0")
        target = device_str.split(",")[0].strip()
        return torch.device(f"cuda:{target}")

    @staticmethod
    def _resolve_dtype(dtype_str: str) -> torch.dtype:
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        if dtype_str not in dtype_map:
            raise ValueError(f"不支持的 dtype: {dtype_str}，可选: {list(dtype_map.keys())}")
        return dtype_map[dtype_str]

    def setup_candidates(self, text_txt: Optional[Union[str, Path]] = None):
        """解析候选分类体系并预先抽取归一化文本原型矩阵 [Num_Candidates, Dim]。"""
        if text_txt is not None:
            txt_path = Path(text_txt).resolve()
            if not txt_path.is_file():
                raise FileNotFoundError(f"未找到指定的候选文本文件: {txt_path}")
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            if not lines:
                raise ValueError(f"候选文本文件内容为空: {txt_path}")
            self.cand_classes = lines
            self.cand_texts = lines
            _log.info(f"[Setup] 从文件加载候选文本: {txt_path.name} (共 {len(lines)} 条)")
        else:
            # 默认回退至 class_map_csv
            csv_path = Path(self.data_cfg.class_map_csv)
            if not csv_path.is_absolute():
                csv_path = Path(self.data_cfg.root) / csv_path

            if not csv_path.is_file():
                raise FileNotFoundError(f"未找到物种映射表 class_map_csv: {csv_path}")

            df = pd.read_csv(csv_path)
            required_cols = {"pest_cname", "pest_latin_name"}
            if not required_cols.issubset(df.columns):
                raise ValueError(f"class_map_csv 缺失必要列: {required_cols}")

            cnames, latins = [], []
            for idx, row in df.iterrows():
                cname = str(row["pest_cname"]).strip()
                latin = str(row["pest_latin_name"]).strip()
                if not cname or not latin or cname.lower() == "nan" or latin.lower() == "nan":
                    raise ValueError(f"class_map_csv 第 {idx + 2} 行存在空值或非法 'nan': cname='{cname}', latin='{latin}'")
                cnames.append(cname)
                latins.append(latin)

            tmpl = self.model_cfg.eval.prompt_template or "{}"
            self.cand_classes = cnames
            self.cand_texts = [tmpl.format(latin) for latin in latins]
            _log.info(f"[Setup] 默认加载 class_map_csv 闭集类别: 共 {len(self.cand_classes)} 类")

        # 批量抽取文本特征原型
        feats = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(self.cand_texts), batch_size):
                chunk = self.cand_texts[i : i + batch_size]
                tok = self.model.tokenize_text(chunk)
                tok = {k: v.to(self.device) for k, v in tok.items()}
                emb = self.model.get_embeddings(texts=tok, to_float32=False)
                feats.append(F.normalize(emb, dim=-1))
            self.txt_prototypes = torch.cat(feats, dim=0)

    def predict_batches(
        self,
        image_paths: List[Path],
        batch_size: int = 32,
        topk: int = 5,
        embed_img: bool = False,
    ) -> Iterator[Tuple[List[dict], float]]:
        """以生成器形式按批次推理，每次产生 (当前批次结果列表, 当前批次纯推理耗时)。"""
        topk = min(topk, len(self.cand_classes)) if self.cand_classes else 0
        model_param_dtype = next(self.model.parameters()).dtype

        for i in range(0, len(image_paths), batch_size):
            chunk_paths = image_paths[i : i + batch_size]
            tensor_list: List[torch.Tensor] = []
            valid_paths: List[Path] = []

            for p in chunk_paths:
                try:
                    with Image.open(p) as img:
                        rgb_img = img.convert("RGB")
                        np_arr = np.asarray(rgb_img)
                        t = self.transforms(np_arr, train=False)
                        tensor_list.append(t)
                        valid_paths.append(p)
                except Exception as e:
                    _log.error(f"图像读取失败，跳过: {p} ({e})")

            if not tensor_list:
                continue

            pixel_batch = torch.stack(tensor_list, dim=0).to(device=self.device, dtype=model_param_dtype)

            # ---------------- 推理计时 ---------------- #
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.no_grad():
                img_embs = self.model.get_embeddings(
                    images={"pixel_values": pixel_batch},
                    to_float32=False,
                )
                img_embs = F.normalize(img_embs, dim=-1)

                sim_matrix = None
                if self.txt_prototypes is not None:
                    sim_matrix = img_embs @ self.txt_prototypes.t()

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            batch_infer_time = time.perf_counter() - t0
            # -------------------------------------------- #

            sim_matrix_np = sim_matrix.cpu().float().numpy() if sim_matrix is not None else None
            img_embs_np = img_embs.cpu().float().numpy() if embed_img else None

            batch_results = []
            for b_idx, p in enumerate(valid_paths):
                rec = {
                    "image_name": p.name,
                    "image_path": str(p),
                }

                if embed_img and img_embs_np is not None:
                    rec["embedding"] = img_embs_np[b_idx].tolist()

                if sim_matrix_np is not None and topk > 0:
                    sims = sim_matrix_np[b_idx]
                    order = np.argsort(-sims)[:topk]
                    ranking = [
                        {
                            "class": self.cand_classes[idx],
                            "text": self.cand_texts[idx],
                            "conf": float(sims[idx]),
                        }
                        for idx in order
                    ]
                    rec["ranking"] = ranking
                    rec["top1_class"] = self.cand_classes[order[0]]
                    rec["top1_conf"] = float(sims[order[0]])

                batch_results.append(rec)

            yield batch_results, batch_infer_time

    def export_excel(
        self,
        sheet_records: List[dict],
        output_path: Path,
        topk: int,
    ):
        """将轻量级分类统计数据与候选集导出为双 Sheet Excel 报表。"""
        pred_rows = []
        for r in sheet_records:
            row = {"image_name": r["image_name"]}
            ranking = r.get("ranking", [])
            for k in range(topk):
                cls_val = ranking[k]["class"] if k < len(ranking) else ""
                conf_val = ranking[k]["conf"] if k < len(ranking) else None
                row[f"top{k+1}_class"] = cls_val
                row[f"top{k+1}_conf"] = conf_val
            pred_rows.append(row)

        df_preds = pd.DataFrame(pred_rows)
        df_cands = pd.DataFrame({
            "index": list(range(len(self.cand_classes))),
            "class_name": self.cand_classes,
            "candidate_text": self.cand_texts,
        })

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_preds.to_excel(writer, sheet_name="predictions", index=False)
            df_cands.to_excel(writer, sheet_name="candidate_texts", index=False)
        _log.info(f"[Export] Excel 报表已保存至: {output_path}")


def collect_images(input_path: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted([p for p in input_path.rglob("*") if p.suffix.lower() in exts])
    raise FileNotFoundError(f"指定的图像路径不存在: {input_path}")


def resolve_output_paths(output_arg: str) -> Tuple[Path, Path]:
    """解析输出路径，返回 (jsonl_path, excel_path)。"""
    out_path = Path(output_arg)
    if out_path.suffix.lower() in (".json", ".jsonl"):
        out_dir = out_path.parent
        jsonl_path = out_path.with_suffix(".jsonl")
    else:
        out_dir = out_path
        jsonl_path = out_dir / "result.jsonl"
    excel_path = out_dir / "result_sheet.xlsx"
    return jsonl_path, excel_path


def build_zeroshot_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Dual-Tower Zero-Shot Standalone Inference")
    p.add_argument("--ckpt_path", required=True, help="safetensors 权重文件路径")
    p.add_argument("--images", required=True, help="单张图像文件或包含图像的文件夹路径")
    p.add_argument(
        "--text_txt",
        default=None,
        help="候选文本文件路径 (.txt，每行一条); 不传则默认使用 data 配置中的 class_map_csv 闭集",
    )
    p.add_argument("--output", default="./outputs/infer", help="结果输出目录或指定的 .json/.jsonl 文件路径")
    p.add_argument("--save_sheet", action="store_true", help="是否在输出目录下导出 result_sheet.xlsx")
    p.add_argument("--batch_size", type=int, default=32, help="批处理大小 (单张输入时等价于 1)")
    p.add_argument("--device", default="auto", help="推理设备: 'auto', '0', '1', '-1'(cpu)")
    p.add_argument("--topk", type=int, default=5, help="Top-K 截断数")
    p.add_argument("--dtype", default=None, choices=["fp16", "bf16", "fp32"], help="模型推断精度，未设置时不进行精度转换")
    p.add_argument("--embed_img", action="store_true", help="是否在 JSONL 中附加图像 embedding 向量")
    return p


def zeroshot_infer(argv: Optional[List[str]] = None):
    args = build_zeroshot_parser().parse_args(argv)

    image_paths = collect_images(Path(args.images))
    if not image_paths:
        _log.warning(f"未检索到有效图像文件: {args.images}")
        return

    classifier = ZeroShotImageClassifier(
        ckpt_path=args.ckpt_path,
        device=args.device,
        dtype=args.dtype,
    )
    classifier.setup_candidates(args.text_txt)

    jsonl_path, excel_path = resolve_output_paths(args.output)
    total_pure_infer_time = 0.0
    total_samples = 0
    sheet_records = []

    batch_gen = classifier.predict_batches(
        image_paths=image_paths,
        batch_size=args.batch_size,
        topk=args.topk,
        embed_img=args.embed_img,
    )

    num_batches = (len(image_paths) + args.batch_size - 1) // args.batch_size
    with FastJSONLWriter(jsonl_path) as writer:
        for batch_records, batch_time in progress(batch_gen, total=num_batches, desc="[Infer] Batches"):
            total_pure_infer_time += batch_time
            total_samples += len(batch_records)

            writer.write_batch(batch_records)

            if args.save_sheet:
                for r in batch_records:
                    sheet_records.append({
                        "image_name": r["image_name"],
                        "ranking": r.get("ranking", []),
                    })
    _log.info(f"[Export] 结果已流式保存至: {jsonl_path}")

    if args.save_sheet and classifier.cand_classes:
        classifier.export_excel(
            sheet_records=sheet_records,
            output_path=excel_path,
            topk=min(args.topk, len(classifier.cand_classes)),
        )

    avg_ms = (total_pure_infer_time / max(1, total_samples)) * 1000.0
    fps = total_samples / max(1e-6, total_pure_infer_time)
    _log.info(
        f"[Benchmark] 样本总量: {total_samples} | "
        f"网络推理总耗时: {total_pure_infer_time:.4f}s | "
        f"单图平均耗时: {avg_ms:.2f}ms | "
        f"吞吐速率: {fps:.2f} FPS"
    )


if __name__ == "__main__":
    zeroshot_infer()
