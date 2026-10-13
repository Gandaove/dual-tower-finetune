from __future__ import annotations

import io
import json
import os
import random
from pathlib import Path
from collections import defaultdict, Counter
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterator

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler

from dataset.augment import DualTowerTransforms
from utils.config import DataConfig, TrainConfig, TaxonomyClasses
from utils.logger import get_logger

_log = get_logger("data")


def load_manifest(path: Union[str, Path]) -> Dict[str, List[str]]:
    """读取包含多条 Caption 的 Manifest JSON 文件。
    
    格式: {"{alias}+{image_name}": ["caption1", "caption2"]}
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Manifest 未找到: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = {}
    for k, v in data.items():
        if isinstance(v, list):
            records[k] = v
        elif isinstance(v, dict):
            records[k] = v.get("captions", [])
        else:
            records[k] = [str(v)]
    return records


class LazyLMDBReader:
    """跨进程安全的单文件 LMDB 懒加载读取器 (subdir=False)"""

    def __init__(self, lmdb_path: Optional[str]):
        self.lmdb_path = lmdb_path
        self._env = None
        self._txn = None
        self._pid: Optional[int] = None
        self._is_valid = bool(lmdb_path and Path(lmdb_path).is_file())

    def _init_db(self):
        if not self._is_valid:
            return
        cur_pid = os.getpid()
        if self._env is not None and self._pid == cur_pid:
            return

        self._pid = cur_pid
        try:
            import lmdb
            self._env = lmdb.open(
                self.lmdb_path,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
            self._txn = self._env.begin(write=False)
        except Exception as e:
            _log.warning(f"[LMDB] 打开失败, 退化为普通磁盘寻址: {e}")
            self._is_valid = False
            self._env = None
            self._txn = None

    def read(self, key: str) -> Optional[bytes]:
        if not self._is_valid:
            return None
        self._init_db()
        if self._txn is None:
            return None
        return self._txn.get(key.encode("utf-8"))

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
            self._txn = None

    def __del__(self):
        self.close()


class ImageIOHandler:
    @staticmethod
    def decode(raw_bytes: bytes) -> np.ndarray:
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if cv_img is not None:
            return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        with Image.open(io.BytesIO(raw_bytes)) as pil_img:
            return np.array(pil_img.convert("RGB"))


class HardNegativeBatchSampler(Sampler[List[int]]):
    """基于混淆度矩阵的动态 PK 批次采样器 (P=batch_size, K=1)。
      1. 每个 Batch 随机挑选一个锚点物种 (Anchor Class);
      2. 依据混淆强度转移分布，挑选与其最易混淆的 (batch_size - 1) 个不同物种;
      3. 各类别内仅随机抽取 1 个样本索引，杜绝 Batch 内同类误杀与类内碰撞。(不足batch_size则随机取样补全)
    """

    def __init__(
        self,
        dataset_items: List[Tuple[str, List[str]]],
        batch_size: int,
        confusion_json: Optional[str | Path] = None,
        hard_prob: float = 0.8,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.hard_prob = hard_prob
        self.rng = random.Random(seed)

        self.alias_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, (key, _) in enumerate(dataset_items):
            alias = key.split("+", 1)[0] if "+" in key else Path(key).parent.name
            self.alias_to_indices[alias].append(idx)

        self.all_aliases = list(self.alias_to_indices.keys())
        self.confusion_distributions: Dict[str, Tuple[List[str], List[float]]] = {}
        if confusion_json:
            self._load_confusion_json(Path(confusion_json))

        self.num_batches = max(1, len(dataset_items) // self.batch_size)

    def _load_confusion_json(self, json_path: Path):
        if not json_path.is_file():
            return
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data: Dict[str, Dict[str, float]] = json.load(f)

        for src, targets in raw_data.items():
            src_clean = src.strip().lower()
            if src_clean not in self.alias_to_indices:
                continue

            cands, weights = [], []
            for dst, score in targets.items():
                dst_clean = dst.strip().lower()
                if dst_clean in self.alias_to_indices and dst_clean != src_clean:
                    cands.append(dst_clean)
                    weights.append(float(score))

            if cands:
                self.confusion_distributions[src_clean] = (cands, weights)

    def _weighted_sample_no_replace(self, pop: List[str], weights: List[float], k: int) -> List[str]:
        p, w = list(pop), list(weights)
        res = []
        for _ in range(k):
            chosen = self.rng.choices(p, weights=w, k=1)[0]
            idx = p.index(chosen)
            res.append(chosen)
            p.pop(idx)
            w.pop(idx)
        return res

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.num_batches):
            anchor = self.rng.choice(self.all_aliases)
            selected_aliases = [anchor]

            # 优先依据混淆权重挑选竞争物种
            if self.rng.random() < self.hard_prob and anchor in self.confusion_distributions:
                cands, weights = self.confusion_distributions[anchor]
                k = min(len(cands), self.batch_size - 1)
                selected_aliases.extend(self._weighted_sample_no_replace(cands, weights, k))

            # 补齐剩余类别，确保类别去重最大化
            remains = [a for a in self.all_aliases if a not in selected_aliases]
            self.rng.shuffle(remains)
            selected_aliases.extend(remains)

            # 截断或自适应平铺至 batch_size
            if len(selected_aliases) >= self.batch_size:
                final_aliases = selected_aliases[: self.batch_size]
            else:
                # 兼容类别数小于 batch_size: 优先复用高混淆类与锚点
                extra_needed = self.batch_size - len(selected_aliases)
                extra = self.rng.choices(selected_aliases, k=extra_needed)
                final_aliases = selected_aliases + extra

            # 类别内抽样 (优先无放回以压低图像重复率)
            batch_indices = []
            counts = Counter(final_aliases)
            for alias, count in counts.items():
                pool = self.alias_to_indices[alias]
                if len(pool) >= count:
                    batch_indices.extend(self.rng.sample(pool, count))
                else:
                    batch_indices.extend(self.rng.choices(pool, k=count))

            self.rng.shuffle(batch_indices)
            yield batch_indices


class DualTowerDataset(Dataset):
    def __init__(
        self,
        manifest: Dict[str, List[str]],
        root: Union[str, Path],
        transform: Callable[[np.ndarray, bool], torch.Tensor],
        is_train: bool = False,
        images_path: Optional[str] = None,
        caption_sample_mode: str = "random",
    ):
        self.items: List[Tuple[str, List[str]]] = list(manifest.items())
        self.root = Path(root)
        self.transform = transform
        self.is_train = is_train
        self.caption_sample_mode = caption_sample_mode
        assert self.caption_sample_mode in ('random', 'first', 'concat'), f"Unsupport caption sample mode: {self.caption_sample_mode}"
        self._aug_enabled = torch.tensor([1 if is_train else 0], dtype=torch.uint8).share_memory_()

        # 判断读取存储介质
        self.use_lmdb = bool(images_path and images_path.lower().endswith(".lmdb"))
        self.lmdb = LazyLMDBReader(images_path) if self.use_lmdb else None
        self.raw_images_dir = Path(images_path) if (images_path and not self.use_lmdb) else self.root
        self.io = ImageIOHandler()

    @property
    def aug_enabled(self) -> bool:
        return bool(self._aug_enabled[0].item())

    @aug_enabled.setter
    def aug_enabled(self, value: bool):
        self._aug_enabled[0] = 1 if value else 0

    def _read_bytes(self, key: str) -> bytes:
        if self.lmdb is not None:
            raw = self.lmdb.read(key)
            if raw is not None:
                return raw
            
        if "+" in key:
            cname, img_name = key.split("+", 1)
            file_path = self.raw_images_dir / cname / img_name
            if not file_path.is_file():
                file_path = self.root / cname / img_name
            if not file_path.is_file():
                file_path = self.raw_images_dir / img_name
        else:
            file_path = self.raw_images_dir / cname / key

        if file_path.is_file():
            return file_path.read_bytes()

        raise FileNotFoundError(f"未找到图像文件: {key} (路径: {file_path})")

    def _sample_caption(self, captions: List[str]) -> str:
        if not captions:
            return ""
        if self.caption_sample_mode == "first":
            return captions[0]
        elif self.caption_sample_mode == "concat":
            return "; ".join(captions)
        elif self.caption_sample_mode == "random":
            return random.choice(captions)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        key, captions = self.items[idx]

        raw_bytes = self._read_bytes(key)
        img_rgb = self.io.decode(raw_bytes)

        use_aug = self.is_train and self.aug_enabled
        pixel_tensor = self.transform(img_rgb, train=use_aug)

        caption = self._sample_caption(captions)
        # 从规范 Key "{alias}+{image_name}" 中剥离中文全拼名作为分类 Ground Truth
        label = key.split("+", 1)[0] if "+" in key else Path(key).parent.name

        return {
            "image": pixel_tensor,
            "text": caption,
            "label": label,
            "path": key,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "image": torch.stack([b["image"] for b in batch], dim=0),
        "text": [b["text"] for b in batch],
        "label": [b["label"] for b in batch],
        "path": [b["path"] for b in batch],
    }


class DualTowerDataModule(LightningDataModule):
    def __init__(
        self,
        data_cfg: DataConfig,
        train_cfg: TrainConfig,
    ):
        super().__init__()
        self.cfg = data_cfg
        self.train_cfg = train_cfg
        self.root = Path(data_cfg.root)
        self.transform = DualTowerTransforms(data_cfg)

        self.taxonomy: Optional[TaxonomyClasses] = None
        self.classes: List[str] = []

        self._train_ds: Optional[DualTowerDataset] = None
        self._val_ds: Optional[DualTowerDataset] = None
        self._resolved_images_path: Optional[str] = None

    @property
    def cname_classes(self) -> List[str]:
        return self.taxonomy.cnames if self.taxonomy else []

    @property
    def latin_classes(self) -> List[str]:
        return self.taxonomy.latins if self.taxonomy else []

    def _resolve_images_path(self) -> Optional[str]:
        if not self.cfg.images:
            return None
        cand = Path(self.cfg.images)
        if cand.is_absolute() and cand.exists():
            return str(cand)
        cand_root = self.root / self.cfg.images
        return str(cand_root)

    def setup(self, stage: Optional[str] = None):
        self._resolved_images_path = self._resolve_images_path()

        # 解析物种类别映射
        csv_path = Path(self.cfg.class_map_csv)
        if not csv_path.is_absolute():
            csv_path = self.root / csv_path
        self.taxonomy = TaxonomyClasses.from_csv(csv_path)
        self.classes = self.taxonomy.cnames

        # 路由挂载数据集
        is_fit = stage in (None, "fit")
        is_eval = stage in (None, "fit", "validate", "test")

        if is_fit:
            train_manifest = load_manifest(self.root / self.cfg.train)
            self._train_ds = DualTowerDataset(
                manifest=train_manifest,
                root=self.root,
                transform=self.transform,
                is_train=True,
                images_path=self._resolved_images_path,
                caption_sample_mode=self.cfg.caption_sample,
            )

        if is_eval:
            val_manifest = load_manifest(self.root / self.cfg.val)
            self._val_ds = DualTowerDataset(
                manifest=val_manifest,
                root=self.root,
                transform=self.transform,
                is_train=False,
                images_path=self._resolved_images_path,
                caption_sample_mode=self.cfg.val_caption_sample,
            )

        train_len = len(self._train_ds) if self._train_ds else 0
        val_len = len(self._val_ds) if self._val_ds else 0
        _log.info(
            f"[DataModule] 类别基准总数={len(self.taxonomy.pos_latins)} | "
            f"train_samples={train_len} | val_samples={val_len} | "
            f"images_mode={'LMDB' if (self._resolved_images_path and self._resolved_images_path.endswith('.lmdb')) else 'Filesystem'}"
        )

    def set_augment_status(self, enabled: bool):
        if self._train_ds is not None:
            self._train_ds.aug_enabled = enabled

    def train_dataloader(self) -> DataLoader:
        if self._train_ds is None:
            raise RuntimeError("Train dataset 未初始化，请先调用 setup('fit')")
        
        confusion_json = getattr(self.cfg, "confusion_json", None)
        if confusion_json and Path(confusion_json).is_file():
            # 使用基于混淆度矩阵的动态 PK 采样器
            batch_sampler = HardNegativeBatchSampler(
                dataset_items=self._train_ds.items,
                batch_size=self.train_cfg.batch_size,
                confusion_json=confusion_json,
                hard_prob=getattr(self.cfg, "hard_prob", 0.8),
                seed=self.train_cfg.seed,
            )
            return DataLoader(
                self._train_ds,
                batch_sampler=batch_sampler,
                num_workers=self.train_cfg.num_workers,
                collate_fn=collate_fn,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=self.train_cfg.num_workers > 0,
            )
        
        return DataLoader(
            self._train_ds,
            batch_size=self.train_cfg.batch_size,
            shuffle=True,
            num_workers=self.train_cfg.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.train_cfg.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_ds is None:
            raise RuntimeError("Val dataset 未初始化，请先调用 setup('validate')")
        return DataLoader(
            self._val_ds,
            batch_size=self.train_cfg.batch_size,
            shuffle=False,
            num_workers=self.train_cfg.num_workers,
            collate_fn=collate_fn,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.train_cfg.num_workers > 0,
        )


def build_dual_tower_datamodule(data_cfg: DataConfig, train_cfg: TrainConfig):
    return DualTowerDataModule(data_cfg, train_cfg)
