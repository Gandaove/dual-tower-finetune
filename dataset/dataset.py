from __future__ import annotations

import io
import json
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from dataset.augment import DualTowerTransforms
from utils.config import DataConfig, TrainConfig
from utils.logger import get_logger

_log = get_logger("data")


def load_manifest(path: Union[str, Path]) -> Dict[str, List[str]]:
    """读取包含多条 Caption 的 Manifest JSON 文件。
    
    格式: {"{pest_cname}+{image_name}": ["caption1", "caption2"]}
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
            file_path = self.raw_images_dir / key

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
        # 从规范 Key "{pest_cname}+{image_name}" 中剥离中文全拼名作为分类 Ground Truth
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

        # 核心双轨类别列表
        self.cname_classes: List[str] = []
        self.latin_classes: List[str] = []
        # 向后兼容原 model/evaluator 读取 classes 属性
        self.classes: List[str] = []

        self._train_ds: Optional[DualTowerDataset] = None
        self._val_ds: Optional[DualTowerDataset] = None
        self._resolved_images_path: Optional[str] = None

    def _resolve_images_path(self) -> Optional[str]:
        if not self.cfg.images:
            return None
        cand = Path(self.cfg.images)
        if cand.is_absolute() and cand.exists():
            return str(cand)
        cand_root = self.root / self.cfg.images
        return str(cand_root)

    def _load_taxonomy_classes(self) -> Tuple[List[str], List[str]]:
        """从 class_map_csv 解析标准全拼与拉丁学名列表。"""
        csv_path = Path(self.cfg.class_map_csv)
        if not csv_path.is_absolute():
            csv_path = self.root / csv_path

        if not csv_path.is_file():
            raise FileNotFoundError(f"未找到物种分类表 class_map_csv: {csv_path}")

        df = pd.read_csv(csv_path)
        required_cols = {"pest_cname", "pest_latin_name"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"class_map_csv 必须包含以下列: {required_cols}")

        cnames, latins = [], []
        # 去重并保持顺序稳定
        seen = set()
        for _, row in df.iterrows():
            cname = str(row["pest_cname"]).strip()
            latin = str(row["pest_latin_name"]).strip()
            if cname and latin and cname not in seen and cname.lower() != "nan":
                seen.add(cname)
                cnames.append(cname)
                latins.append(latin)

        return cnames, latins

    def setup(self, stage: Optional[str] = None):
        self._resolved_images_path = self._resolve_images_path()

        # 1. 严格从 class_map_csv 建立全量类别映射
        self.cname_classes, self.latin_classes = self._load_taxonomy_classes()
        self.classes = self.cname_classes

        # 2. 依据阶段路由挂载数据集
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
            f"[DataModule] 类别基准总数={len(self.cname_classes)} | "
            f"train_samples={train_len} | val_samples={val_len} | "
            f"images_mode={'LMDB' if (self._resolved_images_path and self._resolved_images_path.endswith('.lmdb')) else 'Filesystem'}"
        )

    def set_augment_status(self, enabled: bool):
        if self._train_ds is not None:
            self._train_ds.aug_enabled = enabled

    def train_dataloader(self) -> DataLoader:
        if self._train_ds is None:
            raise RuntimeError("Train dataset 未初始化，请先调用 setup('fit')")
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
