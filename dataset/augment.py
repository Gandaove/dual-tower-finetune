"""数据增强与预处理。

设计:
  - 以 Albumentations 为主, letterbox resize 为自定义实现(cv2 完成);
  - 增强在 uint8 numpy 上完成, 最后统一归一化并转为 float32 [0,1];
  - 提供 close_augment 开关: 到达指定 epoch 后仅保留 resize/normalize(由 DataModule 控制)。

图像相关操作统一使用 cv2(Skill §4.1)。
"""
from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from utils.config import AugmentConfig, DataConfig

_INTERP = {
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST,
}


def resolve_interp(name: str) -> int:
    return _INTERP.get(name, cv2.INTER_LINEAR)


class DualTowerTransforms:
    """构建 train / val 两套变换。"""

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.aug: AugmentConfig = cfg.augment
        self.train = self._build_train()
        self.val = self._build_val()

    # ---------------- letterbox ---------------- #
    @staticmethod
    def letterbox(image: np.ndarray, size: int, pad_value: int, interp: int) -> np.ndarray:
        h, w = image.shape[:2]
        scale = min(size / h, size / w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        ih, iw = size - nh, size - nw
        top, left = ih // 2, iw // 2
        bottom, right = ih - top, iw - left
        resized = cv2.resize(image, (nw, nh), interpolation=interp)
        canvas = np.full((size, size, image.shape[2]), pad_value, dtype=image.dtype)
        canvas[top:top + nh, left:left + nw] = resized
        return canvas

    # ---------------- 构建 ---------------- #
    def _common_resize(self, train: bool):
        a = self.aug
        r = a.resize
        interp = resolve_interp(r.interp)
        if r.method == "letterbox":
            def _letterbox_op(image, **kwargs):
                return self.letterbox(image, r.size, r.pad_value, interp)

            return A.Lambda(name="Letterbox", image=_letterbox_op, p=1.0)
        if r.method == "shortest_edge":
            return A.SmallestMaxSize(max_size=r.size, interpolation=interp)
        return A.Resize(height=r.size, width=r.size, interpolation=interp)

    def _build_train(self):
        a = self.aug
        ops = [self._common_resize(train=True)]

        if a.crop.enabled:
            if a.crop.method == "random":
                ops.append(A.RandomCrop(height=a.crop.size, width=a.crop.size))
            else:
                ops.append(A.CenterCrop(height=a.crop.size, width=a.crop.size))

        if a.flip and a.flip > 0:
            ops.append(A.HorizontalFlip(p=a.flip))
        if a.vflip and a.vflip > 0:
            ops.append(A.VerticalFlip(p=a.vflip))

        if a.rotate.prob > 0:
            ops.append(A.Rotate(
                limit=tuple(a.rotate.limit),
                p=a.rotate.prob,
                border_mode=0,
                interpolation=resolve_interp(a.resize.interp),
            ))

        if a.affine.prob > 0:
            ops.append(A.Affine(
                scale=tuple(a.affine.scale) if isinstance(a.affine.scale, list) else a.affine.scale,
                shear=a.affine.shear,
                translate_percent=a.affine.percent,
                p=a.affine.prob,
                interpolation=resolve_interp(a.resize.interp),
                fill=0,
            ))

        if a.perspective.prob > 0:
            ops.append(A.Perspective(
                scale=tuple(a.perspective.scale),
                p=a.perspective.prob,
                border_mode=0,
                interpolation=resolve_interp(a.resize.interp),
            ))

        if a.hsv.prob > 0:
            ops.append(A.HueSaturationValue(
                hue_shift_limit=int(a.hsv.h * 180),
                sat_shift_limit=int(a.hsv.s * 100),
                val_shift_limit=int(a.hsv.v * 100),
                p=a.hsv.prob,
            ))

        if a.gauss_blur.prob > 0:
            ops.append(A.GaussianBlur(
                blur_limit=tuple(a.gauss_blur.blur),
                sigma_limit=tuple(a.gauss_blur.sigma),
                p=a.gauss_blur.prob,
            ))

        if a.erase.prob > 0:
            h_min = max(1, int(a.erase.range[0] * a.resize.size))
            h_max = max(1, int(a.erase.range[1] * a.resize.size))
            try:
                erase_op = A.CoarseDropout(
                    num_holes_range=(1, a.erase.max_holes),
                    hole_height_range=(h_min, h_max),
                    hole_width_range=(h_min, h_max),
                    fill=0,
                    p=a.erase.prob,
                )
            except TypeError:
                erase_op = A.CoarseDropout(
                    num_holes=a.erase.max_holes,
                    max_height=h_max,
                    max_width=h_max,
                    min_height=h_min,
                    min_width=h_min,
                    fill_value=0,
                    p=a.erase.prob,
                )
            ops.append(erase_op)

        ops.append(self._normalize())
        ops.append(ToTensorV2())
        return A.Compose(ops)

    def _build_val(self):
        ops = [self._common_resize(train=False)]
        if self.aug.crop.enabled:
            ops.append(A.CenterCrop(height=self.aug.crop.size, width=self.aug.crop.size))
        ops.append(self._normalize())
        ops.append(ToTensorV2())
        return A.Compose(ops)

    def _normalize(self):
        n = self.aug.normalize
        return A.Normalize(mean=n.mean, std=n.std, max_pixel_value=255.0)

    # ---------------- 调用 ---------------- #
    def __call__(self, image: np.ndarray, train: bool = True) -> torch.Tensor:
        if not image.flags["C_CONTIGUOUS"]:
            image = np.ascontiguousarray(image)
        tf = self.train if train else self.val
        return tf(image=image)["image"].contiguous()
