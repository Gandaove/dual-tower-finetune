'''
Fused on multi2dvision-inferencer.py (engine_base + processor)
'''
import abc
import json
import logging
from collections import OrderedDict, namedtuple
from functools import wraps
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import tensorrt as trt
import torch
import cv2
import numpy as np

from utils.logger import get_logger
_logger = get_logger("engine-base")

# 统一 TRT 到 Torch 的类型映射
TRT_TO_TORCH_DTYPE: Dict[trt.DataType, torch.dtype] = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL: torch.bool,
    trt.DataType.BF16: torch.bfloat16,
    trt.DataType.INT64: torch.int64,
}

TensorBinding = namedtuple("TensorBinding", ("name", "dtype", "shape", "is_input"))


def trt_stream_guard(func):
    """
    全链路流守卫装饰器：
    强制将前处理、H2D 搬运、TRT 推理及后处理的所有 GPU 算子发射至 self.stream，
    杜绝默认流与私有流之间的异步竞态。
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with torch.cuda.stream(self.stream):
            return func(self, *args, **kwargs)
    return wrapper


class TRTLoggerBridge(trt.ILogger):
    """单例复用的 TRT 日志桥接器"""
    SEVERITY_MAP = {
        trt.ILogger.INTERNAL_ERROR: logging.CRITICAL,
        trt.ILogger.ERROR: logging.ERROR,
        trt.ILogger.WARNING: logging.WARNING,
        trt.ILogger.INFO: logging.INFO,
        trt.ILogger.VERBOSE: logging.DEBUG,
    }

    def __init__(self, python_logger: logging.Logger):
        super().__init__()
        self.py_logger = python_logger

    def log(self, severity, msg):
        level = self.SEVERITY_MAP.get(severity, logging.INFO)
        self.py_logger.log(level, f"[TRT] {msg}")


class BaseTRTEngine(abc.ABC):
    """
    TensorRT 10.x 引擎底层基类
    封装统一的反序列化、绑定管理、零拷贝输入绑定和安全动态输出分配
    """
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.device = torch.device(self.cfg.get("device", "cuda:0"))
        log_name = self.cfg.get("log_name", self.__class__.__name__)
        self.logger = _logger.bind(logger_name=log_name)

        model_path = self.cfg.get("model_path")
        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(f"Engine 文件不存在: {model_path}")

        self.trt_logger = TRTLoggerBridge(self.logger)
        trt.init_libnvinfer_plugins(self.trt_logger, "")

        self.engine = self._deserialize_engine(model_path)
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=self.device)

        self.input_meta: OrderedDict[str, TensorBinding] = OrderedDict()
        self.output_meta: OrderedDict[str, TensorBinding] = OrderedDict()
        self.output_buffers: Dict[str, torch.Tensor] = {}
        self.is_dynamic = False

        self._parse_io_metadata()

    def _deserialize_engine(self, model_path: str) -> trt.ICudaEngine:
        """兼容 Ultralytics 头部 Meta JSON 与原生标准 Engine"""
        with open(model_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
            try:
                meta_len = int.from_bytes(f.read(4), byteorder="little")
                metadata_bytes = f.read(meta_len)
                metadata = json.loads(metadata_bytes.decode("utf-8"))
                if metadata and metadata.get("dla") is not None:
                    runtime.DLA_core = int(metadata["dla"])
                self.logger.info("已跳过并解析 Ultralytics metadata 头部")
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                f.seek(0)

            engine = runtime.deserialize_cuda_engine(f.read())

        if not engine:
            raise RuntimeError(f"Engine 反序列化失败: {model_path}")
        return engine

    def _parse_io_metadata(self):
        """修复点：强转 tuple 规避 TRT Dims 对象的 __eq__ 兼容性报错"""
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            dtype = TRT_TO_TORCH_DTYPE.get(self.engine.get_tensor_dtype(name), torch.float32)
            trt_shape = tuple(self.engine.get_tensor_shape(name))

            if is_input:
                is_dyn = False
                opt_shape = trt_shape
                
                if self.engine.num_optimization_profiles > 0:
                    profile_shapes = self.engine.get_tensor_profile_shape(name, 0)
                    if profile_shapes is not None and len(profile_shapes) == 3:
                        min_s = tuple(profile_shapes[0])
                        opt_s = tuple(profile_shapes[1])
                        max_s = tuple(profile_shapes[2])
                        opt_shape = opt_s
                        if min_s != max_s or (-1 in trt_shape):
                            is_dyn = True

                if -1 in trt_shape:
                    is_dyn = True

                if is_dyn:
                    self.is_dynamic = True

                self.input_meta[name] = TensorBinding(name, dtype, opt_shape, True)
                self.context.set_input_shape(name, opt_shape)
            else:
                self.output_meta[name] = TensorBinding(name, dtype, trt_shape, False)

    def _infer_core(self, feed_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        核心底层推理函数：
        实现输入 Tensor 指针的纯零拷贝挂载 + 输出 Buffer 的安全重分配与推理
        """
        for name, tensor in feed_tensors.items():
            if name not in self.input_meta:
                continue
            binding = self.input_meta[name]
            
            if tensor.dtype != binding.dtype:
                tensor = tensor.to(binding.dtype)

            if self.is_dynamic:
                curr_shape = tuple(tensor.shape)
                self.context.set_input_shape(name, curr_shape)
                self.input_meta[name] = binding._replace(shape=curr_shape)

            assert self.context.set_tensor_address(name, tensor.data_ptr()), f"设置输入指针失败: {name}"

        # 为输出张量分配合适的显存
        outputs: Dict[str, torch.Tensor] = {}
        for name, binding in self.output_meta.items():
            req_shape = tuple(self.context.get_tensor_shape(name))
            
            # 只有当 Buffer 不存在或 Shape 变更时才重新分配，杜绝 resize_ 导致的不连续与指针悬空
            if (name not in self.output_buffers) or (self.output_buffers[name].shape != req_shape):
                self.output_buffers[name] = torch.empty(
                    req_shape, dtype=binding.dtype, device=self.device
                ).contiguous()

            buf = self.output_buffers[name]
            assert self.context.set_tensor_address(name, buf.data_ptr()), f"设置输出指针失败: {name}"
            outputs[name] = buf

        # 异步发射
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        
        # 浅拷贝输出字典（指向 Buffer 数据），交由上层做克隆或 CPU 收集
        return outputs


class ImageProcessor:
    """统一图像处理组件：

    封装 Preprocess (几何缩放、颜色空间转换、GPU 搬运与标准化)
    与 Postprocess (检测框尺度与空间几何逆映射还原)。
    """
    INTERPOLATION_MAP = {
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
    }

    def __init__(
        self,
        target_size: Tuple[int, int],  # (height, width)
        resize_method: str = "letterbox",
        interpolation: str = "bilinear",
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        bgr_to_rgb: bool = True,
        pad_val: int = 114,
        device: torch.device = torch.device("cuda:0"),
    ):
        self.target_h, self.target_w = target_size
        self.resize_method = resize_method.lower()
        self.bgr_to_rgb = bgr_to_rgb
        self.interpolation_name = interpolation.lower()
        self.pad_val = pad_val
        self.device = device

        if self.resize_method not in ("direct", "letterbox", "shortest_edge"):
            raise ValueError(f"不支持的 resize_method: {self.resize_method}")
        if self.interpolation_name not in self.INTERPOLATION_MAP:
            raise ValueError(
                f"不支持的 interpolation: {self.interpolation_name}，"
                f"可选: {list(self.INTERPOLATION_MAP.keys())}"
            )
        self.interp_flag = self.INTERPOLATION_MAP[self.interpolation_name]

        # GPU 归一化参数预置
        self.mean_tensor = None
        self.std_tensor = None
        if mean is not None and std is not None:
            self.mean_tensor = torch.tensor(
                mean, dtype=torch.float32, device=self.device
            ).view(1, 3, 1, 1)
            self.std_tensor = torch.tensor(
                std, dtype=torch.float32, device=self.device
            ).view(1, 3, 1, 1)

    @classmethod
    def from_config(
        cls,
        cfg: Dict[str, Any],
        target_size: Tuple[int, int],
        device: torch.device,
        default_resize: str = "letterbox",
        default_interpolation: str = "bilinear",
        default_mean: Optional[List[float]] = None,
        default_std: Optional[List[float]] = None,
    ) -> "ImageProcessor":
        return cls(
            target_size=target_size,
            resize_method=cfg.get("resize_method", default_resize),
            interpolation=cfg.get("interpolation", default_interpolation),
            mean=cfg.get("mean", default_mean),
            std=cfg.get("std", default_std),
            bgr_to_rgb=cfg.get("bgr_to_rgb", True),
            pad_val=cfg.get("pad_val", 114),
            device=device,
        )

    def _resize_direct(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
        h_orig, w_orig = img.shape[:2]
        resized = cv2.resize(
            img,
            (self.target_w, self.target_h),
            interpolation=self.interp_flag,
        )
        return (
            resized,
            (self.target_w / w_orig, self.target_h / h_orig),
            (0.0, 0.0),
        )

    def _resize_letterbox(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
        h_orig, w_orig = img.shape[:2]
        r = min(self.target_h / h_orig, self.target_w / w_orig)
        new_w, new_h = int(round(w_orig * r)), int(round(h_orig * r))

        resized = cv2.resize(
            img, (new_w, new_h), interpolation=self.interp_flag,
        )
        dw = (self.target_w - new_w) / 2.0
        dh = (self.target_h - new_h) / 2.0
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(self.pad_val, self.pad_val, self.pad_val),
        )
        return padded, (r, r), (float(left), float(top))

    def _resize_shortest_edge(
        self, img: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
        h_orig, w_orig = img.shape[:2]
        r = max(self.target_h / h_orig, self.target_w / w_orig)
        new_w, new_h = int(round(w_orig * r)), int(round(h_orig * r))

        resized = cv2.resize(
            img, (new_w, new_h), interpolation=self.interp_flag,
        )
        crop_x = max(0, (new_w - self.target_w) // 2)
        crop_y = max(0, (new_h - self.target_h) // 2)
        cropped = resized[
            crop_y : crop_y + self.target_h, crop_x : crop_x + self.target_w
        ]
        return cropped, (r, r), (-float(crop_x), -float(crop_y))

    def preprocess(
        self,
        imgs: List[np.ndarray],
        target_dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """全流程前处理：几何变换 -> CPU Tensor -> H2D -> In-place 归一化。"""
        tensor_list, ratios, offsets, orig_shapes = [], [], [], []

        for img in imgs:
            orig_shapes.append(img.shape[:2])

            if self.resize_method == "direct":
                p_img, r, offset = self._resize_direct(img)
            elif self.resize_method == "shortest_edge":
                p_img, r, offset = self._resize_shortest_edge(img)
            else:
                p_img, r, offset = self._resize_letterbox(img)

            if self.bgr_to_rgb:
                p_img = cv2.cvtColor(p_img, cv2.COLOR_BGR2RGB)

            tensor_list.append(torch.from_numpy(p_img).permute(2, 0, 1))
            ratios.append(r)
            offsets.append(offset)

        batch_tensor = (
            torch.stack(tensor_list)
            .to(device=self.device, non_blocking=True)
            .to(target_dtype)
            .div_(255.0)
        )

        if self.mean_tensor is not None and self.std_tensor is not None:
            batch_tensor.sub_(self.mean_tensor.to(target_dtype)).div_(
                self.std_tensor.to(target_dtype)
            )

        meta = {
            "ratios": ratios,
            "offsets": offsets,
            "orig_shapes": orig_shapes,
            "model_size": (self.target_h, self.target_w),
            "resize_method": self.resize_method,
        }
        return batch_tensor.contiguous(), meta

    def restore_boxes(
        self,
        boxes: torch.Tensor,
        meta: Dict[str, Any],
        batch_idx: int,
    ) -> torch.Tensor:
        """后处理逆映射：将处于模型尺寸网格下的检测框还原至原始图像尺寸。

        所有缩放模式的数学逆变换完全由本方法自决并闭环，Engine 层无需探查缩放类型。
        """
        if boxes.numel() == 0:
            return boxes

        scale_x, scale_y = meta["ratios"][batch_idx]
        offset_x, offset_y = meta["offsets"][batch_idx]
        orig_h, orig_w = meta["orig_shapes"][batch_idx]

        # 统一逆变换公式：直接兼容 direct / letterbox / shortest_edge
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - offset_x) / scale_x
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - offset_y) / scale_y

        # 原图边界截断
        boxes[:, [0, 2]].clamp_(0, orig_w)
        boxes[:, [1, 3]].clamp_(0, orig_h)
        return boxes
