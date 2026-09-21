"""统一日志与进度工具。

Skill §4.1 规定:
  - 所有控制台输出以 loguru 进行输出;
  - 进度训练以 tqdm 显示。

对外暴露:
  - logger: 已配置好的 loguru 实例(全框架统一用这个, 不要再用 print);
  - progress: tqdm 的便捷封装, 用于训练/评估等循环进度展示。
"""
from __future__ import annotations

import sys
from typing import Iterable, Optional

from loguru import logger as _loguru_logger
from tqdm import tqdm


# ----------------------------- 日志配置 ----------------------------- #
_logger_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[tag]}</cyan> | "
    "<level>{message}</level>"
)

# 移除 loguru 默认 sink, 采用统一格式(可重复调用本模块而不重复绑定)
_logger_logger = _loguru_logger
try:
    _logger_logger.remove()
except Exception:
    pass
_logger_logger.add(
    sys.stderr,
    format=_logger_format,
    level="INFO",
    colorize=True,
    enqueue=False,
)

# 让其它模块 `from utils.logger import logger` 即可; 绑定默认 tag 避免裸调用 KeyError
logger = _logger_logger.bind(tag="main")


def get_logger(tag: str = "framework"):
    """返回绑定了 extra['tag'] 的 logger, 便于区分模块来源。"""
    return logger.bind(tag=tag)


# ----------------------------- 进度条 ----------------------------- #
def progress(iterable: Optional[Iterable] = None, *, desc: str = "progress",
             total: Optional[int] = None, **kwargs):
    """tqdm 便捷封装。

    用法:
        for batch in progress(loader, desc="train"): ...
        pbar = progress(total=n, desc="eval"); pbar.update(1)
    """
    kwargs.setdefault("ncols", 100)
    kwargs.setdefault("leave", True)
    return tqdm(iterable, desc=desc, total=total, **kwargs)


# 显式导出, 避免 `from utils.logger import *` 时漏掉
__all__ = ["logger", "get_logger", "progress", "tqdm"]
