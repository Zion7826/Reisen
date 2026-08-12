# -*- coding: utf-8 -*-
"""日志记录器模块。

封装 Python logging 标准库，配置同时输出到控制台与文件。
ERROR 级别日志含完整堆栈信息供排查，但不向用户展示。
"""

import logging
import sys
from pathlib import Path

from .. import config


def setup_logger() -> logging.Logger:
    """初始化并返回全局日志记录器。

    配置控制台输出与文件输出，日志目录不存在时自动创建。
    重复调用时清理已有 handler 避免重复输出。

    Returns:
        配置好的 Logger 实例
    """
    log_dir = Path(config.RUNTIME_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / config.LOG_FILE

    logger = logging.getLogger("glass_bottle_gui")
    logger.setLevel(logging.DEBUG)

    # 清理已有 handler，避免重复添加
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出（INFO 及以上）
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # 文件输出（DEBUG 及以上，含完整堆栈）
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def get_logger() -> logging.Logger:
    """获取已配置的日志记录器（若未初始化则自动初始化）。"""
    logger = logging.getLogger("glass_bottle_gui")
    if not logger.handlers:
        setup_logger()
    return logger