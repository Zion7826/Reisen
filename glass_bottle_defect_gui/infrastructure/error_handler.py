# -*- coding: utf-8 -*-
"""异常处理器模块。

将内部异常转换为用户友好的中文 QMessageBox 提示，
堆栈仅写入日志不向用户展示。维护异常→中文提示映射表。
"""

from __future__ import annotations

import logging
import traceback
from typing import Optional

from PyQt5.QtWidgets import QMessageBox, QWidget

from .exceptions import (
    ImageDecodeError,
    InferenceError,
    ModelNotFoundError,
    ReportBuildError,
    UnsupportedFormatError,
)

logger = logging.getLogger("glass_bottle_gui")

# 异常类型 → 中文提示文案映射表
_EXCEPTION_MESSAGES: dict[type[Exception], str] = {
    ImageDecodeError: "图片文件已损坏，无法加载",
    UnsupportedFormatError: "仅支持 JPG/JPEG/PNG/BMP 格式",
    ModelNotFoundError: "检测模型未加载，请联系管理员",
    InferenceError: "检测失败，请重试",
    ReportBuildError: "报告生成失败，请重试",
    PermissionError: "导出失败，请检查路径权限或磁盘空间",
}

# 默认提示文案
_DEFAULT_MESSAGE = "操作失败，请重试"


class ErrorHandler:
    """异常处理器，统一将异常转中文提示并记录日志。"""

    @staticmethod
    def handle(exc: Exception, parent: Optional[QWidget] = None) -> None:
        """处理异常：弹出中文 QMessageBox 提示并记录日志。

        Args:
            exc: 捕获的异常对象
            parent: 父窗口，用于 QMessageBox 模态定位
        """
        # 查找映射表（支持子类匹配）
        message = _DEFAULT_MESSAGE
        for exc_type, msg in _EXCEPTION_MESSAGES.items():
            if isinstance(exc, exc_type):
                message = msg
                break

        # 记录日志（含完整堆栈）
        logger.error(f"{message} | 异常: {exc}")
        logger.debug(traceback.format_exc())

        # 弹出中文提示
        QMessageBox.critical(parent, "提示", message)

    @staticmethod
    def handle_message(message: str, parent: Optional[QWidget] = None) -> None:
        """直接用中文消息弹出提示框并记录日志。

        Args:
            message: 中文提示文案
            parent: 父窗口
        """
        logger.warning(message)
        QMessageBox.warning(parent, "提示", message)

    @staticmethod
    def log_operation(op: str, context: Optional[dict] = None) -> None:
        """记录关键操作日志，便于追溯。

        Args:
            op: 操作名称（如 "检测启动"、"报告导出"）
            context: 操作上下文信息字典
        """
        ctx_str = f" | 上下文: {context}" if context else ""
        logger.info(f"[操作] {op}{ctx_str}")