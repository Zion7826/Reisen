# -*- coding: utf-8 -*-
"""自定义异常类型模块。

所有异常均继承 Exception，构造时支持携带上下文信息（如文件路径、原因），
便于日志记录与 ErrorHandler 转换为用户友好的中文提示。
"""


class ImageDecodeError(Exception):
    """图片解码失败异常。"""

    def __init__(self, path: str = "", reason: str = "") -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"图片解码失败: {path} - {reason}" if reason else f"图片解码失败: {path}")


class UnsupportedFormatError(Exception):
    """不支持的图片格式异常。"""

    def __init__(self, path: str = "", fmt: str = "") -> None:
        self.path = path
        self.fmt = fmt
        super().__init__(f"不支持的格式: {fmt or path}")


class ModelNotFoundError(Exception):
    """模型权重未找到异常。"""

    def __init__(self, path: str = "") -> None:
        self.path = path
        super().__init__(f"模型权重未找到: {path}")


class InferenceError(Exception):
    """推理失败异常。"""

    def __init__(self, reason: str = "") -> None:
        self.reason = reason
        super().__init__(f"推理失败: {reason}" if reason else "推理失败")


class ReportBuildError(Exception):
    """报告生成失败异常。"""

    def __init__(self, reason: str = "") -> None:
        self.reason = reason
        super().__init__(f"报告生成失败: {reason}" if reason else "报告生成失败")