# -*- coding: utf-8 -*-
"""异常定义"""


class GlassBottleDefectError(Exception):
    """玻璃瓶缺陷检测基础异常"""
    pass


class ModelLoadError(GlassBottleDefectError):
    """模型加载异常"""
    pass


class ImageLoadError(GlassBottleDefectError):
    """图片加载异常"""
    pass


class DetectionError(GlassBottleDefectError):
    """检测异常"""
    pass


class CameraError(GlassBottleDefectError):
    """摄像头异常"""
    pass