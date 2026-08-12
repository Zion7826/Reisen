# -*- coding: utf-8 -*-
"""图片上传控制器模块。

负责文件选择对话框、格式白名单校验、图片解码加载。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QWidget

from .. import config
from ..infrastructure.exceptions import ImageDecodeError, UnsupportedFormatError

logger = logging.getLogger("glass_bottle_gui")


class ImageUploadController:
    """图片上传控制器，处理文件选择、格式校验与加载。"""

    @staticmethod
    def select_image(parent: Optional[QWidget] = None) -> Optional[str]:
        """弹出文件选择对话框，返回所选图片绝对路径。

        Args:
            parent: 父窗口

        Returns:
            图片绝对路径字符串；用户取消返回 None
        """
        path, _ = QFileDialog.getOpenFileName(
            parent, "选择图片", "", config.IMAGE_FILE_FILTER
        )
        if path:
            logger.info(f"用户选择图片: {path}")
        return path if path else None

    @staticmethod
    def validate_format(path: str) -> bool:
        """校验文件扩展名是否在白名单内。

        Args:
            path: 文件路径

        Returns:
            合法返回 True，否则抛出 UnsupportedFormatError
        """
        ext = Path(path).suffix.lower()
        if ext not in config.IMAGE_FORMAT_WHITELIST:
            raise UnsupportedFormatError(path=path, fmt=ext)
        return True

    @staticmethod
    def load_image(path: str) -> np.ndarray:
        """使用 OpenCV 读取图片为 BGR numpy 数组。

        Args:
            path: 图片文件路径

        Returns:
            BGR 格式的 numpy 数组

        Raises:
            ImageDecodeError: 解码失败时抛出
        """
        image = cv2.imread(path)
        if image is None or image.size == 0:
            raise ImageDecodeError(path=path, reason="cv2.imread 返回空")
        return image