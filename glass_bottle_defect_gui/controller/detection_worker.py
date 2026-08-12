# -*- coding: utf-8 -*-
"""单张图片异步检测线程模块。

使用 QThread 在后台执行检测，避免阻塞 UI 主线程。
通过 Qt 信号槽将检测结果或错误回传主线程。
"""

from __future__ import annotations

import logging

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from ..domain.models import DetectionResult
from ..service.detection_service import DetectionService

logger = logging.getLogger("glass_bottle_gui")


class DetectionWorker(QThread):
    """单张图片检测线程。

    Signals:
        result_ready: 检测成功时发射，携带 DetectionResult
        error_occurred: 检测失败时发射，携带中文错误提示
    """

    result_ready = pyqtSignal(DetectionResult)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        detection_service: DetectionService,
        image: np.ndarray,
        source: str,
        image_name: str,
    ) -> None:
        """初始化检测线程。

        Args:
            detection_service: 检测服务实例
            image: BGR 图片数组
            source: 来源标识（"上传" 或 "摄像头"）
            image_name: 图片名称
        """
        super().__init__()
        self._service = detection_service
        self._image = image
        self._source = source
        self._image_name = image_name

    def run(self) -> None:
        """线程执行体：调用检测服务，异常吞并转为信号。"""
        try:
            result = self._service.detect(
                self._image, self._source, self._image_name
            )
            self.result_ready.emit(result)
        except Exception as e:
            logger.error(f"检测线程异常: {e}")
            # 异常吞并，转为中文提示信号，不向主线程抛出
            self.error_occurred.emit("检测失败，请重试")