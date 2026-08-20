# -*- coding: utf-8 -*-
"""检测工作线程"""

import cv2
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal

from glass_bottle_defect_gui.service.detection_service import DetectionService
from glass_bottle_defect_gui.domain.models import DetectionResponse
from glass_bottle_defect_gui.infrastructure.logger import logger


class DetectionWorker(QThread):
    """图片检测工作线程"""

    finished = pyqtSignal(DetectionResponse)  # 检测完成信号
    error = pyqtSignal(str)  # 错误信号

    def __init__(self, image_path: str, service: DetectionService, conf_threshold: float = 0.25):
        super().__init__()
        self.image_path = image_path
        self.service = service
        self.conf_threshold = conf_threshold

    def run(self):
        """执行检测"""
        try:
            logger.info(f"开始检测图片: {self.image_path}")
            image = self.service.load_image(self.image_path)
            response = self.service.detect(image, self.conf_threshold)
            response.image_path = self.image_path
            self.finished.emit(response)
            logger.info(f"检测完成: {self.image_path}, 检测到 {response.total_count} 个缺陷")
        except Exception as e:
            logger.error(f"检测失败: {e}")
            self.error.emit(str(e))