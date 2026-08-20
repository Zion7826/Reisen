# -*- coding: utf-8 -*-
"""摄像头工作线程"""

import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

from glass_bottle_defect_gui.service.detection_service import DetectionService
from glass_bottle_defect_gui.domain.models import DetectionResponse
from glass_bottle_defect_gui.infrastructure.exceptions import CameraError
from glass_bottle_defect_gui.infrastructure.logger import logger


class CameraWorker(QThread):
    """摄像头实时检测工作线程"""

    frame_ready = pyqtSignal(QImage)  # 帧就绪信号
    detection_ready = pyqtSignal(DetectionResponse)  # 检测结果信号
    error = pyqtSignal(str)  # 错误信号

    def __init__(self, camera_id: int = 0, service: DetectionService = None, conf_threshold: float = 0.25):
        super().__init__()
        self.camera_id = camera_id
        self.service = service
        self.conf_threshold = conf_threshold
        self.running = False
        self.cap = None

    def run(self):
        """运行摄像头捕获"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                raise CameraError(f"无法打开摄像头: {self.camera_id}")

            self.running = True
            logger.info(f"摄像头已启动: {self.camera_id}")

            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("无法读取摄像头帧")
                    break

                # 执行检测
                if self.service:
                    response = self.service.detect(frame, self.conf_threshold)
                    frame = self.service.draw_detections(frame, response)
                    self.detection_ready.emit(response)

                # 转换为 QImage 并发送
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.frame_ready.emit(qt_image)

        except Exception as e:
            logger.error(f"摄像头错误: {e}")
            self.error.emit(str(e))
        finally:
            if self.cap:
                self.cap.release()
            logger.info("摄像头已关闭")

    def stop(self):
        """停止摄像头"""
        self.running = False
        self.wait()