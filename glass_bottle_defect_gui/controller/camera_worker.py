# -*- coding: utf-8 -*-
"""摄像头采集工作线程模块。

使用 QThread 在后台循环采集摄像头帧并执行检测，
通过 Qt 信号槽将帧与检测结果回传主线程。
支持帧率节流、断连检测与资源释放。
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from ..domain.models import DetectionResult
from ..service.detection_service import DetectionService

logger = logging.getLogger("glass_bottle_gui")


class CameraWorker(QThread):
    """摄像头采集与检测线程。

    Signals:
        frame_ready: 采集并检测完成时发射，携带原图帧与检测结果
        error_occurred: 发生错误时发射，携带中文错误提示
    """

    frame_ready = pyqtSignal(np.ndarray, DetectionResult)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        detection_service: DetectionService,
        camera_index: int = 0,
        target_fps: int = 10,
    ) -> None:
        """初始化摄像头线程。

        Args:
            detection_service: 检测服务实例
            camera_index: 摄像头设备索引
            target_fps: 目标帧率
        """
        super().__init__()
        self._service = detection_service
        self._camera_index = camera_index
        self._target_fps = target_fps
        self._stop_flag = False
        self._cap: cv2.VideoCapture | None = None

    def run(self) -> None:
        """线程执行体：循环采集帧并检测。"""
        self._stop_flag = False

        # 打开摄像头
        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            self.error_occurred.emit("未检测到可用摄像头，请检查设备连接")
            self._release_cap()
            return

        logger.info(f"摄像头已启动，索引: {self._camera_index}，目标帧率: {self._target_fps}")
        frame_interval = 1.0 / self._target_fps
        frame_count = 0

        while not self._stop_flag:
            loop_start = time.time()

            ret, frame = self._cap.read()
            if not ret or frame is None or frame.size == 0:
                self.error_occurred.emit("摄像头连接中断，请重新连接")
                logger.warning("摄像头帧无效，疑似断连")
                break

            try:
                result = self._service.detect(
                    frame, "摄像头", f"camera_frame_{frame_count}"
                )
                self.frame_ready.emit(frame, result)
                frame_count += 1
            except Exception as e:
                logger.error(f"摄像头检测异常: {e}")
                # 检测异常不中断采集，继续下一帧

            # 帧率节流
            elapsed = time.time() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._release_cap()
        logger.info(f"摄像头线程结束，共处理 {frame_count} 帧")

    def stop(self) -> None:
        """置位停止标志，线程将在当前帧处理后退出。"""
        self._stop_flag = True

    def _release_cap(self) -> None:
        """释放 VideoCapture 资源。"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None