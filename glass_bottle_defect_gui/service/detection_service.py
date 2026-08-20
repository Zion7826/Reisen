# -*- coding: utf-8 -*-
"""检测服务"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

from ultralytics import YOLO
import torch

from glass_bottle_defect_gui.config import (
    MODEL_PATH, DEFAULT_CONF_THRESHOLD, CLASS_COLORS, DEFAULT_COLOR,
    CLASS_NAMES, CLASS_NAMES_CN
)
from glass_bottle_defect_gui.domain.models import DetectionResult, DetectionResponse
from glass_bottle_defect_gui.infrastructure.exceptions import ModelLoadError, ImageLoadError, DetectionError
from glass_bottle_defect_gui.infrastructure.logger import logger


class DetectionService:
    """缺陷检测服务"""

    def __init__(self, model_path: Optional[Path] = None):
        """
        初始化检测服务

        Args:
            model_path: 模型文件路径，默认使用配置中的路径
        """
        self.model_path = model_path or MODEL_PATH
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        """加载 YOLO 模型"""
        try:
            if not self.model_path.exists():
                logger.warning(f"模型权重不存在: {self.model_path}...进入降级占位检测模式")
                self.model = None
                return

            self.model = YOLO(str(self.model_path))
            logger.info(f"模型加载成功: {self.model_path}, 设备: {self.device}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise ModelLoadError(f"模型加载失败: {e}")

    def detect(self, image: np.ndarray, conf_threshold: float = DEFAULT_CONF_THRESHOLD) -> DetectionResponse:
        """
        执行缺陷检测

        Args:
            image: 输入图像 (BGR 格式)
            conf_threshold: 置信度阈值

        Returns:
            DetectionResponse: 检测结果
        """
        if self.model is None:
            # 降级模式：返回空结果
            return DetectionResponse(
                success=True,
                detections=[],
                class_counts={},
                total_count=0
            )

        try:
            results = self.model(image, device=self.device, conf=conf_threshold, verbose=False)
            detections = []
            class_counts = {}

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    class_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                    class_name_cn = CLASS_NAMES_CN.get(cls_id, class_name)

                    detection = DetectionResult(
                        class_id=cls_id,
                        class_name=class_name,
                        class_name_cn=class_name_cn,
                        confidence=conf,
                        bbox=(int(x1), int(y1), int(x2), int(y2))
                    )
                    detections.append(detection)

                    # 统计各类别数量
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

            return DetectionResponse(
                success=True,
                detections=detections,
                class_counts=class_counts,
                total_count=len(detections)
            )

        except Exception as e:
            logger.error(f"检测失败: {e}")
            raise DetectionError(f"检测失败: {e}")

    def load_image(self, image_path: str) -> np.ndarray:
        """
        加载图片

        Args:
            image_path: 图片路径

        Returns:
            np.ndarray: 图像数据 (BGR 格式)
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ImageLoadError(f"无法读取图片: {image_path}")
            return image
        except Exception as e:
            logger.error(f"图片加载失败: {e}")
            raise ImageLoadError(f"图片加载失败: {e}")

    def draw_detections(self, image: np.ndarray, response: DetectionResponse) -> np.ndarray:
        """
        在图像上绘制检测结果

        Args:
            image: 原始图像
            response: 检测响应

        Returns:
            np.ndarray: 绘制后的图像
        """
        result_image = image.copy()

        for det in response.detections:
            x1, y1, x2, y2 = det.bbox
            color = CLASS_COLORS.get(det.class_id, DEFAULT_COLOR)

            # 绘制边界框
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness=2)

            # 绘制标签
            label = f"{det.class_name_cn}({det.class_name}) {det.confidence:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            label_y1 = max(y1 - text_h - baseline - 4, 0)
            label_y2 = y1

            cv2.rectangle(
                result_image,
                (x1, label_y1),
                (x1 + text_w + 4, label_y2),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                result_image,
                label,
                (x1 + 2, label_y2 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        return result_image