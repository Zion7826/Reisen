# -*- coding: utf-8 -*-
"""检测服务模块。

封装 ultralytics YOLO 推理接口，将 YOLO 结果转换为领域数据模型 DetectionResult。
内部使用 threading.Lock 串行化推理，支持多线程并发调用。
权重缺失时降级为占位检测模式（返回空缺陷），保证 GUI 可演示。
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from ..domain.defect_type_registry import DefectTypeRegistry
from ..domain.models import DefectItem, DetectionResult
from ..infrastructure.exceptions import InferenceError, ModelNotFoundError

logger = logging.getLogger("glass_bottle_gui")


class DetectionService:
    """YOLO 检测服务，封装推理与结果转换。

    Attributes:
        _model: YOLO 模型实例，权重缺失时为 None（降级模式）
        _registry: 缺陷类型注册表
        _lock: 线程锁，串行化推理
        _class_names: 模型类别名列表（索引→类别键）
    """

    def __init__(self, model_path: str, registry: DefectTypeRegistry) -> None:
        """初始化检测服务，加载模型权重。

        Args:
            model_path: 模型权重文件路径（best.pt）
            registry: 缺陷类型注册表实例
        """
        self._registry = registry
        self._lock = threading.Lock()
        self._model = None
        self._class_names: list[str] = []
        self._degraded = False

        self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        """加载模型权重，缺失时降级为占位检测模式。

        Args:
            model_path: 权重文件路径
        """
        path = Path(model_path)
        if not path.is_file():
            logger.warning(f"模型权重不存在: {model_path}，进入降级占位检测模式")
            self._degraded = True
            return

        try:
            from ultralytics import YOLO

            self._model = YOLO(str(path))
            # 获取模型类别名（ultralytics 内部 names 字典，索引→类别名）
            if hasattr(self._model, "names") and self._model.names:
                self._class_names = [
                    self._model.names[i] for i in sorted(self._model.names.keys())
                ]
            logger.info(f"模型加载成功: {model_path}，类别: {self._class_names}")
        except Exception as e:
            logger.warning(f"模型加载失败: {e}，进入降级占位检测模式")
            self._degraded = True

    def detect(
        self,
        image: np.ndarray,
        source: str,
        image_name: str,
    ) -> DetectionResult:
        """执行缺陷检测，返回结构化检测结果。

        Args:
            image: BGR 格式的 numpy 数组（非空）
            source: 来源标识，"上传" 或 "摄像头"
            image_name: 图片名称或标识

        Returns:
            DetectionResult 检测结果对象

        Raises:
            InferenceError: 图片为空或推理失败
        """
        # 前置条件校验
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise InferenceError("输入图片为空或格式不正确")

        detect_time = datetime.now()

        # 降级模式：返回空缺陷结果
        if self._degraded or self._model is None:
            logger.debug("降级占位检测模式，返回空缺陷")
            return DetectionResult(
                detect_time=detect_time,
                source=source,
                image_name=image_name,
                defects=[],
            )

        # 线程安全推理
        with self._lock:
            try:
                results = self._model.predict(
                    image, verbose=False, conf=0.25, save=False
                )
            except Exception as e:
                logger.error(f"YOLO 推理失败: {e}")
                raise InferenceError(str(e)) from e

        defects = self._convert_results(results)
        logger.debug(f"检测完成: {image_name}，发现 {len(defects)} 个缺陷")

        return DetectionResult(
            detect_time=detect_time,
            source=source,
            image_name=image_name,
            defects=defects,
        )

    def _convert_results(self, results: list) -> list[DefectItem]:
        """将 YOLO 推理结果转换为 DefectItem 列表。

        Args:
            results: ultralytics predict 返回的结果列表

        Returns:
            DefectItem 列表
        """
        defects: list[DefectItem] = []
        if not results:
            return defects

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return defects

        # 获取类别名映射
        names = result.names if hasattr(result, "names") else {}

        for i in range(len(boxes)):
            # 类别索引与名称
            cls_id = int(boxes.cls[i].item())
            type_key = names.get(cls_id, str(cls_id))

            # 置信度
            confidence = float(boxes.conf[i].item())

            # 边界框 (x, y, w, h) 像素坐标
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

            # 颜色（从注册表查询）
            color = self._registry.get_color(type_key)

            try:
                item = DefectItem(
                    type_key=type_key,
                    confidence=round(confidence, 4),
                    bbox=bbox,
                    color=color,
                )
                defects.append(item)
            except ValueError as e:
                logger.warning(f"缺陷项构造失败，跳过: {e}")
                continue

        return defects

    @property
    def is_degraded(self) -> bool:
        """是否处于降级模式。"""
        return self._degraded