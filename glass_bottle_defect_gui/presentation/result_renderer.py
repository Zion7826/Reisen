# -*- coding: utf-8 -*-
"""检测结果渲染器模块。

使用 OpenCV 在图像上绘制缺陷框与标签，并转换为 QPixmap 供 QLabel 展示。
支持原图与标注结果并列对比展示，以及无缺陷提示渲染。
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap

from ..domain.defect_type_registry import DefectTypeRegistry
from ..domain.models import DefectItem

logger = logging.getLogger("glass_bottle_gui")


class ResultRenderer:
    """检测结果渲染器，生成原图与标注图 QPixmap。"""

    def __init__(self, registry: DefectTypeRegistry) -> None:
        """初始化渲染器。

        Args:
            registry: 缺陷类型注册表，用于查询中文名
        """
        self._registry = registry

    def draw_boxes(
        self, image: np.ndarray, defects: list[DefectItem]
    ) -> np.ndarray:
        """在图像上绘制缺陷框与类型标签。

        Args:
            image: BGR 格式原图数组
            defects: 缺陷项列表

        Returns:
            标注后的 BGR 图像数组（副本，不修改原图）
        """
        annotated = image.copy()

        for defect in defects:
            x, y, w, h = defect.bbox
            color = defect.color
            chinese_name = self._registry.get_chinese_name(defect.type_key)
            label = f"{chinese_name} {defect.confidence:.2f}"

            # 绘制边界框
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # 绘制标签背景与文字
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated,
                (x, y - th - 6),
                (x + tw, y),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated

    def render_comparison(
        self,
        original: np.ndarray,
        defects: list[DefectItem],
        max_width: int = 480,
        max_height: int = 360,
    ) -> tuple[QPixmap, QPixmap]:
        """渲染原图与标注结果对比。

        Args:
            original: BGR 原图数组
            defects: 缺陷项列表
            max_width: 展示区最大宽度
            max_height: 展示区最大高度

        Returns:
            (原图 QPixmap, 标注结果 QPixmap)
        """
        original_pixmap = self._ndarray_to_qpixmap(original, max_width, max_height)

        if defects:
            annotated = self.draw_boxes(original, defects)
            result_pixmap = self._ndarray_to_qpixmap(
                annotated, max_width, max_height
            )
        else:
            result_pixmap = self.render_no_defect(original, max_width, max_height)

        return original_pixmap, result_pixmap

    def render_no_defect(
        self,
        original: np.ndarray,
        max_width: int = 480,
        max_height: int = 360,
    ) -> QPixmap:
        """渲染无缺陷提示结果图。

        Args:
            original: BGR 原图数组
            max_width: 展示区最大宽度
            max_height: 展示区最大高度

        Returns:
            带"未检测到缺陷"文字提示的 QPixmap
        """
        annotated = original.copy()
        h, w = annotated.shape[:2]

        text = "未检测到缺陷"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = (w - tw) // 2
        y = (h + th) // 2

        # 半透明背景
        overlay = annotated.copy()
        cv2.rectangle(
            overlay,
            (x - 10, y - th - 10),
            (x + tw + 10, y + 10),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
        cv2.putText(
            annotated, text, (x, y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA
        )

        return self._ndarray_to_qpixmap(annotated, max_width, max_height)

    @staticmethod
    def _ndarray_to_qpixmap(
        image: np.ndarray, max_width: int = 480, max_height: int = 360
    ) -> QPixmap:
        """将 BGR numpy 数组转换为 QPixmap，按比例缩放适配展示区。

        Args:
            image: BGR 格式图像数组
            max_width: 最大宽度
            max_height: 最大高度

        Returns:
            缩放后的 QPixmap
        """
        # BGR → RGB，并深拷贝避免 numpy 数组被 GC 后 QImage 数据失效
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        # np.copy() 确保数据独立，QImage 不会因原数组被回收而损坏
        q_image = QImage(rgb.copy(), w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # 按比例缩放
        if pixmap.width() > max_width or pixmap.height() > max_height:
            pixmap = pixmap.scaled(
                max_width, max_height, aspectRatioMode=1  # KeepAspectRatio
            )
        return pixmap