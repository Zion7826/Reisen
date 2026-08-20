# -*- coding: utf-8 -*-
"""数据模型"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime


@dataclass
class DetectionResult:
    """单个检测结果"""
    class_id: int
    class_name: str
    class_name_cn: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)


@dataclass
class DetectionResponse:
    """检测响应"""
    success: bool
    image_path: Optional[str] = None
    detections: List[DetectionResult] = None
    class_counts: dict = None
    total_count: int = 0
    error_message: Optional[str] = None
    timestamp: str = None

    def __post_init__(self):
        if self.detections is None:
            self.detections = []
        if self.class_counts is None:
            self.class_counts = {}
        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")