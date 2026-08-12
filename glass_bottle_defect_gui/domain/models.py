# -*- coding: utf-8 -*-
"""领域数据模型模块。

使用 frozen dataclass 实现不可变数据载体：
- DefectItem：单个缺陷项（类型键、置信度、边界框、颜色）
- DetectionResult：单次检测结果（时间、来源、图片名、缺陷列表）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .defect_type_registry import DefectTypeRegistry


@dataclass(frozen=True)
class DefectItem:
    """单个缺陷项（不可变）。

    Attributes:
        type_key: 缺陷类型键（英文，如 "crack"），对应 data.yaml 的 names 值
        confidence: 置信度，取值范围 [0.00, 1.00]
        bbox: 边界框 (x, y, w, h)，像素坐标四元组
        color: 标注颜色 BGR 三元组，如 (0, 0, 255) 表示红色
    """

    type_key: str
    confidence: float
    bbox: tuple[int, int, int, int]
    color: tuple[int, int, int]

    def __post_init__(self) -> None:
        """构造后校验字段约束。"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence 取值范围应为 [0.00, 1.00]，实际为 {self.confidence}"
            )
        if len(self.bbox) != 4:
            raise ValueError(f"bbox 应为 (x, y, w, h) 四元组，实际长度为 {len(self.bbox)}")
        if len(self.color) != 3:
            raise ValueError(f"color 应为 BGR 三元组，实际长度为 {len(self.color)}")

    def chinese_name(self, registry: "DefectTypeRegistry") -> str:
        """委托注册表查询中文名。

        Args:
            registry: 缺陷类型注册表实例

        Returns:
            缺陷类型的中文名称
        """
        return registry.get_chinese_name(self.type_key)


@dataclass(frozen=True)
class DetectionResult:
    """单次检测结果（不可变）。

    Attributes:
        detect_time: 检测时间
        source: 来源标识，仅允许 "上传" 或 "摄像头"
        image_name: 图片名称或标识
        defects: 缺陷项列表
        has_defect: 是否含缺陷，构造时自动计算
    """

    detect_time: datetime
    source: str
    image_name: str
    defects: list[DefectItem] = field(default_factory=list)
    has_defect: bool = field(default=False)

    def __post_init__(self) -> None:
        """构造后校验约束并自动计算 has_defect。"""
        if self.source not in ("上传", "摄像头"):
            raise ValueError(f"source 仅允许 '上传' 或 '摄像头'，实际为 '{self.source}'")
        # frozen dataclass 无法直接赋值，需通过 object.__setattr__ 绕过
        object.__setattr__(self, "has_defect", len(self.defects) > 0)

    def defect_count(self) -> int:
        """返回缺陷数量。"""
        return len(self.defects)

    def avg_confidence(self) -> float:
        """返回平均置信度，无缺陷时返回 0.0。"""
        if not self.defects:
            return 0.0
        return sum(d.confidence for d in self.defects) / len(self.defects)

    def max_confidence(self) -> float:
        """返回最高置信度，无缺陷时返回 0.0。"""
        if not self.defects:
            return 0.0
        return max(d.confidence for d in self.defects)