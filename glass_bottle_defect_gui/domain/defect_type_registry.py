# -*- coding: utf-8 -*-
"""缺陷类型注册表"""

from typing import Dict, Tuple

from glass_bottle_defect_gui.config import CLASS_NAMES, CLASS_NAMES_CN, CLASS_COLORS, DEFAULT_COLOR


class DefectTypeRegistry:
    """缺陷类型注册表，提供类别索引到名称、颜色的映射"""

    @staticmethod
    def get_class_name(class_id: int) -> str:
        """获取类别英文名称"""
        return CLASS_NAMES.get(class_id, f"class_{class_id}")

    @staticmethod
    def get_class_name_cn(class_id: int) -> str:
        """获取类别中文名称"""
        return CLASS_NAMES_CN.get(class_id, DefectTypeRegistry.get_class_name(class_id))

    @staticmethod
    def get_class_color(class_id: int) -> Tuple[int, int, int]:
        """获取类别颜色 (B, G, R)"""
        return CLASS_COLORS.get(class_id, DEFAULT_COLOR)

    @staticmethod
    def get_all_classes() -> Dict[int, str]:
        """获取所有类别"""
        return CLASS_NAMES.copy()

    @staticmethod
    def get_class_count() -> int:
        """获取类别数量"""
        return len(CLASS_NAMES)