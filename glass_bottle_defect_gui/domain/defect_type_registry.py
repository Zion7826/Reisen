# -*- coding: utf-8 -*-
"""缺陷类型注册表模块。

集中维护缺陷类型的颜色映射与中文名映射，保证 GUI 展示与训练类别一致。
类别键基准从 data.yaml 的 names 字段加载，颜色与中文名作为 GUI 层补充映射。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("glass_bottle_gui")

# 默认颜色映射（BGR 格式，适配 OpenCV）
_DEFAULT_COLORS: dict[str, tuple[int, int, int]] = {
    "crack": (0, 0, 255),       # 红色
    "bubble": (255, 255, 0),    # 青色
    "scratch": (0, 255, 255),   # 黄色
    "stain": (0, 255, 0),       # 绿色
}

# 默认中文名映射
_DEFAULT_CHINESE_NAMES: dict[str, str] = {
    "crack": "裂纹",
    "bubble": "气泡",
    "scratch": "划痕",
    "stain": "污渍",
}

# 未知键兜底值
_DEFAULT_COLOR = (128, 128, 128)   # 灰色
_DEFAULT_CHINESE_NAME = "未知缺陷"


class DefectTypeRegistry:
    """缺陷类型注册表。

    集中管理缺陷类型的颜色与中文名，支持从 data.yaml 加载类别键基准。
    """

    def __init__(self) -> None:
        """初始化注册表，使用默认颜色与中文名映射。"""
        self._colors: dict[str, tuple[int, int, int]] = dict(_DEFAULT_COLORS)
        self._chinese_names: dict[str, str] = dict(_DEFAULT_CHINESE_NAMES)
        self._type_keys: list[str] = []

    def load_from_yaml(self, path: str | Path) -> None:
        """从 data.yaml 的 names 字段加载类别键基准。

        加载后会更新内部类型键列表，并保留已有的颜色/中文名映射；
        对于 yaml 中新增但默认映射未覆盖的键，使用兜底值。

        Args:
            path: data.yaml 文件路径
        """
        path = Path(path)
        if not path.is_file():
            logger.warning(f"data.yaml 不存在: {path}，使用默认类别键")
            self._type_keys = list(_DEFAULT_COLORS.keys())
            return

        # 简易 YAML 解析（仅解析 names 段的 "  index: name" 格式，避免引入 PyYAML 依赖）
        names: list[str] = []
        in_names = False
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("names:"):
                    in_names = True
                    continue
                if in_names:
                    if stripped and not line.startswith(" ") and not line.startswith("\t"):
                        in_names = False
                        continue
                    if ":" in stripped:
                        parts = stripped.split(":", 1)
                        value = parts[1].strip().split("#")[0].strip()
                        if value:
                            names.append(value)

        if names:
            self._type_keys = names
            logger.info(f"从 data.yaml 加载类别键: {names}")
        else:
            self._type_keys = list(_DEFAULT_COLORS.keys())
            logger.warning("data.yaml 未解析到 names 字段，使用默认类别键")

    def get_color(self, type_key: str) -> tuple[int, int, int]:
        """查询缺陷类型的标注颜色（BGR）。

        Args:
            type_key: 缺陷类型键

        Returns:
            BGR 三元组；未知键返回灰色并记录 WARNING
        """
        color = self._colors.get(type_key)
        if color is None:
            logger.warning(f"未知缺陷类型键: {type_key}，使用默认颜色")
            return _DEFAULT_COLOR
        return color

    def get_chinese_name(self, type_key: str) -> str:
        """查询缺陷类型的中文名称。

        Args:
            type_key: 缺陷类型键

        Returns:
            中文名称；未知键返回"未知缺陷"并记录 WARNING
        """
        name = self._chinese_names.get(type_key)
        if name is None:
            logger.warning(f"未知缺陷类型键: {type_key}，使用默认中文名")
            return _DEFAULT_CHINESE_NAME
        return name

    def all_types(self) -> list[str]:
        """返回所有已注册的缺陷类型键列表。"""
        return list(self._type_keys)