# -*- coding: utf-8 -*-
"""资源条目数据类 ResourceEntry。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ResourceType(Enum):
    """资源类型枚举。"""
    MODEL_WEIGHT = "model_weight"
    DATA_CONFIG = "data_config"
    ICON = "icon"
    README = "readme"


class MissingBehavior(Enum):
    """缺失行为枚举。"""
    ABORT = "abort"
    DEGRADE = "degrade"


@dataclass(frozen=True)
class ResourceEntry:
    """资源条目（不可变）。

    Attributes:
        source: 源文件路径
        target_dir: 目标目录（相对产物根）
        file_type: 资源类型
        required: 是否必需
        missing_behavior: 缺失时的行为
    """

    source: Path
    target_dir: str
    file_type: ResourceType
    required: bool
    missing_behavior: MissingBehavior

    def exists(self) -> bool:
        """判断源文件是否存在。"""
        return self.source.is_file()