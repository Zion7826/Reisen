# -*- coding: utf-8 -*-
"""构建结果数据类 BuildResult。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class BuildResult:
    """打包构建结果。

    Attributes:
        success: 是否成功
        artifact_path: 产物路径
        log_path: 构建日志路径
        build_time: 构建时间
        source_version: 源码版本（git commit hash）
        dependency_versions: 依赖版本字典
        error_message: 错误信息
    """

    success: bool
    artifact_path: Path = field(default_factory=Path)
    log_path: Path = field(default_factory=Path)
    build_time: datetime = field(default_factory=datetime.now)
    source_version: str = ""
    dependency_versions: dict[str, str] = field(default_factory=dict)
    error_message: str = ""