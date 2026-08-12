# -*- coding: utf-8 -*-
"""打包参数数据类 PackConfig。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class PackMode(Enum):
    """打包模式枚举。"""
    ONEDIR = "onedir"
    ONEFILE = "onefile"


@dataclass(frozen=True)
class PackConfig:
    """打包配置（不可变）。

    Attributes:
        entry_script: 入口脚本路径
        mode: 打包模式（onedir/onefile）
        name: 产物名称
        model_path: 模型权重路径
        data_yaml_path: 数据配置路径
        excludes: 排除项列表
        skip_verify: 跳过验证
        skip_compress: 跳过压缩
        spec_file_path: spec 文件路径
    """

    entry_script: Path
    mode: PackMode
    name: str
    model_path: Path
    data_yaml_path: Path
    excludes: list[str] = field(default_factory=list)
    skip_verify: bool = False
    skip_compress: bool = False
    spec_file_path: Path = field(default_factory=Path)

    @classmethod
    def default(cls) -> "PackConfig":
        """返回默认配置实例。"""
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        entry = project_root / "glass_bottle_defect_gui" / "main.py"
        model = project_root / "runs" / "detect" / "glass_bottle_defect" / "weights" / "best.pt"
        data_yaml = project_root / "data.yaml"
        spec_file = project_root / "glass_bottle_defect.spec"
        excludes = [
            "datasets", ".git", "__pycache__", "venv", ".venv",
            "*.pyc", "build", "dist", "tests", ".codeartsdoer",
        ]
        return cls(
            entry_script=entry,
            mode=PackMode.ONEDIR,
            name="glass_bottle_defect",
            model_path=model,
            data_yaml_path=data_yaml,
            excludes=excludes,
            spec_file_path=spec_file,
        )