# -*- coding: utf-8 -*-
"""依赖收集与隐藏导入模块。

收集 PyQt5/ultralytics/torch 等依赖的隐藏导入与二进制文件，
并校验关键依赖是否已安装。
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

from ..config.hidden_imports import HiddenImports

logger = logging.getLogger("packaging")

# 关键依赖包名列表
_REQUIRED_PACKAGES = ["PyQt5", "ultralytics", "torch", "cv2", "numpy"]


class DependencyCollector:
    """依赖收集器，收集隐藏导入与 Qt 插件二进制。"""

    def __init__(self) -> None:
        """初始化依赖收集器。"""
        self._hidden = HiddenImports()

    def collect(self) -> tuple[list[str], list[tuple[str, str]]]:
        """收集隐藏导入与二进制依赖。

        Returns:
            (hiddenimports, binaries) 二元组
            hiddenimports: 隐藏导入模块名列表
            binaries: (源路径, 目标目录) 元组列表
        """
        self._verify_installed(_REQUIRED_PACKAGES)

        hiddenimports = self._hidden.all()
        binaries = self._collect_pyqt5_binaries()

        logger.info(f"收集隐藏导入 {len(hiddenimports)} 项，二进制 {len(binaries)} 项")
        return hiddenimports, binaries

    def _collect_pyqt5_binaries(self) -> list[tuple[str, str]]:
        """收集 PyQt5 Qt 插件二进制文件。

        Returns:
            (dll路径, 目标目录) 元组列表
        """
        binaries: list[tuple[str, str]] = []
        try:
            import PyQt5
            pyqt5_dir = Path(PyQt5.__file__).resolve().parent
        except (ImportError, AttributeError):
            logger.warning("无法定位 PyQt5 安装路径，跳过 Qt 插件收集")
            return binaries

        # Qt 插件子目录
        plugin_dirs = [
            ("plugins/platforms", "PyQt5/Qt5/plugins/platforms"),
            ("plugins/imageformats", "PyQt5/Qt5/plugins/imageformats"),
            ("plugins/styles", "PyQt5/Qt5/plugins/styles"),
        ]

        for src_subdir, dst_subdir in plugin_dirs:
            plugin_path = pyqt5_dir / "Qt5" / src_subdir
            if not plugin_path.is_dir():
                plugin_path = pyqt5_dir / src_subdir
            if not plugin_path.is_dir():
                continue
            for dll in plugin_path.glob("*.dll"):
                binaries.append((str(dll), dst_subdir))

        return binaries

    @staticmethod
    def _verify_installed(packages: list[str]) -> None:
        """校验关键依赖是否已安装。

        Args:
            packages: 包名列表

        Raises:
            EnvironmentError: 依赖缺失时抛出
        """
        missing: list[str] = []
        for pkg in packages:
            if importlib.util.find_spec(pkg) is None:
                missing.append(pkg)

        if missing:
            install_cmds = ", ".join(f"pip install {p}" for p in missing)
            raise EnvironmentError(
                f"缺少关键依赖: {', '.join(missing)}。请执行: {install_cmds}"
            )