# -*- coding: utf-8 -*-
"""spec 配置生成模块。

生成 PyInstaller .spec 文件，注入隐藏导入、二进制、数据文件、排除项等参数。
支持 onedir 与 onefile 双模式。
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..config.pack_config import PackConfig, PackMode
from ..exceptions import SpecGenerationError
from .dependency_collector import DependencyCollector
from .resource_manifest_builder import ResourceManifestBuilder

logger = logging.getLogger("packaging")


class SpecGenerator:
    """PyInstaller spec 文件生成器。"""

    def __init__(self, config: PackConfig) -> None:
        """初始化 spec 生成器。

        Args:
            config: 打包配置
        """
        self._config = config

    def generate(self) -> Path:
        """生成 .spec 文件并返回路径。

        Returns:
            生成的 spec 文件路径

        Raises:
            FileNotFoundError: 入口文件不存在
            SpecGenerationError: 渲染失败
        """
        if not self._config.entry_script.is_file():
            raise FileNotFoundError(f"入口文件不存在: {self._config.entry_script}")

        # 收集依赖与资源
        collector = DependencyCollector()
        hiddenimports, binaries = collector.collect()

        manifest_builder = ResourceManifestBuilder(self._config)
        datas = manifest_builder.build()

        # 渲染 spec 内容
        try:
            content = self._render_spec(hiddenimports, binaries, datas)
        except Exception as e:
            raise SpecGenerationError(f"spec 文件渲染失败: {e}") from e

        # 写入 spec 文件
        spec_path = self._config.spec_file_path
        spec_path.write_text(content, encoding="utf-8")
        logger.info(f"spec 文件已生成: {spec_path}")

        return spec_path

    def _render_spec(
        self,
        hiddenimports: list[str],
        binaries: list[tuple[str, str]],
        datas: list[tuple[str, str]],
    ) -> str:
        """渲染 spec 文件内容。

        Args:
            hiddenimports: 隐藏导入列表
            binaries: 二进制元组列表
            datas: 数据文件元组列表

        Returns:
            spec 文件内容字符串
        """
        entry = str(self._config.entry_script).replace("\\", "/")
        name = self._config.name
        mode = self._config.mode

        # 格式化 Python 列表
        def fmt_list(items: list) -> str:
            if not items:
                return "[]"
            return "[\n        " + ",\n        ".join(repr(i) for i in items) + ",\n    ]"

        def fmt_tuples(items: list[tuple[str, str]]) -> str:
            if not items:
                return "[]"
            return "[\n        " + ",\n        ".join(
                f"({repr(s)}, {repr(d)})" for s, d in items
            ) + ",\n    ]"

        hidden_str = fmt_list(hiddenimports)
        binaries_str = fmt_tuples(binaries)
        datas_str = fmt_tuples(datas)
        excludes_str = fmt_list(self._config.excludes)

        # Analysis 段（共用）
        analysis_block = f"""# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec 文件 - 由 scripts/packaging 自动生成

block_cipher = None

a = Analysis(
    [{repr(entry)}],
    pathex=[],
    binaries={binaries_str},
    datas={datas_str},
    hiddenimports={hidden_str},
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes={excludes_str},
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)"""

        if mode == PackMode.ONEDIR:
            return f"""{analysis_block}

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name={repr(name)},
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name={repr(name)},
)
"""
        else:  # ONEFILE
            return f"""{analysis_block}

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name={repr(name)},
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
"""