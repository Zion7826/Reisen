# -*- coding: utf-8 -*-
"""绿色分发组织模块。

整理打包产物结构，生成使用说明，校验关键文件，压缩为绿色免安装分发包。
"""

from __future__ import annotations

import logging
import zipfile
from datetime import datetime
from pathlib import Path

from ..config.pack_config import PackConfig, PackMode
from ..exceptions import DistributionError

logger = logging.getLogger("packaging")

# 分发包中禁止包含的项
_EXCLUDE_PATTERNS = {".git", "build", "__pycache__", "venv", ".venv", ".codeartsdoer"}


class DistributionOrganizer:
    """绿色分发组织器。"""

    def __init__(self, config: PackConfig) -> None:
        """初始化分发组织器。

        Args:
            config: 打包配置
        """
        self._config = config
        self._project_root = config.entry_script.resolve().parent.parent

    def organize(self, artifact_dir: Path) -> Path:
        """整理产物并生成压缩分发包。

        Args:
            artifact_dir: 产物目录（onedir）或 exe 父目录（onefile）

        Returns:
            压缩包路径

        Raises:
            DistributionError: 校验失败
        """
        artifact_dir = Path(artifact_dir)

        # 确定产物根目录
        if self._config.mode == PackMode.ONEDIR:
            product_root = artifact_dir
        else:
            product_root = artifact_dir

        # 生成使用说明
        self._generate_readme(product_root)

        # 分发前校验
        self._validate_product(product_root)

        # 压缩分发包
        if not self._config.skip_compress:
            zip_path = self._compress(product_root)
            logger.info(f"分发包已生成: {zip_path}")
            return zip_path

        logger.info(f"跳过压缩，产物目录: {product_root}")
        return product_root

    def _generate_readme(self, product_root: Path) -> None:
        """生成使用说明文件。"""
        readme_path = product_root / "使用说明.txt"
        content = f"""玻璃瓶缺陷检测系统 - 使用说明
{'=' * 50}

【启动方式】
  双击 {self._config.name}.exe 即可启动程序。

【命令行参数】
  {self._config.name}.exe --model <权重路径>
  可通过 --model 参数指定自定义模型权重文件路径。

【降级模式】
  若模型权重 best.pt 缺失，程序将自动进入降级占位检测模式，
  GUI 仍可正常启动，但检测结果为空。

【运行环境】
  - Windows 10/11 64 位
  - 无需预装 Python 环境，解压即用

【文件说明】
  - {self._config.name}.exe    主程序
  - best.pt              模型权重（可选，缺失则降级）
  - data.yaml            数据配置（缺陷类别定义）
  - logs/                运行日志目录（自动创建）

【构建信息】
  构建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  产物名称: {self._config.name}
  打包模式: {self._config.mode.value}

【技术支持】
  如遇问题请查看 logs/gui.log 日志文件。
"""
        readme_path.write_text(content, encoding="utf-8")
        logger.info(f"使用说明已生成: {readme_path}")

    def _validate_product(self, product_root: Path) -> None:
        """校验产物关键文件存在性。

        Raises:
            DistributionError: 校验失败
        """
        # exe 文件
        exe_name = f"{self._config.name}.exe"
        exe_path = product_root / exe_name
        if not exe_path.exists() and self._config.mode == PackMode.ONEDIR:
            raise DistributionError(f"产物中缺少 exe 文件: {exe_path}")

        # 使用说明
        readme_path = product_root / "使用说明.txt"
        if not readme_path.exists():
            raise DistributionError(f"产物中缺少使用说明: {readme_path}")

        logger.info("分发前校验通过")

    def _compress(self, product_root: Path) -> Path:
        """压缩产物为 zip 分发包。

        Returns:
            压缩包路径
        """
        dist_dir = self._project_root / "dist"
        zip_path = dist_dir / f"{self._config.name}.zip"

        # 删除旧压缩包
        if zip_path.exists():
            zip_path.unlink()

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in product_root.rglob("*"):
                if file_path.is_file():
                    # 排除禁止项
                    rel_path = file_path.relative_to(product_root)
                    if any(p in rel_path.parts for p in _EXCLUDE_PATTERNS):
                        continue
                    # 排除 .pyc 文件
                    if file_path.suffix == ".pyc":
                        continue
                    arcname = str(rel_path)
                    zf.write(file_path, arcname)

        logger.info(f"压缩完成: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return zip_path