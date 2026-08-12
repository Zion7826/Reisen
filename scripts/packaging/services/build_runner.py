# -*- coding: utf-8 -*-
"""打包执行编排模块。

调用 PyInstaller 执行打包，管理构建日志与临时目录清理。
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from ..config.build_result import BuildResult
from ..config.pack_config import PackConfig, PackMode
from ..exceptions import BuildFailureError

logger = logging.getLogger("packaging")


class BuildRunner:
    """PyInstaller 打包执行器。"""

    def __init__(self, config: PackConfig) -> None:
        """初始化打包执行器。

        Args:
            config: 打包配置
        """
        self._config = config
        self._project_root = config.entry_script.resolve().parent.parent

    def run(self, spec_file: Path) -> BuildResult:
        """执行 PyInstaller 打包。

        Args:
            spec_file: spec 文件路径

        Returns:
            BuildResult 构建结果

        Raises:
            BuildFailureError: 构建失败
            OSError: 磁盘空间不足
        """
        # 清理残留目录
        self._clean_dirs()

        # 构建日志目录
        log_dir = self._project_root / "build" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"build_{timestamp}.log"

        # 调用 PyInstaller
        cmd = [sys.executable, "-m", "PyInstaller", str(spec_file), "--noconfirm"]
        logger.info(f"执行打包命令: {' '.join(cmd)}")

        build_time = datetime.now()
        source_version = self._get_source_version()

        try:
            with log_path.open("w", encoding="utf-8") as log_f:
                log_f.write(f"构建时间: {build_time}\n")
                log_f.write(f"源码版本: {source_version}\n")
                log_f.write(f"命令: {' '.join(cmd)}\n")
                log_f.write("=" * 60 + "\n\n")

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=str(self._project_root),
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )

                for line in process.stdout:
                    sys.stdout.write(line)
                    log_f.write(line)

                process.wait()

            if process.returncode != 0:
                error_msg = f"PyInstaller 构建失败，返回码 {process.returncode}，日志: {log_path}"
                logger.error(error_msg)
                raise BuildFailureError(error_msg)

        except OSError as e:
            if "No space left" in str(e) or "磁盘空间不足" in str(e):
                raise
            raise BuildFailureError(f"打包执行异常: {e}") from e

        # 确定产物路径
        if self._config.mode == PackMode.ONEDIR:
            artifact_path = self._project_root / "dist" / self._config.name
        else:
            artifact_path = self._project_root / "dist" / f"{self._config.name}.exe"

        logger.info(f"打包成功，产物: {artifact_path}")
        return BuildResult(
            success=True,
            artifact_path=artifact_path,
            log_path=log_path,
            build_time=build_time,
            source_version=source_version,
        )

    def _clean_dirs(self) -> None:
        """清理 build/ 与 dist/ 残留目录。"""
        for dir_name in ("build", "dist"):
            target = self._project_root / dir_name
            if target.exists():
                logger.info(f"清理残留目录: {target}")
                shutil.rmtree(target, ignore_errors=True)

    def _get_source_version(self) -> str:
        """获取源码版本（git commit hash）。

        Returns:
            commit hash 字符串，获取失败返回 "unknown"
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(self._project_root),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return "unknown"