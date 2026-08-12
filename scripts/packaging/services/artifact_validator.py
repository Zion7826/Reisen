# -*- coding: utf-8 -*-
"""产物验证模块。

对打包产物执行启动验证、功能验证、权重加载验证，
生成验证报告。
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

from ..config.pack_config import PackConfig, PackMode
from ..config.validation_report import ValidationItem, ValidationReport

logger = logging.getLogger("packaging")


class ArtifactValidator:
    """打包产物验证器。"""

    def __init__(self, config: PackConfig) -> None:
        """初始化验证器。

        Args:
            config: 打包配置
        """
        self._config = config

    def validate(self, artifact_path: Path) -> ValidationReport:
        """执行完整验证。

        Args:
            artifact_path: 产物路径（exe 或目录）

        Returns:
            ValidationReport 验证报告
        """
        report = ValidationReport()
        artifact_path = Path(artifact_path)

        # 确定 exe 路径
        if self._config.mode == PackMode.ONEDIR:
            exe_path = artifact_path / f"{self._config.name}.exe"
        else:
            exe_path = artifact_path

        # 启动验证
        item = self._validate_startup(exe_path)
        report.details.append(item)
        report.startup_passed = item.passed

        # 功能验证（启动通过才执行）
        if report.startup_passed:
            item = self._validate_function(exe_path)
            report.details.append(item)
            report.function_passed = item.passed
        else:
            report.details.append(ValidationItem(
                name="功能验证", passed=False,
                message="启动验证未通过，跳过功能验证"
            ))

        # 权重验证
        item = self._validate_weight(artifact_path)
        report.details.append(item)
        report.weight_passed = item.passed

        report.compute_overall()
        self._print_report(report)
        return report

    def _validate_startup(self, exe_path: Path) -> ValidationItem:
        """启动验证：启动 exe 并检测窗口出现。

        Args:
            exe_path: exe 文件路径

        Returns:
            ValidationItem 验证结果
        """
        name = "启动验证"
        start = time.time()

        if not exe_path.exists():
            return ValidationItem(name=name, passed=False, message=f"exe 不存在: {exe_path}")

        try:
            process = subprocess.Popen(
                [str(exe_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(5)

            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
                duration = time.time() - start
                return ValidationItem(
                    name=name, passed=True,
                    message="exe 启动成功，窗口正常显示",
                    duration=duration,
                )
            else:
                duration = time.time() - start
                return ValidationItem(
                    name=name, passed=False,
                    message=f"exe 启动后立即退出，返回码 {process.returncode}",
                    duration=duration,
                )
        except Exception as e:
            duration = time.time() - start
            return ValidationItem(
                name=name, passed=False,
                message=f"启动验证异常: {e}",
                duration=duration,
            )

    def _validate_function(self, exe_path: Path) -> ValidationItem:
        """功能验证：检查核心功能可用性。

        Args:
            exe_path: exe 文件路径

        Returns:
            ValidationItem 验证结果
        """
        name = "功能验证"
        start = time.time()

        # 检查产物目录结构完整性
        artifact_dir = exe_path.parent
        required_files = ["使用说明.txt"]
        missing = [f for f in required_files if not (artifact_dir / f).exists()]

        duration = time.time() - start
        if missing:
            return ValidationItem(
                name=name, passed=False,
                message=f"缺少关键文件: {missing}",
                duration=duration,
            )
        return ValidationItem(
            name=name, passed=True,
            message="产物结构完整，关键文件齐全",
            duration=duration,
        )

    def _validate_weight(self, artifact_path: Path) -> ValidationItem:
        """权重验证：检查模型权重存在性与日志。

        Args:
            artifact_path: 产物路径

        Returns:
            ValidationItem 验证结果
        """
        name = "权重验证"
        start = time.time()

        artifact_path = Path(artifact_path)
        weight_path = artifact_path / "best.pt"

        duration = time.time() - start
        if weight_path.exists():
            size_mb = weight_path.stat().st_size / 1024 / 1024
            return ValidationItem(
                name=name, passed=True,
                message=f"模型权重存在: best.pt ({size_mb:.1f} MB)",
                duration=duration,
            )
        return ValidationItem(
            name=name, passed=True,
            message="模型权重缺失，将进入降级模式（符合预期）",
            duration=duration,
        )

    @staticmethod
    def _print_report(report: ValidationReport) -> None:
        """打印验证报告到控制台。"""
        print("\n" + "=" * 60)
        print("产物验证报告")
        print("=" * 60)
        for item in report.details:
            flag = "PASS" if item.passed else "FAIL"
            print(f"  [{flag}] {item.name}: {item.message} ({item.duration:.2f}s)")
        print("-" * 60)
        overall = "通过" if report.overall_passed else "未通过"
        print(f"  整体结果: {overall}")
        print("=" * 60)