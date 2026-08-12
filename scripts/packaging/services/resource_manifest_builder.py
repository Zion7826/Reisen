# -*- coding: utf-8 -*-
"""资源清单构建模块。

构建 PyInstaller datas 清单，处理模型权重与数据配置文件的打包，
实现资源缺失降级决策（必需资源缺失终止，可选资源缺失降级）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

from ..config.pack_config import PackConfig
from ..config.resource_entry import MissingBehavior, ResourceEntry, ResourceType

logger = logging.getLogger("packaging")


class ValidationIssue(NamedTuple):
    """资源校验问题。"""
    resource: str
    issue: str
    severity: str  # "warning" or "error"


class ResourceManifestBuilder:
    """资源清单构建器。"""

    def __init__(self, config: PackConfig) -> None:
        """初始化资源清单构建器。

        Args:
            config: 打包配置
        """
        self._config = config
        self._entries: list[ResourceEntry] = []
        self._build_entries()

    def _build_entries(self) -> None:
        """构建资源条目清单。"""
        # 模型权重（可选，缺失降级）
        self._entries.append(ResourceEntry(
            source=self._config.model_path,
            target_dir=".",
            file_type=ResourceType.MODEL_WEIGHT,
            required=False,
            missing_behavior=MissingBehavior.DEGRADE,
        ))
        # 数据配置（必需，缺失终止）
        self._entries.append(ResourceEntry(
            source=self._config.data_yaml_path,
            target_dir=".",
            file_type=ResourceType.DATA_CONFIG,
            required=True,
            missing_behavior=MissingBehavior.ABORT,
        ))

    def build(self) -> list[tuple[str, str]]:
        """构建 PyInstaller datas 清单。

        Returns:
            (源路径, 目标目录) 元组列表

        Raises:
            FileNotFoundError: 必需资源缺失时抛出
        """
        issues = self.validate()
        for issue in issues:
            if issue.severity == "error":
                raise FileNotFoundError(f"必需资源缺失: {issue.resource} - {issue.issue}")

        datas: list[tuple[str, str]] = []
        for entry in self._entries:
            if entry.exists():
                datas.append((str(entry.source), entry.target_dir))
                logger.info(f"资源加入打包: {entry.source} -> {entry.target_dir}")
            else:
                logger.warning(f"资源缺失已跳过: {entry.source}")

        return datas

    def validate(self) -> list[ValidationIssue]:
        """校验资源清单，返回问题列表。

        Returns:
            ValidationIssue 列表
        """
        issues: list[ValidationIssue] = []
        for entry in self._entries:
            if not entry.exists():
                if entry.missing_behavior == MissingBehavior.ABORT:
                    issues.append(ValidationIssue(
                        resource=str(entry.source),
                        issue="必需资源缺失，将终止打包",
                        severity="error",
                    ))
                else:
                    issues.append(ValidationIssue(
                        resource=str(entry.source),
                        issue="可选资源缺失，将降级处理",
                        severity="warning",
                    ))
        return issues