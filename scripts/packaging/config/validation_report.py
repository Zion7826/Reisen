# -*- coding: utf-8 -*-
"""验证结果数据类 ValidationReport 与 ValidationItem。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ValidationItem:
    """单项验证结果。

    Attributes:
        name: 验证项名称
        passed: 是否通过
        message: 详情消息
        duration: 耗时（秒）
    """

    name: str
    passed: bool
    message: str
    duration: float = 0.0


@dataclass
class ValidationReport:
    """验证报告。

    Attributes:
        startup_passed: 启动验证是否通过
        function_passed: 功能验证是否通过
        weight_passed: 权重验证是否通过
        details: 各验证项详情列表
        overall_passed: 整体是否通过
    """

    startup_passed: bool = False
    function_passed: bool = False
    weight_passed: bool = False
    details: list[ValidationItem] = field(default_factory=list)
    overall_passed: bool = False

    def compute_overall(self) -> None:
        """计算整体通过状态。"""
        self.overall_passed = (
            self.startup_passed and self.function_passed and self.weight_passed
        )