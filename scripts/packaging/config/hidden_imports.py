# -*- coding: utf-8 -*-
"""隐藏导入清单数据类 HiddenImports。"""

from __future__ import annotations

from dataclasses import dataclass, field


# ultralytics 内部动态导入子模块
ULTRALYTICS_HIDDEN = [
    "ultralytics",
    "ultralytics.nn",
    "ultralytics.nn.tasks",
    "ultralytics.utils",
    "ultralytics.utils.ops",
    "ultralytics.utils.plotting",
    "ultralytics.utils.checks",
    "ultralytics.utils.downloads",
    "ultralytics.models",
    "ultralytics.models.yolo",
    "ultralytics.models.yolo.detect",
    "ultralytics.engine",
    "ultralytics.engine.predictor",
    "ultralytics.engine.validator",
    "ultralytics.engine.trainer",
    "ultralytics.data",
    "ultralytics.data.augment",
    "ultralytics.data.dataloaders",
]

# torch 子模块
TORCH_HIDDEN = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.utils",
    "torch.utils.data",
    "torch.cuda",
    "torch._C",
    "torch.version",
]

# PyQt5 插件模块
PYQT5_HIDDEN = [
    "PyQt5",
    "PyQt5.QtCore",
    "PyQt5.QtGui",
    "PyQt5.QtWidgets",
    "PyQt5.sip",
]


@dataclass
class HiddenImports:
    """隐藏导入清单，按依赖包分组。"""

    ultralytics: list[str] = field(default_factory=lambda: list(ULTRALYTICS_HIDDEN))
    torch: list[str] = field(default_factory=lambda: list(TORCH_HIDDEN))
    pyqt5: list[str] = field(default_factory=lambda: list(PYQT5_HIDDEN))
    custom: list[str] = field(default_factory=list)

    def all(self) -> list[str]:
        """合并所有分组并去重后返回完整隐藏导入列表。"""
        combined = self.ultralytics + self.torch + self.pyqt5 + self.custom
        seen: set[str] = set()
        result: list[str] = []
        for item in combined:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result