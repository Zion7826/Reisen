# -*- coding: utf-8 -*-
"""打包流程专用异常类型。"""


class SpecGenerationError(Exception):
    """spec 文件生成失败异常。"""


class BuildFailureError(Exception):
    """打包构建失败异常。"""


class DistributionError(Exception):
    """分发组织失败异常。"""


class ValidationError(Exception):
    """产物验证失败异常。"""