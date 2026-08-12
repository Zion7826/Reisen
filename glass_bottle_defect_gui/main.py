# -*- coding: utf-8 -*-
"""玻璃瓶缺陷检测系统 - 启动入口。

初始化各依赖并注入主窗口，支持命令行参数覆盖模型权重路径。
安装全局异常钩子，避免未处理异常导致 GUI 崩溃。

用法：
    python -m glass_bottle_defect_gui.main                    # 使用默认权重路径
    python -m glass_bottle_defect_gui.main --model path/to/best.pt  # 指定权重
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
import os

# 【关键修复】在 PyQt5 之前预加载 torch，避免 DLL 冲突
# PyQt5 的 Qt 库与 torch 的 c10.dll 存在初始化顺序冲突
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    import torch
    # 预初始化 torch CPU backend
    _ = torch.zeros(1)
    print(f"[INFO] torch 预加载成功: {torch.__version__}")
except Exception as e:
    print(f"[WARNING] torch 预加载失败: {e}")

from PyQt5.QtWidgets import QApplication

from . import config
from .controller.image_upload_controller import ImageUploadController
from .domain.defect_type_registry import DefectTypeRegistry
from .infrastructure.error_handler import ErrorHandler
from .infrastructure.logger import setup_logger
from .presentation.main_window import MainWindow
from .presentation.result_renderer import ResultRenderer
from .service.detection_service import DetectionService
from .service.report_exporter import ReportExporter


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="玻璃瓶缺陷检测系统")
    parser.add_argument(
        "--model",
        type=str,
        default=config.DEFAULT_MODEL_PATH,
        help="模型权重文件路径 (默认: %(default)s)",
    )
    return parser.parse_args()


def install_excepthook(logger: logging.Logger) -> None:
    """安装全局异常钩子，捕获未处理异常并记录日志。"""

    def excepthook(exc_type, exc_value, exc_tb):
        logger.error(f"未处理异常: {exc_value}")
        logger.debug("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))

    sys.excepthook = excepthook


def main() -> int:
    """应用启动入口。

    Returns:
        退出码
    """
    args = parse_args()

    # 初始化日志
    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("玻璃瓶缺陷检测系统启动")
    logger.info("=" * 60)

    install_excepthook(logger)

    # 创建 QApplication
    app = QApplication(sys.argv)

    # 初始化各依赖
    registry = DefectTypeRegistry()
    registry.load_from_yaml(config.DATA_YAML_PATH)
    logger.info(f"缺陷类型注册表加载完成: {registry.all_types()}")

    detection_service = DetectionService(args.model, registry)
    if detection_service.is_degraded:
        logger.warning("检测服务处于降级模式（模型权重未加载）")

    report_exporter = ReportExporter(registry)
    result_renderer = ResultRenderer(registry)
    image_upload_controller = ImageUploadController()
    error_handler = ErrorHandler()

    # 实例化主窗口并注入依赖
    window = MainWindow(
        detection_service=detection_service,
        report_exporter=report_exporter,
        result_renderer=result_renderer,
        image_upload_controller=image_upload_controller,
        error_handler=error_handler,
    )
    window.show()

    logger.info("主窗口已显示，进入事件循环")
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())