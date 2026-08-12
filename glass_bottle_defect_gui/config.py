# -*- coding: utf-8 -*-
"""全局配置常量模块。

集中维护模型权重路径、摄像头参数、图片格式白名单、报告格式、日志路径等配置项，
供各层模块导入使用，避免硬编码散落各处。

支持 PyInstaller 冻结模式（sys.frozen）下的资源路径适配：
- 资源文件（模型权重、data.yaml）从 sys._MEIPASS 读取
- 运行时可写文件（日志）写入 exe 同级目录
"""

import sys
from pathlib import Path


# ============================================================
# 资源路径解析函数（适配 PyInstaller 冻结模式）
# ============================================================

def resolve_resource_root() -> Path:
    """返回资源文件根目录。

    冻结模式下返回 sys._MEIPASS（PyInstaller 临时解压目录），
    兜底返回 exe 同级目录；开发模式下返回项目根目录。
    """
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


def resolve_runtime_root() -> Path:
    """返回运行时可写根目录。

    冻结模式下返回 exe 同级目录（可写），
    开发模式下返回项目根目录。
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


def resolve_resource_path(relative_path: str) -> Path:
    """拼接资源根目录与相对路径，返回资源文件绝对路径。

    Args:
        relative_path: 相对于资源根的路径

    Returns:
        资源文件绝对路径
    """
    return resolve_resource_root() / relative_path


# ============================================================
# 路径常量
# ============================================================

# 项目根目录（开发模式下使用，保留兼容旧导入）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------- 模型相关 ----------
# 训练产出的最优权重默认路径（适配冻结模式）
DEFAULT_MODEL_PATH = str(resolve_resource_path("runs/detect/glass_bottle_defect/weights/best.pt"))
# 数据配置文件路径（缺陷类别权威来源，适配冻结模式）
DATA_YAML_PATH = str(resolve_resource_path("data.yaml"))

# ---------- 摄像头相关 ----------
# 默认摄像头设备索引（0 = 内置/首个 USB 摄像头）
DEFAULT_CAMERA_INDEX = 0
# 实时检测目标帧率（FPS）
TARGET_FPS = 10

# ---------- 图片格式 ----------
# 支持的图片扩展名白名单（小写，含点号）
IMAGE_FORMAT_WHITELIST = {".jpg", ".jpeg", ".png", ".bmp"}
# QFileDialog 过滤器文本
IMAGE_FILE_FILTER = "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*.*)"

# ---------- 报告导出 ----------
# 报告默认格式
DEFAULT_REPORT_FORMAT = "html"
# 报告文件过滤器
REPORT_FILE_FILTER = "HTML 报告 (*.html);;所有文件 (*.*)"

# ---------- 日志 ----------
# 运行时可写日志目录（冻结模式下为 exe 同级 logs/，开发模式下为项目根 logs/）
RUNTIME_LOG_DIR = str(resolve_runtime_root() / "logs")
# Deprecated: 请使用 RUNTIME_LOG_DIR（保留兼容旧导入）
LOG_DIR = RUNTIME_LOG_DIR
# 日志文件名
LOG_FILE = "gui.log"
