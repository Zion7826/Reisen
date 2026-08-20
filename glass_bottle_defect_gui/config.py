# -*- coding: utf-8 -*-
"""全局配置"""

import os
from pathlib import Path


# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 模型路径
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "glass_bottle_defect" / "weights" / "best.pt"

# 缺陷类别定义（与 data.yaml 保持一致）
CLASS_NAMES = {
    0: "crack",    # 裂纹
    1: "bubble",   # 气泡
    2: "scratch",  # 划痕
    3: "stain",    # 污渍
}

CLASS_NAMES_CN = {
    0: "裂纹",
    1: "气泡",
    2: "划痕",
    3: "污渍",
}

# 缺陷类别颜色 (B, G, R)
CLASS_COLORS = {
    0: (0, 0, 255),      # 裂纹 crack   - 红色
    1: (0, 255, 255),    # 气泡 bubble  - 黄色
    2: (255, 0, 0),      # 划痕 scratch - 蓝色
    3: (0, 255, 0),      # 污渍 stain   - 绿色
}

DEFAULT_COLOR = (255, 255, 255)  # 白色

# 检测参数
DEFAULT_CONF_THRESHOLD = 0.25

# 结果保存目录
RESULTS_DIR = PROJECT_ROOT / "detection_results"
RESULTS_DIR.mkdir(exist_ok=True)