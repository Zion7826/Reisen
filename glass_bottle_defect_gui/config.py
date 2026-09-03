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
    0: "obvious_damage",  # 明显破损
    1: "subtle_damage",   # 非明显破损
}

CLASS_NAMES_CN = {
    0: "明显破损",
    1: "非明显破损",
}

# 缺陷类别颜色 (B, G, R)
CLASS_COLORS = {
    0: (0, 0, 255),      # 明显破损 obvious_damage - 红色
    1: (0, 255, 255),    # 非明显破损 subtle_damage - 黄色
}

DEFAULT_COLOR = (255, 255, 255)  # 白色

# 检测参数
DEFAULT_CONF_THRESHOLD = 0.25

# 结果保存目录
RESULTS_DIR = PROJECT_ROOT / "detection_results"
RESULTS_DIR.mkdir(exist_ok=True)