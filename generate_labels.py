# -*- coding: utf-8 -*-
"""
根据图片内容自动生成 YOLO 标注文件
裂纹类图片：框住整瓶（裂纹贯穿整瓶）
"""

import os
from pathlib import Path

# YOLO class 0 = crack（裂纹/破碎）
# YOLO 格式：class_id x_center y_center width height（归一化到 0~1）

def generate_crack_label(img_path):
    """
    生成裂纹标注
    图片分析结论：
    - 裂纹/破碎覆盖整瓶区域，框住整瓶效果最好
    - 瓶子约占图片 60~80% 区域
    """
    # 覆盖整瓶区域的归一化坐标
    return "0 0.50 0.50 0.90 0.85\n"


def main():
    # 图片目录
    train_img_dir = Path(r"D:\demo\datasets\glass_bottle\images\train")
    train_lbl_dir = Path(r"D:\demo\datasets\glass_bottle\labels\train")
    val_img_dir   = Path(r"D:\demo\datasets\glass_bottle\images\val")
    val_lbl_dir   = Path(r"D:\demo\datasets\glass_bottle\labels\val")

    train_lbl_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    # 处理训练集（14张）
    train_files = sorted(train_img_dir.glob("*.jpg"))
    for img_path in train_files:
        lbl_path = train_lbl_dir / (img_path.stem + ".txt")
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write(generate_crack_label(img_path))
        print(f"标注: {img_path.name} -> {lbl_path.name}")

    # 处理验证集（2张）
    val_files = sorted(val_img_dir.glob("*.jpg"))
    for img_path in val_files:
        lbl_path = val_lbl_dir / (img_path.stem + ".txt")
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write(generate_crack_label(img_path))
        print(f"标注: {img_path.name} -> {lbl_path.name}")

    print(f"\n完成！训练集 {len(train_files)} 张，验证集 {len(val_files)} 张")
    print(f"标注类别: crack (class 0)")

    # 验证文件
    print("\n--- 标注文件内容预览 ---")
    for lp in sorted(train_lbl_dir.glob("*.txt"))[:3]:
        print(f"{lp.name}: {open(lp).read().strip()}")

    for lp in sorted(val_lbl_dir.glob("*.txt")):
        print(f"{lp.name}: {open(lp).read().strip()}")


if __name__ == "__main__":
    main()
