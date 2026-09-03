# -*- coding: utf-8 -*-
"""
MVTec bottle 数据集转换为 YOLO 格式
将缺陷图片和mask标注转换为YOLO检测格式
"""

import os
import cv2
import numpy as np
import shutil
from pathlib import Path
import random


# 类别定义（CLASS_MAP 的键为 MVTec 源数据集的异常类型目录名，不可修改）
CLASS_MAP = {
    "broken_large": 0,    # MVTec源目录名，对应: 明显破损
    "broken_small": 1,    # MVTec源目录名，对应: 非明显破损
    "contamination": 2,   # MVTec源目录名，对应: 污染（数据中存在污染素材，保留）
}

CLASS_NAMES = {
    0: "obvious_damage",
    1: "subtle_damage",
    2: "contamination",
}


def mask_to_yolo_bbox(mask_path, class_id):
    """
    将mask转换为YOLO格式的边界框标注

    Args:
        mask_path: mask图片路径
        class_id: 类别ID

    Returns:
        YOLO格式的标注字符串列表
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return []

    h, w = mask.shape

    # 二值化mask
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 找到轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for contour in contours:
        if cv2.contourArea(contour) < 50:  # 过滤太小的区域
            continue

        # 获取边界框
        x, y, bw, bh = cv2.boundingRect(contour)

        # 转换为YOLO格式（归一化）
        center_x = (x + bw / 2) / w
        center_y = (y + bh / 2) / h
        norm_w = bw / w
        norm_h = bh / h

        bboxes.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")

    return bboxes


def prepare_dataset():
    """准备YOLO格式数据集"""

    # 源数据路径
    src_root = Path("D:/bottle")
    test_dir = src_root / "test"
    gt_dir = src_root / "ground_truth"

    # 目标数据集路径
    dst_root = Path("D:/demo/datasets/bottle_defect")
    dst_images_train = dst_root / "images" / "train"
    dst_images_val = dst_root / "images" / "val"
    dst_labels_train = dst_root / "labels" / "train"
    dst_labels_val = dst_root / "labels" / "val"

    # 创建目录
    for d in [dst_images_train, dst_images_val, dst_labels_train, dst_labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    all_data = []

    # 处理每个缺陷类别
    for defect_type, class_id in CLASS_MAP.items():
        print(f"处理类别: {defect_type} (class_id={class_id})")

        # 测试图片目录
        img_dir = test_dir / defect_type
        # mask目录
        mask_dir = gt_dir / defect_type

        if not img_dir.exists():
            print(f"  警告: 目录不存在 {img_dir}")
            continue

        # 遍历所有图片
        for img_file in sorted(img_dir.glob("*.png")):
            img_name = img_file.stem  # 例如 "000"
            mask_file = mask_dir / f"{img_name}_mask.png"

            if not mask_file.exists():
                print(f"  警告: mask不存在 {mask_file}")
                continue

            # 转换mask为YOLO标注
            bboxes = mask_to_yolo_bbox(str(mask_file), class_id)

            if not bboxes:
                print(f"  警告: 未找到有效标注 {img_file.name}")
                continue

            # 保存数据信息
            new_name = f"{defect_type}_{img_name}"
            all_data.append({
                "img_path": str(img_file),
                "new_name": new_name,
                "bboxes": bboxes
            })

            print(f"  处理: {img_file.name} -> {new_name} ({len(bboxes)}个标注)")

    print(f"\n总共处理 {len(all_data)} 张图片")

    # 随机打乱并分割训练集和验证集 (80%训练, 20%验证)
    random.seed(42)
    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    print(f"训练集: {len(train_data)} 张")
    print(f"验证集: {len(val_data)} 张")

    # 复制图片和生成标注文件
    def process_dataset(data_list, img_dst, label_dst, dataset_name):
        for item in data_list:
            # 复制图片
            img_src = item["img_path"]
            img_dst_path = img_dst / f"{item['new_name']}.png"
            shutil.copy2(img_src, img_dst_path)

            # 生成标注文件
            label_path = label_dst / f"{item['new_name']}.txt"
            with open(label_path, "w", encoding="utf-8") as f:
                for bbox in item["bboxes"]:
                    f.write(bbox + "\n")

        print(f"{dataset_name}: 已处理 {len(data_list)} 张图片")

    process_dataset(train_data, dst_images_train, dst_labels_train, "训练集")
    process_dataset(val_data, dst_images_val, dst_labels_val, "验证集")

    # 生成data.yaml
    yaml_content = f"""# 玻璃瓶缺陷检测数据集 (MVTec bottle)
path: {dst_root}
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names:
  0: obvious_damage  # 明显破损
  1: subtle_damage   # 非明显破损
  2: contamination   # 污染
"""

    yaml_path = dst_root / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\n数据集配置已保存至: {yaml_path}")
    print(f"数据集目录: {dst_root}")

    return dst_root


if __name__ == "__main__":
    print("=" * 60)
    print("MVTec bottle 数据集转换为 YOLO 格式")
    print("=" * 60)

    dataset_path = prepare_dataset()

    print("\n" + "=" * 60)
    print("数据集准备完成!")
    print("=" * 60)
    print(f"数据集路径: {dataset_path}")
    print(f"配置文件: {dataset_path / 'data.yaml'}")
    print("\n类别说明:")
    for class_id, name in CLASS_NAMES.items():
        print(f"  {class_id}: {name}")