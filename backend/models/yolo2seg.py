# -*- coding: utf-8 -*-
"""
YOLO 标签转分割 PNG 工具
将 YOLO 格式的边界框标签转换为分割掩码 PNG 图像

输入: YOLO 格式标签 (class_id cx cy w h 归一化坐标)
输出: 单通道 PNG 掩码 (像素值=类别ID, 0=背景)
"""

import os
import cv2
import numpy as np
from pathlib import Path
import yaml


def yolo_bbox_to_mask(label_path, img_w, img_h, num_classes):
    """
    将单个 YOLO 标签文件转换为分割掩码

    Args:
        label_path: YOLO 标签文件路径
        img_w: 图片宽度
        img_h: 图片高度
        num_classes: 类别数

    Returns:
        mask: 单通道掩码 (uint8), 像素值=类别ID
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    if not os.path.exists(label_path):
        return mask

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)

            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = cls_id + 1

    return mask


def convert_dataset(images_dir, labels_dir, output_dir, num_classes=3):
    """
    批量转换 YOLO 数据集为分割掩码

    Args:
        images_dir: 图片目录
        labels_dir: YOLO 标签目录
        output_dir: 掩码输出目录
        num_classes: 类别数
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    count = 0

    for ext in extensions:
        for img_path in images_dir.glob(f'*{ext}'):
            stem = img_path.stem
            label_path = labels_dir / f'{stem}.txt'

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            mask = yolo_bbox_to_mask(str(label_path), w, h, num_classes)

            mask_path = output_dir / f'{stem}.png'
            cv2.imwrite(str(mask_path), mask)
            count += 1

    print(f'转换完成: {count} 张图片 -> {output_dir}')
    return count


def convert_from_mvtec(mvtec_root, output_dir, num_classes=3):
    """
    从 MVTec AD 格式直接生成分割掩码
    MVTec ground_truth 已有 mask_png 文件，直接复制并重映射像素值

    Args:
        mvtec_root: MVTec bottle 根目录 (含 test/ 和 ground_truth/)
        output_dir: 掩码输出目录
        num_classes: 类别数
    """
    mvtec_root = Path(mvtec_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_map = {
        'broken_large': 1,
        'broken_small': 2,
        'contamination': 3,
    }

    gt_dir = mvtec_root / 'ground_truth'
    test_dir = mvtec_root / 'test'

    count = 0
    for defect_type, mask_value in class_map.items():
        gt_type_dir = gt_dir / defect_type
        if not gt_type_dir.exists():
            continue

        for mask_file in sorted(gt_type_dir.glob('*_mask.png')):
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            result_mask = np.where(mask > 127, mask_value, 0).astype(np.uint8)

            stem = mask_file.stem.replace('_mask', '')
            out_path = output_dir / f'{stem}.png'
            cv2.imwrite(str(out_path), result_mask)
            count += 1

    print(f'MVTec 掩码转换完成: {count} 张 -> {output_dir}')
    return count


if __name__ == '__main__':
    dataset_root = Path(__file__).parent.parent.parent / 'demo' / 'datasets' / 'bottle_defect'

    seg_output = Path(__file__).parent.parent / 'seg_labels'

    print('=' * 60)
    print('YOLO 标签 -> 分割掩码 PNG 转换工具')
    print('=' * 60)

    train_count = convert_dataset(
        dataset_root / 'images' / 'train',
        dataset_root / 'labels' / 'train',
        seg_output / 'train',
        num_classes=3
    )

    val_count = convert_dataset(
        dataset_root / 'images' / 'val',
        dataset_root / 'labels' / 'val',
        seg_output / 'val',
        num_classes=3
    )

    print(f'\n训练集掩码: {train_count} 张')
    print(f'验证集掩码: {val_count} 张')
    print('=' * 60)