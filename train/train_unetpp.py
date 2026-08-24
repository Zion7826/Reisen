# -*- coding: utf-8 -*-
"""
UNet++ 分割模型训练脚本

数据来源: D:\demo\datasets\bottle_defect\ (YOLO格式)
流程: yolo2seg 转换 -> 训练 UNet++ -> 保存权重
"""

import os
import sys
import cv2
import time
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.models.unetpp import UNetPP, CombinedLoss
from backend.models.yolo2seg import convert_dataset


class SegDataset(Dataset):
    """分割数据集: 图片 + 掩码"""

    def __init__(self, images_dir, masks_dir, img_size=256, augment=False):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        self.augment = augment

        exts = ['.jpg', '.jpeg', '.png', '.bmp']
        self.samples = []
        for ext in exts:
            for img_path in self.images_dir.glob(f'*{ext}'):
                mask_path = self.masks_dir / f'{img_path.stem}.png'
                if mask_path.exists():
                    self.samples.append((img_path, mask_path))

        print(f'  数据集: {len(self.samples)} 样本 ({images_dir})')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.augment and np.random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()
        if self.augment and np.random.random() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask.astype(np.int64))

        return image, mask


def train_unetpp(
    images_train, labels_train,
    images_val, labels_val,
    output_dir,
    epochs=100,
    batch_size=4,
    lr=1e-3,
    img_size=256,
    device='cpu',
):
    """
    训练 UNet++ 模型

    Args:
        images_train/val: 图片目录
        labels_train/val: YOLO标签目录
        output_dir: 权重输出目录
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        img_size: 输入尺寸
        device: cpu/cuda
    """
    print('=' * 60)
    print('UNet++ 分割模型训练')
    print('=' * 60)

    seg_train = Path(output_dir).parent / 'seg_labels' / 'train'
    seg_val = Path(output_dir).parent / 'seg_labels' / 'val'

    print('\n[1/4] 转换 YOLO 标签 -> 分割掩码')
    convert_dataset(images_train, labels_train, seg_train, num_classes=3)
    convert_dataset(images_val, labels_val, seg_val, num_classes=3)

    print('\n[2/4] 构建数据集')
    train_ds = SegDataset(images_train, seg_train, img_size, augment=True)
    val_ds = SegDataset(images_val, seg_val, img_size, augment=False)

    if len(train_ds) == 0:
        print('错误: 训练集为空!')
        return

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    print('\n[3/4] 初始化模型')
    model = UNetPP(in_ch=3, num_classes=4, base_ch=32, deep_supervision=True)
    model = model.to(device)
    criterion = CombinedLoss(weight_ce=1.0, weight_dice=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    params = sum(p.numel() for p in model.parameters())
    print(f'  参数量: {params / 1e6:.2f}M')
    print(f'  设备: {device}, 轮数: {epochs}, 批大小: {batch_size}, 学习率: {lr}')

    best_val_loss = float('inf')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('\n[4/4] 开始训练')
    print('-' * 60)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_steps = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_steps += 1

        train_loss /= max(train_steps, 1)

        val_loss = 0.0
        val_steps = 0
        model.eval()
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item()
                val_steps += 1
        val_loss /= max(val_steps, 1)

        scheduler.step()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(output_dir / 'unetpp_best.pth'))

        if epoch % 5 == 0 or epoch == 1 or is_best:
            tag = ' *BEST*' if is_best else ''
            print(f'  Epoch {epoch:3d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}{tag}')

        if epoch % 20 == 0:
            torch.save(model.state_dict(), str(output_dir / f'unetpp_epoch{epoch}.pth'))

    torch.save(model.state_dict(), str(output_dir / 'unetpp_last.pth'))
    print('-' * 60)
    print(f'训练完成! 最佳验证损失: {best_val_loss:.4f}')
    print(f'权重保存至: {output_dir}')
    print('=' * 60)


if __name__ == '__main__':
    dataset_root = Path(r'D:\demo\datasets\bottle_defect')
    weights_dir = Path(__file__).parent / 'weights'

    train_unetpp(
        images_train=dataset_root / 'images' / 'train',
        labels_train=dataset_root / 'labels' / 'train',
        images_val=dataset_root / 'images' / 'val',
        labels_val=dataset_root / 'labels' / 'val',
        output_dir=weights_dir,
        epochs=100,
        batch_size=4,
        lr=1e-3,
        img_size=256,
        device='cpu',
    )