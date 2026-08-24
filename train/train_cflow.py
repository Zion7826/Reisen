# -*- coding: utf-8 -*-
"""
Cflow 异常检测模型训练脚本

数据来源: D:\bottle\train\good\ (MVTec AD 正常样本, 209张)
原理: 仅用正常样本训练，模型学习正常分布，推理时偏离即为异常
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

from backend.models.cflow import CflowAD


class NormalDataset(Dataset):
    """正常样本数据集 (仅正常图片，无标签)"""

    def __init__(self, images_dir, img_size=256, augment=True):
        self.images_dir = Path(images_dir)
        self.img_size = img_size
        self.augment = augment

        exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tif']
        self.samples = []
        for ext in exts:
            self.samples.extend(self.images_dir.glob(f'*{ext}'))
        self.samples = sorted(self.samples)

        print(f'  正常样本数据集: {len(self.samples)} 张 ({images_dir})')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        if self.augment:
            if np.random.random() > 0.5:
                image = np.fliplr(image).copy()
            if np.random.random() > 0.5:
                image = np.flipud(image).copy()
            angle = np.random.choice([0, 90, 180, 270])
            if angle > 0:
                M = cv2.getRotationMatrix2D((self.img_size / 2, self.img_size / 2), angle, 1)
                image = cv2.warpAffine(image, M, (self.img_size, self.img_size))

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        return image


def train_cflow(
    normal_dir,
    output_dir,
    epochs=50,
    batch_size=8,
    lr=1e-3,
    img_size=256,
    device='cpu',
):
    """
    训练 Cflow 异常检测模型

    Args:
        normal_dir: 正常样本图片目录
        output_dir: 权重输出目录
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        img_size: 输入尺寸
        device: cpu/cuda
    """
    print('=' * 60)
    print('Cflow 异常检测模型训练')
    print('=' * 60)

    print('\n[1/3] 构建数据集')
    dataset = NormalDataset(normal_dir, img_size=img_size, augment=True)

    if len(dataset) == 0:
        print('错误: 正常样本数据集为空!')
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print('\n[2/3] 初始化模型')
    model = CflowAD(flow_layers=4, hidden_ch=128)
    model = model.to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  可训练参数量: {params / 1e6:.2f}M')
    print(f'  设备: {device}, 轮数: {epochs}, 批大小: {batch_size}, 学习率: {lr}')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('\n[3/3] 开始训练')
    print('-' * 60)

    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0

        for images in loader:
            images = images.to(device)
            optimizer.zero_grad()

            _, nll = model(images)
            loss = nll.mean()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        epoch_loss /= max(steps, 1)
        scheduler.step()

        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss
            torch.save(model.state_dict(), str(output_dir / 'cflow_best.pth'))

        if epoch % 5 == 0 or epoch == 1 or is_best:
            tag = ' *BEST*' if is_best else ''
            print(f'  Epoch {epoch:3d}/{epochs} | NLL: {epoch_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}{tag}')

        if epoch % 20 == 0:
            torch.save(model.state_dict(), str(output_dir / f'cflow_epoch{epoch}.pth'))

    torch.save(model.state_dict(), str(output_dir / 'cflow_last.pth'))
    print('-' * 60)
    print(f'训练完成! 最佳损失: {best_loss:.4f}')
    print(f'权重保存至: {output_dir}')
    print('=' * 60)


if __name__ == '__main__':
    normal_dir = Path(r'D:\bottle\train\good')
    weights_dir = Path(__file__).parent / 'weights'

    train_cflow(
        normal_dir=normal_dir,
        output_dir=weights_dir,
        epochs=50,
        batch_size=8,
        lr=1e-3,
        img_size=256,
        device='cpu',
    )