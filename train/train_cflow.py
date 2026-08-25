# -*- coding: utf-8 -*-
"""
Cflow 异常检测模型训练脚本 (多轮渐进式)

数据来源: D:\bottle\train\good\ (MVTec AD 正常样本, 209张)
原理: 仅用正常样本训练，模型学习正常分布，推理时偏离即为异常

多轮训练: lr=1e-3 -> 1e-4 -> 1e-5
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

BOTTLE_ROOT = Path(r'D:\bottle')


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
            k = np.random.randint(0, 4)
            if k > 0:
                image = np.rot90(image, k).copy()

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        return image


def train_one_round(model, loader, epochs, lr, device, output_dir, round_name, best_loss=float('inf')):
    """训练一轮"""
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f'\n--- {round_name}: epochs={epochs}, lr={lr} ---')

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
            print(f'  Epoch {epoch:3d}/{epochs} | NLL: {epoch_loss:.4f}{tag}')

    torch.save(model.state_dict(), str(output_dir / 'cflow_last.pth'))
    return best_loss


def main():
    print('=' * 60)
    print('Cflow 异常检测模型训练 (MVTec AD bottle 正常样本)')
    print('=' * 60)

    device = 'cpu'
    img_size = 128
    batch_size = 16
    output_dir = Path(__file__).parent / 'weights'
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(12)

    normal_dir = BOTTLE_ROOT / 'train' / 'good'

    print('\n[1/3] 构建数据集')
    dataset = NormalDataset(normal_dir, img_size=img_size, augment=True)

    if len(dataset) == 0:
        print('错误: 正常样本数据集为空!')
        return

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    print('\n[2/3] 初始化模型')
    model = CflowAD(flow_layers=4, hidden_ch=128)
    model = model.to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  可训练参数量: {params / 1e6:.2f}M')

    print('\n[3/3] 多轮渐进式训练')
    best_loss = float('inf')

    best_loss = train_one_round(model, loader,
                                epochs=25, lr=1e-3, device=device,
                                output_dir=output_dir, round_name='第1轮(粗训)',
                                best_loss=best_loss)

    best_loss = train_one_round(model, loader,
                                epochs=15, lr=1e-4, device=device,
                                output_dir=output_dir, round_name='第2轮(精调)',
                                best_loss=best_loss)

    best_loss = train_one_round(model, loader,
                                epochs=10, lr=1e-5, device=device,
                                output_dir=output_dir, round_name='第3轮(微调)',
                                best_loss=best_loss)

    print('\n' + '=' * 60)
    print(f'训练完成! 最佳损失: {best_loss:.4f}')
    print(f'权重保存至: {output_dir}')
    print(f'  - cflow_best.pth (最佳)')
    print(f'  - cflow_last.pth (最后一轮)')
    print('=' * 60)


if __name__ == '__main__':
    main()
