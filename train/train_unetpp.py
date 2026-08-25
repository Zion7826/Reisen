# -*- coding: utf-8 -*-
"""
UNet++ 分割模型训练脚本 (直接使用 MVTec AD bottle 数据集)

数据来源: D:\bottle\
    train/good/         -> 正常样本 (空掩码)
    test/good/          -> 正常样本 (空掩码)
    test/broken_large/  -> 大破损 + ground_truth/broken_large/ 掩码
    test/broken_small/  -> 小破损 + ground_truth/broken_small/ 掩码
    test/contamination/ -> 污染   + ground_truth/contamination/ 掩码

多轮渐进式训练: lr=1e-3 -> 1e-4 -> 1e-5
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

BOTTLE_ROOT = Path(r'D:\bottle')

CLASS_MAP = {
    'broken_large': 1,
    'broken_small': 2,
    'contamination': 3,
}


class MVTecSegDataset(Dataset):
    """MVTec AD 分割数据集"""

    def __init__(self, samples, img_size=256, augment=False):
        self.samples = samples
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, mask_value = self.samples[idx]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if mask_path is not None:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = np.where(mask > 127, mask_value, 0).astype(np.uint8)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if np.random.random() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            if np.random.random() > 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()
            k = np.random.randint(0, 4)
            if k > 0:
                image = np.rot90(image, k).copy()
                mask = np.rot90(mask, k).copy()

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask.astype(np.int64))

        return image, mask


def collect_samples():
    """收集所有训练样本 (img_path, mask_path, mask_value)"""
    samples = []

    for defect_type, mask_value in CLASS_MAP.items():
        img_dir = BOTTLE_ROOT / 'test' / defect_type
        gt_dir = BOTTLE_ROOT / 'ground_truth' / defect_type

        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.glob('*.png')):
            mask_path = gt_dir / f'{img_path.stem}_mask.png'
            if mask_path.exists():
                samples.append((img_path, mask_path, mask_value))
            else:
                samples.append((img_path, None, mask_value))

    for good_dir in [BOTTLE_ROOT / 'train' / 'good', BOTTLE_ROOT / 'test' / 'good']:
        if not good_dir.exists():
            continue
        for img_path in sorted(good_dir.glob('*.png')):
            samples.append((img_path, None, 0))

    return samples


def split_train_val(samples, val_ratio=0.2, seed=42):
    """分割训练集和验证集"""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(samples))
    val_count = int(len(samples) * val_ratio)
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]
    return [samples[i] for i in train_idx], [samples[i] for i in val_idx]


def train_one_round(model, train_loader, val_loader, epochs, lr, device, output_dir, round_name, best_loss=float('inf')):
    """训练一轮"""
    criterion = CombinedLoss(weight_ce=1.0, weight_dice=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f'\n--- {round_name}: epochs={epochs}, lr={lr} ---')

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        steps = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            steps += 1

        train_loss /= max(steps, 1)

        val_loss = 0.0
        steps = 0
        model.eval()
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item()
                steps += 1
        val_loss /= max(steps, 1)

        scheduler.step()

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            torch.save(model.state_dict(), str(output_dir / 'unetpp_best.pth'))

        if epoch % 5 == 0 or epoch == 1 or is_best:
            tag = ' *BEST*' if is_best else ''
            print(f'  Epoch {epoch:3d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}{tag}')

    torch.save(model.state_dict(), str(output_dir / 'unetpp_last.pth'))
    return best_loss


def main():
    print('=' * 60)
    print('UNet++ 分割模型训练 (MVTec AD bottle 数据集)')
    print('=' * 60)

    device = 'cpu'
    img_size = 128
    batch_size = 8
    base_ch = 16
    output_dir = Path(__file__).parent / 'weights'
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(12)

    print('\n[1/3] 收集数据样本')
    all_samples = collect_samples()
    train_samples, val_samples = split_train_val(all_samples, val_ratio=0.2)

    defect_count = sum(1 for s in train_samples if s[1] is not None)
    normal_count = sum(1 for s in train_samples if s[1] is None)
    print(f'  总样本: {len(all_samples)} (训练: {len(train_samples)}, 验证: {len(val_samples)})')
    print(f'  训练集: {defect_count} 缺陷 + {normal_count} 正常 = {len(train_samples)}')
    print(f'  验证集: {len(val_samples)}')

    train_ds = MVTecSegDataset(train_samples, img_size, augment=True)
    val_ds = MVTecSegDataset(val_samples, img_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    print('\n[2/3] 初始化模型')
    model = UNetPP(in_ch=3, num_classes=4, base_ch=base_ch, deep_supervision=True)
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'  参数量: {params / 1e6:.2f}M')

    print('\n[3/3] 多轮渐进式训练')
    best_loss = float('inf')

    best_loss = train_one_round(model, train_loader, val_loader,
                                epochs=30, lr=1e-3, device=device,
                                output_dir=output_dir, round_name='第1轮(粗训)',
                                best_loss=best_loss)

    best_loss = train_one_round(model, train_loader, val_loader,
                                epochs=20, lr=1e-4, device=device,
                                output_dir=output_dir, round_name='第2轮(精调)',
                                best_loss=best_loss)

    best_loss = train_one_round(model, train_loader, val_loader,
                                epochs=10, lr=1e-5, device=device,
                                output_dir=output_dir, round_name='第3轮(微调)',
                                best_loss=best_loss)

    print('\n' + '=' * 60)
    print(f'训练完成! 最佳验证损失: {best_loss:.4f}')
    print(f'权重保存至: {output_dir}')
    print(f'  - unetpp_best.pth (最佳)')
    print(f'  - unetpp_last.pth (最后一轮)')
    print('=' * 60)


if __name__ == '__main__':
    main()
