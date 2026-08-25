# -*- coding: utf-8 -*-
"""UNet++ 短轮次训练 - 直接运行，加载已有权重继续"""

import sys
sys.path.insert(0, r'D:\dlgd-pharma')

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

torch.set_num_threads(12)

from backend.models.unetpp import UNetPP, CombinedLoss
from train.train_unetpp import collect_samples, split_train_val, MVTecSegDataset

device = 'cpu'
img_size = 128
batch_size = 8
base_ch = 16
output_dir = Path(r'D:\dlgd-pharma\train\weights')

all_samples = collect_samples()
train_samples, val_samples = split_train_val(all_samples, val_ratio=0.2)
train_ds = MVTecSegDataset(train_samples, img_size, augment=True)
val_ds = MVTecSegDataset(val_samples, img_size, augment=False)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

model = UNetPP(in_ch=3, num_classes=4, base_ch=base_ch, deep_supervision=True)
model = model.to(device)

best_path = output_dir / 'unetpp_best.pth'
if best_path.exists():
    state = torch.load(str(best_path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f'已加载权重: {best_path}')

criterion = CombinedLoss()

# 渐进式学习率
lr_schedule = [(5, 1e-4), (5, 5e-5)]
best_loss = float('inf')

# 评估当前模型
model.eval()
with torch.no_grad():
    val_loss = sum(criterion(model(imgs.to(device)), masks.to(device)).item()
                   for imgs, masks in val_loader) / len(val_loader)
print(f'当前验证损失: {val_loss:.4f}')
best_loss = val_loss

for epochs, lr in lr_schedule:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    print(f'\n--- epochs={epochs}, lr={lr} ---')

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
        train_loss /= steps

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
        val_loss /= steps
        scheduler.step()

        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            torch.save(model.state_dict(), str(output_dir / 'unetpp_best.pth'))

        tag = ' *BEST*' if is_best else ''
        print(f'  Epoch {epoch:2d}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}{tag}')

torch.save(model.state_dict(), str(output_dir / 'unetpp_last.pth'))
print(f'\n训练完成! 最佳验证损失: {best_loss:.4f}')