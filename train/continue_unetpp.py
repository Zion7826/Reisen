# -*- coding: utf-8 -*-
"""
UNet++ 续训脚本 - 从已保存的权重继续训练
"""

import sys
sys.path.insert(0, r'D:\dlgd-pharma')

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

torch.set_num_threads(12)

from backend.models.unetpp import UNetPP, CombinedLoss
from train.train_unetpp import collect_samples, split_train_val, MVTecSegDataset, train_one_round

device = 'cpu'
img_size = 128
batch_size = 8
base_ch = 16
output_dir = Path(r'D:\dlgd-pharma\train\weights')

print('=' * 60)
print('UNet++ 续训 (从断点继续)')
print('=' * 60)

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
else:
    print('未找到权重，从头训练')

best_loss = float('inf')

best_loss = train_one_round(model, train_loader, val_loader,
                            epochs=20, lr=5e-4, device=device,
                            output_dir=output_dir, round_name='续训第1轮',
                            best_loss=best_loss)

best_loss = train_one_round(model, train_loader, val_loader,
                            epochs=15, lr=1e-4, device=device,
                            output_dir=output_dir, round_name='续训第2轮',
                            best_loss=best_loss)

best_loss = train_one_round(model, train_loader, val_loader,
                            epochs=10, lr=1e-5, device=device,
                            output_dir=output_dir, round_name='续训第3轮',
                            best_loss=best_loss)

print('\n' + '=' * 60)
print(f'续训完成! 最佳验证损失: {best_loss:.4f}')
print('=' * 60)