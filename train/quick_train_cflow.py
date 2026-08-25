# -*- coding: utf-8 -*-
"""Cflow 短轮次训练 - 直接运行"""

import sys
sys.path.insert(0, r'D:\dlgd-pharma')

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

torch.set_num_threads(12)

from backend.models.cflow import CflowAD
from train.train_cflow import NormalDataset

device = 'cpu'
img_size = 128
batch_size = 16
output_dir = Path(r'D:\dlgd-pharma\train\weights')
output_dir.mkdir(parents=True, exist_ok=True)

normal_dir = Path(r'D:\bottle\train\good')
dataset = NormalDataset(normal_dir, img_size=img_size, augment=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

model = CflowAD(flow_layers=4, hidden_ch=128)
model = model.to(device)

best_path = output_dir / 'cflow_best.pth'
if best_path.exists():
    state = torch.load(str(best_path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f'已加载权重: {best_path}')

# 评估当前模型
model.eval()
with torch.no_grad():
    total_nll = 0.0
    steps = 0
    for images in loader:
        images = images.to(device)
        _, nll = model(images)
        total_nll += nll.mean().item()
        steps += 1
    current_loss = total_nll / steps
print(f'当前NLL: {current_loss:.4f}')
best_loss = current_loss

lr_schedule = [(8, 1e-3), (8, 5e-4), (4, 1e-4)]

for epochs, lr in lr_schedule:
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    print(f'\n--- epochs={epochs}, lr={lr} ---')

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
        epoch_loss /= steps
        scheduler.step()

        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss
            torch.save(model.state_dict(), str(output_dir / 'cflow_best.pth'))

        tag = ' *BEST*' if is_best else ''
        print(f'  Epoch {epoch:2d}/{epochs} | NLL: {epoch_loss:.4f}{tag}')

torch.save(model.state_dict(), str(output_dir / 'cflow_last.pth'))
print(f'\n训练完成! 最佳NLL: {best_loss:.4f}')