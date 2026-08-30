import sys
sys.path.insert(0, r'D:\dlgd-pharma')

import torch
import time
torch.set_num_threads(12)

from pathlib import Path
from torch.utils.data import DataLoader
from backend.models.unetpp import UNetPP, CombinedLoss
from train.train_unetpp import collect_samples, split_train_val, MVTecSegDataset

device = 'cpu'
img_size = 128
batch_size = 8
base_ch = 16

all_samples = collect_samples()
train_samples, val_samples = split_train_val(all_samples, val_ratio=0.2)

train_ds = MVTecSegDataset(train_samples, img_size, augment=True)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

model = UNetPP(in_ch=3, num_classes=4, base_ch=base_ch, deep_supervision=True)
model = model.to(device)
criterion = CombinedLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

params = sum(p.numel() for p in model.parameters())
print(f'参数量: {params/1e6:.2f}M, 图片尺寸: {img_size}, 批大小: {batch_size}')
print(f'每epoch步数: {len(train_loader)}')

print('开始训练1个epoch...')
start = time.time()
model.train()
total_loss = 0.0
steps = 0
for images, masks in train_loader:
    images, masks = images.to(device), masks.to(device)
    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, masks)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    steps += 1
    if steps % 5 == 0:
        elapsed = time.time() - start
        print(f'  Step {steps}/{len(train_loader)} | Loss: {total_loss/steps:.4f} | 用时: {elapsed:.1f}s')

elapsed = time.time() - start
print(f'\n1个epoch完成! 用时: {elapsed:.1f}s, 平均损失: {total_loss/steps:.4f}')
print(f'预计60轮总用时: {elapsed*60/60:.0f}分钟 = {elapsed*60/3600:.1f}小时')

output_dir = Path(r'D:\dlgd-pharma\train\weights')
output_dir.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), str(output_dir / 'unetpp_best.pth'))
print('权重已保存!')
