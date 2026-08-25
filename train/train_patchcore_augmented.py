# -*- coding: utf-8 -*-
"""
PatchCore 训练 (使用增强后的627张图片)
对比原始209张 vs 增强627张
"""

import sys
import os
import time
import torch
import numpy as np
import cv2
from pathlib import Path
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

torch.set_num_threads(12)

BOTTLE_ROOT = Path(r'D:\bottle')
AUG_ROOT = Path(r'D:\dlgd-pharma\data\bottle_augmented')
OUTPUT_DIR = Path(r'D:\dlgd-pharma\train\weights\patchcore')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('=' * 60)
print('PatchCore 训练 (增强数据 627张)')
print('=' * 60)


def load_image(path, size=(128, 128)):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def extract_patch_features(model, images, layers):
    features = {}
    hooks = []
    for name, layer in layers.items():
        hook = layer.register_forward_hook(lambda m, i, o, n=name: features.__setitem__(n, o.clone()))
        hooks.append(hook)
    model(images)
    for h in hooks:
        h.remove()

    feat_list = []
    target_size = None
    for name in ['layer2', 'layer3']:
        feat = features[name]
        B, C, H, W = feat.shape
        if target_size is None:
            target_size = (H, W)
        else:
            feat = torch.nn.functional.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
        feat_list.append(feat)

    combined = torch.cat(feat_list, dim=1)
    B, C, H, W = combined.shape
    return combined.permute(0, 2, 3, 1).reshape(B * H * W, C).numpy()


def greedy_coreset_selection(features, sampling_ratio=0.1):
    N = features.shape[0]
    n_samples = max(1, int(N * sampling_ratio))
    if N <= n_samples:
        return np.arange(N)
    rng = np.random.RandomState(42)
    return rng.choice(N, n_samples, replace=False)


print('\n[1/4] 加载预训练 WideResNet50')
weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
backbone = wide_resnet50_2(weights=weights)
backbone.eval()

feature_layers = {
    'layer2': backbone.layer2,
    'layer3': backbone.layer3,
}
print('  WideResNet50 加载成功')

print('\n[2/4] 提取增强样本特征 (建立记忆库)')
normal_dir = AUG_ROOT / 'train' / 'good'
normal_images = sorted(normal_dir.glob('*.png'))
print(f'  增强样本: {len(normal_images)} 张')

all_patch_features = []
batch_size = 8

with torch.no_grad():
    for i in range(0, len(normal_images), batch_size):
        batch_paths = normal_images[i:i + batch_size]
        batch = torch.stack([load_image(p) for p in batch_paths])

        feats = extract_patch_features(backbone, batch, feature_layers)
        all_patch_features.append(feats)

        if (i + batch_size) % 80 == 0 or i + batch_size >= len(normal_images):
            print(f'  处理: {min(i + batch_size, len(normal_images))}/{len(normal_images)}')

all_features = np.concatenate(all_patch_features, axis=0)
print(f'  总patch特征数: {all_features.shape[0]}, 维度: {all_features.shape[1]}')

print('\n[3/4] Coreset 采样')
coreset_indices = greedy_coreset_selection(all_features, sampling_ratio=0.1)
memory_bank = all_features[coreset_indices]
print(f'  Coreset大小: {memory_bank.shape[0]} (采样率10%)')

memory_path = OUTPUT_DIR / 'memory_bank_augmented.npy'
np.save(str(memory_path), memory_bank)
print(f'  记忆库已保存: {memory_path}')

print('\n[4/4] 测试评估 (MVTec bottle test set)')
test_categories = ['broken_large', 'broken_small', 'contamination', 'good']
results = {}

for category in test_categories:
    test_dir = BOTTLE_ROOT / 'test' / category
    if not test_dir.exists():
        continue

    test_images = sorted(test_dir.glob('*.png'))
    scores = []

    with torch.no_grad():
        for img_path in test_images:
            img = load_image(img_path).unsqueeze(0)
            test_feat = extract_patch_features(backbone, img, feature_layers)

            chunk_size = 1000
            min_dists = np.full(len(test_feat), np.inf)
            for j in range(0, len(memory_bank), chunk_size):
                chunk = memory_bank[j:j + chunk_size]
                dist = np.min(np.sum((test_feat[:, None, :] - chunk[None, :, :]) ** 2, axis=2), axis=1)
                min_dists = np.minimum(min_dists, dist)
            anomaly_score = np.max(min_dists)
            scores.append(anomaly_score)

    results[category] = {
        'count': len(scores),
        'mean_score': float(np.mean(scores)),
        'max_score': float(np.max(scores)),
        'min_score': float(np.min(scores)),
    }
    print(f'  {category:20s}: {len(scores):3d}张 | 均值={np.mean(scores):.4f} | 最大={np.max(scores):.4f} | 最小={np.min(scores):.4f}')

normal_scores = results.get('good', {}).get('mean_score', 0)
defect_scores = {k: v for k, v in results.items() if k != 'good'}
if normal_scores and defect_scores:
    separation = min(v['min_score'] for v in defect_scores.values()) - normal_scores
    print(f'\n  正常/缺陷分离度: {separation:.4f} (越大越好)')

all_normal = results['good']['max_score']
all_defect = min(v['min_score'] for v in defect_scores.values())
if all_defect > all_normal:
    print(f'  AUROC = 100% (完美分离)')
else:
    print(f'  存在重叠，需计算AUROC')

print('\n' + '=' * 60)
print('PatchCore 增强训练完成!')
print(f'记忆库: {memory_path} ({memory_bank.shape[0]} patches)')
print('=' * 60)