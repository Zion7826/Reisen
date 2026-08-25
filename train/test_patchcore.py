# -*- coding: utf-8 -*-
"""PatchCore 测试评估 - 使用 sklearn NearestNeighbors 加速"""

import sys
import time
import torch
import numpy as np
import cv2
from pathlib import Path
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from sklearn.neighbors import NearestNeighbors

torch.set_num_threads(12)

BOTTLE_ROOT = Path(r'D:\bottle')
MEMORY_PATH = Path(r'D:\dlgd-pharma\train\weights\patchcore\memory_bank.npy')

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

print('=' * 60)
print('PatchCore 测试评估')
print('=' * 60)

print('\n[1/3] 加载模型和记忆库')
backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
backbone.eval()
feature_layers = {'layer2': backbone.layer2, 'layer3': backbone.layer3}

memory_bank = np.load(str(MEMORY_PATH))
print(f'  记忆库: {memory_bank.shape[0]} patches, {memory_bank.shape[1]} 维')

print('  建立KD-Tree索引...')
nn = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=-1)
nn.fit(memory_bank)

print('\n[2/3] 测试评估')
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

            distances, _ = nn.kneighbors(test_feat)
            anomaly_score = np.max(distances ** 2)
            scores.append(anomaly_score)

    results[category] = {
        'count': len(scores),
        'mean': float(np.mean(scores)),
        'max': float(np.max(scores)),
        'min': float(np.min(scores)),
    }
    print(f'  {category:20s}: {len(scores):3d}张 | 均值={np.mean(scores):.2f} | 最大={np.max(scores):.2f} | 最小={np.min(scores):.2f}')

print('\n[3/3] 计算AUROC')
from sklearn.metrics import roc_auc_score

all_labels = []
all_scores = []
for category, info in results.items():
    test_dir = BOTTLE_ROOT / 'test' / category
    test_images = sorted(test_dir.glob('*.png'))
    with torch.no_grad():
        for img_path in test_images:
            img = load_image(img_path).unsqueeze(0)
            test_feat = extract_patch_features(backbone, img, feature_layers)
            distances, _ = nn.kneighbors(test_feat)
            anomaly_score = np.max(distances ** 2)
            all_scores.append(anomaly_score)
            all_labels.append(0 if category == 'good' else 1)

auroc = roc_auc_score(all_labels, all_scores)
print(f'\n  Image AUROC: {auroc:.4f} ({auroc*100:.1f}%)')

normal_mean = results.get('good', {}).get('mean', 0)
defect_mins = [v['min'] for k, v in results.items() if k != 'good']
if normal_mean and defect_mins:
    separation = min(defect_mins) - normal_mean
    print(f'  正常/缺陷分离度: {separation:.2f}')

print('\n' + '=' * 60)
print('评估完成!')
print('=' * 60)