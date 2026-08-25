import torch
import numpy as np
import cv2
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

torch.set_num_threads(12)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

print('加载记忆库...')
mem_orig = np.load(r'D:\dlgd-pharma\train\weights\patchcore\memory_bank.npy')
mem_aug = np.load(r'D:\dlgd-pharma\train\weights\patchcore\memory_bank_augmented.npy')
print(f'  原始: {mem_orig.shape[0]} patches')
print(f'  增强: {mem_aug.shape[0]} patches')

print('加载WideResNet50...')
backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
backbone.eval()
feature_layers = {'layer2': backbone.layer2, 'layer3': backbone.layer3}

def load_image(path, size=128):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)

def extract_features(tensor):
    features = {}
    hooks = []
    for name, layer in feature_layers.items():
        hook = layer.register_forward_hook(lambda m, i, o, n=name: features.__setitem__(n, o.clone()))
        hooks.append(hook)
    backbone(tensor)
    for h in hooks:
        h.remove()
    feat_list = []
    target_size = None
    for name in ['layer2', 'layer3']:
        feat = features[name]
        if target_size is None:
            target_size = (feat.shape[2], feat.shape[3])
        else:
            feat = torch.nn.functional.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
        feat_list.append(feat)
    combined = torch.cat(feat_list, dim=1)
    B, C, H, W = combined.shape
    return combined.permute(0, 2, 3, 1).reshape(H * W, C).numpy()

print('建立KD-Tree索引...')
nn_orig = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=-1)
nn_orig.fit(mem_orig)
nn_aug = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=-1)
nn_aug.fit(mem_aug)

bottle_root = Path(r'D:\bottle\test')
categories = ['broken_large', 'broken_small', 'contamination', 'good']

print('\n评估结果:')
print(f'{"类别":20s} | {"原始均值":>10s} {"原始最小":>10s} | {"增强均值":>10s} {"增强最小":>10s}')
print('-' * 70)

all_scores_orig = []
all_scores_aug = []
all_labels = []

for cat in categories:
    test_dir = bottle_root / cat
    imgs = sorted(test_dir.glob('*.png'))
    label = 0 if cat == 'good' else 1
    
    scores_o, scores_a = [], []
    with torch.no_grad():
        for p in imgs:
            tensor = load_image(p)
            feat = extract_features(tensor)
            
            d_o, _ = nn_orig.kneighbors(feat)
            s_o = float(np.max(d_o.flatten() ** 2))
            scores_o.append(s_o)
            
            d_a, _ = nn_aug.kneighbors(feat)
            s_a = float(np.max(d_a.flatten() ** 2))
            scores_a.append(s_a)
    
    all_scores_orig.extend(scores_o)
    all_scores_aug.extend(scores_a)
    all_labels.extend([label] * len(imgs))
    
    print(f'{cat:20s} | {np.mean(scores_o):10.4f} {np.min(scores_o):10.4f} | {np.mean(scores_a):10.4f} {np.min(scores_a):10.4f}')

all_scores_orig = np.array(all_scores_orig)
all_scores_aug = np.array(all_scores_aug)
all_labels = np.array(all_labels)

def compute_auroc(scores, labels):
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    total = 0
    for p in pos:
        total += np.sum(neg < p) + 0.5 * np.sum(neg == p)
    return total / (n_pos * n_neg)

auroc_orig = compute_auroc(all_scores_orig, all_labels)
auroc_aug = compute_auroc(all_scores_aug, all_labels)

print(f'\n原始记忆库 AUROC: {auroc_orig*100:.1f}%')
print(f'增强记忆库 AUROC: {auroc_aug*100:.1f}%')
print(f'改善: {(auroc_aug - auroc_orig)*100:+.1f}%')

sep_orig = np.min(all_scores_orig[all_labels == 1]) - np.max(all_scores_orig[all_labels == 0])
sep_aug = np.min(all_scores_aug[all_labels == 1]) - np.max(all_scores_aug[all_labels == 0])
print(f'\n原始分离度: {sep_orig:.4f}')
print(f'增强分离度: {sep_aug:.4f}')