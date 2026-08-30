import torch
import numpy as np
import cv2
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

torch.set_num_threads(12)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
backbone.eval()
feature_layers = {'layer2': backbone.layer2, 'layer3': backbone.layer3}

mem = np.load(r'D:\dlgd-pharma\train\weights\patchcore\memory_bank_augmented.npy')
nn = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=-1)
nn.fit(mem)

def load_image(path, size=128):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0), img.shape[:2]

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
    return combined.permute(0, 2, 3, 1).reshape(H * W, C).numpy(), H, W

bottle_root = Path(r'D:\bottle\test')
categories = ['broken_large', 'broken_small', 'contamination', 'good']

print(f'{"类别":<15} {"图片":<10} {"分数":>7} {"面积占比":>8} {"强度":>6} {"长宽比":>6} {"充实度":>6} {"区域数":>6}')
print('-' * 80)

for cat in categories:
    test_dir = bottle_root / cat
    imgs = sorted(test_dir.glob('*.png'))[:3]
    
    for p in imgs:
        img_bgr = cv2.imread(str(p))
        orig_h, orig_w = img_bgr.shape[:2]
        
        tensor, _ = load_image(p)
        with torch.no_grad():
            test_feat, H, W = extract_features(tensor)
        
        distances, _ = nn.kneighbors(test_feat)
        dist_sq = distances.flatten() ** 2
        anomaly_score = float(np.max(dist_sq))
        
        amap = dist_sq.reshape(H, W)
        amap = amap - amap.min()
        if amap.max() > 0:
            amap = amap / amap.max()
        amap = (amap * 255).astype(np.uint8)
        amap = cv2.resize(amap, (orig_w, orig_h))
        
        threshold_map = amap > (amap.max() * 0.5)
        contours, _ = cv2.findContours(threshold_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid = [c for c in contours if cv2.contourArea(c) >= 50]
        total_area = sum(cv2.contourArea(c) for c in valid)
        total_ratio = total_area / (orig_h * orig_w)
        
        if valid:
            c = max(valid, key=cv2.contourArea)
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            area_ratio = area / (orig_h * orig_w)
            
            mask_r = np.zeros((orig_h, orig_w), dtype=np.uint8)
            cv2.drawContours(mask_r, [c], -1, 255, -1)
            intensity = float(np.mean(amap[mask_r > 0])) / 255.0
            
            aspect = max(w, h) / max(min(w, h), 1)
            hull = cv2.convexHull(c)
            solidity = area / max(cv2.contourArea(hull), 1)
            
            print(f'{cat:<15} {p.name:<10} {anomaly_score:>7.2f} {area_ratio:>8.4f} {intensity:>6.3f} {aspect:>6.2f} {solidity:>6.3f} {len(valid):>6}')
        else:
            print(f'{cat:<15} {p.name:<10} {anomaly_score:>7.2f} {"--":>8} {"--":>6} {"--":>6} {"--":>6} {0:>6}')