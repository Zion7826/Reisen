# -*- coding: utf-8 -*-
"""
PatchCore 记忆库训练（自定义玻璃制品数据）
只需正常样本图片，输出新的记忆库 npy，不覆盖原有 bottle 记忆库
用法:
    python train/train_patchcore_custom.py --input data/glass_normal --output train/weights/patchcore/memory_bank_custom.npy
"""

import argparse
import os
import tempfile
import torch
import numpy as np
import cv2
from pathlib import Path
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

torch.set_num_threads(12)

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def load_raw(path):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'无法读取图片: {path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def augment_image(img):
    h, w = img.shape[:2]
    out = cv2.convertScaleAbs(img, alpha=np.random.uniform(0.7, 1.3), beta=np.random.uniform(-25, 25))
    if np.random.rand() < 0.3:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    if np.random.rand() < 0.5:
        out = cv2.flip(out, 1)
    scale = np.random.uniform(0.8, 1.0)
    nw, nh = max(8, int(w * scale)), max(8, int(h * scale))
    ox = np.random.randint(0, w - nw + 1)
    oy = np.random.randint(0, h - nh + 1)
    out = out[oy:oy + nh, ox:ox + nw]
    return cv2.resize(out, (w, h))


def load_image(path, size=(128, 128), do_augment=False):
    img = load_raw(path)
    if do_augment:
        img = augment_image(img)
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
        if target_size is None:
            target_size = (feat.shape[2], feat.shape[3])
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


def main():
    parser = argparse.ArgumentParser(description='PatchCore 自定义记忆库训练')
    parser.add_argument('--input', required=True, help='正常样本图片目录')
    parser.add_argument('--output', default='train/weights/patchcore/memory_bank_custom.npy',
                        help='输出记忆库路径')
    parser.add_argument('--img-size', type=int, default=128, help='训练图像尺寸')
    parser.add_argument('--sampling-ratio', type=float, default=0.1, help='Coreset 采样率')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--augment', type=int, default=0,
                        help='每张图额外生成的增强副本数(亮度/翻转/裁剪扰动)')
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise SystemExit(f'目录不存在: {input_dir}')

    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file())
    if not images:
        raise SystemExit(f'目录中没有图片(jpg/png/bmp): {input_dir}')

    samples = [(p, False) for p in images]
    samples += [(p, True) for p in images for _ in range(args.augment)]
    total = len(samples)

    print('=' * 60)
    print(f'PatchCore 自定义记忆库训练')
    print(f'  正常样本: {len(images)} 张 (来源: {input_dir})')
    print(f'  增强副本: {args.augment} 份/张, 共处理 {total} 次')
    print(f'  图像尺寸: {args.img_size}x{args.img_size}')
    print('=' * 60)

    size = (args.img_size, args.img_size)
    backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    backbone.eval()
    feature_layers = {'layer2': backbone.layer2, 'layer3': backbone.layer3}
    print('  WideResNet50 加载成功')

    patches_per_image = (args.img_size // 8) ** 2
    total_patches = total * patches_per_image
    tmp_dir = Path(args.output).parent
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f'patchcore_feats_{os.getpid()}.npy'
    feat_mm = np.memmap(tmp_path, dtype=np.float16, mode='w+', shape=(total_patches, 1536))
    print(f'  特征缓冲(memmap): {total_patches} patches')

    write_pos = 0
    with torch.no_grad():
        for i in range(0, total, args.batch_size):
            batch_samples = samples[i:i + args.batch_size]
            batch = torch.stack([load_image(p, size, do_augment=aug) for p, aug in batch_samples])
            feats = extract_patch_features(backbone, batch, feature_layers)
            feat_mm[write_pos:write_pos + feats.shape[0]] = feats.astype(np.float16)
            write_pos += feats.shape[0]
            print(f'  处理: {min(i + args.batch_size, total)}/{total}', flush=True)

    print(f'  总patch特征数: {write_pos}, 维度: 1536')
    coreset_indices = greedy_coreset_selection(feat_mm[:write_pos], sampling_ratio=args.sampling_ratio)
    memory_bank = np.asarray(feat_mm[:write_pos][coreset_indices], dtype=np.float32)
    print(f'  Coreset大小: {memory_bank.shape[0]} (采样率{args.sampling_ratio:.0%})')

    del feat_mm
    tmp_path.unlink(missing_ok=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), memory_bank)
    print(f'  记忆库已保存: {out_path}')


if __name__ == '__main__':
    main()