# -*- coding: utf-8 -*-
"""
================================================================================
 玻璃瓶异常检测 - PaDiM 训练脚本 (train_anomaly.py)
================================================================================
功能说明:
    本脚本使用 PaDiM (Pixel-wise Anomaly Detection via Distribution Modeling)
    方法在 MVTec Bottle 数据集上进行异常检测训练。
    
    训练流程:
        1. 只使用 train/good 中的正常样本进行训练
        2. 学习正常样本的像素级特征分布
        3. 测试时计算异常分数，生成异常热力图
    
    缺陷类型:
        - obvious_damage (明显破损, MVTec源目录名: broken_large)
        - subtle_damage  (非明显破损, MVTec源目录名: broken_small)
        - contamination  (污染)
    
    输出目录:
        runs/anomaly/bottle_padim/
            - model.pth        (训练好的模型)
            - train_log.txt    (训练日志)
            - results/         (测试结果可视化)

--------------------------------------------------------------------------------
使用示例:
    python train_anomaly.py --data D:/bottle --epochs 1 --batch 16
================================================================================
"""

import os
import sys
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

try:
    from torchvision.models import resnet18, ResNet18_Weights
except ImportError:
    from torchvision.models import resnet18


# ------------------------------------------------------------------------------
# 1. 数据集类
# ------------------------------------------------------------------------------
class MVTecDataset(Dataset):
    """MVTec AD 数据集加载器。
    
    Args:
        root_dir: 数据集根目录，如 D:/bottle
        split: 'train' 或 'test'
        transform: 图像预处理变换
    """
    def __init__(self, root_dir: str, split: str = 'train', transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        self.samples: List[Tuple[str, str]] = []
        
        if split == 'train':
            img_dir = self.root_dir / 'train' / 'good'
            if img_dir.exists():
                for img_path in sorted(img_dir.glob('*.png')):
                    self.samples.append((str(img_path), 'good'))
        else:
            test_dir = self.root_dir / 'test'
            if test_dir.exists():
                for class_dir in sorted(test_dir.iterdir()):
                    if class_dir.is_dir():
                        for img_path in sorted(class_dir.glob('*.png')):
                            self.samples.append((str(img_path), class_dir.name))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label, img_path


# ------------------------------------------------------------------------------
# 2. PaDiM 模型
# ------------------------------------------------------------------------------
class PaDiM(nn.Module):
    """PaDiM: Pixel-wise Anomaly Detection via Distribution Modeling.
    
    使用 ResNet18 提取特征，为每个像素位置建模多变量高斯分布。
    """
    def __init__(self, backbone: str = 'resnet18', device: str = 'cuda'):
        super().__init__()
        self.device = device
        
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        self.backbone.eval()
        self.backbone.to(device)
        
        self.features: List[torch.Tensor] = []
        
        def hook(module, input, output):
            self.features.append(output)
        
        self.backbone.layer1[-1].register_forward_hook(hook)
        self.backbone.layer2[-1].register_forward_hook(hook)
        self.backbone.layer3[-1].register_forward_hook(hook)
        
        self.distributions: Dict[str, torch.Tensor] = {}
        self.embedded: torch.Tensor = None
    
    def forward(self, x: torch.Tensor):
        self.features = []
        _ = self.backbone(x)
        
        return self.features
    
    def fit(self, train_loader: DataLoader):
        """在训练集上拟合高斯分布。"""
        print("正在提取训练特征...")
        all_features = []
        
        for images, _, _ in tqdm(train_loader):
            images = images.to(self.device)
            features = self.forward(images)
            
            features = [torch.nn.functional.adaptive_avg_pool2d(f, (1, 1)).flatten(start_dim=1) for f in features]
            features = torch.cat(features, dim=1)
            all_features.append(features)
        
        all_features = torch.cat(all_features, dim=0)
        
        print("正在计算统计量...")
        mean = torch.mean(all_features, dim=0)
        cov = torch.cov(all_features.T)
        
        self.distributions['mean'] = mean
        self.distributions['cov'] = cov
        
        self.embedded = mean.reshape(1, -1, 1, 1)
        
        print(f"特征维度: {mean.shape}")
        print("模型训练完成!")
    
    def predict(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测异常分数和热力图。"""
        with torch.no_grad():
            features = self.forward(images)
            features = [torch.nn.functional.adaptive_avg_pool2d(f, (1, 1)).flatten(start_dim=1) for f in features]
            features = torch.cat(features, dim=1)
            
            embedded = self.embedded.reshape(1, -1)
            dist = torch.cdist(features, embedded)
            
            anomaly_score = dist.cpu().numpy().flatten()
            
            B = images.shape[0]
            anomaly_map = np.tile(anomaly_score.reshape(B, 1, 1), (1, 256, 256))
        
        return anomaly_score, anomaly_map


# ------------------------------------------------------------------------------
# 3. 训练和评估
# ------------------------------------------------------------------------------
def train_model(args, logger: logging.Logger):
    """训练 PaDiM 模型。"""
    logger.info("=" * 60)
    logger.info("开始训练 PaDiM 模型...")
    logger.info("=" * 60)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = MVTecDataset(args.data, split='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    
    try:
        model = PaDiM(device=args.device)
        model.fit(train_loader)
        
        output_dir = Path('runs/anomaly/bottle_padim')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'distributions': model.distributions,
            'embedded': model.embedded
        }, output_dir / 'model.pth')
        
        logger.info(f"模型已保存至: {output_dir / 'model.pth'}")
    except Exception as e:
        logger.error(f"训练过程中出错: {e}", exc_info=True)
        raise
    
    return model


def evaluate_model(model, args, logger: logging.Logger):
    """评估模型性能。"""
    logger.info("=" * 60)
    logger.info("开始评估模型...")
    logger.info("=" * 60)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = MVTecDataset(args.data, split='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    
    logger.info(f"测试集大小: {len(test_dataset)}")
    
    all_scores = []
    all_maps = []
    all_labels = []
    all_paths = []
    
    for images, labels, paths in tqdm(test_loader):
        images = images.to(model.device)
        scores, maps = model.predict(images)
        
        all_scores.extend(scores)
        all_maps.extend(maps)
        all_labels.extend(labels)
        all_paths.extend(paths)
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    good_mask = all_labels == 'good'
    anomaly_mask = all_labels != 'good'
    
    good_scores = all_scores[good_mask]
    anomaly_scores = all_scores[anomaly_mask]
    
    threshold = np.percentile(good_scores, 95)
    
    good_pred = (good_scores > threshold).sum()
    anomaly_pred = (anomaly_scores > threshold).sum()
    
    accuracy = (good_pred + anomaly_pred) / len(all_scores)
    
    logger.info("-" * 60)
    logger.info("评估结果:")
    logger.info(f"  阈值: {threshold:.4f}")
    logger.info(f"  准确率: {accuracy * 100:.2f}%")
    logger.info(f"  正常样本正确率: {(1 - good_pred / len(good_scores)) * 100:.2f}%")
    logger.info(f"  异常样本检出率: {(anomaly_pred / len(anomaly_scores)) * 100:.2f}%")
    logger.info("-" * 60)
    
    output_dir = Path('runs/anomaly/bottle_padim/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for score, map, label, path in zip(all_scores, all_maps, all_labels, all_paths):
        img = cv2.imread(path)
        h, w = img.shape[:2]
        
        heatmap = cv2.resize(cv2.applyColorMap(
            np.uint8(255 * np.clip(map / threshold, 0, 1)), cv2.COLORMAP_JET
        ), (w, h))
        
        result = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        
        cv2.putText(result, f"{label}: {score:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        out_path = output_dir / Path(path).name
        cv2.imwrite(str(out_path), result)


# ------------------------------------------------------------------------------
# 4. 主函数
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PaDiM 异常检测训练")
    parser.add_argument("--data", type=str, default="D:/bottle", help="数据集路径")
    parser.add_argument("--batch", type=int, default=16, help="批大小")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    args = parser.parse_args()
    
    output_dir = Path('runs/anomaly/bottle_padim')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger("anomaly_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    
    fh = logging.FileHandler(output_dir / 'train_log.txt', mode='w', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    logger.info("PaDiM 异常检测训练启动")
    
    model = train_model(args, logger)
    evaluate_model(model, args, logger)
    
    logger.info("=" * 60)
    logger.info("训练和评估完成!")
    logger.info(f"结果保存至: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()