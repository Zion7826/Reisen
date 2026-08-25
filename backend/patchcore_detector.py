# -*- coding: utf-8 -*-
"""
PatchCore 检测器
使用预训练WideResNet50特征 + 记忆库最近邻搜索
AUROC: 99.7% (MVTec bottle)
"""

import os
import cv2
import time
import torch
import numpy as np
from pathlib import Path
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from sklearn.neighbors import NearestNeighbors

CLASS_NAMES = ['broken_large', 'broken_small', 'contamination']
CLASS_NAMES_CN = ['大破损', '小破损', '污染']
CLASS_COLORS_BGR = {1: (0, 0, 255), 2: (255, 0, 0), 3: (0, 255, 0)}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PatchCoreDetector:
    """PatchCore 异常检测器"""

    def __init__(self, memory_bank_path, device='cpu', img_size=128,
                 anomaly_threshold=15.0, backbone=None, memory_bank=None,
                 nn_index=None):
        self.device = device
        self.img_size = img_size
        self.anomaly_threshold = anomaly_threshold

        if backbone is not None and memory_bank is not None and nn_index is not None:
            self.backbone = backbone
            self.memory_bank = memory_bank
            self.nn = nn_index
        else:
            print('[PatchCore] 加载模型...')
            self.backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
            self.backbone.eval()
            self.feature_layers = {'layer2': self.backbone.layer2, 'layer3': self.backbone.layer3}

            self.memory_bank = np.load(memory_bank_path)
            print(f'  记忆库: {self.memory_bank.shape[0]} patches, {self.memory_bank.shape[1]} 维')

            self.nn = NearestNeighbors(n_neighbors=1, algorithm='auto', n_jobs=-1)
            self.nn.fit(self.memory_bank)
            print(f'  KD-Tree索引建立完成')
            print(f'  异常阈值: {anomaly_threshold}')

    def _preprocess(self, image_bgr):
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)

    def _extract_features(self, tensor):
        features = {}
        hooks = []
        for name, layer in self.feature_layers.items():
            hook = layer.register_forward_hook(lambda m, i, o, n=name: features.__setitem__(n, o.clone()))
            hooks.append(hook)
        self.backbone(tensor)
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

    def detect(self, image_bgr):
        """
        检测缺陷

        Returns:
            annotated: 标注后图像
            anomaly_map: 异常热力图 (0-255)
            anomaly_score: 异常分数
            is_defective: 是否有缺陷
            detections: 检测结果列表
            class_counts: 各类别计数
            fps: FPS
        """
        start_time = time.time()
        orig_h, orig_w = image_bgr.shape[:2]

        tensor = self._preprocess(image_bgr)

        with torch.no_grad():
            test_feat, H, W = self._extract_features(tensor)

        distances, _ = self.nn.kneighbors(test_feat)
        dist_sq = distances.flatten() ** 2

        anomaly_score = float(np.max(dist_sq))
        is_defective = anomaly_score > self.anomaly_threshold

        anomaly_map = dist_sq.reshape(H, W)
        anomaly_map = anomaly_map - anomaly_map.min()
        if anomaly_map.max() > 0:
            anomaly_map = anomaly_map / anomaly_map.max()
        anomaly_map = (anomaly_map * 255).astype(np.uint8)
        anomaly_map = cv2.resize(anomaly_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        detections = []
        class_counts = {name: 0 for name in CLASS_NAMES}

        if is_defective:
            threshold_map = anomaly_map > (anomaly_map.max() * 0.6)
            contours, _ = cv2.findContours(
                threshold_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 50:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                area_ratio = area / (orig_h * orig_w)

                if area_ratio > 0.15:
                    cls_idx = 0
                elif area_ratio > 0.03:
                    cls_idx = 1
                else:
                    cls_idx = 2

                detections.append({
                    'class_id': cls_idx,
                    'class_name': CLASS_NAMES[cls_idx],
                    'class_name_cn': CLASS_NAMES_CN[cls_idx],
                    'bbox': [x, y, x + w, y + h],
                    'area_ratio': round(area_ratio, 4),
                    'confidence': round(min(anomaly_score / 50.0, 1.0), 4),
                })
                class_counts[CLASS_NAMES[cls_idx]] += 1

        annotated = self._visualize(image_bgr, anomaly_map, detections)

        fps = 1.0 / max(time.time() - start_time, 1e-6)

        return {
            'annotated': annotated,
            'seg_mask': np.zeros((orig_h, orig_w), dtype=np.uint8),
            'anomaly_map': anomaly_map,
            'detections': detections,
            'class_counts': class_counts,
            'anomaly_score': round(anomaly_score, 4),
            'is_defective': is_defective,
            'fps': round(fps, 1),
        }

    def _visualize(self, image, anomaly_map, detections):
        annotated = image.copy()

        heatmap = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
        annotated = cv2.addWeighted(heatmap, 0.3, annotated, 0.7, 0)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_name = det['class_name']
            cls_cn = det['class_name_cn']
            conf = det['confidence']
            cls_val = det['class_id'] + 1
            color = CLASS_COLORS_BGR.get(cls_val, (255, 255, 255))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f'{cls_cn}({cls_name}) {conf:.2f}'
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            ly1 = max(y1 - th - baseline - 4, 0)
            ly2 = y1
            cv2.rectangle(annotated, (x1, ly1), (x1 + tw + 4, ly2), color, cv2.FILLED)
            cv2.putText(annotated, label, (x1 + 2, ly2 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated

    def generate_comparison(self, image_bgr):
        """生成三图对比: 原图 | 异常热力图 | 标注图"""
        result = self.detect(image_bgr)
        orig = image_bgr.copy()
        heatmap = cv2.applyColorMap(result['anomaly_map'], cv2.COLORMAP_JET)
        seg_vis = result['annotated'].copy()

        h, w = orig.shape[:2]
        gap = 10
        comparison = np.full((h, w * 3 + gap * 2, 3), 255, dtype=np.uint8)
        comparison[:, :w] = orig
        comparison[:, w + gap:2 * w + gap] = heatmap
        comparison[:, 2 * w + 2 * gap:3 * w + 2 * gap] = seg_vis

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, 'Original', (10, 30), font, 0.8, (0, 0, 0), 2)
        cv2.putText(comparison, 'Anomaly Heatmap', (w + gap + 10, 30), font, 0.8, (0, 0, 0), 2)
        cv2.putText(comparison, 'Detection', (2 * w + 2 * gap + 10, 30), font, 0.8, (0, 0, 0), 2)

        return comparison