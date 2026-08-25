# -*- coding: utf-8 -*-
"""
推理检测器
加载 UNet++ 和 Cflow 模型，执行分割 + 异常检测，融合结果并可视化
"""

import os
import cv2
import time
import torch
import numpy as np
from pathlib import Path

from .preprocessing import (
    preprocess_for_segmentation,
    preprocess_for_anomaly,
    postprocess_mask,
    overlay_mask,
    clahe_enhance,
    denoise,
)
from .models.unetpp import UNetPP
from .models.cflow import CflowAD


CLASS_NAMES = ['broken_large', 'broken_small', 'contamination']
CLASS_NAMES_CN = ['大破损', '小破损', '污染']
CLASS_COLORS_BGR = {
    1: (0, 0, 255),
    2: (255, 0, 0),
    3: (0, 255, 0),
}


class DefectDetector:
    """
    缺陷检测器: UNet++ (分割) + Cflow (异常检测)
    """

    def __init__(self, unetpp_path=None, cflow_path=None, device='cpu',
                 img_size=256, anomaly_threshold=0.5):
        self.device = device
        self.img_size = img_size
        self.anomaly_threshold = anomaly_threshold

        self.unetpp = None
        self.cflow = None

        if unetpp_path and os.path.exists(unetpp_path):
            self.unetpp = UNetPP(in_ch=3, num_classes=4, base_ch=16, deep_supervision=True)
            state = torch.load(unetpp_path, map_location=device, weights_only=True)
            self.unetpp.load_state_dict(state)
            self.unetpp.to(device).eval()
            print(f'[UNet++] 加载成功: {unetpp_path}')
        else:
            print(f'[UNet++] 未加载 (路径: {unetpp_path})')

        if cflow_path and os.path.exists(cflow_path):
            self.cflow = CflowAD(flow_layers=4, hidden_ch=128)
            state = torch.load(cflow_path, map_location=device, weights_only=True)
            self.cflow.load_state_dict(state)
            self.cflow.to(device).eval()
            print(f'[Cflow] 加载成功: {cflow_path}')
        else:
            print(f'[Cflow] 未加载 (路径: {cflow_path})')

    def _to_device(self, tensor):
        return tensor.to(self.device)

    def detect(self, image_bgr):
        """
        执行完整检测流程

        Args:
            image_bgr: BGR 格式 numpy 数组

        Returns:
            result dict:
                annotated: 标注后图像
                seg_mask: 分割掩码
                anomaly_map: 异常热力图 (0-255)
                detections: 检测结果列表
                class_counts: 各类别计数
                anomaly_score: 总异常分数
                fps: 推理耗时
        """
        start_time = time.time()

        orig_h, orig_w = image_bgr.shape[:2]
        detections = []
        class_counts = {name: 0 for name in CLASS_NAMES}
        anomaly_score = 0.0

        seg_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        anomaly_map = np.zeros((orig_h, orig_w), dtype=np.uint8)

        if self.unetpp is not None:
            tensor, oh, ow = preprocess_for_segmentation(image_bgr, (self.img_size, self.img_size))
            tensor = self._to_device(tensor)

            with torch.no_grad():
                logits = self.unetpp(tensor)
                pred = torch.argmax(logits, dim=1)

            seg_mask = postprocess_mask(pred, oh, ow)

            for cls_val in range(1, 4):
                count = int((seg_mask == cls_val).sum())
                if count > 50:
                    cls_idx = cls_val - 1
                    cls_name = CLASS_NAMES[cls_idx]

                    ys, xs = np.where(seg_mask == cls_val)
                    if len(xs) > 0:
                        x1, y1 = int(xs.min()), int(ys.min())
                        x2, y2 = int(xs.max()), int(ys.max())
                        area_ratio = count / (orig_h * orig_w)
                        detections.append({
                            'class_id': cls_idx,
                            'class_name': cls_name,
                            'class_name_cn': CLASS_NAMES_CN[cls_idx],
                            'bbox': [x1, y1, x2, y2],
                            'area_ratio': round(area_ratio, 4),
                            'pixel_count': count,
                        })
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        if self.cflow is not None:
            tensor = preprocess_for_anomaly(image_bgr, (self.img_size, self.img_size))
            tensor = self._to_device(tensor)

            with torch.no_grad():
                amap = self.cflow.get_anomaly_map(tensor)

            amap_np = amap[0, 0].cpu().numpy()
            anomaly_map = (amap_np * 255).astype(np.uint8)
            anomaly_map = cv2.resize(anomaly_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            anomaly_score = float(amap_np.mean())

        annotated = self._visualize(image_bgr, seg_mask, anomaly_map, detections)

        fps = 1.0 / max(time.time() - start_time, 1e-6)

        return {
            'annotated': annotated,
            'seg_mask': seg_mask,
            'anomaly_map': anomaly_map,
            'detections': detections,
            'class_counts': class_counts,
            'anomaly_score': round(anomaly_score, 4),
            'is_defective': len(detections) > 0 or anomaly_score > self.anomaly_threshold,
            'fps': round(fps, 1),
        }

    def _visualize(self, image, seg_mask, anomaly_map, detections):
        """可视化: 边界框 + 分割叠加 + 标签"""
        annotated = image.copy()

        if seg_mask.max() > 0:
            annotated = overlay_mask(annotated, seg_mask, alpha=0.4, colors=CLASS_COLORS_BGR)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls_name = det['class_name']
            cls_cn = det['class_name_cn']
            cls_val = det['class_id'] + 1
            color = CLASS_COLORS_BGR.get(cls_val, (255, 255, 255))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f'{cls_cn}({cls_name})'
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            ly1 = max(y1 - th - baseline - 4, 0)
            ly2 = y1
            cv2.rectangle(annotated, (x1, ly1), (x1 + tw + 4, ly2), color, cv2.FILLED)
            cv2.putText(annotated, label, (x1 + 2, ly2 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated

    def generate_comparison(self, image_bgr):
        """
        生成三图对比: 原图 | 分割遮罩 | 异常热力图

        Returns:
            comparison: 拼接后的 BGR 图像
        """
        result = self.detect(image_bgr)
        orig = image_bgr.copy()
        seg_vis = result['annotated'].copy()
        anomaly_vis = self._anomaly_to_colormap(result['anomaly_map'])

        h, w = orig.shape[:2]
        gap = 10
        comparison = np.full((h, w * 3 + gap * 2, 3), 255, dtype=np.uint8)
        comparison[:, :w] = orig
        comparison[:, w + gap:2 * w + gap] = seg_vis
        comparison[:, 2 * w + 2 * gap:3 * w + 2 * gap] = anomaly_vis

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, 'Original', (10, 30), font, 0.8, (0, 0, 0), 2)
        cv2.putText(comparison, 'Segmentation', (w + gap + 10, 30), font, 0.8, (0, 0, 0), 2)
        cv2.putText(comparison, 'Anomaly Heatmap', (2 * w + 2 * gap + 10, 30), font, 0.8, (0, 0, 0), 2)

        return comparison

    def _anomaly_to_colormap(self, anomaly_map):
        """异常图转 JET 伪彩色"""
        colored = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
        return colored