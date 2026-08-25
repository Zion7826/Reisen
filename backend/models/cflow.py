# -*- coding: utf-8 -*-
"""
Cflow (Conditional Normalizing Flow) 异常检测模型
基于 EfficientNet-B3 特征提取 + 多尺度 Conditioned Flow

原理:
    1. 用预训练 CNN 提取多尺度特征
    2. 对每个特征层训练 Conditioned Flow (可逆变换)
    3. 正常样本在潜空间服从高斯分布，异常样本偏离
    4. 推理时计算负对数似然 (NLL) 作为异常分数 -> 热力图
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


class AffineCoupling(nn.Module):
    """
    仿射耦合层 (Affine Coupling Layer)
    将输入分成两半，一半不变，另一半做仿射变换
    """

    def __init__(self, in_ch, hidden_ch=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch // 2, hidden_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, in_ch, 3, padding=1),
        )
        self.in_ch = in_ch

    def forward(self, x, reverse=False):
        x1, x2 = x.chunk(2, dim=1)

        if not reverse:
            st = self.net(x1)
            s, t = st.chunk(2, dim=1)
            s = torch.tanh(s)
            y2 = x2 * torch.exp(s) + t
            log_det = s.sum(dim=[1, 2, 3])
            y = torch.cat([x1, y2], dim=1)
            return y, log_det
        else:
            st = self.net(x1)
            s, t = st.chunk(2, dim=1)
            s = torch.tanh(s)
            x2_rec = (x2 - t) * torch.exp(-s)
            log_det = -s.sum(dim=[1, 2, 3])
            x_rec = torch.cat([x1, x2_rec], dim=1)
            return x_rec, log_det


class CflowBlock(nn.Module):
    """单层 Cflow: 多个仿射耦合层堆叠"""

    def __init__(self, in_ch, num_layers=4, hidden_ch=128):
        super().__init__()
        self.layers = nn.ModuleList([
            AffineCoupling(in_ch, hidden_ch) for _ in range(num_layers)
        ])

    def forward(self, x):
        total_log_det = 0
        for layer in self.layers:
            x, log_det = layer(x, reverse=False)
            total_log_det = total_log_det + log_det
        return x, total_log_det

    def reverse(self, y):
        total_log_det = 0
        for layer in reversed(self.layers):
            y, log_det = layer(y, reverse=True)
            total_log_det = total_log_det + log_det
        return y, total_log_det


class FeatureExtractor(nn.Module):
    """EfficientNet-B3 多尺度特征提取器"""

    def __init__(self, out_indices=(4, 6, 7)):
        super().__init__()
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        backbone = efficientnet_b3(weights=weights)
        self.features = backbone.features
        self.out_indices = out_indices

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        feats = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_indices:
                feats.append(x)
        return feats


class CflowAD(nn.Module):
    """
    Cflow 异常检测模型

    结构:
        特征提取: EfficientNet-B3 (3个尺度)
        Flow: 每个尺度一个 CflowBlock
        输出: 每个像素的异常分数 (NLL)
    """

    def __init__(self, flow_layers=4, hidden_ch=128):
        super().__init__()
        self.extractor = FeatureExtractor(out_indices=(4, 6, 7))

        feature_channels = [96, 232, 384]

        self.flows = nn.ModuleList([
            CflowBlock(ch, num_layers=flow_layers, hidden_ch=hidden_ch)
            for ch in feature_channels
        ])

        self.feature_channels = feature_channels

    def forward(self, x):
        """
        前向传播: 计算异常分数

        Args:
            x: [B, 3, H, W] 输入图像

        Returns:
            anomaly_maps: [B, 1, H, W] 异常分数热力图
            total_nll: [B] 总负对数似然
        """
        feats = self.extractor(x)

        anomaly_maps = []
        total_nll = torch.zeros(x.shape[0], device=x.device)

        for i, (feat, flow) in enumerate(zip(feats, self.flows)):
            z, log_det = flow(feat)

            nll = 0.5 * (z ** 2).sum(dim=1) - log_det.unsqueeze(1).unsqueeze(2)
            nll_per_pixel = nll / (feat.shape[2] * feat.shape[3])

            anomaly_map = F.interpolate(
                nll_per_pixel.unsqueeze(1),
                size=x.shape[2:],
                mode='bilinear',
                align_corners=False
            )
            anomaly_maps.append(anomaly_map)
            total_nll = total_nll + nll.mean(dim=[1, 2])

        anomaly_map = torch.stack(anomaly_maps, dim=0).mean(dim=0)

        return anomaly_map, total_nll

    def get_anomaly_map(self, x):
        """获取归一化的异常热力图 [0, 1]"""
        with torch.no_grad():
            anomaly_map, _ = self.forward(x)
        anomaly_map = anomaly_map - anomaly_map.min()
        if anomaly_map.max() > 0:
            anomaly_map = anomaly_map / anomaly_map.max()
        return anomaly_map


if __name__ == '__main__':
    model = CflowAD(flow_layers=4, hidden_ch=128)
    x = torch.randn(2, 3, 256, 256)
    amap, nll = model(x)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'输入: {x.shape}')
    print(f'异常热力图: {amap.shape}')
    print(f'NLL: {nll.shape}')
    print(f'可训练参数量: {params / 1e6:.2f}M')