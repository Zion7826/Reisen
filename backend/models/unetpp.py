# -*- coding: utf-8 -*-
"""
UNet++ (Nested UNet) 分割模型
编码器: ResNet34 (简化版)
解码器: 嵌套跳跃连接
输出: 多类别分割掩码

UNet++ 结构 (i=深度, j=嵌套层级):
    X^{i,0} = 编码器第 i 层输出
    X^{i,j} = H(X^{i,0}, X^{i,1}, ..., X^{i,j-1}, up(X^{i+1,j-1}))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    """基础卷积块: Conv2d -> BatchNorm -> ReLU"""

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class ResBlock(nn.Module):
    """ResNet 残差块"""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = BasicConv(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class Encoder(nn.Module):
    """ResNet34 编码器 (简化版)

    输出 5 层特征:
        x0: [B, C0, H, W]      (stem)
        x1: [B, C1, H, W]      (enc1, 无stride)
        x2: [B, C2, H/2, W/2]  (enc2)
        x3: [B, C3, H/4, W/4]  (enc3)
        x4: [B, C4, H/8, W/8]  (enc4)
    """

    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()
        self.stem = BasicConv(in_ch, base_ch, kernel_size=3, padding=1)

        self.enc1 = nn.Sequential(ResBlock(base_ch, base_ch), ResBlock(base_ch, base_ch))
        self.enc2 = nn.Sequential(ResBlock(base_ch, base_ch * 2, stride=2), ResBlock(base_ch * 2, base_ch * 2))
        self.enc3 = nn.Sequential(ResBlock(base_ch * 2, base_ch * 4, stride=2), ResBlock(base_ch * 4, base_ch * 4))
        self.enc4 = nn.Sequential(ResBlock(base_ch * 4, base_ch * 8, stride=2), ResBlock(base_ch * 8, base_ch * 8))

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        return x0, x1, x2, x3, x4


class DecoderBlock(nn.Module):
    """UNet++ 解码节点: 两层卷积"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            BasicConv(in_ch, out_ch),
            BasicConv(out_ch, out_ch),
        )

    def forward(self, x):
        return self.conv(x)


def _upsample(x, size):
    """上采样到指定尺寸"""
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class UNetPP(nn.Module):
    """
    UNet++ 分割模型

    编码器: ResNet34 (5层特征, 通道数 [C, C, 2C, 4C, 8C])
    解码器: 4层嵌套跳跃连接
    输出: num_classes 通道

    嵌套结构:
        X^{3,1} = H(x3, up(x4))           → C3 通道
        X^{2,1} = H(x2, up(x3))           → C2 通道
        X^{1,1} = H(x1, up(x2))           → C1 通道
        X^{3,2} = H(x3, X^{3,1}, up(X^{2,1}))  → C3 通道
        X^{2,2} = H(x2, X^{2,1}, up(X^{3,1}))  → C2 通道
        X^{1,2} = H(x1, X^{1,1}, up(X^{2,1}))  → C1 通道
        X^{2,3} = H(x2, X^{2,1}, X^{2,2}, up(X^{3,2}))  → C2 通道
        X^{1,3} = H(x1, X^{1,1}, X^{1,2}, up(X^{2,2}))  → C1 通道
    """

    def __init__(self, in_ch=3, num_classes=4, base_ch=32, deep_supervision=False):
        super().__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        self.encoder = Encoder(in_ch, base_ch)

        C0, C1, C2, C3, C4 = base_ch, base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        # 第 1 列 (j=1)
        self.dec3_1 = DecoderBlock(C3 + C4, C3)  # x3 + up(x4)
        self.dec2_1 = DecoderBlock(C2 + C3, C2)  # x2 + up(x3)
        self.dec1_1 = DecoderBlock(C1 + C2, C1)  # x1 + up(x2)

        # 第 2 列 (j=2)
        self.dec3_2 = DecoderBlock(C3 + C3 + C2, C3)  # x3 + X^{3,1} + up(X^{2,1})
        self.dec2_2 = DecoderBlock(C2 + C2 + C3, C2)  # x2 + X^{2,1} + up(X^{3,1})
        self.dec1_2 = DecoderBlock(C1 + C1 + C2, C1)  # x1 + X^{1,1} + up(X^{2,1})

        # 第 3 列 (j=3)
        self.dec2_3 = DecoderBlock(C2 + C2 + C2 + C3, C2)  # x2 + X^{2,1} + X^{2,2} + up(X^{3,2})
        self.dec1_3 = DecoderBlock(C1 + C1 + C1 + C2, C1)  # x1 + X^{1,1} + X^{1,2} + up(X^{2,2})

        # 输出头
        if deep_supervision:
            self.final1 = nn.Conv2d(C1, num_classes, 1)
            self.final2 = nn.Conv2d(C1, num_classes, 1)
            self.final3 = nn.Conv2d(C1, num_classes, 1)
        else:
            self.final = nn.Conv2d(C1, num_classes, 1)

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)

        # j=1
        d3_1 = self.dec3_1(torch.cat([x3, _upsample(x4, x3.shape[2:])], dim=1))
        d2_1 = self.dec2_1(torch.cat([x2, _upsample(x3, x2.shape[2:])], dim=1))
        d1_1 = self.dec1_1(torch.cat([x1, _upsample(x2, x1.shape[2:])], dim=1))

        # j=2
        d3_2 = self.dec3_2(torch.cat([x3, d3_1, _upsample(d2_1, x3.shape[2:])], dim=1))
        d2_2 = self.dec2_2(torch.cat([x2, d2_1, _upsample(d3_1, x2.shape[2:])], dim=1))
        d1_2 = self.dec1_2(torch.cat([x1, d1_1, _upsample(d2_1, x1.shape[2:])], dim=1))

        # j=3
        d2_3 = self.dec2_3(torch.cat([x2, d2_1, d2_2, _upsample(d3_2, x2.shape[2:])], dim=1))
        d1_3 = self.dec1_3(torch.cat([x1, d1_1, d1_2, _upsample(d2_2, x1.shape[2:])], dim=1))

        # 输出
        if self.deep_supervision:
            out1 = self.final1(d1_1)
            out2 = self.final2(d1_2)
            out3 = self.final3(d1_3)
            out = (out1 + out2 + out3) / 3.0
        else:
            out = self.final(d1_3)

        out = _upsample(out, x.shape[2:])
        return out


class DiceLoss(nn.Module):
    """Dice 损失函数"""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target = target.unsqueeze(1)
        target_onehot = torch.zeros_like(pred)
        target_onehot.scatter_(1, target, 1.0)

        dims = (2, 3)
        intersection = (pred * target_onehot).sum(dims)
        union = pred.sum(dims) + target_onehot.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """组合损失: CE + Dice"""

    def __init__(self, weight_ce=1.0, weight_dice=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.w_ce = weight_ce
        self.w_dice = weight_dice

    def forward(self, pred, target):
        return self.w_ce * self.ce(pred, target) + self.w_dice * self.dice(pred, target)


if __name__ == '__main__':
    model = UNetPP(in_ch=3, num_classes=4, base_ch=32)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    params = sum(p.numel() for p in model.parameters())
    print(f'输入: {x.shape}')
    print(f'输出: {out.shape}')
    print(f'参数量: {params / 1e6:.2f}M')
