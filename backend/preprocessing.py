# -*- coding: utf-8 -*-
"""
图像预处理模块
包含 CLAHE 对比度增强、去噪、归一化等功能
"""

import cv2
import numpy as np
import torch


def clahe_enhance(image, clip_limit=2.0, tile_grid=(8, 8)):
    """
    CLAHE 自适应直方图均衡化增强对比度

    Args:
        image: BGR 图像
        clip_limit: 对比度限制
        tile_grid: 分块大小

    Returns:
        增强后的 BGR 图像
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l = clahe.apply(l)

    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return enhanced


def denoise(image, method='bilateral', h=10):
    """
    图像去噪

    Args:
        image: BGR 图像
        method: 'bilateral' | 'gaussian' | 'median' | 'nlm'
        h: 滤波强度

    Returns:
        去噪后的图像
    """
    if method == 'bilateral':
        return cv2.bilateralFilter(image, d=5, sigmaColor=h * 5, sigmaSpace=h * 5)
    elif method == 'gaussian':
        return cv2.GaussianBlur(image, (3, 3), 0)
    elif method == 'median':
        return cv2.medianBlur(image, 3)
    elif method == 'nlm':
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
    return image


def normalize_image(image, target_size=(256, 256)):
    """
    归一化图像: resize + 归一化到 [0, 1]

    Args:
        image: BGR 图像
        target_size: (width, height)

    Returns:
        归一化后的 RGB float32 图像
    """
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    return normalized


def preprocess_for_segmentation(image, target_size=(256, 256)):
    """
    分割模型预处理: CLAHE -> 去噪 -> 归一化 -> Tensor

    Args:
        image: BGR 图像
        target_size: 模型输入尺寸

    Returns:
        tensor: [1, 3, H, W] float32
        orig_h, orig_w: 原始尺寸
    """
    orig_h, orig_w = image.shape[:2]

    enhanced = clahe_enhance(image)
    denoised = denoise(enhanced, method='bilateral')
    normalized = normalize_image(denoised, target_size)

    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    return tensor, orig_h, orig_w


def preprocess_for_anomaly(image, target_size=(256, 256)):
    """
    异常检测模型预处理: 归一化 -> Tensor

    Args:
        image: BGR 图像
        target_size: 模型输入尺寸

    Returns:
        tensor: [1, 3, H, W] float32
    """
    normalized = normalize_image(image, target_size)
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    return tensor


def postprocess_mask(mask, orig_h, orig_w):
    """
    掩码后处理: resize 回原始尺寸

    Args:
        mask: [H, W] 或 [1, H, W] 预测掩码
        orig_h, orig_w: 原始尺寸

    Returns:
        [orig_h, orig_w] uint8 掩码
    """
    if mask.ndim == 3:
        mask = mask[0]
    mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    mask_resized = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return mask_resized.astype(np.uint8)


def overlay_mask(image, mask, alpha=0.5, colors=None):
    """
    在图像上叠加彩色掩码

    Args:
        image: BGR 图像
        mask: [H, W] 掩码 (0=背景, 1/2/3=类别)
        alpha: 透明度
        colors: {mask_value: (B, G, R)} 颜色映射

    Returns:
        叠加后的 BGR 图像
    """
    if colors is None:
        colors = {
            1: (0, 0, 255),
            2: (255, 0, 0),
            3: (0, 255, 0),
        }

    overlay = image.copy()
    for mask_value, color in colors.items():
        binary = (mask == mask_value).astype(np.uint8)
        overlay[binary > 0] = color

    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return result