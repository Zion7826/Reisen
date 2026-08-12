# -*- coding: utf-8 -*-
"""
自动裂纹标注工具
使用图像处理算法自动检测裂纹区域，生成 YOLO 格式标注文件
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def detect_crack_regions(img_path: str, debug: bool = False) -> list[dict]:
    """检测图片中的裂纹区域，返回 YOLO 格式坐标列表
    
    Args:
        img_path: 图片路径
        debug: 是否保存调试图
    
    Returns:
        [{'x_center': float, 'y_center': float, 'width': float, 'height': float}]
    """
    # 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [WARN] 无法读取: {img_path}")
        return []

    h, w = img.shape[:2]
    orig = img.copy()

    # ========== 1. 转灰度图 ==========
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ========== 2. 增强对比度 (CLAHE) ==========
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # ========== 3. 边缘检测 (Canny) ==========
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 1.5)
    edges = cv2.Canny(blurred, 30, 90)

    # ========== 4. 形态学处理：连接断开的边缘 ==========
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # ========== 5. 找轮廓 ==========
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:  # 过滤太小噪点
            continue

        x, y, cw, ch = cv2.boundingRect(cnt)
        # 过滤太小的框
        if cw < 20 or ch < 20:
            continue
        # 过滤太大的框（接近整图 = 背景）
        if cw > w * 0.95 or ch > h * 0.95:
            continue

        # 转 YOLO 归一化格式
        xc = (x + cw / 2) / w
        yc = (y + ch / 2) / h
        nw = cw / w
        nh = ch / h

        regions.append({
            'x': x, 'y': y, 'cw': cw, 'ch': ch,
            'x_center': xc, 'y_center': yc,
            'width': nw, 'height': nh,
            'area': area
        })

    # 如果只找到一个超大的（整瓶），退化为裂纹线检测
    if len(regions) == 0 or (regions[0]['cw'] > w * 0.7 and regions[0]['ch'] > h * 0.7):
        print(f"  [INFO] 全瓶检测，尝试裂纹线检测: {os.path.basename(img_path)}")
        regions = _detect_crack_lines(enhanced, edges, w, h)

    # 调试可视化
    if debug:
        _debug_visualize(orig, edges, regions, img_path)

    return regions


def _detect_crack_lines(gray: np.ndarray, edges: np.ndarray, w: int, h: int) -> list[dict]:
    """基于线段检测找裂纹细线"""
    # HoughLinesP 找细线（裂纹通常是细长的线）
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=30, minLineLength=40, maxLineGap=10
    )

    regions = []
    if lines is not None:
        all_pts = []
        for line in lines:
            # HoughLinesP 返回 (1, 4) 或 (4,)，统一处理
            line_arr = np.asarray(line).flatten()
            if len(line_arr) < 4:
                continue
            x1, y1, x2, y2 = int(line_arr[0]), int(line_arr[1]), int(line_arr[2]), int(line_arr[3])
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length < 30:
                continue
            all_pts.extend([(x1, y1), (x2, y2)])

        if all_pts:
            xs = [p[0] for p in all_pts]
            ys = [p[1] for p in all_pts]
            margin = 20
            rx = max(0, min(xs) - margin)
            ry = max(0, min(ys) - margin)
            rw = min(w - 1, max(xs) + margin) - rx
            rh = min(h - 1, max(ys) + margin) - ry

            if rw > 20 and rh > 20:
                regions.append({
                    'x': int(rx), 'y': int(ry),
                    'cw': int(rw), 'ch': int(rh),
                    'x_center': (rx + rw / 2) / w,
                    'y_center': (ry + rh / 2) / h,
                    'width': rw / w,
                    'height': rh / h,
                    'area': rw * rh
                })

    # 如果 Hough 也没找到，用边缘密集区域
    if not regions:
        print("  [WARN] Hough也未找到，使用边缘密度法")
        regions = _edge_density_regions(edges, w, h)

    return regions


def _edge_density_regions(edges: np.ndarray, w: int, h: int) -> list[dict]:
    """用边缘密度热力图找裂纹区域"""
    # 计算局部边缘密度
    kernel_size = 31
    density = cv2.blur(edges.astype(np.float32), (kernel_size, kernel_size))

    # 找密度峰值点
    _, thresh = cv2.threshold(density, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw < 30 or ch < 30 or cw > w * 0.95 or ch > h * 0.95:
            continue

        regions.append({
            'x': x, 'y': y, 'cw': cw, 'ch': ch,
            'x_center': (x + cw / 2) / w,
            'y_center': (y + ch / 2) / h,
            'width': cw / w,
            'height': ch / h,
            'area': area
        })

    return regions


def _debug_visualize(orig, edges, regions, img_path):
    """保存调试图"""
    dbg = orig.copy()
    for reg in regions:
        x, y, cw, ch = reg['x'], reg['y'], reg['cw'], reg['ch']
        cv2.rectangle(dbg, (x, y), (x + cw, y + ch), (0, 255, 0), 2)
        cv2.putText(dbg, "crack", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out_path = img_path.replace('.jpg', '_debug.jpg').replace(
        'D:\\demo\\datasets\\glass_bottle\\images\\train',
        'D:\\demo\\debug_labels'
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, dbg)
    print(f"  [DEBUG] 保存: {out_path}")


def main():
    print("=" * 60)
    print("自动裂纹标注生成器")
    print("=" * 60)

    train_dir = Path(r"D:\demo\datasets\glass_bottle\images\train")
    val_dir = Path(r"D:\demo\datasets\glass_bottle\images\val")
    train_lbl_dir = Path(r"D:\demo\datasets\glass_bottle\labels\train")
    val_lbl_dir = Path(r"D:\demo\datasets\glass_bottle\labels\val")
    debug_dir = Path(r"D:\demo\debug_labels")

    for d in [train_lbl_dir, val_lbl_dir, debug_dir]:
        d.mkdir(parents=True, exist_ok=True)

    total = 0
    total_regions = 0

    for img_dir, lbl_dir in [(train_dir, train_lbl_dir), (val_dir, val_lbl_dir)]:
        images = sorted(img_dir.glob("*.jpg"))
        for img_path in images:
            print(f"\n处理: {img_path.name}")
            regions = detect_crack_regions(str(img_path), debug=True)

            lbl_path = lbl_dir / (img_path.stem + ".txt")

            with open(lbl_path, "w", encoding="utf-8") as f:
                for reg in regions:
                    # class 0 = crack
                    f.write(f"0 {reg['x_center']:.6f} {reg['y_center']:.6f} "
                            f"{reg['width']:.6f} {reg['height']:.6f}\n")

            if regions:
                print(f"  -> 找到 {len(regions)} 个区域，标注已保存")
                for reg in regions:
                    print(f"     框: ({reg['x']},{reg['y']}) "
                          f"{reg['cw']}x{reg['ch']}  "
                          f"归一化: xc={reg['x_center']:.3f} "
                          f"yc={reg['y_center']:.3f} "
                          f"w={reg['width']:.3f} h={reg['height']:.3f}")
            else:
                print(f"  -> [WARN] 未找到裂纹区域，使用默认框")
                # 默认框：覆盖图像中间80%
                with open(lbl_path, "w", encoding="utf-8") as f:
                    f.write("0 0.50 0.50 0.80 0.80\n")

            total += 1
            total_regions += len(regions)

    print(f"\n{'=' * 60}")
    print(f"完成! 共处理 {total} 张图片，找到 {total_regions} 个裂纹区域")
    print(f"标注目录: D:\\demo\\datasets\\glass_bottle\\labels\\train\\")
    print(f"调试图: D:\\demo\\debug_labels\\")
    print("=" * 60)


if __name__ == "__main__":
    main()
