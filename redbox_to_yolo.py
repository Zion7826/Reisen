# -*- coding: utf-8 -*-
"""
把"AI生成的带红框裂纹图"转换成 YOLO 训练数据
============================================
功能：
  1. 找出图里的红色矩形框（红框 = AI 标的缺陷位置）
  2. 把红框坐标写成 YOLO 标注文件 (.txt)
  3. 把红框从原图抹掉（inpainting），得到"干净裂纹图"给模型看

用法：
  把 AI 红框图放进 INPUT_DIR，运行：
  D:\QClaw\v0.2.33.617\resources\python\python.exe D:\demo\redbox_to_yolo.py

输出：
  OUTPUT_IMG_DIR  -> 干净图（红框已抹）
  OUTPUT_LBL_DIR  -> YOLO 标注 .txt
"""
import sys
from pathlib import Path

import cv2
import numpy as np

# ============ 配置 ============
INPUT_DIR = r"D:\demo\ai_redbox"          # 放 AI 红框图
OUTPUT_IMG_DIR = r"D:\demo\ai_clean"       # 干净图输出
OUTPUT_LBL_DIR = r"D:\demo\ai_labels"      # 标注输出
CLASS_ID = 0                               # 0 = crack
MIN_BOX_RATIO = 0.01                       # 框最小占图比例，过滤噪点
# ============================

# OpenCV 中红色分布在两个 Hue 区间
LOWER1 = np.array([0, 120, 120])
UPPER1 = np.array([10, 255, 255])
LOWER2 = np.array([170, 120, 120])
UPPER2 = np.array([180, 255, 255])


def find_red_boxes(img: np.ndarray):
    """返回 (红框坐标列表, 红像素掩码)
    红框坐标用于写标注；红像素掩码用于只抹红框、保留裂纹。
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, LOWER1, UPPER1)
    mask2 = cv2.inRange(hsv, LOWER2, UPPER2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # 形态学闭运算，补红框断点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    red_mask = cv2.dilate(red_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img.shape[:2]
    min_area = (w * h) * MIN_BOX_RATIO

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        # 过滤几乎全图的框
        if bw > w * 0.98 or bh > h * 0.98:
            continue
        boxes.append((x, y, bw, bh))
    return boxes, red_mask


def to_yolo(boxes, img_w, img_h):
    """转 YOLO 归一化格式字符串"""
    lines = []
    for x, y, bw, bh in boxes:
        xc = (x + bw / 2) / img_w
        yc = (y + bh / 2) / img_h
        nw = bw / img_w
        nh = bh / img_h
        lines.append(f"{CLASS_ID} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
    return "\n".join(lines)


def main():
    in_dir = Path(INPUT_DIR)
    if not in_dir.exists():
        print(f"[提示] 目录不存在: {INPUT_DIR}")
        print("请把 AI 红框图放进这个文件夹，再运行。")
        input("按回车退出...")
        return

    out_img = Path(OUTPUT_IMG_DIR)
    out_lbl = Path(OUTPUT_LBL_DIR)
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    imgs = sorted(list(in_dir.glob("*.jpg")) + list(in_dir.glob("*.png")))
    if not imgs:
        print(f"[提示] {INPUT_DIR} 里没图片")
        input("按回车退出...")
        return

    print(f"找到 {len(imgs)} 张 AI 红框图，开始转换...\n")
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [跳过] 无法读取: {p.name}")
            continue
        h, w = img.shape[:2]
        boxes, red_mask = find_red_boxes(img)

        # 写标注
        lbl_path = out_lbl / (p.stem + ".txt")
        if boxes:
            lbl_path.write_text(to_yolo(boxes, w, h), encoding="utf-8")
        else:
            # 没找到红框也写个空文件，保持对齐
            lbl_path.write_text("", encoding="utf-8")

        # 只抹红框像素 -> 干净图（裂纹如果是深色非红，会保留）
        if boxes:
            clean = cv2.inpaint(img, red_mask, 3, cv2.INPAINT_TELEA)
        else:
            clean = img.copy()
        cv2.imwrite(str(out_img / p.name), clean)

        print(f"  {p.name}: 找到 {len(boxes)} 个红框 -> 标注+干净图已生成")

    print(f"\n完成！")
    print(f"干净图: {OUTPUT_IMG_DIR}")
    print(f"标注:   {OUTPUT_LBL_DIR}")


if __name__ == "__main__":
    main()
    input("\n按回车退出...")
