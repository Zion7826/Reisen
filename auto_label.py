# -*- coding: utf-8 -*-
"""
自动标注脚本 auto_label.py
==========================
使用训练好的 YOLO 模型（best.pt）对指定目录下的所有图片进行批量自动标注，
并将检测结果保存为 YOLO 格式的标签文件，同时生成带标注框的调试图便于人工复核。

YOLO 标签文件格式（每行一个目标，所有坐标均为归一化到 [0, 1] 的相对值）：
    class_id x_center y_center width height

支持检测的缺陷类别（与 data.yaml 保持一致）：
    0: crack      裂纹
    1: bubble     气泡
    2: scratch    划痕
    3: stain      污渍

使用方法
--------
# 1. 使用全部默认参数（最常用）：对 datasets/glass_bottle/images/train 下所有图片标注
python auto_label.py

# 2. 指定模型与图片目录
python auto_label.py --model best.pt --image-dir datasets/glass_bottle/images/train

# 3. 自定义置信度阈值与输出目录
python auto_label.py --conf 0.3 --label-dir datasets/glass_bottle/labels/train --debug-dir debug_labels

# 4. 仅处理部分图片（限制数量，用于快速测试）
python auto_label.py --limit 5

# 5. 覆盖已存在的标签文件（默认会跳过已存在的 .txt）
python auto_label.py --overwrite

依赖安装
--------
    pip install ultralytics opencv-python torch
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


# ============================================================
# 类别定义：索引与中英文名称的映射
# 注意：索引顺序必须与 best.pt 训练时的 data.yaml 类别顺序一致
# ============================================================
CLASS_NAMES = {
    0: "crack",    # 裂纹
    1: "bubble",   # 气泡
    2: "scratch",  # 划痕
    3: "stain",    # 污渍
}

# 中文名称映射（用于在调试图上显示）
CLASS_NAMES_CN = {
    0: "裂纹",
    1: "气泡",
    2: "划痕",
    3: "污渍",
}

# ============================================================
# 4 类缺陷对应的标注颜色 (B, G, R) 格式，OpenCV 默认使用 BGR
# ============================================================
CLASS_COLORS = {
    0: (0, 0, 255),      # 裂纹 crack   - 红色
    1: (0, 255, 255),    # 气泡 bubble  - 黄色
    2: (255, 0, 0),      # 划痕 scratch - 蓝色
    3: (0, 255, 0),      # 污渍 stain   - 绿色
}

# 默认颜色（当类别索引超出预设范围时使用）
DEFAULT_COLOR = (255, 255, 255)  # 白色

# 支持读取的图片扩展名（小写）
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="YOLO 自动标注脚本：批量推理并生成 YOLO 格式标签文件 + 调试图",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="best.pt",
        help="YOLO 模型权重文件路径",
    )
    parser.add_argument(
        "--image-dir", "-i",
        type=str,
        default="datasets/glass_bottle/images/train",
        help="待标注图片所在目录",
    )
    parser.add_argument(
        "--label-dir", "-l",
        type=str,
        default="datasets/glass_bottle/labels/train",
        help="标签文件 (.txt) 输出目录，默认与 YOLO 数据集结构对应",
    )
    parser.add_argument(
        "--debug-dir", "-d",
        type=str,
        default="debug_labels",
        help="带标注框的调试图保存目录",
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.25,
        help="置信度阈值，低于该值的检测结果将被过滤",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS 的 IoU 阈值，用于去除重叠框",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="推理时输入图片的尺寸（短边缩放至该大小）",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="覆盖已存在的标签文件（默认跳过已存在的 .txt 文件）",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="仅处理前 N 张图片（0 表示处理全部），用于快速测试",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="推理设备，如 cuda / cpu / cuda:0，留空则自动选择",
    )
    return parser.parse_args()


def check_model_exists(path: str) -> None:
    """检查模型文件是否存在，不存在则报错退出"""
    if not os.path.isfile(path):
        print(f"[错误] 模型文件不存在：{path}")
        print("请确认 best.pt 路径是否正确，或先运行 train.py 训练模型。")
        sys.exit(1)


def collect_images(image_dir: str, limit: int = 0) -> list:
    """
    收集目录下所有支持格式的图片文件路径（按文件名排序，保证处理顺序稳定）

    参数：
        image_dir: 图片目录
        limit:     最多返回的图片数量，0 表示不限制

    返回：
        图片文件路径列表（Path 对象）
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        print(f"[错误] 图片目录不存在：{image_dir}")
        sys.exit(1)

    images = []
    for f in image_dir.iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS:
            images.append(f)

    # 按文件名排序，保证每次运行处理顺序一致
    images.sort(key=lambda p: p.name)

    if limit > 0:
        images = images[:limit]

    return images


def write_label_file(label_path: Path, detections: list) -> None:
    """
    将检测结果写入 YOLO 格式的标签文件

    YOLO 标签格式（每行一个目标，坐标均归一化到 [0, 1]）：
        class_id x_center y_center width height

    参数：
        label_path:  标签文件路径 (.txt)
        detections:  检测结果列表，每个元素为
                     (class_id, x_center, y_center, w, h) 的归一化坐标
    """
    with open(label_path, "w", encoding="utf-8") as f:
        for cls_id, xc, yc, w, h in detections:
            # 保险起见，将坐标裁剪到 [0, 1] 区间，避免极少数越界值
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def draw_detections_on_image(image, detections: list, conf_values: list):
    """
    在原图上绘制检测框、类别名称和置信度，用于人工复核

    参数：
        image:        原始图像（numpy 数组，BGR 格式），会被原地修改
        detections:   检测结果列表，每个元素为
                      (class_id, x1, y1, x2, y2) 的像素坐标
        conf_values:  与 detections 一一对应的置信度列表

    返回：
        绘制后的图像
    """
    for (cls_id, x1, y1, x2, y2), conf in zip(detections, conf_values):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        color = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)
        name_en = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        name_cn = CLASS_NAMES_CN.get(cls_id, name_en)

        # ---------- 绘制边界框 ----------
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)

        # ---------- 绘制标签文本 ----------
        label = f"{name_cn}({name_en}) {conf:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        # 标签背景位置（确保不超出图像上边界）
        label_y1 = max(y1 - text_h - baseline - 4, 0)
        label_y2 = y1
        cv2.rectangle(
            image,
            (x1, label_y1),
            (x1 + text_w + 4, label_y2),
            color,
            cv2.FILLED,
        )
        # 标签文字
        cv2.putText(
            image,
            label,
            (x1 + 2, label_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # 白色文字
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return image


def process_single_image(
    model,
    image_path: Path,
    label_dir: Path,
    debug_dir: Path,
    conf_threshold: float,
    iou_threshold: float,
    img_size: int,
    device: str,
    overwrite: bool,
) -> dict:
    """
    处理单张图片：推理 -> 写标签文件 -> 保存调试图

    返回：
        统计字典，包含该图片的检测信息
        {
            "name": 文件名,
            "count": 检测到的目标总数,
            "classes": {类别名: 数量},
            "skipped": 是否因标签已存在而跳过,
        }
    """
    result_info = {
        "name": image_path.name,
        "count": 0,
        "classes": {},
        "skipped": False,
    }

    # 对应的标签文件路径（同名 .txt）
    label_path = label_dir / (image_path.stem + ".txt")

    # 若标签已存在且未指定覆盖，则跳过
    if label_path.exists() and not overwrite:
        result_info["skipped"] = True
        return result_info

    # ---------- 1. 读取原图 ----------
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  [警告] 无法读取图片，跳过：{image_path.name}")
        result_info["skipped"] = True
        return result_info

    img_h, img_w = image.shape[:2]

    # ---------- 2. 执行推理 ----------
    # verbose=False 关闭 ultralytics 默认的逐张打印，保持输出整洁
    results = model(
        image,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=img_size,
        device=device,
        verbose=False,
    )

    # ---------- 3. 解析检测结果 ----------
    # detections_norm: 归一化坐标，用于写入标签文件
    # detections_px:   像素坐标，用于绘制调试图
    # conf_values:     置信度，用于绘制调试图
    detections_norm = []
    detections_px = []
    conf_values = []

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        # xywhn: 归一化的 (x_center, y_center, w, h)，直接用于标签文件
        xywhn = boxes.xywhn.cpu().numpy()
        # xyxy: 像素坐标 (x1, y1, x2, y2)，用于绘图
        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        for i in range(len(cls_ids)):
            cls_id = int(cls_ids[i])
            xc, yc, w, h = xywhn[i]
            x1, y1, x2, y2 = xyxy[i]
            conf = float(confs[i])

            detections_norm.append((cls_id, float(xc), float(yc), float(w), float(h)))
            detections_px.append((cls_id, float(x1), float(y1), float(x2), float(y2)))
            conf_values.append(conf)

            # 统计
            name_en = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            result_info["classes"][name_en] = result_info["classes"].get(name_en, 0) + 1

    result_info["count"] = len(detections_norm)

    # ---------- 4. 写入标签文件 ----------
    # 即使没有检测到任何目标，也写一个空文件（YOLO 训练要求负样本对应空标签）
    write_label_file(label_path, detections_norm)

    # ---------- 5. 生成并保存调试图 ----------
    debug_image = image.copy()
    debug_image = draw_detections_on_image(debug_image, detections_px, conf_values)

    # 在画面左上角叠加统计信息，便于批量复核
    summary_text = f"{image_path.name}  |  size: {img_w}x{img_h}  |  objects: {result_info['count']}"
    cv2.rectangle(debug_image, (0, 0), (img_w, 32), (0, 0, 0), cv2.FILLED)
    cv2.putText(
        debug_image,
        summary_text,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    debug_path = debug_dir / (image_path.stem + ".jpg")
    cv2.imwrite(str(debug_path), debug_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return result_info


def main():
    # 1. 解析命令行参数
    args = parse_args()

    # 2. 检查模型文件
    check_model_exists(args.model)

    # 3. 收集待标注图片
    images = collect_images(args.image_dir, args.limit)
    if len(images) == 0:
        print(f"[提示] 目录下未找到任何图片：{args.image_dir}")
        print(f"       支持的格式：{', '.join(SUPPORTED_EXTS)}")
        sys.exit(0)

    # 4. 创建输出目录
    label_dir = Path(args.label_dir)
    debug_dir = Path(args.debug_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    # 5. 选择推理设备
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 6. 打印运行信息
    print("=" * 60)
    print("YOLO 自动标注脚本")
    print("=" * 60)
    print(f"PyTorch 版本   : {torch.__version__}")
    print(f"CUDA 是否可用  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 设备       : {torch.cuda.get_device_name(0)}")
    print(f"推理设备       : {device}")
    print(f"模型文件       : {args.model}")
    print(f"图片目录       : {args.image_dir}")
    print(f"标签输出目录   : {label_dir}")
    print(f"调试图输出目录 : {debug_dir}")
    print(f"置信度阈值     : {args.conf}")
    print(f"IoU 阈值       : {args.iou}")
    print(f"推理图片尺寸   : {args.img_size}")
    print(f"覆盖已有标签   : {args.overwrite}")
    print(f"待处理图片数量 : {len(images)}")
    print("-" * 60)

    # 7. 加载 YOLO 模型
    print("正在加载模型...")
    model = YOLO(args.model)
    print("模型加载完成。")
    print("-" * 60)

    # 8. 批量处理
    print("开始批量自动标注...")
    start_time = time.time()

    total_count = 0          # 所有图片检测到的目标总数
    total_skipped = 0        # 跳过的图片数量
    total_class_counts = {}  # 各类别目标总数
    processed = 0            # 实际执行推理的图片数量

    for idx, image_path in enumerate(images, start=1):
        # 进度显示
        print(f"[{idx}/{len(images)}] 处理 {image_path.name} ...", end=" ", flush=True)

        info = process_single_image(
            model=model,
            image_path=image_path,
            label_dir=label_dir,
            debug_dir=debug_dir,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            img_size=args.img_size,
            device=device,
            overwrite=args.overwrite,
        )

        if info["skipped"]:
            print("跳过（标签已存在或图片无法读取）")
            total_skipped += 1
            continue

        processed += 1
        total_count += info["count"]
        for name, cnt in info["classes"].items():
            total_class_counts[name] = total_class_counts.get(name, 0) + cnt

        # 单张结果摘要
        if info["count"] == 0:
            print("未检测到目标（已写入空标签文件）")
        else:
            detail = ", ".join(f"{k}:{v}" for k, v in info["classes"].items())
            print(f"检测到 {info['count']} 个目标 ({detail})")

    elapsed = time.time() - start_time

    # 9. 打印汇总统计
    print("=" * 60)
    print("自动标注完成！汇总统计：")
    print("-" * 60)
    print(f"总图片数量     : {len(images)}")
    print(f"实际处理数量   : {processed}")
    print(f"跳过数量       : {total_skipped}")
    print(f"检测目标总数   : {total_count}")
    if total_class_counts:
        print("各类别分布     :")
        for name_en, cnt in total_class_counts.items():
            # 找到对应的中文名
            name_cn = next(
                (cn for en, cn in zip(CLASS_NAMES.values(), CLASS_NAMES_CN.values())
                 if en == name_en),
                name_en,
            )
            print(f"  - {name_cn}({name_en}): {cnt} 个")
    else:
        print("各类别分布     : 无检测目标")
    print(f"总耗时         : {elapsed:.2f} 秒")
    if processed > 0:
        print(f"平均每张耗时   : {elapsed / processed:.3f} 秒")
    print(f"标签文件目录   : {label_dir}")
    print(f"调试图目录     : {debug_dir}")
    print("=" * 60)
    print("提示：请打开 debug_labels 目录下的调试图人工复核标注质量。")
    print("      如需重新标注某些图片，删除对应 .txt 后重跑，或使用 --overwrite 参数。")


if __name__ == "__main__":
    main()