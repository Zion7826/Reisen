# -*- coding: utf-8 -*-
"""
玻璃瓶缺陷检测脚本
==================
基于 PyTorch + ultralytics YOLO 实现对玻璃瓶图像的缺陷检测。

支持检测的缺陷类别：
    0: obvious_damage 明显破损
    1: subtle_damage  非明显破损

使用方法：
    python detect.py --image 图片路径
    python detect.py --image 图片路径 --model best.pt --conf 0.25

依赖安装：
    pip install ultralytics opencv-python torch
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


# ============================================================
# 类别定义：索引与中文名称的映射
# 注意：索引顺序必须与 best.pt 训练时的 data.yaml 类别顺序一致
# ============================================================
CLASS_NAMES = {
    0: "obvious_damage",  # 明显破损
    1: "subtle_damage",   # 非明显破损
}

# 中文名称映射（用于在画面上显示）
CLASS_NAMES_CN = {
    0: "明显破损",
    1: "非明显破损",
}

# ============================================================
# 各类缺陷对应的标注颜色 (B, G, R) 格式，OpenCV 默认使用 BGR
# ============================================================
CLASS_COLORS = {
    0: (0, 0, 255),      # 明显破损 obvious_damage - 红色
    1: (0, 255, 255),    # 非明显破损 subtle_damage - 黄色
}

# 默认颜色（当类别索引超出预设范围时使用）
DEFAULT_COLOR = (255, 255, 255)  # 白色


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="玻璃瓶缺陷检测脚本（基于 YOLO）"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="待检测的图片路径（支持 jpg/png/bmp 等格式）",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="best.pt",
        help="YOLO 模型权重文件路径，默认为当前目录下的 best.pt",
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.25,
        help="置信度阈值，低于该值的检测结果将被过滤，默认 0.25",
    )
    parser.add_argument(
        "--save-dir", "-s",
        type=str,
        default="",
        help="结果保存目录，默认与原图同目录",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="添加此参数则不弹窗显示结果，仅保存文件",
    )
    return parser.parse_args()


def check_file_exists(path: str, desc: str) -> None:
    """检查文件是否存在，不存在则报错退出"""
    if not os.path.isfile(path):
        print(f"[错误] {desc}不存在：{path}")
        sys.exit(1)


def draw_detections(image, results, conf_threshold: float):
    """
    在原图上绘制检测框、类别名称和置信度

    参数：
        image:          原始图像（numpy 数组，BGR 格式）
        results:        YOLO 推理结果对象
        conf_threshold: 置信度阈值

    返回：
        绘制后的图像、检测到的目标数量、各类别数量统计
    """
    det_count = 0
    class_counts = {name: 0 for name in CLASS_NAMES.values()}

    # 遍历每张图片的检测结果（ultralytics 返回的是列表）
    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        # 逐个处理检测框
        for box in boxes:
            # 获取类别索引、置信度
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())

            # 置信度过滤
            if conf < conf_threshold:
                continue

            # 获取边界框坐标 (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 获取颜色与名称
            color = CLASS_COLORS.get(cls_id, DEFAULT_COLOR)
            name_en = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            name_cn = CLASS_NAMES_CN.get(cls_id, name_en)

            # ---------- 绘制边界框 ----------
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)

            # ---------- 绘制标签文本 ----------
            label = f"{name_cn}({name_en}) {conf:.2f}"
            # 计算文本尺寸以绘制背景框
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

            det_count += 1
            class_counts[name_en] = class_counts.get(name_en, 0) + 1

    return image, det_count, class_counts


def main():
    # 1. 解析命令行参数
    args = parse_args()

    # 2. 检查输入文件与模型文件
    check_file_exists(args.image, "图片文件")
    check_file_exists(args.model, "模型文件")

    # 3. 检查 PyTorch 环境信息（便于排查问题）
    print("=" * 50)
    print("玻璃瓶缺陷检测")
    print("=" * 50)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
    print(f"模型文件: {args.model}")
    print(f"输入图片: {args.image}")
    print(f"置信度阈值: {args.conf}")
    print("-" * 50)

    # 4. 加载 YOLO 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    model = YOLO(args.model)
    print("模型加载完成。")

    # 5. 读取原图（使用 OpenCV 读取，保持 BGR 格式）
    image = cv2.imread(args.image)
    if image is None:
        print(f"[错误] 无法读取图片：{args.image}")
        sys.exit(1)
    print(f"图片尺寸: {image.shape[1]} x {image.shape[0]}")

    # 6. 执行推理
    print("正在进行缺陷检测...")
    results = model(image, device=device)

    # 7. 在原图上绘制检测结果
    result_image, det_count, class_counts = draw_detections(
        image, results, args.conf
    )

    # 8. 打印检测结果摘要
    print("-" * 50)
    print(f"检测到 {det_count} 个缺陷目标：")
    for name_en, cnt in class_counts.items():
        name_cn = {
            v: k for k, v in CLASS_NAMES_CN.items()
        }
        cn_name = next(
            (cn for en, cn in zip(CLASS_NAMES.values(), CLASS_NAMES_CN.values())
             if en == name_en),
            name_en,
        )
        print(f"  - {cn_name}({name_en}): {cnt} 个")

    # 9. 保存结果图片：result_原文件名.jpg
    input_path = Path(args.image)
    original_stem = input_path.stem  # 不带扩展名的文件名
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, f"result_{original_stem}.jpg")
    else:
        save_path = str(input_path.parent / f"result_{original_stem}.jpg")
    cv2.imwrite(save_path, result_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("-" * 50)
    print(f"结果已保存至: {save_path}")

    # 10. 显示结果（除非指定 --no-show）
    if not args.no_show:
        cv2.imshow("Glass Bottle Defect Detection - Press any key to close",
                   result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("检测完成。")


if __name__ == "__main__":
    main()