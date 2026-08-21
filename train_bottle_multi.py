# -*- coding: utf-8 -*-
"""
玻璃瓶缺陷检测 - 多轮训练脚本
使用MVTec bottle数据集进行多次训练，提高检测精度
"""

import os
import sys
import time
import logging
from pathlib import Path
from ultralytics import YOLO
import torch


def setup_logger():
    """配置日志"""
    log_dir = "runs/detect/bottle_defect"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, "multi_train_log.txt")

    logger = logging.getLogger("bottle_multi_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def train_round(model_path, round_num, logger, prev_best=None):
    """
    执行一轮训练

    Args:
        model_path: 初始模型路径
        round_num: 训练轮次
        logger: 日志记录器
        prev_best: 上一轮的最佳权重路径

    Returns:
        最佳权重路径
    """
    logger.info("=" * 60)
    logger.info(f"开始第 {round_num} 轮训练")
    logger.info("=" * 60)

    # 加载模型
    if prev_best and os.path.exists(prev_best):
        logger.info(f"从上一轮最佳权重继续: {prev_best}")
        model = YOLO(prev_best)
    else:
        logger.info(f"加载预训练模型: {model_path}")
        model = YOLO(model_path)

    # 训练参数 - 每轮逐渐降低学习率
    base_lr = 0.001
    lr = base_lr * (0.5 ** (round_num - 1))  # 学习率逐轮减半
    epochs = 200 + round_num * 100  # 逐轮增加训练轮数

    logger.info(f"训练配置:")
    logger.info(f"  训练轮数: {epochs}")
    logger.info(f"  学习率: {lr}")
    logger.info(f"  批大小: 16")
    logger.info(f"  图像尺寸: 640")

    start_time = time.time()

    # 执行训练
    model.train(
        data="datasets/bottle_defect/data.yaml",
        epochs=epochs,
        batch=16,
        imgsz=640,
        optimizer="AdamW",
        lr0=lr,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        device="cpu",
        workers=0,
        project="runs/detect",
        name="bottle_defect",
        exist_ok=True,
        verbose=True,
        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        # 提高检测精度
        box=7.5,          # 边界框损失增益
        cls=0.5,          # 分类损失增益
        dfl=1.5,          # 分布焦点损失增益
        patience=50,      # 早停耐心值
    )

    elapsed = time.time() - start_time
    logger.info(f"第 {round_num} 轮训练完成，耗时: {elapsed / 60:.2f} 分钟")

    # 评估模型
    logger.info("评估模型...")
    metrics = model.val(
        data="datasets/bottle_defect/data.yaml",
        imgsz=640,
        batch=16,
        device="cpu",
        workers=0,
        project="runs/detect",
        name="bottle_defect_val",
        exist_ok=True,
        save_json=True,
    )

    # 输出评估结果
    map50 = metrics.box.map50
    map = metrics.box.map
    precision = metrics.box.mp
    recall = metrics.box.mr

    logger.info("-" * 60)
    logger.info(f"第 {round_num} 轮评估结果:")
    logger.info(f"  mAP@0.5      : {map50:.4f}  ({map50 * 100:.2f}%)")
    logger.info(f"  mAP@0.5:0.95 : {map:.4f}  ({map * 100:.2f}%)")
    logger.info(f"  平均 Precision: {precision:.4f}")
    logger.info(f"  平均 Recall   : {recall:.4f}")

    # 各类别mAP
    try:
        map50_per_class = metrics.box.maps
        class_names = ["broken_large", "broken_small", "contamination"]
        logger.info("  各类别 mAP@0.5:")
        for i, (cls, val) in enumerate(zip(class_names, map50_per_class)):
            logger.info(f"    [{i}] {cls:<15}: {val:.4f}  ({val * 100:.2f}%)")
    except Exception:
        pass

    logger.info("-" * 60)

    best_pt = "runs/detect/bottle_defect/weights/best.pt"
    return best_pt, map50, precision, recall


def main():
    logger = setup_logger()

    logger.info("=" * 60)
    logger.info("玻璃瓶缺陷检测 - 多轮训练")
    logger.info("=" * 60)
    logger.info(f"Python: {sys.version.split()[0]}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"设备: cpu")
    logger.info(f"数据集: datasets/bottle_defect/data.yaml")
    logger.info("=" * 60)

    # 训练配置
    initial_model = "yolov8n.pt"  # 初始预训练模型
    num_rounds = 3  # 训练轮数

    results = []

    try:
        prev_best = None
        for round_num in range(1, num_rounds + 1):
            best_pt, map50, precision, recall = train_round(
                initial_model, round_num, logger, prev_best
            )
            results.append({
                "round": round_num,
                "map50": map50,
                "precision": precision,
                "recall": recall
            })
            prev_best = best_pt

            logger.info(f"第 {round_num} 轮训练完成，最佳模型: {best_pt}")

        # 输出总结
        logger.info("=" * 60)
        logger.info("多轮训练完成！训练总结:")
        logger.info("=" * 60)
        logger.info(f"{'轮次':<6} {'mAP@0.5':<12} {'Precision':<12} {'Recall':<12}")
        logger.info("-" * 60)
        for r in results:
            logger.info(f"{r['round']:<6} {r['map50']:<12.4f} {r['precision']:<12.4f} {r['recall']:<12.4f}")

        logger.info("=" * 60)
        logger.info(f"最终最佳模型: runs/detect/bottle_defect/weights/best.pt")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("用户中断训练")
        sys.exit(1)
    except Exception as e:
        logger.error(f"训练出错: {e}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()