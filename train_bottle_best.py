# -*- coding: utf-8 -*-
"""
玻璃瓶缺陷检测 - 最佳状态训练脚本
从已有权重继续训练，直到达到最佳状态
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
    log_dir = "runs/detect/bottle_best"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, "best_train_log.txt")

    logger = logging.getLogger("bottle_best_train")
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


def train_round(model_path, round_num, logger, epochs=300, lr=0.001):
    """
    执行一轮训练
    """
    logger.info("=" * 60)
    logger.info(f"开始第 {round_num} 轮训练 (epochs={epochs}, lr={lr})")
    logger.info("=" * 60)

    # 加载模型
    logger.info(f"加载模型: {model_path}")
    model = YOLO(model_path)

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
        name="bottle_best",
        exist_ok=True,
        verbose=False,  # 禁用详细输出，避免tqdm错误
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
        # 损失函数参数
        box=7.5,
        cls=0.5,
        dfl=1.5,
        patience=100,  # 增加耐心值
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
        name="bottle_best_val",
        exist_ok=True,
        save_json=True,
        verbose=False,
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

    best_pt = "runs/detect/bottle_best/weights/best.pt"
    return best_pt, map50, precision, recall


def main():
    logger = setup_logger()

    logger.info("=" * 60)
    logger.info("玻璃瓶缺陷检测 - 最佳状态训练")
    logger.info("=" * 60)
    logger.info(f"Python: {sys.version.split()[0]}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"设备: cpu")
    logger.info("=" * 60)

    # 检查已有权重
    prev_best = "runs/detect/runs/detect/bottle_defect/weights/best.pt"
    last_pt = "runs/detect/runs/detect/bottle_defect/weights/last.pt"

    if os.path.exists(last_pt):
        initial_model = last_pt
        logger.info(f"从上次训练的last.pt继续: {last_pt}")
    elif os.path.exists(prev_best):
        initial_model = prev_best
        logger.info(f"从已有的best.pt继续: {prev_best}")
    else:
        initial_model = "yolov8n.pt"
        logger.info(f"使用预训练模型: {initial_model}")

    # 多轮训练配置 - 逐渐降低学习率，增加训练轮数
    train_configs = [
        {"epochs": 300, "lr": 0.001},     # 第1轮：基础训练
        {"epochs": 400, "lr": 0.0005},    # 第2轮：精细微调
        {"epochs": 500, "lr": 0.0002},    # 第3轮：进一步优化
        {"epochs": 600, "lr": 0.0001},    # 第4轮：深度优化
        {"epochs": 800, "lr": 0.00005},   # 第5轮：极致优化
    ]

    results = []
    current_model = initial_model

    try:
        for round_num, config in enumerate(train_configs, 1):
            logger.info(f"\n准备第 {round_num} 轮训练，使用模型: {current_model}")

            best_pt, map50, precision, recall = train_round(
                current_model, round_num, logger,
                epochs=config["epochs"],
                lr=config["lr"]
            )

            results.append({
                "round": round_num,
                "epochs": config["epochs"],
                "lr": config["lr"],
                "map50": map50,
                "precision": precision,
                "recall": recall
            })

            # 更新模型路径为最新最佳权重
            if os.path.exists(best_pt):
                current_model = best_pt
                logger.info(f"第 {round_num} 轮完成，最佳模型: {best_pt}")
            else:
                logger.warning(f"最佳权重未找到: {best_pt}，使用last.pt")
                current_model = "runs/detect/bottle_best/weights/last.pt"

            # 检查是否达到很好的性能
            if map50 >= 0.95 and precision >= 0.95 and recall >= 0.95:
                logger.info("模型已达到优秀性能！(mAP@0.5 >= 95%)")
                logger.info("提前结束训练")
                break

        # 输出总结
        logger.info("\n" + "=" * 60)
        logger.info("多轮训练完成！训练总结:")
        logger.info("=" * 60)
        logger.info(f"{'轮次':<6} {'轮数':<6} {'学习率':<10} {'mAP@0.5':<12} {'Precision':<12} {'Recall':<12}")
        logger.info("-" * 60)
        for r in results:
            logger.info(f"{r['round']:<6} {r['epochs']:<6} {r['lr']:<10.5f} {r['map50']:<12.4f} {r['precision']:<12.4f} {r['recall']:<12.4f}")

        logger.info("=" * 60)
        logger.info(f"最终最佳模型: runs/detect/bottle_best/weights/best.pt")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("用户中断训练")
        sys.exit(1)
    except Exception as e:
        logger.error(f"训练出错: {e}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()