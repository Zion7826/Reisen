# -*- coding: utf-8 -*-
"""
继续训练脚本 - 从最佳权重继续训练，提高模型性能
"""

import os
import sys
import time
import logging
from pathlib import Path
from ultralytics import YOLO


def setup_logger():
    """配置日志"""
    log_dir = "runs/detect/glass_bottle_defect"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, "train_log_continue.txt")

    logger = logging.getLogger("glass_bottle_train_continue")
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


def main():
    logger = setup_logger()

    # 从最佳权重继续训练
    best_pt = "runs/detect/glass_bottle_defect/weights/best.pt"

    if not os.path.exists(best_pt):
        logger.error(f"权重文件不存在: {best_pt}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("玻璃瓶缺陷检测 - 继续训练")
    logger.info("=" * 60)
    logger.info(f"加载权重: {best_pt}")

    model = YOLO(best_pt)

    # 训练参数 - 增加轮数，使用更小的学习率
    logger.info("训练配置:")
    logger.info("  数据集     : data.yaml")
    logger.info("  训练轮数   : 300")
    logger.info("  批大小     : 8")
    logger.info("  图像尺寸   : 640")
    logger.info("  优化器     : AdamW")
    logger.info("  初始学习率 : 0.0005 (更小，精细微调)")
    logger.info("  设备       : cpu")
    logger.info("=" * 60)

    start_time = time.time()
    logger.info("开始继续训练 ...")

    # 继续训练
    model.train(
        data="data.yaml",
        epochs=300,            # 增加训练轮数
        batch=8,               # 减小批大小（数据量少）
        imgsz=640,             # 图像尺寸
        optimizer="AdamW",     # 优化器
        lr0=0.0005,            # 更小的学习率，精细微调
        lrf=0.01,              # 最终学习率比例
        momentum=0.937,        # 动量
        weight_decay=0.0005,   # 权重衰减
        warmup_epochs=3,       # 预热轮数
        warmup_momentum=0.8,   # 预热动量
        warmup_bias_lr=0.1,    # 预热偏置学习率
        device="cpu",          # 使用CPU
        workers=0,             # 工作进程数
        project="runs/detect", # 输出目录
        name="glass_bottle_defect",  # 运行名称
        exist_ok=True,         # 允许覆盖
        verbose=True,          # 详细输出
        # 数据增强参数
        hsv_h=0.015,          # 色调增强
        hsv_s=0.7,            # 饱和度增强
        hsv_v=0.4,            # 明度增强
        degrees=10.0,         # 旋转角度范围
        translate=0.1,        # 平移比例
        scale=0.5,            # 缩放比例
        shear=2.0,            # 剪切变换
        perspective=0.0,      # 透视变换
        flipud=0.0,           # 上下翻转概率
        fliplr=0.5,           # 左右翻转概率
        mosaic=1.0,           # mosaic数据增强
        mixup=0.1,            # mixup数据增强
        copy_paste=0.1,       # copy-paste增强
    )

    elapsed = time.time() - start_time
    logger.info(f"训练完成，总耗时: {elapsed / 60:.2f} 分钟")

    # 评估模型
    logger.info("=" * 60)
    logger.info("开始在验证集上评估模型 ...")

    metrics = model.val(
        data="data.yaml",
        imgsz=640,
        batch=8,
        device="cpu",
        workers=0,
        project="runs/detect",
        name="glass_bottle_defect_val",
        exist_ok=True,
        save_json=True,
    )

    # 输出评估结果
    map50 = metrics.box.map50
    map = metrics.box.map
    precision = metrics.box.mp
    recall = metrics.box.mr

    logger.info("-" * 60)
    logger.info("验证集评估结果:")
    logger.info(f"  mAP@0.5      : {map50:.4f}  ({map50 * 100:.2f}%)")
    logger.info(f"  mAP@0.5:0.95 : {map:.4f}  ({map * 100:.2f}%)")
    logger.info(f"  平均 Precision: {precision:.4f}")
    logger.info(f"  平均 Recall   : {recall:.4f}")

    # 各类别mAP
    try:
        map50_per_class = metrics.box.maps
        class_names = ["crack", "bubble", "scratch", "stain"]
        logger.info("  各类别 mAP@0.5:")
        for i, (cls, val) in enumerate(zip(class_names, map50_per_class)):
            logger.info(f"    [{i}] {cls:<8}: {val:.4f}  ({val * 100:.2f}%)")
    except Exception:
        pass

    logger.info("-" * 60)
    logger.info("训练与评估全部完成!")
    logger.info(f"最佳模型: runs/detect/glass_bottle_defect/weights/best.pt")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()