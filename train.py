"""
YOLOv8 玻璃瓶缺陷检测模型训练脚本

功能：
1. 加载 data.yaml 数据配置
2. 配置训练超参数（epochs、batch、imgsz、优化器等）
3. 训练模型并自动保存最优权重
4. 输出训练日志到控制台与文件
5. 训练完成后在验证集上评估并导出 ONNX 模型

依赖：ultralytics (pip install ultralytics)
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
def setup_logger(log_dir: Path) -> logging.Logger:
    """配置同时输出到控制台和文件的日志记录器。"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("yolov8_train")
    logger.setLevel(logging.INFO)
    # 避免重复添加 handler（重复运行脚本时）
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # 文件输出
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"日志文件保存至: {log_file}")
    return logger


# ---------------------------------------------------------------------------
# 训练流程
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace, logger: logging.Logger) -> None:
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("YOLOv8 玻璃瓶缺陷检测模型训练开始")
    logger.info("=" * 70)

    # 1. 加载预训练模型
    logger.info(f"加载预训练模型: {args.weights}")
    model = YOLO(args.weights)

    # 2. 训练参数
    train_params = dict(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        warmup_momentum=args.warmup_momentum,
        warmup_bias_lr=args.warmup_bias_lr,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=True,
        cos_lr=True,
        close_mosaic=10,          # 最后 10 个 epoch 关闭 mosaic 增强
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        rect=False,
        save=True,
        save_period=-1,           # 仅保存最后与最优模型
        val=True,
        plots=True,               # 生成训练曲线图
        verbose=True,
    )

    logger.info("训练参数配置:")
    for k, v in train_params.items():
        logger.info(f"  {k}: {v}")

    # 3. 开始训练
    logger.info("-" * 70)
    logger.info("开始训练...")
    results = model.train(**train_params)

    # 4. 训练耗时
    elapsed = time.time() - start_time
    logger.info("-" * 70)
    logger.info(f"训练完成，总耗时: {elapsed / 3600:.2f} 小时 "
                f"({elapsed / 60:.2f} 分钟)")

    # 5. 训练结果
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else \
        Path(args.project) / args.name
    logger.info(f"模型与结果保存目录: {save_dir}")
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"
    logger.info(f"最优权重: {best_pt}")
    logger.info(f"最后权重: {last_pt}")

    # 6. 在验证集上评估最优模型
    logger.info("-" * 70)
    logger.info("使用最优模型在验证集上评估...")
    best_model = YOLO(str(best_pt))
    metrics = best_model.val(data=args.data, imgsz=args.imgsz, batch=args.batch)

    logger.info("验证集评估指标:")
    logger.info(f"  精确率 Precision: {metrics.box.mp:.4f}")
    logger.info(f"  召回率 Recall:    {metrics.box.mr:.4f}")
    logger.info(f"  mAP@0.5:         {metrics.box.map50:.4f}")
    logger.info(f"  mAP@0.5:0.95:    {metrics.box.map:.4f}")

    # 7. 导出 ONNX 模型（便于部署）
    if args.export_onnx:
        logger.info("-" * 70)
        logger.info("导出 ONNX 模型...")
        onnx_path = best_model.export(format="onnx", imgsz=args.imgsz, simplify=True)
        logger.info(f"ONNX 模型已导出: {onnx_path}")

    logger.info("=" * 70)
    logger.info("全部流程结束")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLOv8 玻璃瓶缺陷检测模型训练脚本"
    )
    # 数据与模型
    p.add_argument("--data", type=str, default="data.yaml",
                   help="数据配置文件路径 (默认: data.yaml)")
    p.add_argument("--weights", type=str, default="yolov8s.pt",
                   help="预训练权重 (默认: yolov8s.pt)")

    # 训练超参数
    p.add_argument("--epochs", type=int, default=200, help="训练轮数")
    p.add_argument("--batch", type=int, default=16, help="批大小")
    p.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    p.add_argument("--optimizer", type=str, default="AdamW",
                   choices=["SGD", "Adam", "AdamW"],
                   help="优化器")
    p.add_argument("--lr0", type=float, default=0.001, help="初始学习率")
    p.add_argument("--lrf", type=float, default=0.01, help="最终学习率衰减系数")
    p.add_argument("--momentum", type=float, default=0.937, help="动量")
    p.add_argument("--weight_decay", type=float, default=0.0005,
                   help="权重衰减")
    p.add_argument("--warmup_epochs", type=float, default=3.0,
                   help="预热轮数")
    p.add_argument("--warmup_momentum", type=float, default=0.8,
                   help="预热初始动量")
    p.add_argument("--warmup_bias_lr", type=float, default=0.1,
                   help="预热偏置学习率")

    # 早停与设备
    p.add_argument("--patience", type=int, default=50,
                   help="早停耐心值 (无提升的 epoch 数)")
    p.add_argument("--device", type=str, default="0",
                   help="训练设备 (如 0, 0,1, cpu)")
    p.add_argument("--workers", type=int, default=8,
                   help="数据加载线程数")

    # 输出
    p.add_argument("--project", type=str, default="runs/detect",
                   help="结果保存根目录")
    p.add_argument("--name", type=str, default="glass_bottle_defect",
                   help="本次训练实验名称")
    p.add_argument("--export_onnx", action="store_true",
                   help="训练完成后导出 ONNX 模型")
    return p.parse_args()


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger(Path(args.project) / args.name)
    try:
        train(args, logger)
    except Exception as e:
        logger.exception(f"训练过程中发生错误: {e}")
        sys.exit(1)