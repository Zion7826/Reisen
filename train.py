# -*- coding: utf-8 -*-
"""
================================================================================
 玻璃瓶缺陷检测 - YOLOv8 训练脚本 (train.py)
================================================================================
功能说明:
    本脚本基于 ultralytics 官方库,使用 YOLOv8n 预训练模型在玻璃瓶缺陷数据集上
    进行迁移学习,训练完成后自动在验证集上评估,并输出 mAP@0.5 与 mAP@0.5:0.95 指标。

    缺陷类别(共 2 类):
        0 - obvious_damage (明显破损)
        1 - subtle_damage  (非明显破损)

默认训练超参数:
    epochs  = 200      训练轮数
    batch   = 16       批大小
    imgsz   = 640      输入图像尺寸
    optimizer = AdamW  优化器
    lr0     = 0.001    初始学习率

输出目录:
    runs/detect/glass_bottle_defect/   (训练日志、权重、混淆矩阵等)

--------------------------------------------------------------------------------
使用示例:
    # 1. 使用全部默认参数(GPU 训练)
    python train.py

    # 2. 指定数据集与设备
    python train.py --data data.yaml --device cuda

    # 3. 自定义训练轮数与批大小(CPU 训练,建议调小 batch)
    python train.py --epochs 100 --batch 8 --device cpu

    # 4. 指定预训练权重路径
    python train.py --weights yolov8n.pt --data data.yaml --epochs 200

    # 5. 从断点继续训练(resume)
    python train.py --resume runs/detect/glass_bottle_defect/weights/last.pt
================================================================================
"""

import os
import sys
import time
import logging
from pathlib import Path

# ------------------------------------------------------------------------------
# 1. 命令行参数解析
# ------------------------------------------------------------------------------
import argparse


def parse_args():
    """解析命令行参数。

    支持的参数:
        --weights : 预训练权重路径,默认 yolov8n.pt
        --data    : 数据集配置文件(yaml),默认 data.yaml
        --epochs  : 训练轮数,默认 200
        --batch   : 批大小,默认 16
        --imgsz   : 输入图像尺寸,默认 640
        --device  : 训练设备(cuda / cpu / 0,0,1 等),默认 cuda
        --workers : DataLoader 工作进程数,默认 8
        --resume  : 从指定 last.pt 断点续训
    """
    parser = argparse.ArgumentParser(
        description="YOLOv8 玻璃瓶缺陷检测训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights", type=str, default="yolov8n.pt",
        help="预训练权重路径(默认 yolov8n.pt)"
    )
    parser.add_argument(
        "--data", type=str, default="data.yaml",
        help="数据集配置文件路径(默认 data.yaml)"
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="训练总轮数(默认 200)"
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="批大小(默认 16,CPU 训练建议调小)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="输入图像尺寸(默认 640)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="训练设备,如 cuda / cpu / 0 / 0,1(默认 cuda)"
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="DataLoader 工作进程数(默认 8,Windows 下建议设为 0 或较小值)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="从指定 last.pt 断点继续训练(如 runs/detect/glass_bottle_defect/weights/last.pt)"
    )
    return parser.parse_args()


# ------------------------------------------------------------------------------
# 2. 日志配置
# ------------------------------------------------------------------------------
def setup_logger(log_dir: str) -> logging.Logger:
    """配置日志记录器,同时输出到控制台和日志文件。

    日志文件保存路径: {log_dir}/train_log.txt

    Args:
        log_dir: 日志输出目录。

    Returns:
        配置好的 logging.Logger 对象。
    """
    # 确保日志目录存在
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, "train_log.txt")

    logger = logging.getLogger("glass_bottle_train")
    logger.setLevel(logging.INFO)
    # 避免重复添加 handler(多次调用时)
    logger.handlers.clear()

    # 日志格式:时间 - 级别 - 消息
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # 控制台输出
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # 文件输出
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"日志文件将保存至: {log_file}")
    return logger


# ------------------------------------------------------------------------------
# 3. 训练主流程
# ------------------------------------------------------------------------------
def train(args, logger: logging.Logger):
    """执行 YOLOv8 训练流程。

    使用 ultralytics 的 YOLO 接口加载预训练模型并调用 model.train() 进行训练。
    训练产物(权重、曲线、混淆矩阵等)保存至 runs/detect/glass_bottle_defect/。

    Args:
        args:   parse_args() 返回的命名空间对象。
        logger: 日志记录器。

    Returns:
        训练完成后的模型对象(ultralytics YOLO)。
    """
    # 延迟导入,避免 argparse 解析失败时也加载 ultralytics
    from ultralytics import YOLO

    # 输出目录设置:project/name 组合为 runs/detect/glass_bottle_defect/
    project_name = "runs/detect"
    run_name = "glass_bottle_defect"

    # ---------- 断点续训 ----------
    if args.resume:
        logger.info(f"从断点继续训练: {args.resume}")
        model = YOLO(args.resume)
        train_results = model.train(resume=True)
    else:
        # ---------- 加载预训练模型 ----------
        logger.info(f"加载预训练模型: {args.weights}")
        model = YOLO(args.weights)

        # ---------- 打印训练配置 ----------
        logger.info("=" * 60)
        logger.info("训练配置:")
        logger.info(f"  数据集配置   : {args.data}")
        logger.info(f"  预训练权重   : {args.weights}")
        logger.info(f"  训练轮数     : {args.epochs}")
        logger.info(f"  批大小       : {args.batch}")
        logger.info(f"  图像尺寸     : {args.imgsz}")
        logger.info(f"  优化器       : AdamW")
        logger.info(f"  初始学习率   : 0.001")
        logger.info(f"  设备         : {args.device}")
        logger.info(f"  工作进程数   : {args.workers}")
        logger.info(f"  输出目录     : {project_name}/{run_name}/")
        logger.info("=" * 60)

        # ---------- 开始训练 ----------
        start_time = time.time()
        logger.info("开始训练 ...")
        train_results = model.train(
            data=args.data,            # 数据集配置文件
            epochs=args.epochs,        # 训练轮数
            batch=args.batch,          # 批大小
            imgsz=args.imgsz,          # 输入图像尺寸
            optimizer="AdamW",         # 优化器:AdamW
            lr0=0.001,                 # 初始学习率
            device=args.device,        # 训练设备
            workers=args.workers,      # DataLoader 工作进程数
            project=project_name,      # 项目输出根目录
            name=run_name,             # 本次运行名称
            exist_ok=True,             # 允许覆盖同名目录,便于重复实验
            verbose=True,              # 打印详细训练信息
        )
        elapsed = time.time() - start_time
        logger.info(f"训练完成,总耗时: {elapsed / 3600:.2f} 小时 "
                    f"({elapsed / 60:.2f} 分钟)")

    return model, run_name, project_name


# ------------------------------------------------------------------------------
# 4. 验证集评估
# ------------------------------------------------------------------------------
def evaluate(model, args, run_name: str, project_name: str, logger: logging.Logger):
    """在验证集上评估训练好的模型,输出 mAP@0.5 与 mAP@0.5:0.95 指标。

    调用 model.val() 进行验证,ultralytics 会自动使用 data.yaml 中定义的 val 集。
    评估结果(含 PR 曲线、F1 曲线等)保存至 {project}/{name}_val/ 目录。

    Args:
        model:       训练完成后的 YOLO 模型对象。
        args:        命令行参数。
        run_name:    训练运行名称。
        project_name:训练项目目录。
        logger:      日志记录器。
    """
    logger.info("=" * 60)
    logger.info("开始在验证集上评估模型 ...")

    # 调用 val 进行验证;save_json=True 保存 COCO 格式结果便于后续分析
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=project_name,
        name=f"{run_name}_val",     # 评估结果目录:runs/detect/glass_bottle_defect_val/
        exist_ok=True,
        save_json=True,
    )

    # ---------- 提取关键指标 ----------
    # metrics.box.map50     -> mAP@0.5
    # metrics.box.map       -> mAP@0.5:0.95
    # metrics.box.mp        -> 所有类别平均 Precision
    # metrics.box.mr        -> 所有类别平均 Recall
    map50 = metrics.box.map50       # mAP@0.5
    map = metrics.box.map           # mAP@0.5:0.95
    precision = metrics.box.mp      # 平均精确率
    recall = metrics.box.mr         # 平均召回率

    # 每个类别单独的 mAP@0.5(用于分析哪类缺陷检测效果差)
    try:
        map50_per_class = metrics.box.maps  # 各类别 mAP@0.5 数组
    except Exception:
        map50_per_class = None

    logger.info("-" * 60)
    logger.info("验证集评估结果:")
    logger.info(f"  mAP@0.5      : {map50:.4f}  ({map50 * 100:.2f}%)")
    logger.info(f"  mAP@0.5:0.95 : {map:.4f}  ({map * 100:.2f}%)")
    logger.info(f"  平均 Precision: {precision:.4f}")
    logger.info(f"  平均 Recall   : {recall:.4f}")

    if map50_per_class is not None:
        class_names = ["obvious_damage", "subtle_damage"]
        logger.info("  各类别 mAP@0.5:")
        for i, (cls, val) in enumerate(zip(class_names, map50_per_class)):
            logger.info(f"    [{i}] {cls:<8}: {val:.4f}  ({val * 100:.2f}%)")
    logger.info("-" * 60)

    return metrics


# ------------------------------------------------------------------------------
# 5. 程序入口
# ------------------------------------------------------------------------------
def main():
    """脚本主入口:解析参数 -> 配置日志 -> 训练 -> 评估。"""
    # 解析命令行参数
    args = parse_args()

    # 输出目录(与训练输出保持一致)
    output_dir = os.path.join("runs", "detect", "glass_bottle_defect")
    logger = setup_logger(output_dir)

    logger.info("玻璃瓶缺陷检测 YOLOv8 训练脚本启动")
    logger.info(f"Python 版本: {sys.version.split()[0]}")

    try:
        # ---------- 训练 ----------
        model, run_name, project_name = train(args, logger)

        # ---------- 评估 ----------
        evaluate(model, args, run_name, project_name, logger)

        # ---------- 提示最佳模型位置 ----------
        best_pt = os.path.join(project_name, run_name, "weights", "best.pt")
        last_pt = os.path.join(project_name, run_name, "weights", "last.pt")
        logger.info("=" * 60)
        logger.info("训练与评估全部完成!")
        logger.info(f"最佳模型权重 : {best_pt}")
        logger.info(f"最后一轮权重 : {last_pt}")
        logger.info(f"训练日志目录 : {output_dir}")
        logger.info("=" * 60)
        logger.info("提示: 可使用以下命令进行推理预测:")
        logger.info(f'  yolo predict model="{best_pt}" source="你的图片路径"')

    except KeyboardInterrupt:
        logger.warning("用户中断训练(Ctrl+C),已保存当前进度。")
        logger.warning(f"可使用 --resume {os.path.join(output_dir, 'weights', 'last.pt')} 继续训练。")
        sys.exit(1)
    except Exception as e:
        logger.error(f"训练/评估过程出错: {e}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()