# -*- coding: utf-8 -*-
"""
玻璃瓶缺陷智能检测系统 - Flask + YOLOv8
支持实时摄像头动态检测和图片上传静态检测
"""

import os
import cv2
import time
import yaml
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# ============================================================
# Flask 应用初始化
# ============================================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB 上传限制

# 目录配置
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
WEIGHTS_DIR = BASE_DIR / "weights"
STATIC_DIR = BASE_DIR / "static"

# 确保目录存在
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
WEIGHTS_DIR.mkdir(exist_ok=True)

# ============================================================
# 读取 data.yaml 获取类别配置
# ============================================================
def load_data_yaml():
    """动态读取 data.yaml 获取类名"""
    yaml_path = BASE_DIR / "data.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["names"], data["nc"]

CLASS_NAMES, NUM_CLASSES = load_data_yaml()

# 类名 → 颜色映射 (B, G, R)
CLASS_COLORS = {
    "broken_large":     (0, 0, 255),    # 红色
    "broken_small":     (255, 0, 0),    # 蓝色
    "contamination":    (0, 255, 0),    # 绿色
}

# 类名 → 中文名映射
CLASS_NAMES_CN = {
    "broken_large":     "大破损",
    "broken_small":     "小破损",
    "contamination":    "污染",
}

# ============================================================
# 全局加载 YOLOv8 模型
# ============================================================
MODEL_PATH = WEIGHTS_DIR / "best.pt"
DEVICE = "cpu"
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMGSZ = 640

print("=" * 60)
print("玻璃瓶缺陷智能检测系统")
print("=" * 60)
print(f"类别数: {NUM_CLASSES}")
print(f"类别名: {CLASS_NAMES}")

model = None
if MODEL_PATH.exists():
    model = YOLO(str(MODEL_PATH))
    model.conf = CONF_THRESHOLD
    model.iou = IOU_THRESHOLD
    print(f"模型加载成功: {MODEL_PATH}")
    print(f"设备: {DEVICE}, 置信度: {CONF_THRESHOLD}, IOU: {IOU_THRESHOLD}")
else:
    print(f"警告: 模型文件不存在 {MODEL_PATH}")
print("=" * 60)


# ============================================================
# 检测核心函数
# ============================================================
def detect_image(image):
    """
    对单张图片执行检测并绘制结果

    Args:
        image: BGR 格式的 numpy 数组

    Returns:
        annotated_image: 标注后的图像
        detections: 检测结果列表
        class_counts: 各类别计数
    """
    annotated = image.copy()
    detections = []
    class_counts = {name: 0 for name in CLASS_NAMES}

    if model is None:
        return annotated, detections, class_counts

    results = model(image, device=DEVICE, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                    imgsz=IMGSZ, verbose=False)

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            color = CLASS_COLORS.get(cls_name, (255, 255, 255))
            cls_cn = CLASS_NAMES_CN.get(cls_name, cls_name)

            # 绘制边界框
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # 绘制标签背景
            label = f"{cls_cn}({cls_name}) {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            ly1 = max(y1 - th - baseline - 4, 0)
            ly2 = y1
            cv2.rectangle(annotated, (x1, ly1), (x1 + tw + 4, ly2), color, cv2.FILLED)
            cv2.putText(annotated, label, (x1 + 2, ly2 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            detections.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "class_name_cn": cls_cn,
                "confidence": round(conf, 4),
                "bbox": [x1, y1, x2, y2]
            })
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    return annotated, detections, class_counts


def clear_output_dir():
    """清空 output 目录中的旧结果"""
    if OUTPUT_DIR.exists():
        for f in OUTPUT_DIR.glob("*"):
            if f.is_file():
                f.unlink()


# ============================================================
# 路由：主页
# ============================================================
@app.route("/")
def index():
    return render_template("index.html", class_names=CLASS_NAMES,
                           class_names_cn=CLASS_NAMES_CN)


# ============================================================
# 路由：摄像头实时视频流
# ============================================================
def generate_video_feed():
    """生成 MJPEG 视频流"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # 摄像头打开失败，返回提示帧
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "未检测到摄像头", (160, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', placeholder)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return

    fps_time = time.time()
    frame_count = 0
    fps = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 执行检测
            annotated, detections, class_counts = detect_image(frame)

            # 计算 FPS
            frame_count += 1
            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_time = time.time()

            # 在画面左上角显示 FPS
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 编码为 JPEG
            _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()


@app.route("/video_feed")
def video_feed():
    """视频流端点"""
    return Response(generate_video_feed(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ============================================================
# 路由：图片上传检测
# ============================================================
@app.route("/upload", methods=["POST"])
def upload():
    """接收图片文件（支持多张），逐张推理"""
    if "files" not in request.files:
        return jsonify({"success": False, "error": "未收到文件"})

    files = request.files.getlist("files")
    if not files or files[0].filename == "":
        return jsonify({"success": False, "error": "未选择文件"})

    # 每次推理前清空 output 目录
    clear_output_dir()

    results = []
    total_counts = {name: 0 for name in CLASS_NAMES}

    for file in files:
        filename = secure_filename(file.filename)
        if not filename:
            continue

        # 保存原始图片
        upload_path = UPLOAD_DIR / filename
        file.save(str(upload_path))

        # 读取图片
        image = cv2.imread(str(upload_path))
        if image is None:
            results.append({
                "filename": filename,
                "success": False,
                "error": "无法读取图片"
            })
            continue

        # 执行检测
        annotated, detections, class_counts = detect_image(image)

        # 保存标注结果
        output_filename = f"result_{filename}"
        output_path = OUTPUT_DIR / output_filename
        cv2.imwrite(str(output_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # 统计
        for name, count in class_counts.items():
            total_counts[name] = total_counts.get(name, 0) + count

        results.append({
            "filename": filename,
            "success": True,
            "original_url": f"/static/uploads/{filename}",
            "result_url": f"/static/output/{output_filename}",
            "detections": detections,
            "detection_count": len(detections),
            "class_counts": class_counts
        })

    return jsonify({
        "success": True,
        "results": results,
        "total_counts": total_counts,
        "total_defects": sum(total_counts.values())
    })


# ============================================================
# 路由：静态文件服务
# ============================================================
@app.route("/static/uploads/<filename>")
def serve_upload(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)


@app.route("/static/output/<filename>")
def serve_output(filename):
    return send_from_directory(str(OUTPUT_DIR), filename)


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    print(f"\n访问地址: http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)