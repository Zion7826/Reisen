# -*- coding: utf-8 -*-
"""
玻璃瓶缺陷智能检测系统 - Flask + UNet++ + Cflow
支持实时摄像头动态检测和图片上传静态检测
"""

import os
import sys
import cv2
import time
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename

sys.path.insert(0, str(Path(__file__).parent))
from backend.patchcore_detector import PatchCoreDetector

# ============================================================
# Flask 应用初始化
# ============================================================
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
WEIGHTS_DIR = BASE_DIR / "train" / "weights"
STATIC_DIR = BASE_DIR / "static"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# 类别配置
# ============================================================
CLASS_NAMES = ['broken_large', 'broken_small', 'contamination']
NUM_CLASSES = 3

CLASS_COLORS = {
    "broken_large":     (0, 0, 255),
    "broken_small":     (255, 0, 0),
    "contamination":    (0, 255, 0),
}

CLASS_NAMES_CN = {
    "broken_large":     "大破损",
    "broken_small":     "小破损",
    "contamination":    "污染",
}

# ============================================================
# 加载检测模型
# ============================================================
UNETPP_PATH = WEIGHTS_DIR / "unetpp_best.pth"
CFLOW_PATH = WEIGHTS_DIR / "cflow_best.pth"
PATCHCORE_BANK_PATH = WEIGHTS_DIR / "patchcore" / "memory_bank_glass_normal.npy"

print("=" * 60)
print("玻璃瓶缺陷智能检测系统 (PatchCore)")
print("=" * 60)
print(f"类别数: {NUM_CLASSES}")
print(f"类别名: {CLASS_NAMES}")

torch.set_num_threads(12)

detector = PatchCoreDetector(
    memory_bank_path=str(PATCHCORE_BANK_PATH),
    device='cpu',
    img_size=128,
    anomaly_threshold=15.0,
)
print("=" * 60)


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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
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

            result = detector.detect(frame)
            annotated = result['annotated']

            frame_count += 1
            elapsed = time.time() - fps_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_time = time.time()

            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()


@app.route("/video_feed")
def video_feed():
    return Response(generate_video_feed(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ============================================================
# 路由：图片上传检测
# ============================================================
@app.route("/upload", methods=["POST"])
def upload():
    if "files" not in request.files:
        return jsonify({"success": False, "error": "未收到文件"})

    files = request.files.getlist("files")
    if not files or files[0].filename == "":
        return jsonify({"success": False, "error": "未选择文件"})

    results = []
    total_counts = {name: 0 for name in CLASS_NAMES}

    for file in files:
        filename = secure_filename(file.filename)
        if not filename:
            continue

        upload_path = UPLOAD_DIR / filename
        file.save(str(upload_path))

        image = cv2.imread(str(upload_path))
        if image is None:
            results.append({"filename": filename, "success": False, "error": "无法读取图片"})
            continue

        result = detector.detect(image)

        output_filename = f"result_{filename}"
        output_path = OUTPUT_DIR / output_filename
        cv2.imwrite(str(output_path), result['annotated'], [cv2.IMWRITE_JPEG_QUALITY, 95])

        for name, count in result['class_counts'].items():
            total_counts[name] = total_counts.get(name, 0) + count

        results.append({
            "filename": filename,
            "success": True,
            "original_url": f"/static/uploads/{filename}",
            "result_url": f"/static/output/{output_filename}",
            "detections": result['detections'],
            "detection_count": len(result['detections']),
            "class_counts": result['class_counts'],
            "anomaly_score": result['anomaly_score'],
            "is_defective": result['is_defective'],
        })

    return jsonify({
        "success": True,
        "results": results,
        "total_counts": total_counts,
        "total_defects": sum(total_counts.values())
    })


# ============================================================
# 路由：三图对比
# ============================================================
@app.route("/compare", methods=["POST"])
def compare():
    if "file" not in request.files:
        return jsonify({"success": False, "error": "未收到文件"})

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"success": False, "error": "未选择文件"})

    filename = secure_filename(file.filename)
    upload_path = UPLOAD_DIR / filename
    file.save(str(upload_path))

    image = cv2.imread(str(upload_path))
    if image is None:
        return jsonify({"success": False, "error": "无法读取图片"})

    comparison = detector.generate_comparison(image)
    output_filename = f"compare_{filename}"
    output_path = OUTPUT_DIR / output_filename
    cv2.imwrite(str(output_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return jsonify({
        "success": True,
        "comparison_url": f"/static/output/{output_filename}",
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
