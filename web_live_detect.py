# -*- coding: utf-8 -*-
"""
玻璃瓶缺陷检测 - 实时摄像头检测Web应用
支持摄像头实时捕获和检测
"""

import os
import cv2
import base64
import time
import threading
import numpy as np
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, Response
from ultralytics import YOLO
import torch

app = Flask(__name__)

# 配置
MODEL_PATH = "runs/detect/glass_bottle_defect/weights/best.pt"
UPLOAD_FOLDER = "detection_results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 类别定义
CLASS_NAMES = {0: "crack", 1: "bubble", 2: "scratch", 3: "stain"}
CLASS_NAMES_CN = {0: "裂纹", 1: "气泡", 2: "划痕", 3: "污渍"}
CLASS_COLORS = {
    0: (0, 0, 255),    # 裂纹 - 红色
    1: (0, 255, 255),  # 气泡 - 黄色
    2: (255, 0, 0),    # 划痕 - 蓝色
    3: (0, 255, 0),    # 污渍 - 绿色
}

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
    print(f"模型加载成功: {MODEL_PATH}, 设备: {device}")
else:
    print(f"模型文件不存在: {MODEL_PATH}")

# 全局变量
camera = None
camera_lock = threading.Lock()
detection_active = False
latest_frame = None
latest_detections = []
detection_count = 0


def draw_detections(image, results, conf_threshold=0.25):
    """绘制检测结果"""
    result_image = image.copy()
    detections = []
    class_counts = {}

    if model is None:
        return result_image, detections, class_counts

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            name_en = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            name_cn = CLASS_NAMES_CN.get(cls_id, name_en)

            # 绘制边界框
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness=2)

            # 绘制标签
            label = f"{name_cn}({name_en}) {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            label_y1 = max(y1 - text_h - baseline - 4, 0)
            label_y2 = y1
            cv2.rectangle(result_image, (x1, label_y1), (x1 + text_w + 4, label_y2), color, cv2.FILLED)
            cv2.putText(result_image, label, (x1 + 2, label_y2 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

            detections.append({
                "class_id": cls_id,
                "class_name": name_en,
                "class_name_cn": name_cn,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })
            class_counts[name_cn] = class_counts.get(name_cn, 0) + 1

    return result_image, detections, class_counts


def generate_frames():
    """生成视频流帧"""
    global camera, latest_frame, latest_detections, detection_active, detection_count

    while True:
        if camera is None or not camera.isOpened():
            time.sleep(0.1)
            continue

        with camera_lock:
            success, frame = camera.read()

        if not success:
            time.sleep(0.1)
            continue

        # 执行检测
        if detection_active and model is not None:
            results = model(frame, device=device, conf=0.25, verbose=False)
            frame, detections, class_counts = draw_detections(frame, results)
            latest_detections = detections
            detection_count = sum(class_counts.values())

        latest_frame = frame

        # 编码为JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()

        # 生成MJPEG流
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(0.033)  # ~30 FPS


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>玻璃瓶实时检测系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            text-align: center;
        }
        .header h1 { font-size: 28px; margin-bottom: 5px; }
        .header p { font-size: 14px; opacity: 0.9; }
        .content { padding: 25px; }
        .main-layout {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 25px;
        }
        @media (max-width: 900px) {
            .main-layout { grid-template-columns: 1fr; }
        }
        .video-section {
            background: #f8f9ff;
            border-radius: 15px;
            padding: 20px;
        }
        .video-container {
            position: relative;
            background: #000;
            border-radius: 10px;
            overflow: hidden;
            aspect-ratio: 4/3;
        }
        .video-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .video-placeholder {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #999;
            font-size: 18px;
            text-align: center;
        }
        .controls {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        .btn {
            padding: 12px 25px;
            border: none;
            border-radius: 8px;
            font-size: 15px;
            cursor: pointer;
            transition: all 0.3s;
            color: white;
            font-weight: bold;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
        .btn:disabled { background: #cbd5e0; cursor: not-allowed; transform: none; }
        .btn-start { background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); }
        .btn-stop { background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%); }
        .btn-detect { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .btn-capture { background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%); }
        .info-section {
            background: #f8f9ff;
            border-radius: 15px;
            padding: 20px;
        }
        .info-section h2 {
            color: #667eea;
            font-size: 20px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }
        .stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .stat-card .value {
            font-size: 28px;
            font-weight: bold;
            color: #667eea;
        }
        .stat-card .label {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        .detection-list {
            background: white;
            border-radius: 10px;
            padding: 15px;
            max-height: 300px;
            overflow-y: auto;
        }
        .detection-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            border-bottom: 1px solid #eee;
        }
        .detection-item:last-child { border-bottom: none; }
        .detection-badge {
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 12px;
            color: white;
            font-weight: bold;
        }
        .color-legend {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .color-item {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 13px;
        }
        .color-box {
            width: 16px;
            height: 16px;
            border-radius: 3px;
            border: 2px solid #333;
        }
        .capture-gallery {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }
        .capture-item {
            position: relative;
            border-radius: 8px;
            overflow: hidden;
            cursor: pointer;
            aspect-ratio: 4/3;
        }
        .capture-item img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        .capture-item .time {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(0,0,0,0.7);
            color: white;
            font-size: 10px;
            padding: 3px;
            text-align: center;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-on { background: #48bb78; animation: pulse 1.5s infinite; }
        .status-off { background: #cbd5e0; }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(72, 187, 120, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(72, 187, 120, 0); }
            100% { box-shadow: 0 0 0 0 rgba(72, 187, 120, 0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 玻璃瓶实时检测系统</h1>
            <p>摄像头实时捕获 · 动态缺陷检测 · 自动截图保存</p>
        </div>
        
        <div class="content">
            <div class="color-legend">
                <div class="color-item"><div class="color-box" style="background: #ff0000;"></div>裂纹</div>
                <div class="color-item"><div class="color-box" style="background: #ffff00;"></div>气泡</div>
                <div class="color-item"><div class="color-box" style="background: #0000ff;"></div>划痕</div>
                <div class="color-item"><div class="color-box" style="background: #00ff00;"></div>污渍</div>
            </div>
            
            <div class="main-layout">
                <div class="video-section">
                    <h2 style="color: #667eea; margin-bottom: 15px;">
                        <span class="status-indicator status-off" id="statusIndicator"></span>
                        实时画面
                    </h2>
                    <div class="video-container">
                        <img id="videoFeed" src="" alt="视频流" style="display: none;">
                        <div class="video-placeholder" id="videoPlaceholder">
                            摄像头未开启<br>请点击"开启摄像头"按钮
                        </div>
                    </div>
                    <div class="controls">
                        <button class="btn btn-start" id="startBtn" onclick="startCamera()">开启摄像头</button>
                        <button class="btn btn-stop" id="stopBtn" onclick="stopCamera()" disabled>关闭摄像头</button>
                        <button class="btn btn-detect" id="detectBtn" onclick="toggleDetection()" disabled>开启检测</button>
                        <button class="btn btn-capture" id="captureBtn" onclick="captureFrame()" disabled>截图保存</button>
                    </div>
                </div>
                
                <div class="info-section">
                    <h2>检测统计</h2>
                    <div class="stats">
                        <div class="stat-card">
                            <div class="value" id="totalCount">0</div>
                            <div class="label">当前缺陷数</div>
                        </div>
                        <div class="stat-card">
                            <div class="value" id="captureCount">0</div>
                            <div class="label">截图数量</div>
                        </div>
                    </div>
                    
                    <h2>检测结果</h2>
                    <div class="detection-list" id="detectionList">
                        <div style="text-align: center; color: #999; padding: 20px;">等待检测...</div>
                    </div>
                    
                    <h2 style="margin-top: 20px;">截图记录</h2>
                    <div class="capture-gallery" id="captureGallery">
                        <div style="text-align: center; color: #999; padding: 20px; grid-column: 1/-1;">暂无截图</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let cameraActive = false;
        let detectionOn = false;
        let captureCount = 0;
        let captures = [];
        
        function startCamera() {
            fetch('/camera/start', { method: 'POST' })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    cameraActive = true;
                    document.getElementById('videoFeed').src = '/video/feed';
                    document.getElementById('videoFeed').style.display = 'block';
                    document.getElementById('videoPlaceholder').style.display = 'none';
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    document.getElementById('detectBtn').disabled = false;
                    document.getElementById('captureBtn').disabled = false;
                    document.getElementById('statusIndicator').className = 'status-indicator status-on';
                    
                    // 开始更新检测结果
                    updateDetections();
                } else {
                    alert('开启摄像头失败: ' + data.error);
                }
            });
        }
        
        function stopCamera() {
            fetch('/camera/stop', { method: 'POST' })
            .then(r => r.json())
            .then(data => {
                cameraActive = false;
                detectionOn = false;
                document.getElementById('videoFeed').src = '';
                document.getElementById('videoFeed').style.display = 'none';
                document.getElementById('videoPlaceholder').style.display = 'block';
                document.getElementById('startBtn').disabled = false;
                document.getElementById('stopBtn').disabled = true;
                document.getElementById('detectBtn').disabled = true;
                document.getElementById('captureBtn').disabled = true;
                document.getElementById('detectBtn').textContent = '开启检测';
                document.getElementById('statusIndicator').className = 'status-indicator status-off';
            });
        }
        
        function toggleDetection() {
            detectionOn = !detectionOn;
            fetch('/detection/toggle', { 
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ active: detectionOn })
            })
            .then(r => r.json())
            .then(data => {
                document.getElementById('detectBtn').textContent = detectionOn ? '停止检测' : '开启检测';
            });
        }
        
        function captureFrame() {
            fetch('/capture', { method: 'POST' })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    captureCount++;
                    document.getElementById('captureCount').textContent = captureCount;
                    captures.unshift(data);
                    updateGallery();
                }
            });
        }
        
        function updateDetections() {
            if (!cameraActive) return;
            
            fetch('/detections')
            .then(r => r.json())
            .then(data => {
                document.getElementById('totalCount').textContent = data.total_count;
                
                const list = document.getElementById('detectionList');
                if (data.detections.length > 0) {
                    list.innerHTML = data.detections.map(d => {
                        const colors = ['#ff0000', '#ffff00', '#0000ff', '#00ff00'];
                        const color = colors[d.class_id] || '#999';
                        return `<div class="detection-item">
                            <span>${d.class_name_cn} (${d.class_name})</span>
                            <span class="detection-badge" style="background: ${color}">${d.confidence.toFixed(2)}</span>
                        </div>`;
                    }).join('');
                } else {
                    list.innerHTML = '<div style="text-align: center; color: #999; padding: 20px;">未检测到缺陷</div>';
                }
                
                setTimeout(updateDetections, 500);
            })
            .catch(() => setTimeout(updateDetections, 1000));
        }
        
        function updateGallery() {
            const gallery = document.getElementById('captureGallery');
            if (captures.length > 0) {
                gallery.innerHTML = captures.slice(0, 12).map(c => 
                    `<div class="capture-item">
                        <img src="data:image/jpeg;base64,${c.image}" onclick="window.open('data:image/jpeg;base64,${c.image}')">
                        <div class="time">${c.time}</div>
                    </div>`
                ).join('');
            }
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/video/feed')
def video_feed():
    """视频流端点"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera/start', methods=['POST'])
def camera_start():
    """开启摄像头"""
    global camera
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                return jsonify({"success": False, "error": "无法打开摄像头"})
    return jsonify({"success": True})


@app.route('/camera/stop', methods=['POST'])
def camera_stop():
    """关闭摄像头"""
    global camera, detection_active
    detection_active = False
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
    return jsonify({"success": True})


@app.route('/detection/toggle', methods=['POST'])
def detection_toggle():
    """切换检测"""
    global detection_active
    detection_active = request.json.get('active', False)
    return jsonify({"success": True, "active": detection_active})


@app.route('/detections')
def get_detections():
    """获取最新检测结果"""
    return jsonify({
        "detections": latest_detections,
        "total_count": detection_count
    })


@app.route('/capture', methods=['POST'])
def capture():
    """截图保存"""
    global latest_frame
    if latest_frame is None:
        return jsonify({"success": False, "error": "没有画面"})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(UPLOAD_FOLDER, f"capture_{timestamp}.jpg")
    cv2.imwrite(save_path, latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # 转换为base64用于显示
    _, buffer = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "success": True,
        "path": save_path,
        "image": img_base64,
        "time": datetime.now().strftime("%H:%M:%S")
    })


if __name__ == '__main__':
    print("=" * 50)
    print("玻璃瓶实时检测系统")
    print("=" * 50)
    print(f"模型: {'已加载' if model else '未加载'}")
    print(f"设备: {device}")
    print(f"访问地址: http://localhost:5001")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)