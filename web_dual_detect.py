# -*- coding: utf-8 -*-
"""
玻璃瓶缺陷检测 - 综合Web应用
支持两种模式：实时画面检测 + 图片上传检测
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
CLASS_NAMES = {0: "obvious_damage", 1: "subtle_damage"}
CLASS_NAMES_CN = {0: "明显破损", 1: "非明显破损"}
CLASS_COLORS = {
    0: (0, 0, 255),      # 明显破损 - 红色
    1: (0, 255, 255),    # 非明显破损 - 黄色
}

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
    print(f"模型加载成功: {MODEL_PATH}, 设备: {device}")
else:
    print(f"模型文件不存在: {MODEL_PATH}")

# 摄像头全局变量
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

            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, thickness=2)
            label = f"{name_cn}({name_en}) {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            label_y1 = max(y1 - text_h - baseline - 4, 0)
            label_y2 = y1
            cv2.rectangle(result_image, (x1, label_y1), (x1 + text_w + 4, label_y2), color, cv2.FILLED)
            cv2.putText(result_image, label, (x1 + 2, label_y2 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

            detections.append({
                "class_id": cls_id, "class_name": name_en, "class_name_cn": name_cn,
                "confidence": conf, "bbox": [x1, y1, x2, y2]
            })
            class_counts[name_cn] = class_counts.get(name_cn, 0) + 1

    return result_image, detections, class_counts


def image_to_base64(image, quality=95):
    """图片转base64"""
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')


def base64_to_image(base64_str):
    """base64转图片"""
    img_data = base64.b64decode(base64_str.split(',')[1] if ',' in base64_str else base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


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

        if detection_active and model is not None:
            results = model(frame, device=device, conf=0.25, verbose=False)
            frame, detections, class_counts = draw_detections(frame, results)
            latest_detections = detections
            detection_count = sum(class_counts.values())

        latest_frame = frame

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>玻璃瓶缺陷检测系统</title>
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
        .tabs {
            display: flex;
            background: #f8f9ff;
            border-bottom: 2px solid #e0e0e0;
        }
        .tab {
            flex: 1;
            padding: 18px;
            text-align: center;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            color: #666;
            transition: all 0.3s;
            border-bottom: 3px solid transparent;
        }
        .tab.active {
            color: #667eea;
            border-bottom-color: #667eea;
            background: white;
        }
        .tab:hover { background: #f0f0ff; }
        .content { padding: 25px; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        
        /* 通用样式 */
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
        .btn-save { background: linear-gradient(135deg, #38b2ac 0%, #319795 100%); }
        
        .color-legend {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .color-item { display: flex; align-items: center; gap: 6px; font-size: 13px; }
        .color-box { width: 16px; height: 16px; border-radius: 3px; border: 2px solid #333; }
        
        /* 实时检测样式 */
        .live-layout {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 25px;
        }
        @media (max-width: 900px) { .live-layout { grid-template-columns: 1fr; } }
        .video-section, .info-section {
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
        .video-container img { width: 100%; height: 100%; object-fit: cover; }
        .video-placeholder {
            position: absolute;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            color: #999;
            font-size: 18px;
            text-align: center;
        }
        .controls {
            display: flex;
            justify-content: center;
            gap: 12px;
            margin-top: 20px;
            flex-wrap: wrap;
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
        .stat-card .value { font-size: 28px; font-weight: bold; color: #667eea; }
        .stat-card .label { font-size: 12px; color: #666; margin-top: 5px; }
        .detection-list {
            background: white;
            border-radius: 10px;
            padding: 15px;
            max-height: 250px;
            overflow-y: auto;
        }
        .detection-item {
            display: flex;
            justify-content: space-between;
            padding: 8px;
            border-bottom: 1px solid #eee;
        }
        .detection-badge {
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 12px;
            color: white;
            font-weight: bold;
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
        
        /* 图片上传样式 */
        .upload-layout {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
        }
        @media (max-width: 900px) { .upload-layout { grid-template-columns: 1fr; } }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            transition: all 0.3s;
            cursor: pointer;
            background: #f8f9ff;
        }
        .upload-area:hover { border-color: #764ba2; background: #f0f0ff; }
        .upload-area.dragover { border-color: #48bb78; background: #f0fff4; }
        .upload-icon { font-size: 48px; margin-bottom: 15px; }
        .image-box {
            background: #f8f9ff;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
        }
        .image-box h3 { color: #667eea; margin-bottom: 15px; font-size: 18px; }
        .image-box img {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .placeholder {
            color: #999;
            font-size: 16px;
            padding: 60px 20px;
            background: #f0f0f0;
            border-radius: 10px;
        }
        .conf-control {
            display: flex;
            align-items: center;
            gap: 10px;
            background: #f8f9ff;
            padding: 15px 25px;
            border-radius: 10px;
        }
        .conf-control label { font-weight: bold; color: #555; }
        .conf-control input[type="range"] { width: 200px; }
        .conf-value { font-weight: bold; color: #667eea; min-width: 40px; }
        .loading { display: none; text-align: center; padding: 30px; }
        .loading.show { display: block; }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .table-container {
            margin-top: 20px;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .table-container h3 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 20px;
            font-size: 16px;
        }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px 20px; text-align: left; border-bottom: 1px solid #e0e0e0; }
        th { background: #f8f9ff; color: #667eea; font-weight: bold; }
        tr:hover { background: #f8f9ff; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 玻璃瓶缺陷检测系统</h1>
            <p>支持实时画面检测和图片上传检测两种模式</p>
        </div>
        
        <div class="tabs">
            <div class="tab active" onclick="switchTab('live')">📹 实时画面检测</div>
            <div class="tab" onclick="switchTab('upload')">📁 图片上传检测</div>
        </div>
        
        <div class="content">
            <div class="color-legend">
                <div class="color-item"><div class="color-box" style="background: #ff0000;"></div>明显破损</div>
                <div class="color-item"><div class="color-box" style="background: #ffff00;"></div>非明显破损</div>
            </div>
            
            <!-- 实时检测 -->
            <div class="tab-content active" id="liveTab">
                <div class="live-layout">
                    <div class="video-section">
                        <h2 style="color: #667eea; margin-bottom: 15px;">
                            <span class="status-indicator status-off" id="statusIndicator"></span>
                            实时画面
                        </h2>
                        <div class="video-container">
                            <img id="videoFeed" src="" style="display: none;">
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
                        <h2 style="color: #667eea; margin-bottom: 15px;">检测统计</h2>
                        <div class="stats">
                            <div class="stat-card">
                                <div class="value" id="liveTotalCount">0</div>
                                <div class="label">当前缺陷数</div>
                            </div>
                            <div class="stat-card">
                                <div class="value" id="captureCount">0</div>
                                <div class="label">截图数量</div>
                            </div>
                        </div>
                        <h2 style="color: #667eea; margin-bottom: 15px;">检测结果</h2>
                        <div class="detection-list" id="liveDetectionList">
                            <div style="text-align: center; color: #999; padding: 20px;">等待检测...</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- 图片上传检测 -->
            <div class="tab-content" id="uploadTab">
                <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                    <div class="upload-icon">📁</div>
                    <div style="font-size: 18px; color: #555; margin-bottom: 15px;">点击或拖拽图片到此处上传</div>
                    <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="handleFileSelect(this)">
                    <button class="btn btn-detect">选择图片</button>
                </div>
                
                <div class="controls" style="margin: 20px 0;">
                    <div class="conf-control">
                        <label>置信度阈值:</label>
                        <input type="range" id="confSlider" min="1" max="100" value="25" oninput="updateConfLabel()">
                        <span class="conf-value" id="confValue">0.25</span>
                    </div>
                    <button class="btn btn-detect" id="uploadDetectBtn" onclick="detectImage()" disabled>开始检测</button>
                    <button class="btn btn-save" id="saveBtn" onclick="saveResult()" disabled>保存结果</button>
                </div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>正在检测中，请稍候...</p>
                </div>
                
                <div class="upload-layout">
                    <div class="image-box">
                        <h3>原图</h3>
                        <div class="placeholder" id="originalPlaceholder">请上传图片</div>
                        <img id="originalImage" style="display: none;">
                    </div>
                    <div class="image-box">
                        <h3>检测结果</h3>
                        <div class="placeholder" id="resultPlaceholder">等待检测</div>
                        <img id="resultImage" style="display: none;">
                    </div>
                </div>
                
                <div class="stats" style="margin-top: 20px;">
                    <div class="stat-card">
                        <div class="value" id="uploadTotalCount">0</div>
                        <div class="label">检测缺陷数</div>
                    </div>
                    <div class="stat-card">
                        <div class="value" id="uploadDefectTypes">0</div>
                        <div class="label">缺陷类型数</div>
                    </div>
                </div>
                
                <div class="table-container" id="tableContainer" style="display: none;">
                    <h3>检测详情</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>序号</th>
                                <th>类别</th>
                                <th>中文名</th>
                                <th>置信度</th>
                                <th>位置</th>
                            </tr>
                        </thead>
                        <tbody id="detailTableBody"></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // 标签切换
        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            if (tab === 'live') {
                document.querySelectorAll('.tab')[0].classList.add('active');
                document.getElementById('liveTab').classList.add('active');
            } else {
                document.querySelectorAll('.tab')[1].classList.add('active');
                document.getElementById('uploadTab').classList.add('active');
            }
        }
        
        // ========== 实时检测 ==========
        let cameraActive = false;
        let detectionOn = false;
        let captureCount = 0;
        
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
                }
            });
        }
        
        function updateDetections() {
            if (!cameraActive) return;
            fetch('/detections')
            .then(r => r.json())
            .then(data => {
                document.getElementById('liveTotalCount').textContent = data.total_count;
                const list = document.getElementById('liveDetectionList');
                if (data.detections.length > 0) {
                    const colors = ['#ff0000', '#ffff00', '#0000ff', '#00ff00'];
                    list.innerHTML = data.detections.map(d => {
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
        
        // ========== 图片上传检测 ==========
        let currentImage = null;
        let currentResult = null;
        
        const uploadArea = document.querySelector('.upload-area');
        uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('dragover'); });
        uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
        });
        
        function handleFileSelect(input) {
            if (input.files.length > 0) handleFile(input.files[0]);
        }
        
        function handleFile(file) {
            if (!file.type.startsWith('image/')) { alert('请选择图片文件！'); return; }
            const reader = new FileReader();
            reader.onload = (e) => {
                currentImage = e.target.result;
                document.getElementById('originalImage').src = currentImage;
                document.getElementById('originalImage').style.display = 'block';
                document.getElementById('originalPlaceholder').style.display = 'none';
                document.getElementById('uploadDetectBtn').disabled = false;
                document.getElementById('saveBtn').disabled = true;
                document.getElementById('resultPlaceholder').style.display = 'block';
                document.getElementById('resultPlaceholder').textContent = '等待检测';
                document.getElementById('resultImage').style.display = 'none';
                document.getElementById('tableContainer').style.display = 'none';
                document.getElementById('uploadTotalCount').textContent = '0';
                document.getElementById('uploadDefectTypes').textContent = '0';
            };
            reader.readAsDataURL(file);
        }
        
        function updateConfLabel() {
            document.getElementById('confValue').textContent = (document.getElementById('confSlider').value / 100).toFixed(2);
        }
        
        function detectImage() {
            if (!currentImage) return;
            document.getElementById('loading').classList.add('show');
            document.getElementById('uploadDetectBtn').disabled = true;
            const conf = document.getElementById('confSlider').value / 100;
            
            fetch('/detect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: currentImage, conf: conf })
            })
            .then(r => r.json())
            .then(data => {
                document.getElementById('loading').classList.remove('show');
                document.getElementById('uploadDetectBtn').disabled = false;
                if (data.success) {
                    currentResult = data;
                    document.getElementById('resultImage').src = 'data:image/jpeg;base64,' + data.result_image;
                    document.getElementById('resultImage').style.display = 'block';
                    document.getElementById('resultPlaceholder').style.display = 'none';
                    document.getElementById('saveBtn').disabled = false;
                    document.getElementById('uploadTotalCount').textContent = data.total_count;
                    document.getElementById('uploadDefectTypes').textContent = Object.keys(data.class_counts).length;
                    if (data.detections.length > 0) {
                        document.getElementById('tableContainer').style.display = 'block';
                        document.getElementById('detailTableBody').innerHTML = data.detections.map((det, i) => 
                            `<tr>
                                <td>${i + 1}</td>
                                <td>${det.class_name}</td>
                                <td>${det.class_name_cn}</td>
                                <td>${det.confidence.toFixed(2)}</td>
                                <td>(${det.bbox[0]}, ${det.bbox[1]}, ${det.bbox[2]}, ${det.bbox[3]})</td>
                            </tr>`
                        ).join('');
                    } else {
                        document.getElementById('tableContainer').style.display = 'none';
                    }
                } else {
                    alert('检测失败: ' + data.error);
                }
            })
            .catch(error => {
                document.getElementById('loading').classList.remove('show');
                document.getElementById('uploadDetectBtn').disabled = false;
                alert('请求失败: ' + error);
            });
        }
        
        function saveResult() {
            if (!currentResult) return;
            fetch('/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: currentResult.result_image })
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) alert('结果已保存至: ' + data.path);
                else alert('保存失败: ' + data.error);
            });
        }
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


# ========== 实时检测路由 ==========
@app.route('/video/feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera/start', methods=['POST'])
def camera_start():
    global camera
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                return jsonify({"success": False, "error": "无法打开摄像头"})
    return jsonify({"success": True})

@app.route('/camera/stop', methods=['POST'])
def camera_stop():
    global camera, detection_active
    detection_active = False
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
    return jsonify({"success": True})

@app.route('/detection/toggle', methods=['POST'])
def detection_toggle():
    global detection_active
    detection_active = request.json.get('active', False)
    return jsonify({"success": True, "active": detection_active})

@app.route('/detections')
def get_detections():
    return jsonify({"detections": latest_detections, "total_count": detection_count})

@app.route('/capture', methods=['POST'])
def capture():
    global latest_frame
    if latest_frame is None:
        return jsonify({"success": False, "error": "没有画面"})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(UPLOAD_FOLDER, f"capture_{timestamp}.jpg")
    cv2.imwrite(save_path, latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return jsonify({"success": True, "path": save_path})


# ========== 图片上传检测路由 ==========
@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        image = base64_to_image(data['image'])
        conf = data.get('conf', 0.25)
        
        if model is not None:
            results = model(image, device=device, conf=conf, verbose=False)
        else:
            results = []
        
        result_image, detections, class_counts = draw_detections(image, results)
        
        return jsonify({
            "success": True,
            "result_image": image_to_base64(result_image),
            "detections": detections,
            "class_counts": class_counts,
            "total_count": len(detections)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/save', methods=['POST'])
def save():
    try:
        data = request.json
        image = base64_to_image(f"data:image/jpeg;base64,{data['image']}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(UPLOAD_FOLDER, f"result_{timestamp}.jpg")
        cv2.imwrite(save_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return jsonify({"success": True, "path": save_path})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == '__main__':
    print("=" * 50)
    print("玻璃瓶缺陷检测系统（双模式）")
    print("=" * 50)
    print(f"模型: {'已加载' if model else '未加载'}")
    print(f"设备: {device}")
    print(f"访问地址: http://localhost:5002")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)