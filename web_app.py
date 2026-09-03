# -*- coding: utf-8 -*-
"""玻璃瓶缺陷检测 Web 应用"""

import os
import base64
import io
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template_string, request, jsonify, send_file
import cv2
import numpy as np
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

# HTML模板
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
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { font-size: 32px; margin-bottom: 10px; }
        .header p { font-size: 16px; opacity: 0.9; }
        .content { padding: 30px; }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s;
            cursor: pointer;
            background: #f8f9ff;
        }
        .upload-area:hover {
            border-color: #764ba2;
            background: #f0f0ff;
        }
        .upload-area.dragover {
            border-color: #48bb78;
            background: #f0fff4;
        }
        .upload-icon { font-size: 64px; margin-bottom: 20px; }
        .upload-text { font-size: 18px; color: #555; margin-bottom: 20px; }
        .btn {
            display: inline-block;
            padding: 12px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s;
            margin: 5px;
        }
        .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
        .btn:disabled { background: #cbd5e0; cursor: not-allowed; transform: none; }
        .btn-success { background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); }
        .controls {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            margin-bottom: 30px;
            flex-wrap: wrap;
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
        .conf-value {
            font-weight: bold;
            color: #667eea;
            min-width: 40px;
        }
        .result-area {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-top: 30px;
        }
        @media (max-width: 768px) {
            .result-area { grid-template-columns: 1fr; }
        }
        .image-box {
            background: #f8f9ff;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
        }
        .image-box h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 18px;
        }
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
        .stats {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            margin-top: 30px;
            text-align: center;
        }
        .stats h2 { font-size: 24px; margin-bottom: 15px; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }
        .stat-item {
            background: rgba(255,255,255,0.2);
            padding: 15px;
            border-radius: 10px;
        }
        .stat-item .count { font-size: 32px; font-weight: bold; }
        .stat-item .label { font-size: 14px; opacity: 0.9; }
        .table-container {
            margin-top: 30px;
            background: white;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .table-container h3 {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 25px;
            font-size: 18px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px 25px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }
        th {
            background: #f8f9ff;
            color: #667eea;
            font-weight: bold;
        }
        tr:hover { background: #f8f9ff; }
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        .loading.show { display: block; }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .color-legend {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .color-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
        }
        .color-box {
            width: 20px;
            height: 20px;
            border-radius: 4px;
            border: 2px solid #333;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 玻璃瓶缺陷检测系统</h1>
            <p>基于 YOLOv8 的智能缺陷检测平台</p>
        </div>
        
        <div class="content">
            <div class="color-legend">
                <div class="color-item"><div class="color-box" style="background: #ff0000;"></div>明显破损 (obvious_damage)</div>
                <div class="color-item"><div class="color-box" style="background: #ffff00;"></div>非明显破损 (subtle_damage)</div>
            </div>
            
            <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">📁</div>
                <div class="upload-text">点击或拖拽图片到此处上传</div>
                <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="handleFileSelect(this)">
                <button class="btn" type="button">选择图片</button>
            </div>
            
            <div class="controls">
                <div class="conf-control">
                    <label>置信度阈值:</label>
                    <input type="range" id="confSlider" min="1" max="100" value="25" oninput="updateConfLabel()">
                    <span class="conf-value" id="confValue">0.25</span>
                </div>
                <button class="btn btn-success" id="detectBtn" onclick="detectImage()" disabled>开始检测</button>
                <button class="btn" id="saveBtn" onclick="saveResult()" disabled>保存结果</button>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>正在检测中，请稍候...</p>
            </div>
            
            <div class="result-area">
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
            
            <div class="stats" id="stats" style="display: none;">
                <h2>检测统计</h2>
                <div class="stats-grid" id="statsGrid"></div>
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
                            <th>位置 (x1, y1, x2, y2)</th>
                        </tr>
                    </thead>
                    <tbody id="detailTableBody"></tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script>
        let currentImage = null;
        let currentResult = null;
        
        // 上传区域拖拽支持
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0]);
            }
        });
        
        function handleFileSelect(input) {
            if (input.files.length > 0) {
                handleFile(input.files[0]);
            }
        }
        
        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                alert('请选择图片文件！');
                return;
            }
            
            const reader = new FileReader();
            reader.onload = (e) => {
                currentImage = e.target.result;
                document.getElementById('originalImage').src = currentImage;
                document.getElementById('originalImage').style.display = 'block';
                document.getElementById('originalPlaceholder').style.display = 'none';
                document.getElementById('detectBtn').disabled = false;
                document.getElementById('saveBtn').disabled = true;
                document.getElementById('resultPlaceholder').style.display = 'block';
                document.getElementById('resultPlaceholder').textContent = '等待检测';
                document.getElementById('resultImage').style.display = 'none';
                document.getElementById('stats').style.display = 'none';
                document.getElementById('tableContainer').style.display = 'none';
            };
            reader.readAsDataURL(file);
        }
        
        function updateConfLabel() {
            const value = document.getElementById('confSlider').value / 100;
            document.getElementById('confValue').textContent = value.toFixed(2);
        }
        
        function detectImage() {
            if (!currentImage) return;
            
            document.getElementById('loading').classList.add('show');
            document.getElementById('detectBtn').disabled = true;
            
            const conf = document.getElementById('confSlider').value / 100;
            
            fetch('/detect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: currentImage,
                    conf: conf
                })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').classList.remove('show');
                document.getElementById('detectBtn').disabled = false;
                
                if (data.success) {
                    currentResult = data;
                    document.getElementById('resultImage').src = 'data:image/jpeg;base64,' + data.result_image;
                    document.getElementById('resultImage').style.display = 'block';
                    document.getElementById('resultPlaceholder').style.display = 'none';
                    document.getElementById('saveBtn').disabled = false;
                    
                    // 显示统计
                    if (data.total_count > 0) {
                        document.getElementById('stats').style.display = 'block';
                        const statsGrid = document.getElementById('statsGrid');
                        statsGrid.innerHTML = `
                            <div class="stat-item">
                                <div class="count">${data.total_count}</div>
                                <div class="label">总缺陷数</div>
                            </div>
                            ${Object.entries(data.class_counts).map(([name, count]) => `
                                <div class="stat-item">
                                    <div class="count">${count}</div>
                                    <div class="label">${name}</div>
                                </div>
                            `).join('')}
                        `;
                    } else {
                        document.getElementById('stats').style.display = 'block';
                        document.getElementById('statsGrid').innerHTML = `
                            <div class="stat-item">
                                <div class="count">0</div>
                                <div class="label">未检测到缺陷</div>
                            </div>
                        `;
                    }
                    
                    // 显示详情表格
                    if (data.detections.length > 0) {
                        document.getElementById('tableContainer').style.display = 'block';
                        const tbody = document.getElementById('detailTableBody');
                        tbody.innerHTML = data.detections.map((det, i) => `
                            <tr>
                                <td>${i + 1}</td>
                                <td>${det.class_name}</td>
                                <td>${det.class_name_cn}</td>
                                <td>${det.confidence.toFixed(2)}</td>
                                <td>(${det.bbox[0]}, ${det.bbox[1]}, ${det.bbox[2]}, ${det.bbox[3]})</td>
                            </tr>
                        `).join('');
                    } else {
                        document.getElementById('tableContainer').style.display = 'none';
                    }
                } else {
                    alert('检测失败: ' + data.error);
                }
            })
            .catch(error => {
                document.getElementById('loading').classList.remove('show');
                document.getElementById('detectBtn').disabled = false;
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
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('结果已保存至: ' + data.path);
                } else {
                    alert('保存失败: ' + data.error);
                }
            })
            .catch(error => alert('保存失败: ' + error));
        }
    </script>
</body>
</html>
"""


def draw_detections(image, results, conf_threshold):
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


def image_to_base64(image):
    """将图片转换为base64编码"""
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.b64encode(buffer).decode('utf-8')


def base64_to_image(base64_str):
    """将base64编码转换为图片"""
    img_data = base64.b64decode(base64_str.split(',')[1] if ',' in base64_str else base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        image_data = data['image']
        conf_threshold = data.get('conf', 0.25)
        
        # 解码图片
        image = base64_to_image(image_data)
        
        # 执行检测
        if model is not None:
            results = model(image, device=device, conf=conf_threshold, verbose=False)
        else:
            results = []
        
        # 绘制检测结果
        result_image, detections, class_counts = draw_detections(image, results, conf_threshold)
        
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
        image_data = data['image']
        image = base64_to_image(f"data:image/jpeg;base64,{image_data}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(UPLOAD_FOLDER, f"result_{timestamp}.jpg")
        cv2.imwrite(save_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return jsonify({"success": True, "path": save_path})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


if __name__ == '__main__':
    print("=" * 50)
    print("玻璃瓶缺陷检测系统")
    print("=" * 50)
    print(f"模型: {'已加载' if model else '未加载（降级模式）'}")
    print(f"设备: {device}")
    print("访问地址: http://localhost:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False)