# 玻璃瓶缺陷检测系统 - 项目文档

<div align="center">

**项目名称**：玻璃瓶缺陷检测系统 (Glass Bottle Defect Detection System)

**版本**：v1.0

**开发语言**：Python 3.11

**核心框架**：YOLOv8 + PyQt5

**作者**：Reisen

**仓库地址**：[https://gitee.com/ccfun_studio/Reisen](https://gitee.com/ccfun_studio/Reisen)

**最后更新**：2026-08-19

</div>

---

## 📋 目录

- [项目概述](#项目概述)
- [缺陷类别定义](#缺陷类别定义)
- [环境要求与安装](#环境要求与安装)
- [项目目录结构](#项目目录结构)
- [数据集准备](#数据集准备)
- [模型训练](#模型训练)
- [模型推理](#模型推理)
- [自动标注](#自动标注)
- [GUI 使用说明](#gui-使用说明)
- [异常检测训练](#异常检测训练)
- [模型导出与部署](#模型导出与部署)
- [性能优化建议](#性能优化建议)
- [常见问题](#常见问题)
- [技术支持](#技术支持)
- [许可证](#许可证)

---

## 项目概述

本项目是一个基于 **YOLOv8** 深度学习模型的工业级玻璃瓶缺陷检测系统，集成了 **PyQt5** 桌面 GUI 界面，提供完整的缺陷检测解决方案。

### 核心功能

- ✅ **图片上传检测**：支持批量图片上传，自动进行缺陷检测
- ✅ **摄像头实时检测**：支持 USB 摄像头实时视频流检测
- ✅ **结果报告导出**：生成详细的检测报告，包含缺陷统计和可视化结果
- ✅ **自动标注工具**：基于预训练模型自动生成标注数据
- ✅ **模型训练管理**：完整的训练、验证、评估流程
- ✅ **降级运行模式**：无模型时仍可正常启动 GUI 进行界面测试

### 检测缺陷类型

系统可自动识别玻璃瓶表面的 **4 类缺陷**：

| 缺陷类型 | 英文标识 | 检测框颜色 | 典型特征 |
|---------|---------|-----------|---------|
| 🔴 裂纹 | crack | 红色 (0,0,255) | 玻璃表面线性或网状裂纹 |
| 🟡 气泡 | bubble | 青色 (255,255,0) | 透明或半透明气泡缺陷 |
| 🟢 划痕 | scratch | 黄色 (0,255,255) | 表面线性划痕 |
| 🔵 污渍 | stain | 绿色 (0,255,0) | 表面污渍或杂质 |

### 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                      PyQt5 GUI 界面层                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │图片上传  │  │摄像头检测│  │结果展示  │  │报告导出  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                        业务逻辑层                            │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │  DetectionWorker │  │  CameraWorker    │               │
│  │  (检测控制器)     │  │  (摄像头控制器)   │               │
│  └──────────────────┘  └──────────────────┘               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                        服务层                                │
│  ┌──────────────────┐  ┌──────────────────┐               │
│  │ DetectionService │  │ReportExportService│              │
│  │  (检测服务)       │  │  (报告导出服务)   │               │
│  └──────────────────┘  └──────────────────┘               │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      YOLOv8 推理引擎                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  YOLOv8n Model (best.pt)                              │  │
│  │  - 目标检测                                           │  │
│  │  - 置信度过滤                                         │  │
│  │  - NMS 非极大值抑制                                   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 缺陷类别定义

### 类别映射表

| 类别索引 (class_id) | 英文名称 (name) | 中文名称 | BGR 颜色 | RGB 颜色 | 十六进制 |
|---------------------|----------------|---------|---------|---------|---------|
| 0 | `crack` | 裂纹 | (0, 0, 255) | (255, 0, 0) | #FF0000 |
| 1 | `bubble` | 气泡 | (255, 255, 0) | (0, 255, 255) | #00FFFF |
| 2 | `scratch` | 划痕 | (0, 255, 255) | (255, 255, 0) | #FFFF00 |
| 3 | `stain` | 污渍 | (0, 255, 0) | (0, 255, 0) | #00FF00 |

### 配置文件说明

类别定义在 `data.yaml` 文件中：

```yaml
# data.yaml
path: datasets/glass_bottle  # 数据集根目录
train: images/train           # 训练集图片路径（相对）
val: images/val               # 验证集图片路径（相对）

names:
  0: crack     # 裂纹
  1: bubble    # 气泡
  2: scratch   # 划痕
  3: stain     # 污渍
```

**⚠️ 重要提示**：
- 类别索引顺序必须与 `data.yaml` 中的 `names` 字段严格一致
- 修改类别顺序会导致模型训练和推理结果错误
- GUI 层的颜色映射在 `glass_bottle_defect_gui/domain/defect_type_registry.py` 中维护

---

## 环境要求与安装

### 系统要求

| 组件 | 最低要求 | 推荐配置 |
|-----|---------|---------|
| 操作系统 | Windows 10 / Linux / macOS | Windows 11 / Ubuntu 20.04+ |
| Python | 3.10 | 3.11 |
| 内存 | 8 GB | 16 GB+ |
| GPU | 无（CPU 模式） | NVIDIA RTX 3060+ (6 GB+ VRAM) |
| CUDA | - | 11.8 / 12.1 |

### 依赖安装步骤

#### 步骤 1：创建虚拟环境（推荐）

```bash
# 使用 venv
python -m venv venv

# Windows 激活
venv\Scripts\activate

# Linux/macOS 激活
source venv/bin/activate
```

#### 步骤 2：安装 PyTorch（必须先安装）

**CPU 版本**（适用于无 GPU 环境）：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**GPU 版本**（CUDA 11.8）：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**GPU 版本**（CUDA 12.1）：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> 💡 **提示**：安装完成后可通过 `python -c "import torch; print(torch.cuda.is_available())"` 验证 GPU 是否可用。

#### 步骤 3：安装项目依赖

```bash
pip install -r requirements.txt
```

### 依赖清单详解

| 依赖包 | 版本 | 用途说明 |
|-------|------|---------|
| `opencv-python` | 4.10.0.84 | 图像读取、预处理、结果可视化 |
| `ultralytics` | 8.3.78 | YOLOv8 官方实现库 |
| `PyQt5` | 5.15.11 | 桌面 GUI 框架 |
| `numpy` | ≥1.26.0 | 数值计算基础库 |
| `pandas` | ≥2.0.0 | 数据处理与分析 |
| `matplotlib` | ≥3.7.0 | 训练曲线绘制 |
| `Pillow` | ≥10.0.0 | 图像格式转换 |
| `PyYAML` | ≥6.0 | 配置文件解析 |
| `requests` | ≥2.31.0 | HTTP 请求（模型下载等） |
| `scipy` | ≥1.11.0 | 科学计算辅助 |
| `tqdm` | ≥4.66.0 | 进度条显示 |
| `seaborn` | ≥0.12.0 | 数据可视化 |
| `psutil` | ≥5.9.0 | 系统资源监控 |
| `py-cpuinfo` | ≥9.0.0 | CPU 信息获取 |
| `onnx` | ≥1.16.0 | ONNX 模型格式支持 |
| `onnxruntime` | ≥1.18.0 | ONNX 模型推理 |

### 环境验证

安装完成后，运行以下命令验证环境：

```bash
python -c "
import torch
import cv2
import ultralytics
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'OpenCV: {cv2.__version__}')
print(f'Ultralytics: {ultralytics.__version__}')
"
```

---

## 项目目录结构

```
demo/
│
├── 📄 data.yaml                          # YOLO 数据配置文件（4类缺陷定义）
├── 📄 train.py                           # 模型训练脚本
├── 📄 train_anomaly.py                   # 异常检测训练脚本
├── 📄 detect.py                          # 单张图片检测脚本
├── 📄 auto_label.py                      # 自动标注脚本
├── 📄 requirements.txt                   # Python 依赖清单
├── 📄 best.pt                            # 训练好的最佳模型权重
├── 📄 yolov8n.pt                         # YOLOv8n 预训练权重
├── 📄 LICENSE                            # 开源许可证
│
├── 📁 datasets/glass_bottle/             # 数据集目录
│   ├── 📁 images/
│   │   ├── 📁 train/                     # 训练集图片
│   │   ├── 📁 val/                       # 验证集图片
│   │   └── 📁 test/                      # 测试集图片（可选）
│   └── 📁 labels/
│       ├── 📁 train/                     # 训练集标签（YOLO格式）
│       ├── 📁 val/                       # 验证集标签
│       └── 📁 test/                      # 测试集标签
│
├── 📁 runs/detect/glass_bottle_defect/   # 训练输出目录
│   ├── 📁 weights/
│   │   ├── 📄 best.pt                    # 最佳模型权重（验证集最优）
│   │   └── 📄 last.pt                    # 最后一轮权重
│   ├── 📄 results.csv                    # 训练过程指标数据
│   ├── 📄 train_log.txt                  # 详细训练日志
│   ├── 📄 confusion_matrix.png           # 混淆矩阵
│   ├── 📄 PR_curve.png                   # 精确率-召回率曲线
│   ├── 📄 F1_curve.png                   # F1分数曲线
│   └── 📄 ...                            # 其他训练可视化文件
│
├── 📁 logs/                              # 运行日志目录
│   └── 📄 gui.log                        # GUI 运行日志
│
├── 📁 detection_results/                 # 检测结果输出目录
├── 📁 debug_labels/                      # 自动标注调试图目录
│
├── 📁 ai_clean/                          # AI 清洗相关模块
├── 📁 ai_labels/                         # AI 标注相关模块
├── 📁 ai_redbox/                         # AI 红框检测模块
│
├── 📁 glass_bottle_defect_gui/           # PyQt5 GUI 应用包
│   ├── 📄 main.py                        # GUI 入口文件
│   ├── 📄 config.py                      # 全局配置
│   │
│   ├── 📁 domain/                        # 领域层
│   │   └── 📄 defect_type_registry.py    # 缺陷类型注册表
│   │
│   ├── 📁 controller/                    # 控制器层
│   │   ├── 📄 camera_worker.py           # 摄像头控制器
│   │   ├── 📄 detection_worker.py        # 检测控制器
│   │   └── 📄 upload_worker.py           # 上传控制器
│   │
│   ├── 📁 service/                       # 服务层
│   │   ├── 📄 detection_service.py       # 检测服务
│   │   └── 📄 report_export_service.py   # 报告导出服务
│   │
│   ├── 📁 presentation/                  # 表示层
│   │   └── 📄 main_window.py             # 主窗口
│   │
│   └── 📁 infrastructure/                # 基础设施层
│       ├── 📄 logger.py                  # 日志工具
│       ├── 📄 exception_handler.py       # 异常处理
│       └── 📄 error_handler.py           # 错误处理
│
└── 📁 doc/                               # 项目文档目录
    └── 📄 README.md                      # 本文档
```

---

## 数据集准备

### 数据集目录结构

```
datasets/glass_bottle/
├── images/
│   ├── train/          # 训练集图片（如：0001.jpg, 0002.jpg, ...）
│   ├── val/            # 验证集图片
│   └── test/           # 测试集图片（可选）
└── labels/
    ├── train/          # 训练集标签（如：0001.txt, 0002.txt, ...）
    ├── val/            # 验证集标签
    └── test/           # 测试集标签（可选）
```

### YOLO 标注格式说明

每个 `.txt` 标签文件对应一张同名图片，每行表示一个检测目标：

```
<class_id> <center_x> <center_y> <width> <height>
```

**参数说明**：

| 参数 | 类型 | 范围 | 说明 |
|-----|------|------|------|
| `class_id` | 整数 | 0-3 | 缺陷类别索引（0=crack, 1=bubble, 2=scratch, 3=stain） |
| `center_x` | 浮点数 | 0.0-1.0 | 目标中心点 X 坐标（归一化） |
| `center_y` | 浮点数 | 0.0-1.0 | 目标中心点 Y 坐标（归一化） |
| `width` | 浮点数 | 0.0-1.0 | 目标宽度（归一化） |
| `height` | 浮点数 | 0.0-1.0 | 目标高度（归一化） |

**坐标归一化公式**：

```python
center_x = (x_min + x_max) / 2 / image_width
center_y = (y_min + y_max) / 2 / image_height
width = (x_max - x_min) / image_width
height = (y_max - y_min) / image_height
```

### 标注示例

**图片文件**：`datasets/glass_bottle/images/train/0001.jpg`

**对应的标签文件**：`datasets/glass_bottle/labels/train/0001.txt`

```
0 0.512345 0.678901 0.024567 0.034567
1 0.234567 0.456789 0.012345 0.023456
3 0.789012 0.890123 0.034567 0.045678
```

**解释**：
- 第 1 行：检测到 1 个裂纹（class_id=0），中心在 (51.23%, 67.89%)，尺寸为 2.46% × 3.46%
- 第 2 行：检测到 1 个气泡（class_id=1），中心在 (23.46%, 45.68%)，尺寸为 1.23% × 2.35%
- 第 3 行：检测到 1 个污渍（class_id=3），中心在 (78.90%, 89.01%)，尺寸为 3.46% × 4.57%

### 数据放置步骤

1. **准备训练图片**
   ```bash
   # 将训练图片放入指定目录
   cp your_images/*.jpg datasets/glass_bottle/images/train/
   ```

2. **准备标注文件**
   ```bash
   # 将对应的标签文件放入指定目录
   cp your_labels/*.txt datasets/glass_bottle/labels/train/
   ```

3. **验证文件对应关系**
   ```bash
   # 确保图片和标签文件同名（仅扩展名不同）
   # 0001.jpg ↔ 0001.txt
   # 0002.jpg ↔ 0002.txt
   # ...
   ```

4. **准备验证集**（同理）
   ```bash
   cp val_images/*.jpg datasets/glass_bottle/images/val/
   cp val_labels/*.txt datasets/glass_bottle/labels/val/
   ```

5. **准备测试集**（可选）
   ```bash
   cp test_images/*.jpg datasets/glass_bottle/images/test/
   cp test_labels/*.txt datasets/glass_bottle/labels/test/
   ```

### 数据集质量建议

| 指标 | 建议值 | 说明 |
|-----|-------|------|
| 训练集图片数量 | ≥ 1000 张 | 每类缺陷至少 200 张 |
| 验证集比例 | 10%-20% | 用于模型选择和超参数调优 |
| 图片分辨率 | ≥ 640×640 | 较高分辨率有助于小目标检测 |
| 缺陷标注精度 | ±2 像素 | 标注框应紧贴缺陷边缘 |
| 类别平衡 | 各类数量相近 | 避免类别严重不平衡 |

---

## 模型训练

### 基本训练命令

#### 使用默认参数训练（GPU）

```bash
python train.py
```

#### CPU 训练（建议调小 batch size）

```bash
python train.py --device cpu --batch 8
```

### 自定义参数训练

```bash
python train.py \
  --weights yolov8n.pt \
  --data data.yaml \
  --epochs 200 \
  --batch 16 \
  --imgsz 640 \
  --device cuda \
  --workers 8
```

### 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `--weights` | str | `yolov8n.pt` | 预训练权重路径 |
| `--data` | str | `data.yaml` | 数据集配置文件路径 |
| `--epochs` | int | `200` | 训练总轮数 |
| `--batch` | int | `16` | 批大小（GPU 显存不足时减小） |
| `--imgsz` | int | `640` | 输入图像尺寸（短边） |
| `--device` | str | `cuda` | 训练设备：`cuda` / `cpu` / `0,1`（多GPU） |
| `--workers` | int | `8` | DataLoader 工作进程数（Windows 建议设为 0） |
| `--resume` | str | `None` | 从断点继续训练，指定 `last.pt` 路径 |

### 训练超参数

默认训练超参数（可在代码中修改）：

| 超参数 | 默认值 | 说明 |
|-------|-------|------|
| `optimizer` | `AdamW` | 优化器类型 |
| `lr0` | `0.001` | 初始学习率 |
| `lrf` | `0.01` | 最终学习率（lr0 * lrf） |
| `momentum` | `0.937` | SGD 动量 |
| `weight_decay` | `0.0005` | 权重衰减（L2 正则化） |
| `warmup_epochs` | `3.0` | 预热轮数 |
| `warmup_momentum` | `0.8` | 预热初始动量 |
| `warmup_bias_lr` | `0.1` | 预热初始偏置学习率 |
| `box` | `7.5` | 边界框损失权重 |
| `cls` | `0.5` | 分类损失权重 |
| `dfl` | `1.5` | 分布焦点损失权重 |
| `mosaic` | `1.0` | Mosaic 数据增强概率 |
| `mixup` | `0.0` | Mixup 数据增强概率 |

### 断点续训

训练过程中按 `Ctrl+C` 中断后，可从断点继续：

```bash
python train.py --resume runs/detect/glass_bottle_defect/weights/last.pt
```

### 训练输出说明

训练完成后，以下文件将生成在 `runs/detect/glass_bottle_defect/`：

```
runs/detect/glass_bottle_defect/
├── weights/
│   ├── best.pt              # 验证集上 mAP 最高的模型权重
│   └── last.pt              # 最后一轮训练的权重（用于断点续训）
├── results.csv              # 训练过程指标（epoch, mAP, loss, ...）
├── train_log.txt            # 详细训练日志
├── confusion_matrix.png     # 混淆矩阵（类别预测分析）
├── PR_curve.png             # 精确率-召回率曲线
├── F1_curve.png             # F1 分数曲线
├── P_curve.png              # 精确率曲线
├── R_curve.png              # 召回率曲线
├── loss.png                 # 损失曲线
├── labels.jpg               # 标签分布可视化
├── train_batch*.jpg         # 训练批次示例（含数据增强）
└── val_batch*.jpg           # 验证批次示例
```

### 评估指标说明

训练完成后会自动在验证集上评估，输出以下指标：

| 指标 | 全称 | 说明 | 目标值 |
|-----|------|------|-------|
| `mAP@0.5` | Mean Average Precision @ IoU=0.5 | IoU 阈值为 0.5 时的平均精度 | > 0.85 |
| `mAP@0.5:0.95` | Mean Average Precision @ IoU=0.5:0.95 | IoU 从 0.5 到 0.95 的平均精度（COCO 标准） | > 0.60 |
| `Precision` | Precision | 平均精确率（预测为正的样本中真正为正的比例） | > 0.85 |
| `Recall` | Recall | 平均召回率（真正为正的样本中被预测为正的比例） | > 0.85 |
| `mAP@0.5 (per class)` | Class-wise mAP@0.5 | 每类缺陷的单独 mAP | 各类 > 0.80 |

### 训练监控

#### 实时监控训练进度

训练过程中，Ultralytics 会自动启动 TensorBoard：

```bash
# 在另一个终端运行
tensorboard --logdir runs/detect
```

然后在浏览器打开 `http://localhost:6006` 查看实时训练曲线。

#### 查看训练日志

```bash
# 实时查看训练日志
tail -f runs/detect/glass_bottle_defect/train_log.txt
```

### 训练技巧

#### 1. 显存不足

```bash
# 减小 batch size
python train.py --batch 8

# 或减小图像尺寸
python train.py --imgsz 512
```

#### 2. 提升检测精度

```bash
# 增加训练轮数
python train.py --epochs 300

# 使用更大的预训练模型
python train.py --weights yolov8s.pt  # 或 yolov8m.pt, yolov8l.pt
```

#### 3. 加速训练

```bash
# 使用多 GPU
python train.py --device 0,1

# 增加 workers 数量（Linux）
python train.py --workers 16
```

---

## 模型推理

### 单张图片检测

#### 基本用法

```bash
python detect.py --image 图片路径.jpg
```

#### 自定义参数

```bash
python detect.py \
  --image 图片路径.jpg \
  --model best.pt \
  --conf 0.25 \
  --save-dir 输出目录
```

### 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `--image` | str | **必需** | 待检测图片路径（支持 jpg/png/bmp 等格式） |
| `--model` | str | `best.pt` | 模型权重文件路径 |
| `--conf` | float | `0.25` | 置信度阈值，低于该值的检测结果将被过滤 |
| `--save-dir` | str | `""` | 结果保存目录，默认与原图同目录 |
| `--no-show` | flag | `False` | 添加此参数则不弹窗显示结果 |

### 检测结果说明

#### 输出文件

- **结果图片**：保存为 `result_原文件名.jpg`
- **可视化内容**：
  - 检测框（彩色，对应缺陷类型）
  - 类别名称（中英文）
  - 置信度分数

#### 控制台输出

```
==================================================
玻璃瓶缺陷检测
==================================================
PyTorch 版本: 2.1.0+cu118
CUDA 是否可用: True
GPU 设备: NVIDIA GeForce RTX 3060
模型文件: best.pt
输入图片: test.jpg
置信度阈值: 0.25
--------------------------------------------------
使用设备: cuda
模型加载完成。
图片尺寸: 1920 x 1080
正在进行缺陷检测...
--------------------------------------------------
检测到 3 个缺陷目标：
  - 裂纹(crack): 1 个
  - 气泡(bubble): 1 个
  - 污渍(stain): 1 个
--------------------------------------------------
结果已保存至: result_test.jpg
检测完成。
```

### 批量检测

如需批量检测多张图片，可使用以下脚本：

```bash
# 创建批量检测脚本 batch_detect.sh
for img in images/*.jpg; do
  python detect.py --image "$img" --save-dir results/
done
```

---

## 自动标注

`auto_label.py` 可以使用训练好的模型对图片进行批量自动标注，生成 YOLO 格式的标签文件和带标注框的调试图。

### 基本用法

#### 使用默认参数（标注训练集）

```bash
python auto_label.py
```

#### 指定图片目录

```bash
python auto_label.py --image-dir datasets/glass_bottle/images/train
```

### 自定义参数

```bash
python auto_label.py \
  --model best.pt \
  --image-dir datasets/glass_bottle/images/train \
  --label-dir datasets/glass_bottle/labels/train \
  --debug-dir debug_labels \
  --conf 0.25 \
  --iou 0.45 \
  --img-size 640 \
  --overwrite \
  --limit 10
```

### 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `--model` | str | `best.pt` | 模型权重文件路径 |
| `--image-dir` | str | `datasets/glass_bottle/images/train` | 待标注图片目录 |
| `--label-dir` | str | `datasets/glass_bottle/labels/train` | 标签文件输出目录 |
| `--debug-dir` | str | `debug_labels` | 调试图输出目录 |
| `--conf` | float | `0.25` | 置信度阈值 |
| `--iou` | float | `0.45` | NMS IoU 阈值（去除重叠框） |
| `--img-size` | int | `640` | 推理时输入图片尺寸 |
| `--overwrite` | flag | `False` | 覆盖已存在的标签文件 |
| `--limit` | int | `0` | 仅处理前 N 张图片（0 表示全部） |
| `--device` | str | `""` | 推理设备（cuda/cpu，留空自动选择） |

### 输出说明

#### 标签文件

- 保存为 YOLO 格式的 `.txt` 文件
- 每行一个目标：`class_id center_x center_y width height`
- 坐标已归一化到 [0, 1]

#### 调试图

- 保存为带检测框的 `.jpg` 图片
- 包含检测框、类别名称、置信度
- 左上角显示统计信息（文件名、尺寸、目标数量）

#### 负样本处理

即使没有检测到任何目标，也会生成空标签文件（YOLO 训练要求负样本对应空标签）。

### 使用场景

1. **数据集扩充**：使用模型自动标注新图片，然后人工修正
2. **半监督学习**：自动标注 + 人工复核，提高标注效率
3. **质量检查**：检查已有标注是否遗漏

### 示例输出

```
============================================================
YOLO 自动标注脚本
============================================================
PyTorch 版本   : 2.1.0+cu118
CUDA 是否可用  : True
GPU 设备       : NVIDIA GeForce RTX 3060
推理设备       : cuda
模型文件       : best.pt
图片目录       : datasets/glass_bottle/images/train
标签输出目录   : datasets/glass_bottle/labels/train
调试图输出目录 : debug_labels
置信度阈值     : 0.25
IoU 阈值       : 0.45
推理图片尺寸   : 640
覆盖已有标签   : False
待处理图片数量 : 100
------------------------------------------------------------
正在加载模型...
模型加载完成。
------------------------------------------------------------
开始批量自动标注...
[1/100] 处理 0001.jpg ... 检测到 2 个目标 (crack:1, bubble:1)
[2/100] 处理 0002.jpg ... 检测到 0 个目标（已写入空标签文件）
[3/100] 处理 0003.jpg ... 检测到 1 个目标 (stain:1)
...
============================================================
自动标注完成！汇总统计：
------------------------------------------------------------
总图片数量     : 100
实际处理数量   : 100
跳过数量       : 0
检测目标总数   : 156
各类别分布     :
  - 裂纹(crack): 42 个
  - 气泡(bubble): 38 个
  - 划痕(scratch): 35 个
  - 污渍(stain): 41 个
------------------------------------------------------------
耗时           : 12.34 秒
平均速度       : 8.10 张/秒
============================================================
```

---

## GUI 使用说明

### 启动 GUI

```bash
python -m glass_bottle_defect_gui.main
```

### GUI 功能模块

#### 1. 图片上传检测

- 点击「上传图片」按钮选择图片
- 系统自动进行缺陷检测
- 检测结果实时显示在界面上
- 支持多张图片批量上传

#### 2. 摄像头实时检测

- 点击「打开摄像头」按钮启动摄像头
- 实时显示检测画面和缺陷标注
- 点击「关闭摄像头」停止检测

#### 3. 结果报告导出

- 检测完成后可导出详细报告
- 报告包含：
  - 检测图片（原图 + 标注图）
  - 缺陷统计信息
  - 各类缺陷数量分布
  - 检测时间戳

### 降级模式说明

#### 什么是降级模式？

GUI 默认模型路径为：`runs/detect/glass_bottle_defect/weights/best.pt`

如果该文件不存在（训练前），GUI 会自动进入**降级占位检测模式**：

- ✅ GUI 可正常启动
- ✅ 所有界面功能均可操作（上传图片、打开摄像头、导出报告等）
- ⚠️ 检测结果返回空缺陷列表（不报错、不崩溃）
- 📝 日志输出：`模型权重不存在...进入降级占位检测模式`

#### 退出降级模式

完成训练后，`best.pt` 文件生成，再次启动 GUI 即可自动加载真实检测模型，**无需修改任何代码**。

#### 使用建议

- 训练前：可直接启动 GUI 熟悉界面和功能
- 训练后：重启 GUI 即可进行真实缺陷检测

### GUI 日志

GUI 运行日志保存在 `logs/gui.log`，遇到问题时可查看：

```bash
tail -f logs/gui.log
```

---

## 异常检测训练

`train_anomaly.py` 用于训练异常检测模型（无监督学习）。

### 基本用法

```bash
python train_anomaly.py
```

### 参数说明

与 `train.py` 类似，支持自定义 `epochs`、`batch`、`device` 等参数：

```bash
python train_anomaly.py \
  --epochs 100 \
  --batch 16 \
  --device cuda
```

### 异常检测原理

异常检测模型仅需正常样本进行训练，无需标注数据：

1. **训练阶段**：模型学习正常样本的特征分布
2. **推理阶段**：计算输入样本与正常分布的偏差分数
3. **判定**：偏差分数超过阈值则判定为异常

---

## 模型导出与部署

### 导出 ONNX 模型

#### 方法 1：训练时导出

```bash
python train.py --export_onnx
```

#### 方法 2：使用 Python API

```python
from ultralytics import YOLO

model = YOLO("best.pt")
model.export(format="onnx")
```

### 导出其他格式

```python
from ultralytics import YOLO

model = YOLO("best.pt")

# 导出为 TorchScript
model.export(format="torchscript")

# 导出为 CoreML (iOS/macOS)
model.export(format="coreml")

# 导出为 TFLite (Android/移动端)
model.export(format="tflite")

# 导出为 TensorRT (NVIDIA GPU 加速)
model.export(format="engine")
```

### ONNX 模型推理示例

```python
import cv2
import onnxruntime as ort

# 加载 ONNX 模型
session = ort.InferenceSession("best.onnx")

# 读取并预处理图片
image = cv2.imread("test.jpg")
input_blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True)

# 推理
outputs = session.run(None, {session.get_inputs()[0].name: input_blob})

# 后处理检测结果
boxes = outputs[0]  # [1, 8400, 4]
scores = outputs[1]  # [1, 8400, 4]
```

---

## 性能优化建议

### 训练性能优化

| 优化项 | 方法 | 预期提升 |
|-------|------|---------|
| 数据加载 | 增加 `--workers`（Linux） | 10-30% |
| 混合精度训练 | 使用 AMP（自动混合精度） | 30-50% |
| 多 GPU 训练 | `--device 0,1,2,3` | 3-4x |
| 梯度累积 | 增大有效 batch size | 稳定性提升 |

### 推理性能优化

| 优化项 | 方法 | 预期提升 |
|-------|------|---------|
| 模型量化 | INT8 量化 | 2-4x |
| TensorRT | NVIDIA GPU 推理加速 | 3-5x |
| ONNX Runtime | 跨平台优化 | 1.5-2x |
| 批量推理 | 一次处理多张图片 | 1.5-2x |

### 检测精度优化

| 优化项 | 方法 | 预期提升 |
|-------|------|---------|
| 数据增强 | Mosaic, Mixup | 2-5% mAP |
| 更大模型 | yolov8s → yolov8m → yolov8l | 3-8% mAP |
| 更多训练数据 | 数据集扩充 | 5-15% mAP |
| 测试时增强（TTA） | 多尺度、翻转 | 1-3% mAP |

---

## 常见问题

### Q1: 训练时显存不足怎么办？

**A1:** 减小 batch size 或图像尺寸：

```bash
python train.py --batch 8 --imgsz 512
```

### Q2: CPU 训练速度太慢？

**A2:** 建议使用 GPU 训练，或减小训练规模：

```bash
python train.py --epochs 50 --batch 4
```

### Q3: 检测结果不准确？

**A3:** 尝试以下方法：
- 增加训练数据量
- 调整置信度阈值：`--conf 0.3`
- 重新训练更多轮次：`--epochs 300`
- 使用更大的预训练模型：`--weights yolov8s.pt`

### Q4: GUI 启动后无法检测？

**A4:** 检查以下几点：
- `best.pt` 是否存在：`ls runs/detect/glass_bottle_defect/weights/`
- 查看日志：`cat logs/gui.log`
- 确认已完成模型训练

### Q5: 如何导出 ONNX 模型？

**A5:** 使用以下命令：

```bash
python train.py --export_onnx
```

或使用 Python API：

```python
from ultralytics import YOLO
model = YOLO("best.pt")
model.export(format="onnx")
```

### Q6: Windows 下训练报错 `BrokenPipeError`？

**A6:** 设置 `workers=0`：

```bash
python train.py --workers 0
```

### Q7: 如何使用自定义数据集？

**A7:** 修改 `data.yaml` 文件：

```yaml
path: /path/to/your/dataset
train: images/train
val: images/val

names:
  0: your_class_1
  1: your_class_2
  # ...
```

### Q8: 模型训练中断如何恢复？

**A8:** 使用断点续训：

```bash
python train.py --resume runs/detect/glass_bottle_defect/weights/last.pt
```

### Q9: 如何评估模型在测试集上的性能？

**A9:** 运行验证命令：

```bash
yolo val model=best.pt data=data.yaml
```

### Q10: 检测速度太慢如何优化？

**A10:** 尝试以下方法：
- 使用 GPU 推理：`--device cuda`
- 减小输入尺寸：`--imgsz 480`
- 导出 TensorRT 模型（NVIDIA GPU）

---

## 技术支持

### 获取帮助

- 📖 **项目文档**：参考 `README.md` 和 `doc/README.md`
- 🐛 **问题反馈**：在 [Gitee Issues](https://gitee.com/ccfun_studio/Reisen/issues) 中提交
- 💬 **代码注释**：查看源代码中的详细注释

### 相关资源

- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [PyQt5 官方文档](https://www.riverbankcomputing.com/static/Docs/PyQt5/)
- [OpenCV 官方文档](https://docs.opencv.org/)

---

## 许可证

本项目采用开源许可证，详见 [LICENSE](../LICENSE) 文件。

---

## 更新日志

### v1.0 (2026-08-19)

- ✨ 初始版本发布
- ✅ 支持 4 类缺陷检测（裂纹、气泡、划痕、污渍）
- ✅ 集成 PyQt5 GUI 界面
- ✅ 支持图片上传和摄像头实时检测
- ✅ 提供自动标注工具
- ✅ 完整的训练和评估流程

---

<div align="center">

**感谢使用玻璃瓶缺陷检测系统！** 🎉

如有问题或建议，欢迎反馈。

</div>