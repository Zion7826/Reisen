# 玻璃瓶缺陷检测项目

基于 YOLOv8 的玻璃瓶缺陷检测，配套 PyQt5 桌面 GUI，支持图片上传检测、摄像头实时检测与结果报告导出。

---

## 1. 缺陷类别定义

比赛要求 **4 类**缺陷，类别索引（class id）与 `data.yaml` 的 `names` 字段一一对应，**顺序不可随意调整**：

| class id | 英文键     | 中文名 | 框颜色（BGR）     | 说明       |
|----------|-----------|--------|-------------------|------------|
| 0        | `crack`   | 裂纹   | `(0, 0, 255)` 红  | 玻璃表面裂纹 |
| 1        | `bubble`  | 气泡   | `(255, 255, 0)` 青| 透明气泡缺陷 |
| 2        | `scratch` | 划痕   | `(0, 255, 255)` 黄| 表面划痕   |
| 3        | `stain`   | 污渍   | `(0, 255, 0)` 绿  | 表面污渍   |

类别权威来源为 `data.yaml`，GUI 层颜色与中文名补充映射在
`glass_bottle_defect_gui/domain/defect_type_registry.py` 中维护，二者保持一致。

> 已移除旧版本中多余的 `chip`（缺口）与 `deformation`（变形）两类。

---

## 2. 环境要求与安装

- Python **3.11**（推荐）
- PyTorch 需通过专用 index-url 安装，不写入 requirements.txt

### 2.1 安装 PyTorch（先装）

```bash
# CPU 版
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU 版（以 CUDA 11.8 为例）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2.2 安装其余依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 已声明如下依赖（含常被遗漏的辅助库）：
`opencv-python`、`ultralytics`、`PyQt5`、`numpy`、`pandas`、`matplotlib`、
`Pillow`、`PyYAML`、`requests`、`scipy`、`tqdm`、`seaborn`、`psutil`、
`py-cpuinfo`、`onnx`、`onnxruntime`。

若仅需补装曾缺失的辅助库，可单独执行：

```bash
pip install pandas scipy onnx onnxruntime tqdm seaborn py-cpuinfo
```

---

## 3. 数据集准备

数据集目录为 `datasets/glass_bottle/`，已建好占位结构：

```
datasets/glass_bottle/
├── images/{train,val,test}/   # 放图像
└── labels/{train,val,test}/   # 放同名 .txt 标注
```

### 3.1 标注格式（YOLO 归一化）

每个 `.txt` 每行一个目标：

```
class_id  center_x  center_y  width  height
```

`center_x / center_y / width / height` 均为相对图像宽高的归一化值（0~1），
`class_id` 取自上表第 1 节（0=crack, 1=bubble, 2=scratch, 3=stain）。

### 3.2 放数据步骤

1. 训练图像放入 `datasets/glass_bottle/images/train/`，对应标注放入 `labels/train/`，
   图像与标注**同名**（仅扩展名不同）：`0001.jpg` ↔ `0001.txt`。
2. 验证集同理放入 `images/val/` 与 `labels/val/`。
3. 测试集（可选）放入 `images/test/` 与 `labels/test/`。
4. 详细说明见 `datasets/glass_bottle/README.txt`。

---

## 4. 模型训练

在项目根目录执行：

```bash
python train.py
```

常用可选参数：

```bash
python train.py --epochs 200 --batch 16 --imgsz 640 --device 0 --export_onnx
```

训练产出默认保存到：

```
runs/detect/glass_bottle_defect/weights/best.pt   # 最优权重
runs/detect/glass_bottle_defect/weights/last.pt   # 最后权重
```

该路径即 GUI 默认加载的模型路径（见下节）。

---

## 5. GUI 使用与降级模式说明

启动 GUI：

```bash
python -m glass_bottle_defect_gui.main
```

### 5.1 降级模式（重要）

GUI 默认模型路径为：

```
runs/detect/glass_bottle_defect/weights/best.pt
```

**该文件在训练之前不存在属于正常现象**。此时 `DetectionService` 会自动进入
**降级占位检测模式**：

- GUI 可正常启动、上传图片、打开摄像头、导出报告等所有界面功能均可操作；
- 检测结果返回**空缺陷列表**（不报错、不崩溃），并在日志中输出
  `模型权重不存在...进入降级占位检测模式` 提示；
- 完成训练后 `best.pt` 生成，再次启动 GUI 即可自动加载并执行真实检测，**无需改任何代码**。

因此：训练前直接启动 GUI 用于熟悉界面是安全的；要获得真实检测结果，请先按第 4 节完成训练。

---

## 6. 目录结构概览

```
demo/
├── data.yaml                              # YOLO 数据配置（4 类）
├── train.py                               # 训练脚本
├── requirements.txt                       # 依赖清单
├── datasets/glass_bottle/                 # 数据集目录（占位）
├── runs/detect/glass_bottle_defect/...    # 训练产出（训练后生成）
└── glass_bottle_defect_gui/               # PyQt5 GUI 包
    ├── main.py                            # 入口
    ├── config.py                          # 全局配置（模型路径等）
    ├── domain/                            # 领域层（缺陷类型注册表、数据模型）
    ├── controller/                        # 控制器（摄像头/检测/上传 worker）
    ├── service/                           # 服务层（检测服务、报告导出）
    ├── presentation/                      # 表示层（主窗口、结果渲染）
    └── infrastructure/                    # 基础设施（日志、异常、错误处理）
```