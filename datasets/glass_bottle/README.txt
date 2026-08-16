玻璃瓶缺陷检测数据集目录
========================

本目录遵循 YOLOv8 数据集目录结构，由项目根目录的 data.yaml 引用
（data.yaml 中 path: ./datasets/glass_bottle）。

目录结构：
  datasets/glass_bottle/
  ├── images/
  │   ├── train/   训练集图像（.jpg/.jpeg/.png/.bmp）
  │   ├── val/     验证集图像
  │   └── test/    测试集图像（可选）
  └── labels/
      ├── train/   训练集标注（与图像同名的 .txt）
      ├── val/     验证集标注
      └── test/    测试集标注（可选）

类别定义（共 4 类，class id 与 data.yaml 的 names 一一对应，顺序不可调整）：
  0 -> crack    裂纹
  1 -> bubble   气泡
  2 -> scratch  划痕
  3 -> stain    污渍

标注文件格式（每行一个目标，YOLO 归一化格式）：
  class_id  center_x  center_y  width  height
  其中 center_x / center_y / width / height 均为相对图像宽高的归一化值（0~1）。

示例（一张图含一条裂纹和一个气泡）：
  0 0.512 0.430 0.180 0.120
  1 0.250 0.600 0.080 0.080

放数据方法：
  1. 将训练集图像放入 images/train/，对应标注 .txt 放入 labels/train/，
     图像与标注文件名必须一致（仅扩展名不同），例如：
       images/train/0001.jpg  <->  labels/train/0001.txt
  2. 验证集同理放入 images/val/ 与 labels/val/。
  3. 测试集（可选）放入 images/test/ 与 labels/test/。
  4. 放好数据后，在项目根目录执行：python train.py

注意：当前各子目录仅含 .gitkeep 占位文件用于保留目录结构，
放入真实数据后可保留或删除 .gitkeep，不影响训练。