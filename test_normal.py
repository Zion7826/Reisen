import sys
sys.path.insert(0, r'D:\dlgd-pharma')
import cv2
from backend.detector import DefectDetector

d = DefectDetector(
    unetpp_path=r'D:\dlgd-pharma\train\weights\unetpp_best.pth',
    cflow_path=r'D:\dlgd-pharma\train\weights\cflow_best.pth',
    img_size=128
)

for img_path in [r'D:\bottle\test\broken_large\000.png', r'D:\bottle\test\broken_small\000.png', r'D:\bottle\test\contamination\000.png', r'D:\bottle\test\good\000.png']:
    img = cv2.imread(img_path)
    r = d.detect(img)
    name = img_path.split('\\')[-2]
    total = sum(r["class_counts"].values())
    print(f'{name}: 框数={total}, 各类={r["class_counts"]}, 检测列表={len(r["detections"])}')
