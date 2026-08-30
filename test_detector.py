import sys
sys.path.insert(0, r'D:\dlgd-pharma')

import cv2
import time
from backend.detector import DefectDetector

unetpp_path = r'D:\dlgd-pharma\train\weights\unetpp_best.pth'
cflow_path = r'D:\dlgd-pharma\train\weights\cflow_best.pth'

detector = DefectDetector(
    unetpp_path=unetpp_path,
    cflow_path=cflow_path,
    device='cpu',
    img_size=128,
    anomaly_threshold=0.5,
)

test_img = cv2.imread(r'D:\bottle\test\broken_large\000.png')
if test_img is not None:
    start = time.time()
    result = detector.detect(test_img)
    elapsed = time.time() - start
    
    print(f'图片尺寸: {test_img.shape}')
    print(f'推理用时: {elapsed:.2f}s')
    print(f'是否缺陷: {result["is_defective"]}')
    print(f'异常分数: {result["anomaly_score"]}')
    print(f'检测到 {len(result["detections"])} 个缺陷:')
    for det in result['detections']:
        print(f'  - {det["class_name_cn"]}({det["class_name"]}) 区域: {det["bbox"]}')
    print(f'各类别像素数: {result["class_counts"]}')
    
    cv2.imwrite(r'D:\dlgd-pharma\output\test_result.png', result['annotated'])
    print('结果已保存到 output/test_result.png')
else:
    print('无法读取测试图片')