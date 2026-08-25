# -*- coding: utf-8 -*-
"""
数据增强扩充MVTec bottle训练集
将209张正常图片增强为627张（x3）
"""

import os
import cv2
import numpy as np
from pathlib import Path

src_dir = Path(r'D:\bottle\train\good')
dst_dir = Path(r'D:\dlgd-pharma\data\bottle_augmented\train\good')
dst_dir.mkdir(parents=True, exist_ok=True)

count = 0
for img_path in sorted(src_dir.glob('*.png')):
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    name = img_path.stem

    cv2.imwrite(str(dst_dir / f'{name}.png'), img)
    count += 1

    flipped_h = cv2.flip(img, 1)
    cv2.imwrite(str(dst_dir / f'{name}_fh.png'), flipped_h)
    count += 1

    flipped_v = cv2.flip(img, 0)
    cv2.imwrite(str(dst_dir / f'{name}_fv.png'), flipped_v)
    count += 1

print(f'增强完成: {count} 张图片 (原始 {len(list(src_dir.glob("*.png")))} 张 -> {count} 张)')
print(f'保存到: {dst_dir}')