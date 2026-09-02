# -*- coding: utf-8 -*-
"""从 Open Images test 集筛选 Eyeglasses 并裁剪镜片区域"""

import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
import cv2
import numpy as np

CSV_PATH = Path('data/openimages/test-bbox.csv')
OUT = Path('data/glass_normal')
RAW = Path('data/openimages/raw')
RAW.mkdir(parents=True, exist_ok=True)
S3 = 'https://open-images-dataset.s3.amazonaws.com/test/{}.jpg'
MAX_IMAGES = 300
WORKERS = 12

rows_by_image = {}
with open(CSV_PATH, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['LabelName'] != '/m/0jy4k':
            continue
        rows_by_image.setdefault(row['ImageID'], []).append(row)

print(f'含 Eyeglasses 的图: {len(rows_by_image)}')
selected = sorted(rows_by_image.items())[:MAX_IMAGES]


def process(item):
    image_id, rows = item
    raw_path = RAW / f'{image_id}.jpg'
    try:
        if not raw_path.exists():
            r = requests.get(S3.format(image_id), timeout=30)
            r.raise_for_status()
            raw_path.write_bytes(r.content)
        img = cv2.imread(str(raw_path))
        if img is None:
            return 0
        h, w = img.shape[:2]
        saved = 0
        for row in rows:
            if row.get('IsGroupOf') == '1':
                continue
            x1 = max(0, int(float(row['XMin']) * w))
            x2 = min(w, int(float(row['XMax']) * w))
            y1 = max(0, int(float(row['YMin']) * h))
            y2 = min(h, int(float(row['YMax']) * h))
            crop = img[y1:y2, x1:x2]
            if crop.shape[0] < 80 or crop.shape[1] < 80:
                continue
            cv2.imwrite(str(OUT / f'oi_glasses_{image_id}.jpg'), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
        raw_path.unlink(missing_ok=True)
        return saved
    except Exception as e:
        return 0


with ThreadPoolExecutor(WORKERS) as ex:
    results = list(ex.map(process, selected))
print(f'下载图: {len(selected)}, 裁剪保存: {sum(results)}')