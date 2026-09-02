# -*- coding: utf-8 -*-
"""从 COCO val2017 筛选玻璃制品(wine glass/cup/vase)并裁剪保存"""

import json
from pathlib import Path
import cv2

COCO = Path('data/coco')
OUT = Path('data/glass_normal')
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = {'wine glass', 'cup', 'vase'}
MIN_BBOX = 80

with open(COCO / 'annotations' / 'instances_val2017.json', encoding='utf-8') as f:
    ann = json.load(f)

cats = {c['id']: c['name'] for c in ann['categories']}
imgs = {im['id']: im['file_name'] for im in ann['images']}

saved = 0
for a in ann['annotations']:
    name = cats.get(a['category_id'])
    if name not in TARGETS:
        continue
    x, y, w, h = a['bbox']
    if w < MIN_BBOX or h < MIN_BBOX:
        continue
    f = COCO / 'val2017' / imgs[a['image_id']]
    if not f.exists():
        continue
    img = cv2.imread(str(f))
    if img is None:
        continue
    x, y, w, h = int(x), int(y), int(w), int(h)
    pad_x, pad_y = int(w * 0.1), int(h * 0.1)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(img.shape[1], x + w + pad_x), min(img.shape[0], y + h + pad_y)
    crop = img[y1:y2, x1:x2]
    if crop.shape[0] < 64 or crop.shape[1] < 64:
        continue
    out_name = f'coco_{name.replace(" ", "_")}_{a["id"]}.jpg'
    cv2.imwrite(str(OUT / out_name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    saved += 1

print(f'saved: {saved}')