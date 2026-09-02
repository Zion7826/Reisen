# -*- coding: utf-8 -*-
"""从 COCO train2017 按标注直链下载玻璃制品图并裁剪到 data/glass_normal"""

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
import cv2

COCO = Path('data/coco')
OUT = Path('data/glass_normal')
RAW = Path('data/coco/raw')
RAW.mkdir(parents=True, exist_ok=True)
TARGETS = {'wine glass', 'cup', 'vase'}
MIN_BBOX = 80
MAX_IMAGES = 300
MAX_ANN_PER_IMAGE = 3
WORKERS = 12

print('解析标注...')
with open(COCO / 'annotations' / 'instances_train2017.json', encoding='utf-8') as f:
    ann = json.load(f)

cats = {c['id']: c['name'] for c in ann['categories']}
imgs = {im['id']: im for im in ann['images']}

per_image = {}
for a in ann['annotations']:
    if cats.get(a['category_id']) not in TARGETS:
        continue
    x, y, w, h = a['bbox']
    if w < MIN_BBOX or h < MIN_BBOX:
        continue
    per_image.setdefault(a['image_id'], []).append(a)

print(f'候选图: {len(per_image)}')
selected = sorted(per_image.items(), key=lambda kv: -len(kv[1]))[:MAX_IMAGES]


def process(item):
    image_id, anns = item
    url = imgs[image_id]['coco_url']
    raw_path = RAW / f'{image_id}.jpg'
    try:
        if not raw_path.exists():
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            raw_path.write_bytes(r.content)
        img = cv2.imread(str(raw_path))
        if img is None:
            return 0
        saved = 0
        for a in anns[:MAX_ANN_PER_IMAGE]:
            x, y, w, h = [int(v) for v in a['bbox']]
            pad_x, pad_y = int(w * 0.1), int(h * 0.1)
            x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
            x2, y2 = min(img.shape[1], x + w + pad_x), min(img.shape[0], y + h + pad_y)
            crop = img[y1:y2, x1:x2]
            if crop.shape[0] < 64 or crop.shape[1] < 64:
                continue
            cv2.imwrite(str(OUT / f'coco_t_{a["id"]}.jpg'), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
        raw_path.unlink(missing_ok=True)
        return saved
    except Exception as e:
        print(f'  跳过 {image_id}: {e}')
        return 0


with ThreadPoolExecutor(WORKERS) as ex:
    results = list(ex.map(process, selected))
print(f'下载图: {len(selected)}, 裁剪保存: {sum(results)}')