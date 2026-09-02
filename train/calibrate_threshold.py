# -*- coding: utf-8 -*-
"""用 MVTec bottle test 集标定 PatchCore 异常阈值"""

import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.patchcore_detector import PatchCoreDetector

BASE = Path(__file__).parent.parent
TEST_ROOT = Path(r'D:\Users\谢佳锷\xwechat_files\wxid_1z7ofosin95d12_1878\msg\file\2026-08\bottle\test')
BANK = BASE / 'train' / 'weights' / 'patchcore' / 'memory_bank_glass_normal.npy'

detector = PatchCoreDetector(memory_bank_path=str(BANK), device='cpu', img_size=128, anomaly_threshold=15.0)

results = {}
for cat_dir in sorted(TEST_ROOT.iterdir()):
    if not cat_dir.is_dir():
        continue
    scores = []
    for img_path in sorted(cat_dir.glob('*.png')):
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        r = detector.detect(img)
        scores.append(r['anomaly_score'])
    results[cat_dir.name] = scores
    print(f"{cat_dir.name}: n={len(scores)} min={min(scores):.1f} max={max(scores):.1f} median={sorted(scores)[len(scores)//2]:.1f}", flush=True)

good_max = max(results.get('good', [0]))
defect_scores = [s for k, v in results.items() if k != 'good' for s in v]
defect_min = min(defect_scores)
correct = sum(1 for s in defect_scores if s > good_max)
print(f"\n正常最大分数: {good_max:.1f}")
print(f"缺陷最小分数: {defect_min:.1f}")
print(f"缺陷>正常max的命中率: {correct}/{len(defect_scores)}")
print(f"建议阈值: {(good_max + defect_min) / 2:.1f}")