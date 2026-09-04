# -*- coding: utf-8 -*-
"""拆分记忆库为 <95MB 分片(绕过 Gitee 100MB 单文件限制)"""

import numpy as np
from pathlib import Path

SRC = Path('train/weights/patchcore/memory_bank_glass_normal.npy')
PARTS_DIR = Path('train/weights/patchcore/glass_normal_parts')
CHUNK_ROWS = 14700

bank = np.load(str(SRC))
PARTS_DIR.mkdir(parents=True, exist_ok=True)
n = bank.shape[0]
idx = 0
for i in range(0, n, CHUNK_ROWS):
    out = PARTS_DIR / f'part{idx:02d}.npy'
    np.save(str(out), bank[i:i + CHUNK_ROWS])
    print(f'{out.name}: {bank[i:i+CHUNK_ROWS].shape[0]} rows, {out.stat().st_size/1e6:.1f} MB')
    idx += 1
print(f'共 {idx} 片')