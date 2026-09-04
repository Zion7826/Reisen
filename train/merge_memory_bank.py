# -*- coding: utf-8 -*-
"""合并分片记忆库 -> train/weights/patchcore/memory_bank_glass_normal.npy"""

import numpy as np
from pathlib import Path

PARTS_DIR = Path('train/weights/patchcore/glass_normal_parts')
OUT = Path('train/weights/patchcore/memory_bank_glass_normal.npy')

parts = sorted(PARTS_DIR.glob('part*.npy'))
if not parts:
    raise SystemExit(f'未找到分片: {PARTS}')
bank = np.concatenate([np.load(str(p)) for p in parts], axis=0)
np.save(str(OUT), bank)
print(f'合并 {len(parts)} 片 -> {OUT}, shape={bank.shape}')