#!/usr/bin/python3
"""Coarsen a map: export_grid.py <input prefix> <output prefix> <resolution>

A coarse cell is occupied if any fine cell in it is, free if at least half of it is
free, and unknown otherwise.
"""
import os
import sys

import numpy as np
import yaml
from PIL import Image

src, dst, target = sys.argv[1], sys.argv[2], float(sys.argv[3])
meta = yaml.safe_load(open(src + '.yaml'))
im = np.flipud(np.array(Image.open(src + '.pgm')))
k = round(target / meta['resolution'])
if k < 1 or abs(k * meta['resolution'] - target) > 1e-6:
    sys.exit('resolution must be a multiple of the input resolution')
h, w = [-(-n // k) for n in im.shape]
full = np.full((h * k, w * k), 205, np.uint8)
full[:im.shape[0], :im.shape[1]] = im
blocks = full.reshape(h, k, w, k).transpose(0, 2, 1, 3).reshape(h, w, -1)
out = np.full((h, w), 205, np.uint8)
out[(blocks > 240).mean(2) >= .5] = 254
out[(blocks < 50).any(2)] = 0
Image.fromarray(np.flipud(out)).save(dst + '.pgm')
meta.update(image=os.path.basename(dst) + '.pgm', resolution=target)
yaml.safe_dump(meta, open(dst + '.yaml', 'w'), sort_keys=False, default_flow_style=None)
