"""
Diagonal A* paths on the SLAM map, for DWA to follow in Nav2.

The search is the one of path_search/replicate_5_1.py, imported unchanged: Eq. (8)-(9), p = 0.001,
8-connected, no corner cutting. Grid: 0.2 m, as in the paper's real-robot test. A cell can be used
if its centre region is at least 0.3 m from anything that is not free in the SLAM map (0.12 m robot
radius plus margin; unknown counts as blocked). Only the SLAM map is read.

Run with the Python that has `pathfinding`:   python plan_diagonal_astar.py
"""
import json
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), 'path_search'))
from pathfinding.core.diagonal_movement import DiagonalMovement  # noqa: E402
from replicate_5_1 import algorithms, path_length, run  # noqa: E402

CELL, CLEARANCE, P = 0.2, 0.3, 0.001
base = os.path.join(ROOT, 'maps', 'turtlebot3_house')
m = yaml.safe_load(open(base + '.yaml'))
res, (ox, oy) = m['resolution'], m['origin'][:2]
im = np.array(Image.open(base + '.pgm'))
dist = ndimage.distance_transform_edt(np.flipud(im) > 240) * res
k = round(CELL / res)
h, w = dist.shape[0] // k, dist.shape[1] // k
Y, X = np.indices((h, w))
centre = np.minimum.reduce([dist[Y*k + a, X*k + b] for a in (1, 2) for b in (1, 2)])
occ = (centre < CLEARANCE).astype(int)

cell = lambda x, y: (int((x - ox) / CELL), int((y - oy) / CELL))
world = lambda c: (ox + (c[0] + .5) * CELL, oy + (c[1] + .5) * CELL)
algos = algorithms(DiagonalMovement.only_when_no_obstacle, P)
goals = json.load(open(os.path.join(HERE, 'goals.json')))['test']
out = os.path.join(HERE, 'diagonal_astar')
os.makedirs(out, exist_ok=True)

fig, ax = plt.subplots(figsize=(11, 8))
ax.imshow(im, cmap='gray', vmin=0, vmax=255, extent=[ox, ox + im.shape[1]*res, oy, oy + im.shape[0]*res])
rows = ['goal,algorithm,path_length_m,expanded_nodes,search_time_s']
for (name, g), colour in zip(goals.items(), ('#2a78d6', '#eb6834', '#1baf7a')):
    s, e = cell(0, 0), cell(g['x'], g['y'])
    found = {}
    for algo, make in algos.items():
        path, nodes, _ = run(make(), occ, s, e)
        if not path:
            sys.exit(f'{algo}: no path to {name}')
        t = float(np.median([run(make(), occ, s, e)[2] for _ in range(15)]))
        found[algo] = path
        rows.append(f'{name},{algo},{path_length(path, CELL):.4f},{nodes},{t:.6f}')
    wp = np.array([(0.0, 0.0)] + [world(c) for c in found['Diagonal Distance'][1:-1]] + [(g['x'], g['y'])])
    np.savetxt(os.path.join(out, name + '.csv'), wp, delimiter=',', header='x,y', comments='', fmt='%.3f')
    ax.plot(*wp.T, '-o', ms=3, lw=1.2, c=colour, label=f'{name.replace("_", " ")}: {len(wp) - 1} steps')
    ax.plot(*wp[-1], '*', c='k', ms=15)
    ax.annotate(name[0], wp[-1] + .2, fontsize=14)
with open(os.path.join(out, 'search.csv'), 'w') as f:
    f.write('\n'.join(rows) + '\n')
ax.plot(0, 0, '*', c='#2a78d6', ms=16, label='start')
ax.legend(loc='lower center', fontsize=9)
ax.set_title('Diagonal A* waypoints on the SLAM map (0.2 m grid)')
ax.set_xlabel('map x (m)')
ax.set_ylabel('map y (m)')
fig.tight_layout()
fig.savefig(os.path.join(out, 'paths.png'), dpi=120)
print('\n'.join(rows))
