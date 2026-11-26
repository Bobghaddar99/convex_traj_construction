#!/usr/bin/python3
"""Grade a map against the house SDF: grade_map.py <map prefix> <trajectory.npz> [picture.png]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
sys.dont_write_bytecode = True
import matplotlib
import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

import house

matplotlib.use('Agg')
import matplotlib.pyplot as plt

base, traj = sys.argv[1:3]
meta = yaml.safe_load(open(base + '.yaml'))
im = np.flipud(np.array(Image.open(base + '.pgm')))
res, start = meta['resolution'], np.load(traj)['origin']
org = np.array(meta['origin'][:2]) + start
truth, meshes = house.geometry()
J = np.floor((house.X - org[0]) / res).astype(int)
I = np.floor((house.Y - org[1]) / res).astype(int)
ok = (I >= 0) & (I < im.shape[0]) & (J >= 0) & (J < im.shape[1])
M = np.full(truth.shape, 205, np.uint8)
M[ok] = im[I[ok], J[ok]]
unknown = M == 205

labels, _ = ndimage.label(~truth)
at = lambda x, y: (int((y - house.Y0) / house.RES), int((x - house.X0) / house.RES))
inside = labels == labels[at(*start)]
inside &= ~ndimage.binary_dilation(truth, iterations=3)
print('interior unknown %.2f%%' % (unknown[inside].mean() * 100))
for name, cx, cy, hx, hy in (('living table', -2.657, 2.424, .70, .40), ('marble table', 4.883, 2.926, .90, .65),
                             ('cafe table 1', 6.359, -3.192, .28, .28), ('cafe table 2', 6.359, -2.278, .28, .28)):
    q = (abs(house.X - cx) <= hx) & (abs(house.Y - cy) <= hy)
    print('  under %-13s unknown %5.1f%%' % (name, unknown[q].mean() * 100))

dist = ndimage.distance_transform_edt(~truth) * house.RES
i, j = np.where(im < 50)
x, y = org[0] + (j + .5) * res, org[1] + (i + .5) * res
keep = np.ones(len(x), bool)
for cx, cy in meshes:  # mesh models are not in the reference
    keep &= np.hypot(x - cx, y - cy) > (1.3 if cx > 4 and cy > 2 else .5)
e = dist[np.clip(((y - house.Y0) / house.RES).astype(int), 0, house.NY - 1),
         np.clip(((x - house.X0) / house.RES).astype(int), 0, house.NX - 1)][keep]
print('wall error median %.3f p95 %.3f p99 %.3f max %.3f m; %.1f%% within 0.10 m'
      % (np.median(e), *np.percentile(e, [95, 99]), e.max(), (e <= .1).mean() * 100))
outside = (labels != labels[at(*start)]) & ~truth
front = outside & (abs(house.X - 0) < 4) & (house.Y < -.4) & (house.Y > -5)
print('outdoor area marked free %.2f%%' % ((M[front] > 240).mean() * 100))

if len(sys.argv) > 3:
    f, a = plt.subplots(1, 2, figsize=(17, 7))
    ext = [org[0], org[0] + im.shape[1] * res, org[1], org[1] + im.shape[0] * res]
    a[0].imshow(im, origin='lower', cmap='gray', vmin=0, vmax=255, extent=ext, interpolation='nearest')
    a[0].set_title('map')
    a[1].imshow(truth, origin='lower', cmap='Greys', alpha=.3,
                extent=[house.X0, house.X0 + house.NX * house.RES, house.Y0, house.Y0 + house.NY * house.RES])
    a[1].scatter(x, y, s=.6, c='r')
    a[1].set_title('mapped obstacles (red) on true walls (grey)')
    for ax in a:
        ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(sys.argv[3])
