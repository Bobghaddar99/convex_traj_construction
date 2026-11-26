#!/usr/bin/python3
"""Choose goals from the SLAM map only: the point with most clearance near a point in each room."""
import json
import os

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.realpath(__file__))
base = os.path.join(os.path.dirname(HERE), 'maps', 'turtlebot3_house')
m = yaml.safe_load(open(base + '.yaml'))
free = np.flipud(np.array(Image.open(base + '.pgm'))) > 240
r, (ox, oy) = m['resolution'], m['origin'][:2]
d = ndimage.distance_transform_edt(free) * r
labels, _ = ndimage.label(d > 0.2)
home = labels[int(-oy / r), int(-ox / r)]
Y, X = np.indices(free.shape)
X, Y = ox + (X + .5) * r, oy + (Y + .5) * r


def pick(x, y):
    near = np.hypot(X - x, Y - y)
    s = np.where((near < .6) & (labels == home), d - .05 * near, -np.inf)
    i = np.unravel_index(np.argmax(s), s.shape)
    return dict(x=round(float(X[i]), 3), y=round(float(Y[i]), 3), clearance=round(float(d[i]), 2))


goals = dict(
    test={'A_south_west_room': pick(-4.4, -3.5), 'B_middle_room': pick(3.0, 1.8),
          'C_south_east_room': pick(8.0, -2.0)},
    tuning={'T1_north_west_room': pick(-4.3, 2.4), 'T2_east_room': pick(8.6, 2.0)})  # never reported
with open(os.path.join(HERE, 'goals.json'), 'w') as f:
    json.dump(goals, f, indent=2)
print(json.dumps(goals, indent=2))
