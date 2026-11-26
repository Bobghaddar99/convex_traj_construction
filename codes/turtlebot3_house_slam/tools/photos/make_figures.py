#!/usr/bin/python3
"""Figures for slam_results/, from saved data only (the mapping bag and the Nav2 trial files)."""
import csv
import math
import os
import sqlite3

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
from nav_msgs.msg import Odometry
from PIL import Image
from rclpy.serialization import deserialize_message
from tf2_msgs.msg import TFMessage

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
OUT = os.path.join(ROOT, 'slam_results')
BLUE, ORANGE, GREEN, MUTED, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#52514e', '#e1e0d9'
DT = 0.1


def tidy(ax):
    ax.grid(color=GRID, lw=.5)
    ax.spines[['top', 'right']].set_visible(False)


# the mapping drive (SLAM Toolbox + map-only explorer)
db = sqlite3.connect(f"file:{ROOT}/runs/live_142803/bag/bag_0.db3?mode=ro", uri=True)


def messages(topic, kind):
    for data, in db.execute('select data from messages where topic_id = '
                            '(select id from topics where name = ?) order by timestamp', (topic,)):
        yield deserialize_message(data, kind)


secs = lambda h: h.stamp.sec + h.stamp.nanosec * 1e-9
yaw = lambda q: math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2 * (q.y*q.y + q.z*q.z))
odom = np.array([(secs(m.header), m.twist.twist.linear.x, m.twist.twist.angular.z,
                  m.pose.pose.position.x, m.pose.pose.position.y) for m in messages('/odom', Odometry)])
fix = np.array([(secs(t.header), t.transform.translation.x, t.transform.translation.y, yaw(t.transform.rotation))
                for m in messages('/tf', TFMessage) for t in m.transforms
                if t.header.frame_id == 'map' and t.child_frame_id == 'odom'])
fix = fix[np.argsort(fix[:, 0])]
odom = odom[np.argsort(odom[:, 0])]
t = np.arange(odom[0, 0], odom[-1, 0], DT)
v, w, x, y = [np.interp(t, odom[:, 0], odom[:, k]) for k in (1, 2, 3, 4)]
fx, fy = [np.interp(t, fix[:, 0], fix[:, k]) for k in (1, 2)]
fa = np.interp(t, fix[:, 0], np.unwrap(fix[:, 3]))
mx, my = fx + np.cos(fa) * x - np.sin(fa) * y, fy + np.sin(fa) * x + np.cos(fa) * y
a, alpha = np.diff(v, prepend=v[0]) / DT, np.diff(w, prepend=w[0]) / DT
distance = float(np.abs(v).sum() * DT)

fig, ax = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
for axis, q, label in zip(ax, (v, a, w, alpha), ('v (m/s)', 'dv/dt (m/s²)', 'ω (rad/s)', 'dω/dt (rad/s²)')):
    axis.plot(t - t[0], q, c=BLUE, lw=.8)
    axis.set_ylabel(label)
    tidy(axis)
ax[-1].set_xlabel('simulated time (s)')
fig.suptitle('SLAM mapping drive: measured velocity and acceleration (explorer, not DWA)', fontsize=14)
fig.tight_layout()
fig.savefig(f'{OUT}/4_velocity_acceleration/01_slam_mapping_drive.png', dpi=130)
plt.close(fig)

m = yaml.safe_load(open(f'{ROOT}/maps/turtlebot3_house.yaml'))
im = np.array(Image.open(f'{ROOT}/maps/turtlebot3_house.pgm'))
(ox, oy), res = m['origin'][:2], m['resolution']
fig, ax = plt.subplots(figsize=(12, 8.8))
ax.imshow(im, cmap='gray', vmin=0, vmax=255, extent=[ox, ox + im.shape[1]*res, oy, oy + im.shape[0]*res])
ax.plot(mx, my, c=ORANGE, lw=1, label=f'robot path ({distance:.0f} m), pose as estimated live by SLAM')
ax.plot(mx[0], my[0], '*', c=BLUE, ms=16, label='start and finish')
ax.legend(loc='lower center')
ax.set_title('Path driven while SLAM Toolbox built the map')
ax.set_xlabel('map x (m)')
ax.set_ylabel('map y (m)')
fig.tight_layout()
fig.savefig(f'{OUT}/2_paths/01_slam_mapping_path.png', dpi=130)
plt.close(fig)

with open(f'{OUT}/5_tables/slam_mapping_drive.csv', 'w', newline='') as f:
    out = csv.writer(f)
    out.writerow(['t_s', 'x_map', 'y_map', 'v', 'a', 'omega', 'alpha'])
    out.writerows(np.round(np.c_[t - t[0], mx, my, v, a, w, alpha], 4))
print(f'mapping: {t[-1]-t[0]:.0f} s, {distance:.1f} m, vmax {v.max():.3f}, '
      f'|a|max {abs(a).max():.2f}, |w|max {abs(w).max():.3f}')

# one velocity figure per goal: both planners, both settings, run 1
for goal in ('A_south_west_room', 'B_middle_room', 'C_south_east_room'):
    fig, ax = plt.subplots(4, 2, figsize=(14, 11), sharey='row', sharex='col')
    for col, (title, folder) in enumerate((('NavFn A*', 'results'), ('Diagonal A*', 'results/diagonal'))):
        ax[0, col].set_title(f'{title} + DWA', fontsize=12)
        for setting, name, colour in (('paper', 'paper', ORANGE), ('optimized', 'alternative weights', GREEN)):
            z = np.load(f'{ROOT}/nav2/{folder}/{setting}_run1/{goal}.npz', allow_pickle=True)
            c, n = z['cmd'], len(z['cmd'])
            while n > 1 and c[n-1, 2] == 0 and c[n-1, 3] == 0:
                n -= 1
            cv, cw = c[:n, 2], c[:n, 3]
            series = (cv, np.diff(np.r_[0, cv]) / DT, cw, np.diff(np.r_[0, cw]) / DT)
            label = f"{name}: {'reached' if str(z['result']) == 'reached' else 'failed'}, {float(z['time_s']):.1f} s"
            for row, q in enumerate(series):
                ax[row, col].plot(np.arange(1, n + 1) * DT, q, c=colour, lw=.9, label=label)
        for row, lim in ((1, .5), (2, .35), (3, .87)):
            for s in (-1, 1):
                ax[row, col].axhline(s * lim, c=MUTED, ls='--', lw=.8)
        for row in range(4):
            tidy(ax[row, col])
        ax[0, col].legend(fontsize=9, frameon=False)
        ax[-1, col].set_xlabel('simulated time (s)')
    for row, label in enumerate(('v (m/s)', 'dv/dt (m/s²)', 'ω (rad/s)', 'dω/dt (rad/s²)')):
        ax[row, 0].set_ylabel(label)
    fig.suptitle(f'Goal {goal[0]}: commanded velocity and acceleration on the SLAM map, run 1. '
                 'Dashed: Table 3 limits', fontsize=13)
    fig.tight_layout()
    fig.savefig(f'{OUT}/4_velocity_acceleration/02_goal_{goal[0]}.png', dpi=130)
    plt.close(fig)
