#!/usr/bin/python3
"""Tables and figures from the saved Nav2 trials. Figures show run 1, tables show all runs.

    report.py [results folder] [global planner name]
"""
import csv
import glob
import json
import os
import re
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

HERE = os.path.dirname(os.path.realpath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results')
PLANNER = sys.argv[2] if len(sys.argv) > 2 else 'NavFn A*'
DT = 0.1
MARK_FAILURES = os.environ.get('MARK_FAILURES', '1') == '1'
BLUE, ORANGE, MUTED, GRID = '#2a78d6', '#eb6834', '#52514e', '#e1e0d9'
GOALS = json.load(open(os.path.join(HERE, 'goals.json')))['test']


def load(path):
    z = dict(np.load(path, allow_pickle=True))
    r = dict(result=str(z['result']), time=float(z['time_s']), z=z)
    if z['cmd'].size:
        c = z['cmd']
        n = len(c)
        while n > 1 and c[n-1, 2] == 0 and c[n-1, 3] == 0:
            n -= 1  # the stop that Nav2 sends at the end is not a DWA decision
        o = z['odom']
        t = np.arange(o[0, 0], o[-1, 0], DT)
        t = t[t <= c[n-1, 0]]
        v, w = np.interp(t, o[:, 0], o[:, 1]), np.interp(t, o[:, 0], o[:, 2])
        r.update(t=np.arange(1, n+1) * DT, v=c[:n, 2], w=c[:n, 3],
                 a=np.diff(np.r_[0, c[:n, 2]]) / DT, alpha=np.diff(np.r_[0, c[:n, 3]]) / DT,
                 mt=t - o[0, 0], mv=v, mw=w,
                 ma=np.diff(v, prepend=v[:1]) / DT, malpha=np.diff(w, prepend=w[:1]) / DT,
                 final_stop=float(c[n-1, 2]), step=float((c[-1, 0] - c[0, 0]) / max(1, len(c)-1)))
    return r


runs = []
for setting in ('paper', 'optimized'):
    for folder in sorted(glob.glob(os.path.join(OUT, setting + '_run*'))):
        for goal in GOALS:
            f = os.path.join(folder, goal + '.npz')
            if os.path.exists(f):
                runs.append(dict(setting=setting, run=int(re.search(r'run(\d+)$', folder).group(1)),
                                 goal=goal, **load(f)))

table = []
for r in runs:
    d = {k: r[k] for k in ('setting', 'run', 'goal', 'result')}
    d['time_s'] = r['time']
    if 't' in r:
        mx = lambda k: round(float(np.abs(r[k]).max()), 3) if len(r[k]) else 0.0
        z = r['z']
        d.update(v_mean=round(float(r['v'].mean()), 3), v_max=mx('v'), v_max_measured=mx('mv'),
                 a_max=mx('a'), a_max_measured=mx('ma'),
                 a_rms=round(float(np.sqrt(np.mean(r['a']**2))), 3),
                 w_max=mx('w'), w_max_measured=mx('mw'), alpha_max=mx('alpha'), alpha_max_measured=mx('malpha'),
                 speed_at_stop=round(r['final_stop'], 3), control_step_s=round(r['step'], 4),
                 closest_lidar_m=round(float(z['scan'][:, 1].min()), 3) if z['scan'].size else '',
                 travelled_m=round(float(np.abs(r['mv']).sum() * DT), 2))
    table.append(d)
    print(d)
keys = list(dict.fromkeys(k for d in table for k in d))
with open(os.path.join(OUT, 'velocity_acceleration.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, keys)
    w.writeheader()
    w.writerows(table)

COLS = (('v', 'mv', 'Linear velocity (m/s)', None), ('a', 'ma', 'Linear acceleration (m/s²)', .5),
        ('w', 'mw', 'Angular velocity (rad/s)', .35), ('alpha', 'malpha', 'Angular acceleration (rad/s²)', .87))
first = {(r['setting'], r['goal']): r for r in runs if r['run'] == 1}
for setting in ('paper', 'optimized'):
    fig, ax = plt.subplots(len(GOALS), 4, figsize=(19, 9.5), squeeze=False, sharey='col')
    for i, goal in enumerate(GOALS):
        r = first.get((setting, goal))
        if not r:
            continue
        ax[i, 0].set_ylabel(f"{goal.replace('_', ' ')}\n{r['result']}, {r['time']:.1f} s", fontsize=10)
        for j, (cmd, meas, title, lim) in enumerate(COLS):
            a = ax[i, j]
            if 't' in r:
                a.plot(r['t'], r[cmd], c=BLUE, lw=.9, label='commanded by DWA')
                a.plot(r['mt'], r[meas], c=ORANGE, lw=.8, label='measured')
            if lim:
                for s in (-1, 1):
                    a.axhline(s * lim, c=MUTED, ls='--', lw=.8)
            a.grid(color=GRID, lw=.5)
            a.spines[['top', 'right']].set_visible(False)
            a.tick_params(labelleft=True, labelsize=8)
            if i == 0:
                a.set_title(title)
            if i == len(GOALS) - 1:
                a.set_xlabel('simulated time (s)')
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, frameon=False)
    label = 'paper Table 3' if setting == 'paper' else 'optimized existing parameters, same limits'
    fig.suptitle(f'{PLANNER} + Nav2 DWA on the SLAM map, {label} (run 1). Dashed: Table 3 limits', fontsize=14)
    fig.tight_layout(rect=(0, .04, 1, .96))
    fig.savefig(os.path.join(OUT, f'velocity_acceleration_{setting}.png'), dpi=120)
    plt.close(fig)

base = os.path.join(os.path.dirname(HERE), 'maps', 'turtlebot3_house')
m = yaml.safe_load(open(base + '.yaml'))
im = np.array(Image.open(base + '.pgm'))
ox, oy, res = m['origin'][0], m['origin'][1], m['resolution']
fig, axes = plt.subplots(1, 2, figsize=(17, 7))
for a, setting in zip(axes, ('paper', 'optimized')):
    a.imshow(im, cmap='gray', vmin=0, vmax=255, extent=[ox, ox + im.shape[1]*res, oy, oy + im.shape[0]*res])
    for r in runs:
        z = r['z']
        if r['setting'] != setting:
            continue
        if r['run'] == 1 and z['plan'].size:
            a.plot(z['plan'][:, 0], z['plan'][:, 1], ':', c=BLUE, lw=1.2, label=f'global plan ({PLANNER})')
        if z['pose'].size:
            a.plot(z['pose'][:, 1], z['pose'][:, 2], c=ORANGE, lw=1, alpha=.8, label='DWA, 3 runs (AMCL pose)')
            if MARK_FAILURES and r['result'] != 'reached':
                a.plot(*z['pose'][-1, 1:], 'x', c='#d03b3b', ms=9, label='failed here')
    for name, g in GOALS.items():
        a.plot(g['x'], g['y'], '*', c='k', ms=15)
        a.annotate(name[0], (g['x'] + .2, g['y'] + .2), fontsize=13)
    a.plot(0, 0, '*', c=BLUE, ms=15, label='start')
    h, l = a.get_legend_handles_labels()
    u = dict(zip(l, h))
    a.legend(u.values(), u.keys(), fontsize=8, loc='lower center')
    ok = sum(r['result'] == 'reached' for r in runs if r['setting'] == setting)
    a.set_title(f"{PLANNER} + DWA, {setting}: {ok}/{sum(r['setting'] == setting for r in runs)} trials reached the goal")
    a.set_xlabel('map x (m)')
axes[0].set_ylabel('map y (m)')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'trajectories.png' if MARK_FAILURES else 'trajectories_no_marks.png'), dpi=120)
