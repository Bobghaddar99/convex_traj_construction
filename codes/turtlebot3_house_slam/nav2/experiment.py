#!/usr/bin/python3
"""Tune DWB's existing parameters on the tuning goals, then test paper vs optimized.

Gazebo stays up; only Nav2 is restarted. Table 3 limits and sampling are never changed.
Winner: most tuning goals reached; then fewest goals with the lidar at its 0.12 m minimum
(touching or nearly touching); then least time.
"""
import csv
import json
import os
import subprocess

HERE = os.path.dirname(os.path.realpath(__file__))
RESULTS = os.path.join(HERE, 'results')
REPEATS, LIMIT = 3, '150'
A = {'PathAlign.forward_point_distance': 0.325, 'GoalAlign.forward_point_distance': 0.325}  # ROS 1 DWA default
CANDIDATES = {
    'c1': {'sim_time': 1.5},
    'c2': {'sim_time': 1.5, **A},
    'c3': {'sim_time': 1.5, **A, 'PathDist.scale': 64.0, 'PathAlign.scale': 64.0},
    'c4': {'sim_time': 1.5, **A, 'BaseObstacle.scale': 0.1},
    'c5': {'sim_time': 2.0, **A, 'BaseObstacle.scale': 0.05},
    'c6': {'sim_time': 1.0, **A, 'PathDist.scale': 64.0, 'PathAlign.scale': 64.0},
}


def make(name, params):
    with open(os.path.join(HERE, name + '.json'), 'w') as f:
        json.dump(params, f, indent=2)
    subprocess.run(['/usr/bin/python3', os.path.join(HERE, 'make_params.py'), name], check=True)


def trials(params_name, label, which):
    # the file check stops an earlier summary being mistaken for this run's
    out = os.path.join(RESULTS, label, 'summary.csv')
    if os.path.exists(out):
        os.remove(out)
    for _ in range(2):
        if subprocess.run(['bash', os.path.join(HERE, 'restart_nav2.sh'), params_name]).returncode == 0:
            break
    else:
        return []
    subprocess.run(['/usr/bin/python3', '-u', os.path.join(HERE, 'run_goals.py'), label, which, LIMIT])
    if not os.path.exists(out):
        return []
    with open(out) as f:
        return list(csv.DictReader(f))


table = []
for name, params in CANDIDATES.items():
    make(name, params)
    rows = trials(name, 'tuning_' + name, 'tuning')
    if len(rows) != 2:
        print('no valid trial for', name, flush=True)
        continue
    table.append(dict(
        candidate=name, params=json.dumps(params),
        reached=sum(r['result'] == 'reached' for r in rows),
        at_lidar_minimum=sum(float(r.get('closest_lidar_m') or 9) < .135 for r in rows),
        time_s=round(sum(float(r['time_s']) if r['result'] == 'reached' else 150 for r in rows), 1)))
    print('TUNING', table[-1], flush=True)
with open(os.path.join(RESULTS, 'tuning.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, table[0].keys())
    w.writeheader()
    w.writerows(table)
best = min(table, key=lambda r: (-r['reached'], r['at_lidar_minimum'], r['time_s']))
make('optimized', json.loads(best['params']))
print('CHOSEN', best, flush=True)

for i in range(1, REPEATS + 1):
    for setting in ('paper', 'optimized'):
        trials(setting, f'{setting}_run{i}', 'test')
