"""
Tune pure DWA. NOTHING is added to the algorithm and dwa_matlab_port.py is not modified.

Only evalParam is chosen, i.e. the four inputs the MATLAB sample already has:
    [heading weight, distance weight, velocity weight, prediction time]
Table 3 limits, obstacle R = 0.5 m and reach = 0.37 m stay as before. Only the ratios of the three
weights matter, so heading stays at 0.1.

    python dwa_tune_pure.py tune       80 settings on 10 tuning maps, the best 10 again on all 40
    python dwa_tune_pure.py compare    original vs tuned on the six maps and on 100 unseen maps

Tuning maps: seeds 100-139. Test maps: seeds 1000-1099, not used for choosing.
Success: reaches the goal, never enters an obstacle cell, never leaves the map. While counting
successes a run is stopped once it has failed; this cannot change the count.

Outputs in results/dwa_pure_tuned/.
"""
import csv
import itertools
import json
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

import dwa_matlab_port as dwa
import dwa_random_maps as common
from replicate_5_1 import OUT_DIR, algorithms, random_grid_map, run

OUT = os.path.join(OUT_DIR, "dwa_pure_tuned")
PARAMS = os.path.join(OUT, "eval_param.json")
ORIGINAL = [0.1, 0.2, 0.1, 3.0]
REACH = 0.37
TUNING, SIX, TEST = range(100, 140), (1, 30, 2, 3, 4, 5), range(1000, 1100)
WORKERS = 6


def problem(seed):
    occ = random_grid_map(common.SIZE, common.DENSITY, common.START, common.GOAL,
                          np.random.default_rng(seed), common.DIAG)
    astar = algorithms(common.DIAG, common.P)["Diagonal Distance"]
    path, _, _ = run(astar(), occ, common.START, common.GOAL)
    ob = np.column_stack(np.nonzero(occ)[::-1]).astype(float) + 0.5
    return occ, np.array(path, float) + 0.5, ob


def outcome(job):
    """Same loop as dwa_matlab_port.run_waypoints, calling the same unmodified functions."""
    seed, ep, max_steps = job
    occ, wp, ob = problem(seed)
    x = np.array([*wp[0], math.atan2(wp[1, 1] - wp[0, 1], wp[1, 0] - wp[0, 0]), 0.0, 0.0])
    k = 1
    for i in range(1, max_steps + 1):
        u = dwa.dynamic_window_approach(x, dwa.TABLE3_KINEMATIC, wp[k], ep, ob, common.OBSTACLE_R)
        x = dwa.f(x, u)
        if not (0 <= x[0] < common.SIZE and 0 <= x[1] < common.SIZE):
            return "left map", i * dwa.DT
        if occ[int(x[1]), int(x[0])]:
            return "collision", i * dwa.DT
        if dwa.norm2(x[0] - wp[k, 0], x[1] - wp[k, 1]) < REACH:
            k += 1
            if k == len(wp):
                return "success", i * dwa.DT
    return "timeout", max_steps * dwa.DT


def score(pool, ep, seeds, max_steps):
    rs = pool.map(outcome, [(s, ep, max_steps) for s in seeds], chunksize=1)
    times = [t for r, t in rs if r == "success"]
    n = {k: sum(r == k for r, _ in rs) for k in ("success", "collision", "left map", "timeout")}
    return dict(n, maps=len(rs), median_time_s=round(float(np.median(times)), 1) if times else 999.0)


def tune():
    grid = [[0.1, d, v, t] for d, v, t in itertools.product(
        (0.05, 0.1, 0.2, 0.4), (0.01, 0.025, 0.05, 0.1, 0.2), (1.0, 2.0, 3.0, 4.0))]
    best_first = lambda r: (-r["success"], r["median_time_s"])
    rows = []
    with Pool(WORKERS) as pool:
        first = [dict(ep=ep, **score(pool, ep, TUNING[:10], 1500)) for ep in grid]
        rows += [dict(stage=1, **r) for r in first]
        print("stage 1 done", flush=True)
        second = []
        for r in sorted(first, key=best_first)[:10]:
            second.append(dict(ep=r["ep"], **score(pool, r["ep"], TUNING, 1500)))
            rows.append(dict(stage=2, **second[-1]))
            print(second[-1], flush=True)
    with open(os.path.join(OUT, "tuning.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    best = min(second, key=best_first)
    with open(PARAMS, "w") as f:
        json.dump(best["ep"], f)
    print("CHOSEN", best)


def compare():
    with open(PARAMS) as f:
        tuned = json.load(f)
    rows = []
    with Pool(WORKERS) as pool:
        for set_name, seeds in (("six", SIX), ("test", TEST)):
            for name, ep in (("original", ORIGINAL), ("tuned", tuned)):
                rows.append(dict(map_set=set_name, setting=name, ep=ep,
                                 **score(pool, ep, seeds, common.MAX_STEPS)))
                print(rows[-1], flush=True)
    with open(os.path.join(OUT, "comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    runs = []  # figures: the complete run, by the unmodified run_waypoints
    for i, seed in enumerate(SIX, 1):
        occ, wp, ob = problem(seed)
        log = dwa.run_waypoints(wp, ob, evalParam=tuned, obstacleR=common.OBSTACLE_R, reach=REACH)
        m, bad = common.evaluate(occ, wp, ob, log, REACH)
        r = dict(id=i, seed=seed, occ=occ, wp=wp, log=log, m=m, bad=bad)
        runs.append(r)
        base = os.path.join(OUT, f"map{i}_seed{seed}")
        np.savetxt(base + "_trajectory.csv", log, delimiter=",", header="t,x,y,yaw,v,w,target",
                   comments="", fmt="%.8g")
        common.plot_one(r, base + ".png")
        print(f"map {i}: {common.verdict(m)}; v max {m['v_max']}", flush=True)
    common.plot_overview(runs, REACH, os.path.join(OUT, "overview.png"), "pure DWA (tuned weights)")


if __name__ == "__main__":
    dwa.RANGE_MODE = "matlab"
    os.makedirs(OUT, exist_ok=True)
    commands = {"tune": tune, "compare": compare}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit("usage: python dwa_tune_pure.py tune|compare")
    commands[sys.argv[1]]()
