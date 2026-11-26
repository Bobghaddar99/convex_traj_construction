"""
Tune the improved DWA, then compare it with the baseline on maps it has not seen.

    python dwa_tune_compare.py tune       grid search on the tuning maps  -> tuned_params.json, tuning.csv
    python dwa_tune_compare.py compare    six presentation maps + 100 unseen maps

Map sets (generator, A* and waypoints are identical to dwa_random_maps.py):
    tuning         seeds 100-139      the only maps used to choose the weights
    presentation   seeds 1 30 2 3 4 5 seen while designing the method, so not proof by themselves
    test           seeds 1000-1099    never used for design or tuning: quote these numbers

Methods:
    baseline               the MATLAB sample with Table 3 (reach 0.37 m)
    baseline_otpwr_limits  the same, with OTPWR's 0.5 m/s and 0.2 m/s^2, to isolate the effect of the limits
    improved               dwa_improved.py

A run is a success only if it reaches the goal, never enters an obstacle cell and never leaves the map.
Outputs in results/dwa_improved/.
"""
import csv
import itertools
import json
import os
import sys
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dwa_improved as improved
import dwa_matlab_port as baseline
import dwa_random_maps as common
from replicate_5_1 import OUT_DIR, algorithms, random_grid_map, run

OUT = os.path.join(OUT_DIR, "dwa_improved")
PARAMS = os.path.join(OUT, "tuned_params.json")
REACH = 0.37
TUNING, PRESENTATION, TEST = range(100, 140), (1, 30, 2, 3, 4, 5), range(1000, 1100)
METHODS = ("baseline", "baseline_otpwr_limits", "improved")
WORKERS = 6


def problem(seed):
    occ = random_grid_map(common.SIZE, common.DENSITY, common.START, common.GOAL,
                          np.random.default_rng(seed), common.DIAG)
    astar = algorithms(common.DIAG, common.P)["Diagonal Distance"]
    path, _, _ = run(astar(), occ, common.START, common.GOAL)
    ob = np.column_stack(np.nonzero(occ)[::-1]).astype(float) + 0.5
    return occ, np.array(path, float) + 0.5, ob


def solve(job):
    method, seed, params, max_steps = job
    occ, wp, ob = problem(seed)
    if method == "improved":
        log = improved.run_waypoints(wp, ob, params, REACH, max_steps, common.SIZE)
    else:
        baseline.RANGE_MODE = "matlab"
        k = list(baseline.TABLE3_KINEMATIC)
        if method == "baseline_otpwr_limits":
            k[0], k[2] = improved.V_MAX, improved.A_MAX
        log = baseline.run_waypoints(wp, ob, kinematic=k, evalParam=common.EVAL,
                                     obstacleR=common.OBSTACLE_R, reach=REACH, max_steps=max_steps)
    m, bad = common.evaluate(occ, wp, ob, log, REACH)
    m["success"] = m["goal_reached"] and not m["entered_obstacle_cell"] and not m["left_map"]
    return dict(method=method, seed=seed, occ=occ, wp=wp, log=log, bad=bad, m=m)


def tune():
    grid = [dict(heading=1.0, progress=g, clearance=c, horizon=t)
            for g, c, t in itertools.product((0.5, 1.0, 2.0, 4.0), (0.0, 0.1), (1.0, 2.0, 3.0))]
    rows = []
    with Pool(WORKERS) as pool:
        for p in grid:
            rs = pool.map(solve, [("improved", s, p, 2000) for s in TUNING])
            n = sum(r["m"]["success"] for r in rs)
            t = float(np.mean([r["m"]["time_s"] if r["m"]["success"] else 200.0 for r in rs]))
            rows.append({**p, "successes": n, "maps": len(TUNING), "mean_time_s": round(t, 2)})
            print(rows[-1], flush=True)
    with open(os.path.join(OUT, "tuning.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    best = min(rows, key=lambda r: (-r["successes"], r["mean_time_s"]))
    with open(PARAMS, "w") as f:
        json.dump({k: best[k] for k in improved.DEFAULTS}, f, indent=2)
    print("CHOSEN", best)


def summarise(rs):
    n, ok = len(rs), [r for r in rs if r["m"]["success"]]
    return dict(
        maps=n,
        success=len(ok),
        reached_goal=sum(r["m"]["goal_reached"] for r in rs),
        collided=sum(r["m"]["entered_obstacle_cell"] for r in rs),
        left_map=sum(r["m"]["left_map"] for r in rs),
        median_time_s=round(float(np.median([r["m"]["time_s"] for r in ok])), 1) if ok else "",
        v_max=max(r["m"]["v_max"] for r in rs),
        accel_max=max(r["m"]["accel_abs_max"] for r in rs),
    )


def plot_before_after(before, after, fname):
    fig, axes = plt.subplots(len(before) // 3, 6, figsize=(21, 9.6))
    for i, (b, a) in enumerate(zip(before, after)):
        for j, (r, name) in enumerate(((b, "baseline"), (a, "improved"))):
            ax = axes.flat[2 * i + j]
            common.draw_map(ax, r)
            m = r["m"]
            state = f"OK, {m['time_s']:.1f} s" if m["success"] else "FAILED"
            ax.set_title(f"Map {r['id']} {name}\n{state}", fontsize=11, color=common.INK)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(length=0)
    handles = {}
    for ax in axes.flat:
        for h, label in zip(*ax.get_legend_handles_labels()):
            handles[label] = h
    fig.suptitle("Baseline vs improved DWA on the same maps and Diagonal A* waypoints", fontsize=15)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, .05, 1, .96), h_pad=3)
    fig.savefig(fname, dpi=130)
    plt.close(fig)


def compare():
    with open(PARAMS) as f:
        params = json.load(f)
    table, per_map = [], []
    for set_name, seeds in (("presentation", PRESENTATION), ("test", TEST)):
        with Pool(WORKERS) as pool:
            jobs = [(m, s, params, common.MAX_STEPS) for m in METHODS for s in seeds]
            rs = pool.map(solve, jobs, chunksize=1)
        for method in METHODS:
            mine = [r for r in rs if r["method"] == method]
            table.append(dict(map_set=set_name, method=method, **summarise(mine)))
            print(table[-1], flush=True)
            per_map += [dict(map_set=set_name, method=method, seed=r["seed"], **r["m"]) for r in mine]
        b = {r["seed"]: r["m"] for r in rs if r["method"] == "baseline"}
        both = [(b[r["seed"]]["time_s"], r["m"]["time_s"]) for r in rs
                if r["method"] == "improved" and r["m"]["success"] and b[r["seed"]]["success"]]
        if both:
            t0, t1 = np.median(both, axis=0)
            print(f"  maps solved by both: {len(both)}, median {t0:.1f} s -> {t1:.1f} s")
        if set_name != "presentation":
            continue
        picked = {}
        for method in ("baseline", "improved"):
            picked[method] = [dict(r, id=i) for i, r in
                              enumerate((r for r in rs if r["method"] == method), 1)]
        for r in picked["improved"]:
            base = os.path.join(OUT, f"map{r['id']}_seed{r['seed']}")
            common.plot_one(r, base + ".png", v_max=improved.V_MAX)
            np.savetxt(base + "_trajectory.csv", r["log"], delimiter=",",
                       header="t,x,y,yaw,v,w,target", comments="", fmt="%.8g")
        common.plot_overview(picked["improved"], REACH, os.path.join(OUT, "overview.png"), "improved DWA")
        plot_before_after(picked["baseline"], picked["improved"], os.path.join(OUT, "before_after.png"))
    for name, rows in (("comparison.csv", table), ("per_map.csv", per_map)):
        with open(os.path.join(OUT, name), "w", newline="") as f:
            w = csv.DictWriter(f, rows[0].keys())
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    commands = {"tune": tune, "compare": compare}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit("usage: python dwa_tune_compare.py tune|compare")
    commands[sys.argv[1]]()
