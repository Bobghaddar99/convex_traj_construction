"""
Diagonal A* + DWA on random 10 x 10 m grid maps (Sections 5.1 and 5.2 of Luo et al. 2024).

Pipeline, for every map:
    1. random map       replicate_5_1.random_grid_map (same size, density and corner-cutting rule)
    2. waypoints        Diagonal-distance A*, Eq. (8)-(9) with p = 0.001
    3. trajectory       DWA through the waypoints, one after the other (dwa_matlab_port.run_waypoints)
    4. checks           goal reached?  collisions?  left the map?  speed and acceleration?

No algorithm is implemented or modified here: search and DWA are imported from the existing scripts.

Maps: seeds 1 and 30 are the two maps of replicate_5_1.py; seeds 2, 3, 4, 5 are four more.

DWA settings (those used to reproduce the paper's Fig. 9):
    Table 3 of the paper     v <= 2 m/s, |w| <= 0.35 rad/s, 0.5 m/s^2, 0.87 rad/s^2,
                             steps 0.02 m/s and 0.017 rad/s, dt = 0.1 s, 5000 steps
    not given in the paper   weights [heading, distance, velocity] = [0.1, 0.2, 0.1],
                             prediction 3.0 s, obstacle R = 0.5 m, reach = 0.37 m, MATLAB range rounding

Two checks run before anything is written:
    * DWA on the Fig. 9 map must equal results/sweep/matlab_reach0.37.csv at every step
    * seeds 1 and 30 must give exactly results/grid_map_{1,2}.txt

Outputs, in results/dwa_random/:
    summary.csv                  one row per map
    overview.png                 all trajectories
    mapN_seedS.png               trajectory, v(t) and w(t)
    mapN_seedS_map.txt           occupancy grid, top row = highest y
    mapN_seedS_waypoints.csv     A* waypoints in metres (the input for the QP as well)
    mapN_seedS_trajectory.csv    t, x, y, yaw, v, w, index of the current waypoint

Usage:
    python dwa_random_maps.py
    python dwa_random_maps.py --seeds 1 30 2 3 4 5 --reach 0.37
"""
import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from pathfinding.core.diagonal_movement import DiagonalMovement

import dwa_matlab_port as dwa
from replicate_5_1 import OUT_DIR, algorithms, path_length, random_grid_map, run

OUT = os.path.join(OUT_DIR, "dwa_random")
SIZE, DENSITY, P = 10, 0.25, 0.001
START, GOAL = (0, 0), (SIZE - 1, SIZE - 1)
DIAG = DiagonalMovement.only_when_no_obstacle  # no cutting of obstacle corners
EVAL = (0.1, 0.2, 0.1, 3.0)                    # heading, distance, velocity weights; prediction time
OBSTACLE_R = 0.5
MAX_STEPS = 5000
V_MAX, W_MAX, A_MAX = dwa.TABLE3_KINEMATIC[:3]

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"


def check_fig9_regression():
    """Same settings on the paper's Fig. 9 map must reproduce the saved run exactly."""
    with open(os.path.join(dwa.HERE, "fig9_map.json")) as f:
        m = json.load(f)
    log = dwa.run_waypoints(np.array(m["paper_path"], float) + 0.5, np.array(m["cells"], float) + 0.5,
                            evalParam=EVAL, obstacleR=OBSTACLE_R, reach=0.37)
    saved = np.loadtxt(os.path.join(OUT_DIR, "sweep", "matlab_reach0.37.csv"), delimiter=",")
    if log.shape != saved.shape or not np.allclose(log, saved, rtol=1e-6, atol=1e-6):
        raise SystemExit("Fig. 9 regression FAILED: the settings differ from the saved run")
    print(f"check 1 passed: Fig. 9 run reproduced at all {len(log)} steps ({log[-1, 0]:.1f} s)")


def check_saved_maps():
    for map_id, seed in ((1, 1), (2, 30)):
        occ = random_grid_map(SIZE, DENSITY, START, GOAL, np.random.default_rng(seed), DIAG)
        saved = np.loadtxt(os.path.join(OUT_DIR, f"grid_map_{map_id}.txt"), dtype=int)
        if not np.array_equal(np.flipud(occ), saved):
            raise SystemExit(f"seed {seed} does not reproduce grid_map_{map_id}.txt")
    print("check 2 passed: seeds 1 and 30 reproduce the two saved maps")


def evaluate(occ, wp, ob, log, reach):
    """Measure what DWA produced. Nothing here feeds back into the planner."""
    t, x, y, v, w = log[:, 0], log[:, 1], log[:, 2], log[:, 4], log[:, 5]
    # the index in the log is the waypoint being targeted during that step
    last = int(log[-1, 6])
    hit = dwa.norm2(x[-1] - wp[last, 0], y[-1] - wp[last, 1]) < reach
    reached = last - 1 + int(hit)

    # an obstacle fills its whole 1 m cell
    i, j = np.floor(y).astype(int), np.floor(x).astype(int)
    inside = (x >= 0) & (x < SIZE) & (y >= 0) & (y < SIZE)
    in_obstacle = np.zeros(len(x), bool)
    in_obstacle[inside] = occ[i[inside], j[inside]] == 1
    centre_dist = np.hypot(x[:, None] - ob[:, 0], y[:, None] - ob[:, 1]).min(axis=1)

    xs, ys = np.r_[wp[0, 0], x], np.r_[wp[0, 1], y]
    return dict(
        waypoints_reached=reached,
        waypoints_total=len(wp) - 1,
        goal_reached=reached == len(wp) - 1,
        time_s=round(float(t[-1]), 1),
        astar_length_m=round(path_length([tuple(p) for p in wp], 1.0), 4),
        dwa_length_m=round(float(np.hypot(np.diff(xs), np.diff(ys)).sum()), 3),
        entered_obstacle_cell=bool(in_obstacle.any()),
        steps_in_obstacle_cell=int(in_obstacle.sum()),
        steps_within_R=int((centre_dist < OBSTACLE_R).sum()),
        min_centre_distance_m=round(float(centre_dist.min()), 3),
        left_map=bool((~inside).any()),
        v_mean=round(float(v.mean()), 3),
        v_max=round(float(v.max()), 3),
        w_abs_max=round(float(np.abs(w).max()), 3),
        accel_abs_max=round(float(np.abs(np.diff(np.r_[0.0, v])).max() / dwa.DT), 3),
    ), in_obstacle


def verdict(m):
    parts = [f"goal in {m['time_s']:.1f} s" if m["goal_reached"]
             else f"FAILED: {m['waypoints_reached']}/{m['waypoints_total']} waypoints"]
    if m["entered_obstacle_cell"]:
        parts.append("collision")
    if m["left_map"]:
        parts.append("left map")
    return ", ".join(parts)


def draw_map(ax, r):
    for cy, cx in zip(*np.nonzero(r["occ"])):
        ax.add_patch(Rectangle((cx + .04, cy + .04), .92, .92, color=GREEN, alpha=.55, lw=0))
    ax.add_patch(Rectangle((0, 0), SIZE, SIZE, fill=False, ec=MUTED, lw=1))
    wp, log, bad = r["wp"], r["log"], r["bad"]
    ax.plot(wp[:, 0], wp[:, 1], ":", c=MUTED, lw=1)
    ax.plot(wp[:, 0], wp[:, 1], "o", mfc="none", mec=INK, ms=6, label="Diagonal A* waypoints", zorder=4)
    ax.plot(np.r_[wp[0, 0], log[:, 1]], np.r_[wp[0, 1], log[:, 2]], c=ORANGE, lw=1.8,
            label="DWA trajectory", zorder=3)
    ax.plot(log[bad, 1], log[bad, 2], "x", c="#d03b3b", ms=4, label="inside obstacle cell", zorder=5)
    ax.plot(*wp[0], "*", c=BLUE, ms=16, label="start", zorder=6)
    ax.plot(*wp[-1], "*", c=INK, ms=16, label="goal", zorder=6)
    ax.set_xlim(-.2, SIZE + .2)  # the view stays on the map; departures are named in the title
    ax.set_ylim(-.2, SIZE + .2)
    ax.set_xticks(range(SIZE + 1))
    ax.set_yticks(range(SIZE + 1))
    ax.set_aspect("equal")
    ax.grid(color=GRID, lw=.5)
    ax.set_axisbelow(True)


def plot_one(r, fname, v_max=V_MAX):
    fig = plt.figure(figsize=(14, 6.5))
    g = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(g[:, 0])
    draw_map(ax, r)
    ax.set_title(f"Map {r['id']} (seed {r['seed']}): {verdict(r['m'])}", color=INK)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.1), ncol=3, fontsize=8, frameon=False)

    a1 = fig.add_subplot(g[0, 1])
    a2 = fig.add_subplot(g[1, 1], sharex=a1)
    t = r["log"][:, 0]
    a1.plot(t, r["log"][:, 4], c=BLUE, lw=1)
    a1.set_title(f"Linear velocity (limit {v_max:g} m/s)", fontsize=10)
    a1.set_ylabel("v (m/s)")
    a2.plot(t, r["log"][:, 5], c=BLUE, lw=1)
    for s in (-1, 1):
        a2.axhline(s * W_MAX, c=MUTED, lw=.8, ls="--")
    a2.set_title(f"Angular velocity (dashed: limit ±{W_MAX:g} rad/s)", fontsize=10)
    a2.set_ylabel("ω (rad/s)")
    a2.set_xlabel("time (s)")
    for a in (a1, a2):
        a.grid(color=GRID, lw=.5)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def plot_overview(runs, reach, fname, method="DWA"):
    cols = min(3, len(runs))
    rows = -(-len(runs) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.7 * cols, 5.3 * rows + 1), squeeze=False)
    for ax in axes.flat[len(runs):]:
        ax.axis("off")
    handles = {}
    for ax, r in zip(axes.flat, runs):
        draw_map(ax, r)
        ax.set_title(f"Map {r['id']} (seed {r['seed']})\n{verdict(r['m'])}", fontsize=11, color=INK)
        ax.tick_params(labelsize=8, colors=MUTED)
        for h, label in zip(*ax.get_legend_handles_labels()):
            handles[label] = h
    ok = sum(r["m"]["goal_reached"] for r in runs)
    clean = sum(r["m"]["goal_reached"] and not r["m"]["entered_obstacle_cell"] and not r["m"]["left_map"]
                for r in runs)
    fig.suptitle(f"Diagonal A* + {method} on random maps: {ok}/{len(runs)} reach the goal, "
                 f"{clean}/{len(runs)} do so without collision or leaving the map (reach = {reach:g} m)", fontsize=13)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, .05, 1, .96), h_pad=3)
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 30, 2, 3, 4, 5])
    ap.add_argument("--reach", type=float, default=0.37, help="waypoint counts as reached within this (m)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    dwa.RANGE_MODE = "matlab"
    check_fig9_regression()
    check_saved_maps()
    os.makedirs(a.out, exist_ok=True)

    astar = algorithms(DIAG, P)["Diagonal Distance"]
    runs = []
    for map_id, seed in enumerate(a.seeds, 1):
        occ = random_grid_map(SIZE, DENSITY, START, GOAL, np.random.default_rng(seed), DIAG)
        path, _, _ = run(astar(), occ, START, GOAL)
        wp = np.array(path, float) + 0.5  # cell centres, in metres
        ob = np.column_stack(np.nonzero(occ)[::-1]).astype(float) + 0.5
        log = dwa.run_waypoints(wp, ob, evalParam=EVAL, obstacleR=OBSTACLE_R, reach=a.reach,
                                max_steps=MAX_STEPS)
        m, bad = evaluate(occ, wp, ob, log, a.reach)
        r = dict(id=map_id, seed=seed, occ=occ, wp=wp, log=log, m=m, bad=bad)
        runs.append(r)

        base = os.path.join(a.out, f"map{map_id}_seed{seed}")
        np.savetxt(base + "_map.txt", np.flipud(occ), fmt="%d")
        np.savetxt(base + "_waypoints.csv", wp, delimiter=",", header="x,y", comments="", fmt="%.2f")
        np.savetxt(base + "_trajectory.csv", log, delimiter=",", header="t,x,y,yaw,v,w,target",
                   comments="", fmt="%.8g")
        plot_one(r, base + ".png")
        print(f"map {map_id} seed {seed:>2}: {verdict(m)} | {m}", flush=True)

    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["map", "seed", *runs[0]["m"]])
        for r in runs:
            out.writerow([r["id"], r["seed"], *r["m"].values()])
    plot_overview(runs, a.reach, os.path.join(a.out, "overview.png"))
    print("written to", a.out)


if __name__ == "__main__":
    main()
