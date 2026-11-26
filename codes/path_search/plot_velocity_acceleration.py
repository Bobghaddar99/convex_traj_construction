"""
Velocity and acceleration of saved pure-DWA runs. Reads the trajectory CSVs; runs no planner.

    python plot_velocity_acceleration.py [results/dwa_random | results/dwa_pure_tuned]

Writes, in that folder:
    velocity_acceleration.png     v, a = dv/dt, w, alpha = dw/dt     (cf. Figs. 11-12 of the paper)
    xy_velocity_acceleration.png  vx, vy, ax, ay                     (cf. Fig. 10 of the paper)
    velocity_acceleration.csv     peak values per map

A run that leaves the map is drawn up to that moment only: afterwards it circles outside until 500 s.
Accelerations are exact backward differences of the commands (dt = 0.1 s), without smoothing.
"""
import csv
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dwa_matlab_port as dwa
import dwa_random_maps as common

BLUE, ORANGE, MUTED, GRID, INK = common.BLUE, common.ORANGE, common.MUTED, common.GRID, common.INK
V_MAX, W_MAX, A_MAX, DW_MAX = dwa.TABLE3_KINEMATIC[:4]


def load(path):
    map_id, seed = map(int, re.search(r"map(\d+)_seed(\d+)_trajectory", path).groups())
    log = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    wp = np.loadtxt(os.path.join(common.OUT, f"map{map_id}_seed{seed}_waypoints.csv"),
                    delimiter=",", skiprows=1)
    t, x, y, yaw, v, w = log[:, :6].T
    # the position update of a step uses the yaw from before that step
    d = wp[1] - wp[0]
    old_yaw = np.r_[np.arctan2(d[1], d[0]), yaw[:-1]]
    vx, vy = v * np.cos(old_yaw), v * np.sin(old_yaw)
    diff = lambda q: np.diff(np.r_[0.0, q]) / dwa.DT
    s = dict(t=t, v=v, w=w, a=diff(v), alpha=diff(w), vx=vx, vy=vy, ax=diff(vx), ay=diff(vy))
    out = np.flatnonzero((x < 0) | (x >= common.SIZE) | (y < 0) | (y >= common.SIZE))
    n = out[0] + 1 if len(out) else len(t)
    last = int(log[-1, 6])
    goal = last == len(wp) - 1 and np.hypot(x[-1] - wp[-1, 0], y[-1] - wp[-1, 1]) < 0.37
    if len(out):
        state = f"FAILED, left map at {t[out[0]]:.1f} s"
    else:
        state = f"goal in {t[-1]:.1f} s" if goal else "FAILED, goal not reached"
    return dict(id=map_id, seed=seed, state=state, full=s, shown={k: q[:n] for k, q in s.items()})


def figure(runs, columns, title, fname):
    fig, axes = plt.subplots(len(runs), len(columns), figsize=(4.6 * len(columns), 2.2 * len(runs) + 1),
                             squeeze=False, sharey="col")
    for row, r in zip(axes, runs):
        s = r["shown"]
        for ax, (heading, series, limit) in zip(row, columns):
            for key, colour, label in series:
                ax.plot(s["t"], s[key], c=colour, lw=0.9, label=label)
            for y in limit:
                ax.axhline(y, c=MUTED, lw=0.8, ls="--")
            ax.grid(color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8, colors=MUTED, labelleft=True)
            ax.set_xlim(0, s["t"][-1])
            if r is runs[0]:
                ax.set_title(heading, fontsize=10, color=INK)
                if len(series) > 1:
                    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
            if r is runs[-1]:
                ax.set_xlabel("time (s)")
        row[0].set_ylabel(f"Map {r['id']} (seed {r['seed']})\n{r['state']}", fontsize=9, color=INK)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(fname, dpi=130)
    plt.close(fig)


def main(folder):
    runs = sorted((load(p) for p in glob.glob(os.path.join(folder, "map*_trajectory.csv"))),
                  key=lambda r: r["id"])
    if not runs:
        sys.exit(f"no trajectories in {folder}")
    name = "tuned weights" if "tuned" in os.path.basename(os.path.normpath(folder)) else "original weights"
    figure(runs, [
        (f"Linear velocity v (m/s), max {V_MAX:g}", [("v", BLUE, "v")], [V_MAX]),
        (f"Linear acceleration dv/dt (m/s²), limit ±{A_MAX:g}", [("a", BLUE, "a")], [-A_MAX, A_MAX]),
        (f"Angular velocity ω (rad/s), limit ±{W_MAX:g}", [("w", BLUE, "w")], [-W_MAX, W_MAX]),
        (f"Angular acceleration dω/dt (rad/s²), limit ±{DW_MAX:g}", [("alpha", BLUE, "alpha")], [-DW_MAX, DW_MAX]),
    ], f"Pure DWA ({name}): velocity and acceleration. Dashed: Table 3 limits",
        os.path.join(folder, "velocity_acceleration.png"))
    figure(runs, [
        ("Velocity components (m/s). Dashed: OTPWR ±0.5", [("vx", BLUE, "x"), ("vy", ORANGE, "y")], [-.5, .5]),
        ("Acceleration components (m/s²). Dashed: OTPWR ±0.2", [("ax", BLUE, "x"), ("ay", ORANGE, "y")], [-.2, .2]),
    ], f"Pure DWA ({name}): x and y components",
        os.path.join(folder, "xy_velocity_acceleration.png"))

    rows = []
    for r in runs:
        s = r["shown"]
        rows.append(dict(
            map=r["id"], seed=r["seed"], result=r["state"],
            v_mean=round(float(s["v"].mean()), 3), v_max=round(float(s["v"].max()), 3),
            a_max=round(float(abs(s["a"]).max()), 3),
            a_sign_changes_per_s=round(float(np.sum(np.diff(np.sign(s["a"])) != 0) / s["t"][-1]), 2),
            w_max=round(float(abs(s["w"]).max()), 3), alpha_max=round(float(abs(s["alpha"]).max()), 3),
            axy_max=round(float(max(abs(s["ax"]).max(), abs(s["ay"]).max())), 3),
            time_pct_axy_over_0p2=round(float(np.mean((abs(s["ax"]) > .2001) | (abs(s["ay"]) > .2001)) * 100), 1),
            v_max_whole_run=round(float(r["full"]["v"].max()), 3),
        ))
        print(rows[-1])
    with open(os.path.join(folder, "velocity_acceleration.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else common.OUT)
