"""
Plot the Octave/MATLAB DWA run on the paper's Figure 9 map (matlab/dwa_fig9.m output) in the
layout of the paper's Figs. 9, 11 and 12, and print summary numbers.

    octave --no-gui --eval "cd matlab; dwa_fig9('../results/dwa_fig9_log.csv')"
    python plot_dwa_fig9.py results/dwa_fig9_log.csv
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main(csv_path, out_png=None):
    m = json.load(open(os.path.join(HERE, "fig9_map.json")))
    ob = np.array(m["cells"], float) + 0.5
    wp = np.array(m["paper_path"], float) + 0.5
    L = np.loadtxt(csv_path, delimiter=",", ndmin=2)   # t, x, y, yaw, v, w, target index
    t, x, y, v, w, k = L[:, 0], L[:, 1], L[:, 2], L[:, 4], L[:, 5], L[:, 6].astype(int)
    n_wp = len(wp) - 1
    reached = int(k[-1] - 1 + (np.hypot(x[-1] - wp[-1, 0], y[-1] - wp[-1, 1]) < 0.5))
    length = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    min_clear = float(np.min(np.hypot(x[:, None] - ob[:, 0], y[:, None] - ob[:, 1])))
    print(f"{os.path.basename(csv_path)}: reached {reached}/{n_wp} path points, {t[-1]:.1f} s, "
          f"{length:.2f} m travelled, v max {v.max():.3f} m/s, |w| max {np.abs(w).max():.3f} rad/s, "
          f"closest approach to an obstacle centre {min_clear:.2f} m")

    fig = plt.figure(figsize=(14, 6.5))
    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(ob[:, 0], ob[:, 1], s=170, c="mediumseagreen")
    ax.plot(x, y, "k-", lw=0.8, label="DWA (MATLABRobotics sample, Table 3 params)")
    ax.plot(wp[:, 0], wp[:, 1], "ko", mfc="none", ms=5, label="Diagonal A* path points")
    ax.scatter(*wp[0], s=200, marker="*", facecolors="none", edgecolors="tab:blue", zorder=4)
    ax.scatter(*wp[-1], s=200, marker="*", facecolors="none", edgecolors="tab:red", zorder=4)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_xticks(range(11)); ax.set_yticks(range(11))
    ax.set_title(f"Paper's Fig. 9 map — DWA: {reached}/{n_wp} points, {t[-1]:.1f} s", fontsize=10)
    ax.legend(loc="upper left", fontsize=7)
    a1 = fig.add_subplot(2, 2, 2)
    a1.plot(t, v, lw=0.9); a1.set_ylabel("linear velocity (m/s)"); a1.grid(True, alpha=0.3)
    a1.set_title("Robot linear velocity obtained by the DWA method (cf. Fig. 11)", fontsize=9)
    a2 = fig.add_subplot(2, 2, 4, sharex=a1)
    a2.plot(t, w, lw=0.9); a2.set_ylabel("angular velocity (rad/s)"); a2.set_xlabel("time(s)")
    a2.grid(True, alpha=0.3)
    a2.set_title("Robot angular velocity obtained by the DWA method (cf. Fig. 12)", fontsize=9)
    fig.tight_layout()
    out_png = out_png or os.path.splitext(csv_path)[0] + ".png"
    fig.savefig(out_png, dpi=150); plt.close(fig)
    return out_png


if __name__ == "__main__":
    main(*sys.argv[1:])
