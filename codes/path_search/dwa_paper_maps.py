"""
Section 5.2 of Luo et al. (Machines 2024, 12, 668): DWA tracking the Diagonal-A* path points,
run on the paper's own Grid Map 1 and Grid Map 2 (paper_maps.json).

DWA is NOT implemented here. It is PythonRobotics' Dynamic Window Approach
(https://github.com/AtsushiSakai/PythonRobotics, PathPlanning/DynamicWindowApproach,
MIT licence), vendored unmodified in third_party/pythonrobotics/. This script only:
  * sets the Config to the paper's Table 3 parameters,
  * feeds the A* path points to dwa_control() one at a time as successive goals
    (switching to the next point when the robot is within --reach m of the current one),
  * logs the trajectory and v / w commands, and plots them (cf. Figs. 9, 11, 12).

Unspecified in the paper (assumptions, all CLI-adjustable):
  predict_time 3.0 s and cost gains (PythonRobotics defaults), min_speed 0,
  collision radius 0.3 m around each obstacle-cell centre, reach tolerance 0.3 m,
  initial heading pointed at the first path point.
"""
import argparse
import json
import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from pathfinding.core.diagonal_movement import DiagonalMovement

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "third_party", "pythonrobotics"))
import dynamic_window_approach as pr_dwa  # noqa: E402  (PythonRobotics)

from paper_maps import GOAL, START, load_maps  # noqa: E402
from replicate_5_1 import OUT_DIR, algorithms, run  # noqa: E402

pr_dwa.show_animation = False


def table3_config(a):
    c = pr_dwa.Config()
    c.dt = 0.1                       # Search time step (s)
    c.v_resolution = 0.02            # Linear velocity resolution (m/s)
    c.yaw_rate_resolution = 0.017    # Angular velocity resolution (rad/s)
    c.max_speed = 2.0                # Maximum linear velocity (m/s)
    c.max_yaw_rate = 0.35            # Maximum angular velocity (rad/s)
    c.max_accel = 0.5                # Maximum linear acceleration (m/s^2)
    c.max_delta_yaw_rate = 0.87      # Maximum angular acceleration (rad/s^2)
    c.min_speed = 0.0
    c.predict_time = a.predict_time
    c.robot_type = pr_dwa.RobotType.circle
    c.robot_radius = a.radius
    return c


def run_dwa(waypoints, ob, cfg, max_steps, reach):
    """Drive through waypoints with PythonRobotics dwa_control. Returns log dict."""
    wp = np.asarray(waypoints, float)
    yaw0 = math.atan2(wp[1, 1] - wp[0, 1], wp[1, 0] - wp[0, 0])
    x = np.array([wp[0, 0], wp[0, 1], yaw0, 0.0, 0.0])   # [x, y, yaw, v, w]
    k = 1                                                 # index of current target point
    traj, us, reached_at = [x.copy()], [], []
    collided = False
    t0 = time.perf_counter()
    for step in range(max_steps):                         # Table 3: Maximum window steps
        u, _ = pr_dwa.dwa_control(x, cfg, wp[k], ob)
        x = pr_dwa.motion(x, u, cfg.dt)
        traj.append(x.copy()); us.append(list(u))
        if np.min(np.hypot(ob[:, 0] - x[0], ob[:, 1] - x[1])) <= cfg.robot_radius:
            collided = True
        if math.hypot(x[0] - wp[k, 0], x[1] - wp[k, 1]) <= reach:
            reached_at.append((k, (step + 1) * cfg.dt))
            k += 1
            if k == len(wp):
                break
    return dict(traj=np.array(traj), u=np.array(us), reached=reached_at, n_wp=len(wp) - 1,
                done=k == len(wp), steps=len(us), collided=collided,
                wall=time.perf_counter() - t0)


def metrics(log, dt):
    tr, u = log["traj"], log["u"]
    length = float(np.sum(np.hypot(np.diff(tr[:, 0]), np.diff(tr[:, 1]))))
    v, w = u[:, 0], u[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.abs(np.where(v > 0.05, w / v, np.nan))
    return dict(length=length, time=log["steps"] * dt, v_mean=float(v.mean()), v_max=float(v.max()),
                dv_max=float(np.abs(np.diff(v)).max() / dt), dw_max=float(np.abs(np.diff(w)).max() / dt),
                kappa_max=float(np.nanmax(kappa)) if np.any(~np.isnan(kappa)) else float("nan"),
                v_switches=int(np.sum(np.diff(np.sign(np.diff(v))) != 0)))


def plot(occ, wp, log, m, map_id, cfg, fname):
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 2, 1)
    oy, ox = np.nonzero(occ)
    ax.scatter(ox + 0.5, oy + 0.5, s=160, c="mediumseagreen", label="Obstacles")
    for cx, cy in zip(ox + 0.5, oy + 0.5):
        ax.add_patch(plt.Circle((cx, cy), cfg.robot_radius, fill=False, ls=":", ec="gray", lw=0.6))
    ax.plot(wp[:, 0], wp[:, 1], "k+", ms=10, mew=1.5, label="Diagonal A* path points")
    ax.plot(wp[:, 0], wp[:, 1], "k:", lw=0.8)
    tr = log["traj"]
    ax.plot(tr[:, 0], tr[:, 1], "-", c="tab:red", lw=1.8, label="DWA trajectory")
    ax.scatter(*wp[0], s=200, marker="*", facecolors="none", edgecolors="tab:blue", label="starting point", zorder=4)
    ax.scatter(*wp[-1], s=200, marker="*", facecolors="none", edgecolors="tab:red", label="Target point", zorder=4)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_xlabel("X(m)"); ax.set_ylabel("Y(m)")
    status = "reached goal" if log["done"] else f"stopped after {len(log['reached'])}/{log['n_wp']} path points"
    ax.set_title(f"Paper's Grid Map {map_id}: DWA tracking A* points\n{status}, "
                 f"{m['time']:.1f} s, {m['length']:.2f} m", fontsize=10)
    ax.legend(loc="upper left", fontsize=7)

    t = np.arange(len(log["u"])) * cfg.dt
    ax1 = fig.add_subplot(2, 2, 2)
    ax1.step(t, log["u"][:, 0], where="post", c="tab:red", lw=1)
    ax1.set_ylabel("linear velocity v (m/s)"); ax1.grid(True, alpha=0.3)
    ax1.set_title("DWA commands (cf. Figs. 11 & 12)", fontsize=10)
    ax2 = fig.add_subplot(2, 2, 4, sharex=ax1)
    ax2.step(t, log["u"][:, 1], where="post", c="tab:purple", lw=1)
    ax2.set_ylabel("angular velocity w (rad/s)"); ax2.set_xlabel("time (s)"); ax2.grid(True, alpha=0.3)
    for k, tk in log["reached"]:
        for a_ in (ax1, ax2):
            a_.axvline(tk, c="gray", lw=0.5, ls="--")
    fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=5000, help="Table 3: maximum window steps")
    ap.add_argument("--predict-time", type=float, default=3.0)
    ap.add_argument("--radius", type=float, default=0.3, help="collision radius around obstacle centres (m)")
    ap.add_argument("--reach", type=float, default=0.3, help="distance at which a path point counts as reached (m)")
    a = ap.parse_args()
    cfg = table3_config(a)
    diag_astar = algorithms(DiagonalMovement.always, 0.001)["Diagonal Distance"]
    os.makedirs(OUT_DIR, exist_ok=True)

    for map_id, occ in load_maps().items():
        path, _, _ = run(diag_astar(), occ, START, GOAL)
        wp = np.array(path, float) + 0.5                      # cell centres in metres
        ob = np.column_stack(np.nonzero(occ)[::-1]).astype(float) + 0.5
        log = run_dwa(wp, ob, cfg, a.max_steps, a.reach)
        m = metrics(log, cfg.dt)
        plot(occ, wp, log, m, map_id, cfg, os.path.join(OUT_DIR, f"dwa_paper_map_{map_id}.png"))
        np.savetxt(os.path.join(OUT_DIR, f"dwa_paper_map_{map_id}_log.csv"),
                   np.column_stack([np.arange(log["steps"]) * cfg.dt, log["traj"][1:], log["u"]]),
                   delimiter=",", header="t,x,y,yaw,v,w,u_v,u_w", comments="", fmt="%.5f")

        print(f"\nPaper's Grid Map {map_id}: {len(wp) - 1} A* path segments, A* length "
              f"{np.sum(np.hypot(*np.diff(wp, axis=0).T)):.4f} m")
        print(f"  reached {len(log['reached'])}/{log['n_wp']} path points"
              f"{' (GOAL)' if log['done'] else ' — did NOT reach goal'} in {log['steps']} steps; "
              f"collision: {log['collided']}; compute {log['wall']:.1f} s")
        print("  point reached at t =", ", ".join(f"#{k}:{tk:.1f}s" for k, tk in log["reached"]))
        print(f"  travelled {m['length']:.3f} m in {m['time']:.1f} s  |  v mean {m['v_mean']:.3f} max {m['v_max']:.3f} m/s"
              f"  |  max |dv/dt| {m['dv_max']:.2f} m/s²  max |dw/dt| {m['dw_max']:.2f} rad/s²"
              f"  |  max curvature {m['kappa_max']:.2f} 1/m  |  v direction changes {m['v_switches']}")
    print(f"\nFigures and logs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
