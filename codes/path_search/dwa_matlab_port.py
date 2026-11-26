"""
Line-by-line Python port of Atsushi Sakai's MATLAB DWA sample
(AtsushiSakai/MATLABRobotics, PathPlanning/DynamicWindowApproach/DynamicWindowApproachSample.m,
Modified BSD), which the paper's Table 3 parameters appear to come from
(5000 window steps, 0.1 s step, 1 deg = 0.017 rad yaw-rate resolution).

Faithful to the MATLAB, including its quirks:
  * motion model x = F x + B u: position advanced with the OLD yaw, then yaw += w dt
  * heading score in degrees, 180 - |goalTheta - theta|, no angle wrapping
  * dist score = min(2, min_i ||p - ob_i|| - R)   (no hard collision rejection)
  * each score column normalised by its sum, feval = evalParam(1:3) . [heading dist vel]
  * argmax picks the first maximum (MATLAB max)
  * MATLAB colon ranges a:s:b and the `while time <= evaldt` float accumulation

Validated against the original .m run in GNU Octave (see compare_dwa_port.py).
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DT = 0.1


EPS = np.finfo(float).eps
RANGE_MODE = "matlab"   # "matlab" (MathWorks colonop algorithm) or "octave" (liboctave Range)


def _octave_tfloor(x, ct):
    """liboctave xtfloor (Hagerty's FL5 tolerant floor)."""
    q = 1.0 - ct if x < 0.0 else 1.0
    rmax = q / (2.0 - ct)
    t1 = 1.0 + math.floor(x)
    t1 = (ct / q) * (-t1 if t1 < 0.0 else t1)
    t1 = rmax if rmax < t1 else t1
    t1 = ct if ct > t1 else t1
    t1 = math.floor(x + t1)
    return t1 if (x <= 0.0 or (t1 - x) < rmax) else t1 - 1.0


def _octave_teq(u, v, ct=3 * EPS):
    tu, tv = abs(u), abs(v)
    return abs(u - v) < ((tu if tu > tv else tv) * ct)


def _colon_octave(a, d, b):
    ct = 3 * EPS
    tmp = _octave_tfloor((b - a + d) / d, ct)
    n = int(tmp) if tmp > 0 else 0
    if n > 0 and not _octave_teq(a + (n - 1) * d, b):
        if _octave_teq(a + (n - 2) * d, b):
            n -= 1
        elif _octave_teq(a + n * d, b):
            n += 1
    if n <= 0:
        return np.empty(0)
    final = a + (n - 1) * d
    if (d > 0 and final >= b) or (d < 0 and final <= b):
        final = b
    v = np.array([a + i * d for i in range(n)], float)
    v[0] = a
    v[-1] = final
    return v


def _colon_matlab(a, d, b):
    """MathWorks' published colonop.m: how MATLAB builds a:d:b."""
    tol = 2.0 * EPS * max(abs(a), abs(b))
    sig = 1.0 if d > 0 else -1.0
    if d == 0 or (sig > 0 and a > b) or (sig < 0 and a < b):
        return np.empty(0)
    if a == math.floor(a) and d == 1:
        n = math.floor(b) - a
    elif a == math.floor(a) and d == math.floor(d):
        q = math.floor(a / d); r = a - q * d
        n = math.floor((b - r) / d) - q
    else:
        n = math.floor(abs((b - a) / d) + 0.5) * (1 if (b - a) / d >= 0 else -1)
        if sig * (a + n * d - b) > tol:
            n -= 1
    n = int(n)
    c = a + n * d
    if sig * (c - b) > -tol:
        c = b
    v = np.zeros(n + 1)
    k = np.arange(0, n // 2 + 1)
    v[k] = a + k * d
    v[n - k] = c - k * d
    if n % 2 == 0:
        v[n // 2] = (a + c) / 2
    return v


def colon(a, d, b):
    """MATLAB/Octave a:d:b, bit-exact for the selected interpreter."""
    return _colon_matlab(a, d, b) if RANGE_MODE == "matlab" else _colon_octave(a, d, b)


def norm2(dx, dy):
    """Octave vector 2-norm (norm_accumulator_2: scaled sum of squares)."""
    scl, sm = 0.0, 1.0
    for val in (dx, dy):
        t = abs(val)
        if scl == t:
            sm += 1
        elif scl < t:
            sm *= (scl / t) ** 2
            sm += 1
            scl = t
        elif t != 0:
            sm += (t / scl) ** 2
    return scl * math.sqrt(sm)


def to_degree(r):
    return r / math.pi * 180


def f(x, u, dt=DT):
    # F zeroes v, w then B adds [dt cos(yaw) v, dt sin(yaw) v, dt w, v, w] using the current yaw
    return np.array([x[0] + dt * math.cos(x[2]) * u[0],
                     x[1] + dt * math.sin(x[2]) * u[0],
                     x[2] + dt * u[1],
                     u[0],
                     u[1]])


def calc_dynamic_window(x, model, dt=DT):
    Vs = [0, model[0], -model[1], model[1]]
    Vd = [x[3] - model[2] * dt, x[3] + model[2] * dt, x[4] - model[3] * dt, x[4] + model[3] * dt]
    return [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]), max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]


def generate_trajectory(x, vt, ot, evaldt, dt=DT):
    time = 0.0
    u = (vt, ot)
    while time <= evaldt:
        time = time + dt
        x = f(x, u, dt)
    return x


def calc_dist_eval(x, ob, R):
    dist = 2.0
    for ox, oy in ob:
        disttmp = norm2(ox - x[0], oy - x[1]) - R
        if dist > disttmp:
            dist = disttmp
    return dist


def calc_heading_eval(x, goal):
    theta = to_degree(x[2])
    goal_theta = to_degree(math.atan2(goal[1] - x[1], goal[0] - x[0]))
    target = goal_theta - theta if goal_theta > theta else theta - goal_theta
    return 180.0 - target


def evaluation(x, Vr, goal, ob, R, model, evalParam):
    rows = []
    for vt in colon(Vr[0], model[4], Vr[1]):
        for ot in colon(Vr[2], model[5], Vr[3]):
            xt = generate_trajectory(x, vt, ot, evalParam[3])
            rows.append((vt, ot, calc_heading_eval(xt, goal), calc_dist_eval(xt, ob, R), abs(vt)))
    return np.array(rows)


def normalize_eval(E):
    for c in (2, 3, 4):
        s = float(np.cumsum(E[:, c])[-1])      # sequential sum, like MATLAB/Octave sum
        if s != 0:
            E[:, c] = E[:, c] / s
    return E


def dynamic_window_approach(x, model, goal, evalParam, ob, R):
    Vr = calc_dynamic_window(x, model)
    E = evaluation(x, Vr, goal, ob, R, model, evalParam)
    if E.size == 0:
        return np.zeros(2)
    E = normalize_eval(E)
    p = evalParam
    feval = np.array([p[0] * h + p[1] * d + p[2] * v for h, d, v in E[:, 2:5]])
    ind = int(np.argmax(feval))
    return E[ind, :2].copy()


# --------------------------------------------------------------- paper driver
TABLE3_KINEMATIC = [2.0, 0.35, 0.5, 0.87, 0.02, 0.017]  # vmax, wmax, acc, dw, v res, w res


def run_waypoints(wp, ob, kinematic=TABLE3_KINEMATIC, evalParam=(0.1, 0.2, 0.1, 3.0),
                  obstacleR=0.5, reach=0.5, max_steps=5000):
    """Same main loop as matlab/dwa_fig9.m. Returns log rows [t, x, y, yaw, v, w, target_idx]."""
    wp = np.asarray(wp, float)
    ob = np.asarray(ob, float)
    x = np.array([wp[0, 0], wp[0, 1], math.atan2(wp[1, 1] - wp[0, 1], wp[1, 0] - wp[0, 0]), 0.0, 0.0])
    k = 1
    log = []
    for i in range(1, max_steps + 1):
        u = dynamic_window_approach(x, kinematic, wp[k], evalParam, ob, obstacleR)
        x = f(x, u)
        log.append([i * DT, *x, k])
        if norm2(x[0] - wp[k, 0], x[1] - wp[k, 1]) < reach:
            k += 1
            if k == len(wp):
                break
    return np.array(log)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results", "dwa_fig9_port_log.csv"))
    ap.add_argument("--eval", type=float, nargs=4, default=[0.1, 0.2, 0.1, 3.0],
                    help="evalParam [heading dist velocity predictDT] (sample default)")
    ap.add_argument("--R", type=float, default=0.5, help="obstacleR (sample default)")
    ap.add_argument("--reach", type=float, default=0.5, help="goal test radius (sample default)")
    ap.add_argument("--range", choices=["matlab", "octave"], default="matlab",
                    help="emulate MATLAB's or Octave's a:d:b rounding (trajectory is sensitive to it)")
    a = ap.parse_args()
    global RANGE_MODE
    RANGE_MODE = a.range
    m = json.load(open(os.path.join(HERE, "fig9_map.json")))
    ob = np.array(m["cells"], float) + 0.5
    wp = np.array(m["paper_path"], float) + 0.5
    log = run_waypoints(wp, ob, evalParam=a.eval, obstacleR=a.R, reach=a.reach)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savetxt(a.out, log, delimiter=",", fmt="%.8g")
    sys.path.insert(0, HERE)
    from plot_dwa_fig9 import main as plot
    print("figure:", plot(a.out))


if __name__ == "__main__":
    main()
