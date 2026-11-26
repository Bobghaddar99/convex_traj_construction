"""
Improved DWA, to compare fairly with OTPWR. The baseline (dwa_matlab_port.py) is left untouched.

The loop is the same as in the baseline: at every 0.1 s sample (v, w) inside the dynamic window,
predict, score, apply the best command. Five things change:

 1. HEADING    The error is wrapped to [-pi, pi]. The sample's 180 - |goal - yaw| becomes wrong
               as soon as the yaw passes +-180 degrees.

 2. SAFETY     A command is admissible only if the robot can apply it for one step and then brake
               to a stop without coming within MARGIN of an obstacle cell or of the map border.
               (Fox et al. 1997 define admissible velocities this way; the sample omits it.)
               If the current command was admissible, "brake one more step with the same w" is
               admissible now, and it is always among the samples. So a safe command always
               exists: by induction the robot cannot collide in this kinematic model.

 3. PROGRESS   The "go fast" score is replaced by "end the prediction closer to the waypoint".
               Rewarding speed is what made the robot overshoot a waypoint and circle around it.

 4. LIMITS     v <= 0.5 m/s and |dv/dt| <= 0.2 m/s^2, the limits that the paper gives to OTPWR
               (Table 3 gives DWA 2 m/s and 0.5 m/s^2). The angular limits stay as in Table 3.

 5. WINDOW     Evenly spaced samples that include both ends of the window and the current w.

Score of an admissible command, to be maximised:

    J = heading * (1 - |e| / pi)                      e: heading error at the end of the prediction
      + progress * (d_now - d_end) / (V_MAX * T)      d: distance to the waypoint
      + clearance * min(c, 0.5) / 0.5                 c: smallest clearance on the braking path

The weights and the horizon T are chosen by dwa_tune_compare.py on tuning maps only.
DWA is not a convex program: it enumerates about 100 commands per step and keeps the best one.
"""
import math

import numpy as np

DT = 0.1
V_MAX, A_MAX = 0.5, 0.2      # OTPWR's limits
W_MAX, DW_MAX = 0.35, 0.87   # Table 3
V_STEP, W_STEP = 0.005, 0.017
MARGIN = 0.15                # clearance kept from obstacle cells and the border (m)
DEFAULTS = dict(heading=1.0, progress=0.5, clearance=0.1, horizon=2.0)  # tuned on seeds 100-139


def samples(lo, hi, step):
    """Values from lo to hi, both included, at most `step` apart."""
    return np.linspace(lo, hi, max(2, math.ceil((hi - lo) / step - 1e-9) + 1))


def dynamic_window(v, w):
    """All (v, w) that can be reached within one time step."""
    vs = samples(max(0.0, v - A_MAX * DT), min(V_MAX, v + A_MAX * DT), V_STEP)
    ws = np.append(samples(max(-W_MAX, w - DW_MAX * DT), min(W_MAX, w + DW_MAX * DT), W_STEP), w)
    V, W = np.meshgrid(vs, ws, indexing="ij")
    return V.ravel(), W.ravel()


def rollout(state, speeds, W):
    """Positions for speeds[k, n] (command k, step n) and a constant turn rate W[k].
    Same motion model as the baseline: move with the old yaw, then turn."""
    yaw = state[2] + DT * W[:, None] * np.arange(speeds.shape[1])
    x = state[0] + np.cumsum(DT * speeds * np.cos(yaw), axis=1)
    y = state[1] + np.cumsum(DT * speeds * np.sin(yaw), axis=1)
    return x, y


def clearance(x, y, ob, size):
    """Distance to the nearest obstacle cell (a 1 m square) or to the map border."""
    d = np.minimum.reduce([x, y, size - x, size - y])
    if len(ob):
        dx = np.maximum(np.abs(x[..., None] - ob[:, 0]) - 0.5, 0)
        dy = np.maximum(np.abs(y[..., None] - ob[:, 1]) - 0.5, 0)
        d = np.minimum(d, np.sqrt((dx * dx + dy * dy).min(axis=-1)))
    return d


def choose_command(state, goal, ob, size, p):
    V, W = dynamic_window(state[3], state[4])

    # safety: one step at v, then full braking
    n_brake = math.ceil(V.max() / (A_MAX * DT) - 1e-9) + 1
    braking = np.maximum(V[:, None] - A_MAX * DT * np.arange(n_brake), 0.0)
    clear = clearance(*rollout(state, braking, W), ob, size).min(axis=1)
    safe = clear >= MARGIN - 1e-9
    if not safe.any():
        raise RuntimeError("no admissible command: the start pose itself is unsafe")

    # score: prediction with constant (v, w) over the horizon
    n = round(p["horizon"] / DT)
    x, y = rollout(state, np.repeat(V[:, None], n, axis=1), W)
    to_x, to_y = goal[0] - x[:, -1], goal[1] - y[:, -1]
    e = np.arctan2(to_y, to_x) - (state[2] + W * n * DT)
    e = (e + np.pi) % (2 * np.pi) - np.pi                         # wrapped
    progress = math.hypot(goal[0] - state[0], goal[1] - state[1]) - np.hypot(to_x, to_y)
    J = (p["heading"] * (1 - np.abs(e) / np.pi)
         + p["progress"] * progress / (V_MAX * p["horizon"])
         + p["clearance"] * np.minimum(clear, 0.5) / 0.5)
    J[~safe] = -np.inf
    best = int(np.argmax(J))
    return float(V[best]), float(W[best])


def run_waypoints(wp, ob, params=None, reach=0.37, max_steps=5000, size=10):
    """Same inputs and log format as dwa_matlab_port.run_waypoints."""
    p = {**DEFAULTS, **(params or {})}
    wp, ob = np.asarray(wp, float), np.asarray(ob, float).reshape(-1, 2)
    s = [wp[0, 0], wp[0, 1], math.atan2(wp[1, 1] - wp[0, 1], wp[1, 0] - wp[0, 0]), 0.0, 0.0]
    k, log = 1, []
    for i in range(1, max_steps + 1):
        v, w = choose_command(s, wp[k], ob, size, p)
        s = [s[0] + DT * v * math.cos(s[2]), s[1] + DT * v * math.sin(s[2]), s[2] + DT * w, v, w]
        log.append([i * DT, *s, k])
        if math.hypot(s[0] - wp[k, 0], s[1] - wp[k, 1]) < reach:
            k += 1
            if k == len(wp):
                break
    return np.array(log)
