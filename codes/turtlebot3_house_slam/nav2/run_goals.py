#!/usr/bin/python3
"""One trial per goal, each from the spawn point: plan (NavFn A*) then FollowPath (DWB).

    run_goals.py <setting name> [test|tuning] [time limit in sim s]
Logs commands, odometry and the localized path to results/<setting>/<goal>.npz.

Optional environment variables:
    PLAN_DIR     follow <PLAN_DIR>/<goal>.csv (Diagonal A*) instead of asking Nav2's planner
    RVIZ_WINDOW  X window id: save screenshots during and at the end of every trial
"""
import csv
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseArray, PoseStamped, PoseWithCovarianceStamped, Twist
from nav2_msgs.action import ComputePathToPose, FollowPath
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import Odometry, Path
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

HERE = os.path.dirname(os.path.realpath(__file__))
setting = sys.argv[1]
which = sys.argv[2] if len(sys.argv) > 2 else 'test'
limit = float(sys.argv[3]) if len(sys.argv) > 3 else 500.0  # Table 3: 5000 steps of 0.1 s
GZ_ENV = {**os.environ, 'GZ_PARTITION': 'house_slam'}
PLAN_DIR, WINDOW = os.environ.get('PLAN_DIR'), os.environ.get('RVIZ_WINDOW')
SPAWN = (-2.008, 1.0)  # where map (0, 0) is in the simulator; only used to reset the robot

rclpy.init()
n = rclpy.create_node('dwa_trials')
state = dict(rec=False, sim=0.0)
rows = {}


def on_odom(m):
    t = state['sim'] = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
    if state['rec']:
        rows['odom'].append((t, m.twist.twist.linear.x, m.twist.twist.angular.z))


def on_cmd(m):
    if state['rec']:
        rows['cmd'].append((state['sim'], time.monotonic(), m.linear.x, m.angular.z))


def on_scan(m):
    if state['rec']:
        r = np.array(m.ranges)
        r = r[~np.isnan(r)]
        rows['scan'].append((state['sim'], max(0.0, float(r.min()))))


n.create_subscription(Odometry, '/odom', on_odom, 100)
n.create_subscription(Twist, '/house_sim/cmd_vel', on_cmd, 100)
n.create_subscription(LaserScan, '/scan', on_scan, qos_profile_sensor_data)
stop = n.create_publisher(Twist, '/house_sim/cmd_vel', 5)
init = n.create_publisher(PoseWithCovarianceStamped, '/initialpose', 5)
clear = [n.create_client(ClearEntireCostmap, f'/{c}_costmap/clear_entirely_{c}_costmap')
         for c in ('local', 'global')]
planner = ActionClient(n, ComputePathToPose, 'compute_path_to_pose')
follower = ActionClient(n, FollowPath, 'follow_path')
show = {k: n.create_publisher(t, k, 1) for k, t in
        (('/plan', Path), ('/dwa_trail', Path), ('/astar_waypoints', PoseArray))}
tf = Buffer()
TransformListener(tf, n)


def spin(seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(n, timeout_sec=0.05)


def wait(future, seconds=20):
    rclpy.spin_until_future_complete(n, future, timeout_sec=seconds)
    return future.result() if future.done() else None


def reset():
    for _ in range(5):
        stop.publish(Twist())
        spin(.3)
    subprocess.run(['gz', 'service', '-s', '/world/default/set_pose', '--reqtype', 'gz.msgs.Pose',
                    '--reptype', 'gz.msgs.Boolean', '--timeout', '3000', '--req',
                    f'name: "burger", position: {{x: {SPAWN[0]}, y: {SPAWN[1]}, z: 0.01}}, orientation: {{w: 1}}'],
                   env=GZ_ENV, check=True, capture_output=True, timeout=15)
    spin(1.5)
    m = PoseWithCovarianceStamped()
    m.header.frame_id = 'map'
    m.pose.pose.orientation.w = 1.0
    m.pose.covariance[0] = m.pose.covariance[7] = .01
    m.pose.covariance[35] = .005
    init.publish(m)
    spin(2)
    for c in clear:
        if c.wait_for_service(5):
            wait(c.call_async(ClearEntireCostmap.Request()), 5)
    spin(2)


def to_path(xy):
    path = Path()
    path.header.frame_id = 'map'
    for x, y in xy:
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position.x, p.pose.position.y = float(x), float(y)
        p.pose.orientation.w = 1.0
        path.poses.append(p)
    return path


def diagonal_path(name):
    """The A* polyline, sampled every 5 cm and oriented along each segment."""
    wp = np.loadtxt(os.path.join(PLAN_DIR, name + '.csv'), delimiter=',', skiprows=1)
    xy, yaw = [], []
    for a, b in zip(wp[:-1], wp[1:]):
        k = max(1, math.ceil(np.hypot(*(b - a)) / .05))
        xy += [a + (b - a) * i / k for i in range(k)]
        yaw += [math.atan2(b[1] - a[1], b[0] - a[0])] * k
    path = to_path(xy + [wp[-1]])
    for p, t in zip(path.poses, yaw + [yaw[-1]]):
        p.pose.orientation.z, p.pose.orientation.w = math.sin(t / 2), math.cos(t / 2)
    dots = PoseArray()
    dots.header.frame_id = 'map'
    dots.poses = [p.pose for p in to_path(wp).poses]
    show['/astar_waypoints'].publish(dots)
    return path


def screenshot(tag, block=False):
    if not WINDOW:
        return
    os.makedirs(os.path.join(folder, 'screenshots'), exist_ok=True)
    cmd = ['timeout', '15', 'import', '-window', WINDOW,
           os.path.join(folder, 'screenshots', f'{name}_{tag}.png')]
    (subprocess.run if block else subprocess.Popen)(cmd)


def trial(goal):
    out = dict(result='planning failed', time_s=0.0, rtf=1.0)
    req = ComputePathToPose.Goal()
    req.planner_id = 'GridBased'
    req.goal = PoseStamped()
    req.goal.header.frame_id = 'map'
    req.goal.pose.position.x, req.goal.pose.position.y = goal['x'], goal['y']
    req.goal.pose.orientation.w = 1.0
    if PLAN_DIR:
        path = diagonal_path(name)
    else:
        h = wait(planner.send_goal_async(req))
        if not h or not h.accepted:
            return out
        res = wait(h.get_result_async())
        if not res or not res.result.path.poses:
            return out
        path = res.result.path
    show['/plan'].publish(path)
    show['/dwa_trail'].publish(to_path([]))
    shot = 0
    rows['plan'] = [(p.pose.position.x, p.pose.position.y) for p in path.poses]

    req = FollowPath.Goal()
    req.path, req.controller_id, req.goal_checker_id = path, 'FollowPath', 'general_goal_checker'
    state['rec'] = True
    t0, w0, last = state['sim'], time.monotonic(), 0.0
    h = wait(follower.send_goal_async(req))
    if not h or not h.accepted:
        state['rec'] = False
        return {**out, 'result': 'controller rejected'}
    done, timed_out = h.get_result_async(), False
    while not done.done():
        rclpy.spin_once(n, timeout_sec=.05)
        if time.monotonic() - last > .1:
            last = time.monotonic()
            try:
                p = tf.lookup_transform('map', 'base_footprint', rclpy.time.Time()).transform.translation
                rows['pose'].append((state['sim'], p.x, p.y))
            except Exception:
                pass
            if WINDOW and rows['pose'] and len(rows['pose']) % 5 == 0:
                show['/dwa_trail'].publish(to_path([q[1:] for q in rows['pose']]))
                show['/plan'].publish(path)
            if WINDOW and last - w0 > 8 + 16 * shot:  # wall seconds
                screenshot(f'during{shot + 1}')
                shot += 1
        if state['sim'] - t0 > limit or time.monotonic() - w0 > 1.5 * limit:
            timed_out = True
            wait(h.cancel_goal_async(), 5)
            break
    t1, w1 = state['sim'], time.monotonic()
    spin(.3)
    state['rec'] = False
    if WINDOW:
        show['/dwa_trail'].publish(to_path([q[1:] for q in rows['pose']]))
        spin(1.5)
        screenshot('end', block=True)
    ok = not timed_out and done.result().status == GoalStatus.STATUS_SUCCEEDED
    return dict(result='reached' if ok else 'timeout' if timed_out else 'controller aborted',
                time_s=round(t1 - t0, 1), rtf=round((t1 - t0) / (w1 - w0), 3))


if not planner.wait_for_server(40) or not follower.wait_for_server(20):
    sys.exit('Nav2 is not up')
spin(3)
folder = os.path.join(HERE, 'results', setting)
os.makedirs(folder, exist_ok=True)
summary = []
for name, g in json.load(open(os.path.join(HERE, 'goals.json')))[which].items():
    rows.clear()
    rows.update({k: [] for k in ('cmd', 'odom', 'scan', 'pose', 'plan')})
    reset()
    r = trial(g)
    data = {k: np.array(v) for k, v in rows.items()}
    np.savez(os.path.join(folder, name + '.npz'), **data, goal=[g['x'], g['y']], **r)
    r = dict(setting=setting, goal=name, **r)
    if len(data['pose']):
        end = data['pose'][-1, 1:]
        r['end_to_goal_m'] = round(math.hypot(end[0] - g['x'], end[1] - g['y']), 2)
    if len(data['cmd']):
        r['cmd_v_max'] = round(float(data['cmd'][:, 2].max()), 3)
        r['cmd_w_max'] = round(float(abs(data['cmd'][:, 3]).max()), 3)
    if len(data['scan']):
        r['closest_lidar_m'] = round(float(data['scan'][:, 1].min()), 3)
    summary.append(r)
    print(r, flush=True)
for _ in range(5):
    stop.publish(Twist())
    spin(.2)
keys = list(dict.fromkeys(k for r in summary for k in r))
with open(os.path.join(folder, 'summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, keys)
    w.writeheader()
    w.writerows(summary)
