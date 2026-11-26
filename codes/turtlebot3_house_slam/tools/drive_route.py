#!/usr/bin/python3
"""Drive the mapping route on true simulator pose. SLAM only gets lidar and odometry."""
import math
import os
import subprocess
import sys
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.parameter import Parameter
from scipy import ndimage
from tf2_ros import Buffer, TransformListener

import house

out = sys.argv[1]
yaw = lambda q: math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
wrap = lambda a: (a + math.pi) % (2*math.pi) - math.pi
rclpy.init()
node = rclpy.create_node('route_driver', parameter_overrides=[Parameter('use_sim_time', value=True)])
pub = node.create_publisher(Twist, '/house_sim/cmd_vel', 5)
now, log = {}, {k: [] for k in ('truth', 'odom', 'slam')}


def record(key):
    def callback(m):
        p = m.pose.pose
        now[key] = (m.header.stamp.sec + m.header.stamp.nanosec*1e-9,
                    p.position.x, p.position.y, yaw(p.orientation))
        log[key].append(now[key])
    return callback


for key in ('truth', 'odom'):
    node.create_subscription(Odometry, '/' + key, record(key), 50)
tf = Buffer()
TransformListener(tf, node)
threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()


def sample():
    try:
        t = tf.lookup_transform('map', 'base_footprint', rclpy.time.Time())
    except Exception:
        return
    v = (t.header.stamp.sec + t.header.stamp.nanosec*1e-9, t.transform.translation.x,
         t.transform.translation.y, yaw(t.transform.rotation))
    if not log['slam'] or v[0] > log['slam'][-1][0]:
        log['slam'].append(v)


def send(v=0., w=0.):
    m = Twist()
    m.linear.x, m.angular.z = float(v), float(w)
    pub.publish(m)


deadline = time.time() + 60
while len(now) < 2:
    if time.time() > deadline:
        sys.exit('no odometry')
    time.sleep(.2)
time.sleep(5)
origin = now['truth'][1:3]
dist = ndimage.distance_transform_edt(~house.obstacles()) * house.RES
v = w = 0.
saving, reached = None, 0
try:
    for i, g in enumerate(house.GOALS):
        ok = False
        for attempt in range(4):
            path, k = house.plan(dist, now['truth'][1:3], g), 0
            ref, began = now['truth'], time.time()
            while time.time() - began < 180:
                t, x, y, a = now['truth']
                if math.hypot(g[0]-x, g[1]-y) < .12:
                    ok = True
                    break
                window = path[k:k+30]
                k += int(np.argmin(np.hypot(window[:, 0]-x, window[:, 1]-y)))
                p = path[min(k+10, len(path)-1)]
                e = wrap(math.atan2(p[1]-y, p[0]-x) - a)
                tv = 0. if abs(e) > .7 else .22 * max(.35, 1-abs(e))
                tw = max(-.6, min(.6, 1.6*e))
                v += max(-.025, min(.025, tv-v))
                w += max(-.08, min(.08, tw-w))
                send(v, w)
                sample()
                time.sleep(.033)
                if t - ref[0] > 10:
                    if math.hypot(x-ref[1], y-ref[2]) < .05 and abs(wrap(a-ref[3])) < .1:
                        print('STUCK at', x, y, flush=True)
                        end = time.time() + 1.5
                        while time.time() < end:
                            send(-.1)
                            time.sleep(.05)
                        break
                    ref = now['truth']
            if ok:
                break
        reached += ok
        t, o = now['truth'], now['odom']
        drift = math.hypot(o[1]+origin[0]-t[1], o[2]+origin[1]-t[2])
        s = log['slam'][-1] if log['slam'] else (0, math.nan, math.nan)
        err = math.hypot(s[1]+origin[0]-t[1], s[2]+origin[1]-t[2])
        print(f'goal {i+1}/{len(house.GOALS)} ok={ok} sim={t[0]:.0f}s odom_err={drift:.3f} slam_err={err:.3f}', flush=True)
        if (i + 1) % 6 == 0:
            np.savez(os.path.join(out, 'trajectory.npz'), origin=origin, **log)
            if saving is None or saving.poll() is not None:
                saving = subprocess.Popen(
                    ['timeout', '30', 'ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f',
                     os.path.join(out, f'checkpoint_{i+1:02d}'), '--ros-args', '-p', 'save_map_timeout:=15.0'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
finally:
    end = time.time() + 4
    while time.time() < end:
        send()
        sample()
        time.sleep(.05)
    np.savez(os.path.join(out, 'trajectory.npz'), origin=origin, **log)
    print(f'RESULT reached {reached}/{len(house.GOALS)}', flush=True)
os._exit(0)
