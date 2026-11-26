#!/usr/bin/python3
"""Frontier exploration. Inputs: /map, /scan and the SLAM pose from TF. Nothing else."""
import heapq
import math
import os
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from scipy import ndimage
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

CLEARANCE = 0.21   # hard limit from anything mapped
LEG_MARGIN = 0.42  # thin isolated objects are often legs or columns on a wider base
TIME_LIMIT = 1800

rclpy.init()
node = rclpy.create_node('explorer', parameter_overrides=[Parameter('use_sim_time', value=True)])
pub = node.create_publisher(Twist, '/house_sim/cmd_vel', 5)
latest = {}
node.create_subscription(OccupancyGrid, '/map', lambda m: latest.update(map=m),
                         QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
node.create_subscription(LaserScan, '/scan', lambda m: latest.update(scan=m), qos_profile_sensor_data)
tf = Buffer()
TransformListener(tf, node)
threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()


def pose():
    t = tf.lookup_transform('map', 'base_footprint', rclpy.time.Time()).transform
    q = t.rotation
    return (t.translation.x, t.translation.y,
            math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2 * (q.y*q.y + q.z*q.z)))


def send(v=0., w=0.):
    m = Twist()
    m.linear.x, m.angular.z = float(v), float(w)
    pub.publish(m)


def drive_for(seconds, v=0., w=0.):
    end = time.time() + seconds
    while time.time() < end:
        send(v, w)
        time.sleep(.05)


class Grid:
    def __init__(self, blocked):
        m = latest['map']
        i = m.info
        self.r, self.h, self.w = i.resolution, i.height, i.width
        self.ox, self.oy = i.origin.position.x, i.origin.position.y
        d = np.array(m.data, np.int16).reshape(self.h, self.w)
        self.occ, self.free, self.unknown = d >= 50, (d >= 0) & (d < 50), d < 0
        self.dist = ndimage.distance_transform_edt(~self.occ) * self.r
        ok = self.free & (self.dist >= CLEARANCE)

        labels, _ = ndimage.label(self.occ, np.ones((3, 3)))
        for n, box in enumerate(ndimage.find_objects(labels), 1):
            if max(s.stop - s.start for s in box) * self.r <= .22:
                ys, xs = np.where(labels[box] == n)
                ok &= ~self.disc(box[1].start + xs.mean(), box[0].start + ys.mean(), LEG_MARGIN)
        for x, y in blocked:  # places where the robot found itself stuck
            ok &= ~self.disc((x - self.ox) / self.r, (y - self.oy) / self.r, .25)
        self.ok = ok

    def disc(self, cx, cy, radius):
        yy, xx = np.ogrid[:self.h, :self.w]
        return (xx - cx)**2 + (yy - cy)**2 <= (radius / self.r)**2

    def cell(self, x, y):
        return (min(self.h - 1, max(0, int((y - self.oy) / self.r))),
                min(self.w - 1, max(0, int((x - self.ox) / self.r))))

    def world(self, c):
        return self.ox + (c[1] + .5) * self.r, self.oy + (c[0] + .5) * self.r

    def search(self, start):
        s = self.cell(*start)
        cost, prev, queue = np.full(self.ok.shape, np.inf), {}, [(0., s)]
        cost[s] = 0
        weight = 1 + 3 * np.clip((.45 - self.dist) / .25, 0, 1)
        while queue:
            c, u = heapq.heappop(queue)
            if c > cost[u]:
                continue
            for i, j in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                v = u[0] + i, u[1] + j
                if not (0 <= v[0] < self.h and 0 <= v[1] < self.w):
                    continue
                if not self.ok[v]:
                    # only allowed while escaping from the start position
                    if self.occ[v] or math.hypot(v[0]-s[0], v[1]-s[1]) * self.r > .3:
                        continue
                n = c + math.hypot(i, j) * self.r * weight[v]
                if n < cost[v]:
                    cost[v], prev[v] = n, u
                    heapq.heappush(queue, (n, v))
        return cost, prev

    def visible(self, a, b):
        n = max(abs(a[0]-b[0]), abs(a[1]-b[1])) + 1
        ys = np.linspace(a[0], b[0], n).round().astype(int)
        xs = np.linspace(a[1], b[1], n).round().astype(int)
        return not self.occ[ys, xs].any()

    def next_goal(self, start, visited):
        cost, _ = self.search(start)
        edge = self.free & ndimage.binary_dilation(self.unknown)
        labels, _ = ndimage.label(edge, np.ones((3, 3)))
        reachable = np.isfinite(cost) & self.ok
        best = None
        for n, box in enumerate(ndimage.find_objects(labels), 1):
            ys, xs = np.where(labels[box] == n)
            if len(ys) < 7:
                continue
            ys, xs = ys + box[0].start, xs + box[1].start
            k = np.argmin(np.hypot(ys - ys.mean(), xs - xs.mean()))
            f = (int(ys[k]), int(xs[k]))
            p = self.world(f)
            if sum(math.hypot(p[0]-a, p[1]-b) < .35 for a, b in visited) >= 2:
                continue
            near = np.argwhere(reachable & self.disc(f[1], f[0], .8))
            if not len(near):
                continue
            d = np.hypot(near[:, 0]-f[0], near[:, 1]-f[1]) * self.r
            for q in np.argsort(d)[:60]:
                c = tuple(int(v) for v in near[q])
                if self.visible(c, f):
                    score = cost[c] - .03 * min(len(ys), 60)
                    if best is None or score < best[0]:
                        best = score, self.world(c), p, len(ys)
                    break
        return best

    def path(self, start, goal):
        cost, prev = self.search(start)
        c = self.cell(*goal)
        if not np.isfinite(cost[c]):
            return None
        out, s = [c], self.cell(*start)
        while out[-1] != s:
            out.append(prev[out[-1]])
        return np.array([self.world(q) for q in out[::-1]])


def ahead(scan):
    r = np.array(scan.ranges)
    a = scan.angle_min + np.arange(len(r)) * scan.angle_increment
    front = r[np.cos(a) > math.cos(math.radians(28))]
    front = front[np.isfinite(front)]
    return front.min() if len(front) else math.inf


def go_to(goal, blocked):
    v = w = 0.
    began, planned, path, k = time.time(), 0., None, 0
    ref = None
    while time.time() - began < 120:
        x, y, a = pose()
        if math.hypot(goal[0]-x, goal[1]-y) < .15:
            return 'arrived'
        if time.time() - planned > 3:
            path, planned, k = Grid(blocked).path((x, y), goal), time.time(), 0
            if path is None:
                send()
                return 'unreachable'
        window = path[k:k+20]
        k += int(np.argmin(np.hypot(window[:, 0]-x, window[:, 1]-y)))
        p = path[min(k+5, len(path)-1)]
        e = (math.atan2(p[1]-y, p[0]-x) - a + math.pi) % (2*math.pi) - math.pi
        tv = 0. if abs(e) > .7 else .22 * max(.35, 1-abs(e))
        scan = latest['scan']
        if tv and ahead(scan) < .19:
            drive_for(1.2, v=-.1)
            send()
            return 'obstacle ahead'
        v += max(-.025, min(.025, tv-v))
        w += max(-.09, min(.09, max(-.6, min(.6, 1.6*e))-w))
        send(v, w)

        # wheels turning but the lidar view frozen means pushing on something it cannot see
        if v < .08 or abs(w) > .15:
            ref = None
        elif ref is None:
            ref = time.time(), np.array(scan.ranges)
        elif time.time() - ref[0] > 2.5:
            r = np.array(scan.ranges)
            both = np.isfinite(r) & np.isfinite(ref[1])
            if both.sum() > 30 and np.percentile(abs(r[both]-ref[1][both]), 98) < .05:
                blocked.append((x + .25*math.cos(a), y + .25*math.sin(a)))
                drive_for(1.5, v=-.1)
                send()
                return 'blocked by unseen object'
            ref = time.time(), r
        time.sleep(.05)
    return 'timeout'


def main():
    t0 = time.time()
    while True:
        try:
            if 'map' in latest and 'scan' in latest:
                home = pose()
                break
        except Exception:
            pass
        if time.time() - t0 > 60:
            return print('no map, scan or pose', flush=True)
        time.sleep(.5)
    drive_for(11, w=.6)  # look around
    drive_for(1)
    blocked, visited, empty, n = [], [], 0, 0
    while time.time() - t0 < TIME_LIMIT:
        best = Grid(blocked).next_goal(pose()[:2], visited)
        if best is None:
            empty += 1
            if empty >= 3:
                print('no frontiers left', flush=True)
                break
            drive_for(3)  # let the map update
            continue
        empty, n = 0, n + 1
        _, goal, frontier, size = best
        visited.append(frontier)
        result = go_to(goal, blocked)
        if result == 'arrived':
            drive_for(2.5, w=.5)
        drive_for(1.5)
        known = (np.array(latest['map'].data) >= 0).sum() * latest['map'].info.resolution**2
        print(f'{n:3d} frontier=({frontier[0]:.1f},{frontier[1]:.1f}) cells={size} {result} '
              f'mapped={known:.1f} m2 t={time.time()-t0:.0f}s', flush=True)
    else:
        print('time limit', flush=True)
    for _ in range(4):
        if go_to(home[:2], blocked) == 'arrived':
            print('back at start', flush=True)
            break
    print('DONE', flush=True)


if __name__ == '__main__':
    try:
        main()
    finally:
        drive_for(1.5)
    os._exit(0)
