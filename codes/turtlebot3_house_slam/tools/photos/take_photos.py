#!/usr/bin/python3
"""Photograph the recorded trials in RViz: take_photos.py <window id> <output folder>

Replays saved results only (nav2/results/**.npz); the robot is not driven.
"""
import os
import subprocess
import sys
import time

import numpy as np
import rclpy
from PIL import Image, ImageDraw, ImageFont
from geometry_msgs.msg import Point
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
window, out = sys.argv[1:]
MARK_FAILURES = os.environ.get('MARK_FAILURES') == '1'  # red X where a trial stopped short
GOALS = ('A_south_west_room', 'B_middle_room', 'C_south_east_room')
PLANNERS = (('navfn', 'NavFn A*', 'results'), ('diagonal', 'Diagonal A*', 'results/diagonal'))
BLUE, ORANGE, GREEN = (.16, .47, .84), (.92, .41, .20), (.11, .69, .48)
RED, BLACK, WHITE = (.85, .1, .1), (0., 0., 0.), (1., 1., 1.)

rclpy.init()
node = rclpy.create_node('photos')
pub = node.create_publisher(MarkerArray, '/photo',
                            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))


def finish(path, title, legend):
    """Crop to the map and add the caption below (RViz's own text spacing is broken)."""
    im = Image.open(path).convert('RGB').crop((146, 82, 1354, 946))
    out_im = Image.new('RGB', (im.width, im.height + 62 + 34 * len(legend)), 'white')
    out_im.paste(im, (0, 0))
    draw = ImageDraw.Draw(out_im)
    font = '/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf'
    draw.text((16, im.height + 12), title, fill='black', font=ImageFont.truetype(font % '-Bold', 28))
    for i, (rgb, text) in enumerate(legend):
        y = im.height + 56 + 34 * i
        if rgb:
            draw.rectangle((18, y + 11, 62, y + 19), fill=tuple(int(255 * c) for c in rgb))
        draw.text((76 if rgb else 18, y), text, fill='black', font=ImageFont.truetype(font % '', 23))
    out_im.save(path)


class Scene:
    def __init__(self):
        clear = Marker()
        clear.header.frame_id = 'map'
        clear.action = Marker.DELETEALL
        self.items = [clear]

    def add(self, kind, rgb, z):
        m = Marker()
        m.header.frame_id = 'map'
        m.id, m.type = len(self.items), kind
        m.pose.orientation.w = 1.
        m.pose.position.z = float(z)
        m.color = ColorRGBA(r=float(rgb[0]), g=float(rgb[1]), b=float(rgb[2]), a=1.)
        self.items.append(m)
        return m

    def line(self, xy, rgb, width, z):
        if len(xy) > 1:
            m = self.add(Marker.LINE_STRIP, rgb, z)
            m.scale.x = width
            m.points = [Point(x=float(x), y=float(y)) for x, y in xy]

    def dots(self, xy, rgb, size, z):
        m = self.add(Marker.SPHERE_LIST, rgb, z)
        m.scale.x = m.scale.y = m.scale.z = size
        m.points = [Point(x=float(x), y=float(y)) for x, y in xy]

    def text(self, x, y, text, rgb=WHITE, size=.32):
        m = self.add(Marker.TEXT_VIEW_FACING, rgb, 1.)
        m.pose.position.x, m.pose.position.y = float(x), float(y)
        m.scale.z = size
        m.text = text

    def cross(self, x, y):
        for s in (-1, 1):
            self.line([(x-.13, y-.13*s), (x+.13, y+.13*s)], RED, .06, .5)

    def save(self, name, title, legend):
        pub.publish(MarkerArray(markers=self.items))
        end = time.time() + 3
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=.1)
        subprocess.run(['timeout', '15', 'import', '-window', window, os.path.join(out, name + '.png')], check=True)
        finish(os.path.join(out, name + '.png'), title, legend)
        print('photo', name, flush=True)


def trials(folder, setting, goal):
    for run in (1, 2, 3):
        yield np.load(os.path.join(ROOT, 'nav2', folder, f'{setting}_run{run}', goal + '.npz'), allow_pickle=True)


def waypoints(goal):
    return np.loadtxt(os.path.join(ROOT, 'nav2', 'diagonal_astar', goal + '.csv'), delimiter=',', skiprows=1)


def mark_ends(s, goals):
    s.dots([(0, 0)], BLUE, .3, .6)
    s.text(0, .38, 'start', BLACK, .3)
    for g in goals:
        e = waypoints(g)[-1]
        s.dots([e], BLACK, .28, .6)
        s.text(e[0] + .35, e[1] + .25, g[0], BLACK, .42)


def label_of(selection):
    return 'all goals' if len(selection) > 1 else 'goal ' + selection[0][0]


def name_of(n, key, selection):
    return f"{n:02d}_{key}_{label_of(selection).replace(' ', '_')}"


time.sleep(3)
s = Scene()
s.save('01_slam_map', 'SLAM Toolbox map of the house (front door closed, 0.05 m cells)',
       [(None, 'white: free     black: obstacle     grey-green: unknown')])

s = Scene()
for g, c in zip(GOALS, (BLUE, ORANGE, GREEN)):
    w = waypoints(g)
    s.line(w, c, .035, .2)
    s.dots(w, c, .1, .3)
mark_ends(s, GOALS)
s.save('02_diagonal_astar_waypoints', 'Diagonal A* paths and waypoints (0.2 m grid, unknown space blocked)',
       [(c, 'to goal ' + g[0]) for g, c in zip(GOALS, (BLUE, ORANGE, GREEN))])

n = 3
for key, title, folder in PLANNERS:
    for selection in ([GOALS[0]], [GOALS[1]], [GOALS[2]], list(GOALS)):
        s = Scene()
        counts = {}
        for setting, colour, z in (('paper', ORANGE, .3), ('optimized', GREEN, .4)):
            ok = total = 0
            for goal in selection:
                for i, t in enumerate(trials(folder, setting, goal)):
                    if setting == 'paper' and i == 0:
                        s.line(t['plan'], BLUE, .035, .2)
                    xy, good = t['pose'][:, 1:], str(t['result']) == 'reached'
                    s.line(xy, colour, .045, z)
                    ok, total = ok + good, total + 1
                    if MARK_FAILURES and not good:
                        s.cross(*xy[-1])
            counts[setting] = f'{ok}/{total}'
        mark_ends(s, selection)
        s.save(name_of(n, key, selection),
               f'{title} + DWA, {label_of(selection)}, 3 runs per setting',
               [(BLUE, 'planned path'),
                (ORANGE, f"DWA, paper Table 3: reached {counts['paper']}"),
                (GREEN, f"DWA, alternative weights: reached {counts['optimized']}"),
                ] + ([(RED, 'X marks where a trial failed')] if MARK_FAILURES else []))
        n += 1
