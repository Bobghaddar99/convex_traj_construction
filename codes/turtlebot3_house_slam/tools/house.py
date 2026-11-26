"""House geometry from the SDF: used to plan the mapping route and to grade maps."""
import heapq
import os
import xml.etree.ElementTree as ET

import numpy as np
from scipy import ndimage

SDF = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                   'models', 'turtlebot3_house', 'model.sdf')
X0, Y0, RES, NX, NY = -9.0, -7.0, 0.025, 720, 560
X, Y = np.meshgrid(X0 + (np.arange(NX) + .5) * RES, Y0 + (np.arange(NY) + .5) * RES)

# front door, closed (same box as in worlds/turtlebot3_house_closed.world)
DOOR = (0.6622, 1.5622, -0.2499, -0.0999)

# indoors only, starting and finishing at the spawn point
GOALS = [(-3.8,.5),(-4.3,1.8),(-6.2,3.2),(-6.5,4.2),(-5.8,4.4),(-6.2,1.7),(-6.4,0),(-6.5,-2.2),
         (-6.1,-3.2),(-6.9,-1.5),(-6.3,.2),(-6.3,2.2),(-4.2,3.4),(-3.3,4.3),(-1,4.3),(-.8,2.3),
         (-2.5,1.3),(-.8,.35),(1.1,.35),(3.4,.4),(6.2,.5),(6.1,-1.2),(5.5,-2.8),(5.6,-4.5),
         (6.8,-4.5),(7,-2.7),(6.2,-.9),(6.4,1.2),(6.7,3.2),(6.2,4.3),(4.3,4.2),(3,4.5),(1.2,4.6),
         (.7,3),(.8,1.6),(1.1,3.4),(1.4,4.6),(3.1,4.2),(3.3,1.7),(3,.4),(1.1,.35),(-1,.4),
         (-3,.5),(-2,1.0)]


def pose(e):
    p = e.find('pose')
    x, y, z, r, p, q = map(float, p.text.split()) if p is not None else [0] * 6
    c, s = np.cos, np.sin
    Rz = np.array([[c(q), -s(q), 0], [s(q), c(q), 0], [0, 0, 1]])
    Ry = np.array([[c(p), 0, s(p)], [0, 1, 0], [-s(p), 0, c(p)]])
    Rx = np.array([[1, 0, 0], [0, c(r), -s(r)], [0, s(r), c(r)]])
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = Rz @ Ry @ Rx, (x, y, z)
    return M


def geometry(kind='visual', heights=(0.181,)):
    occ, meshes = np.zeros((NY, NX), bool), []

    def walk(element, M):
        for e in element:
            if e.tag in ('model', 'link'):
                walk(e, M @ pose(e))
            elif e.tag == kind:
                V, g = M @ pose(e), e.find('geometry')[0]
                if g.tag == 'mesh':
                    meshes.append(V[:2, 3].copy())
                    continue
                I = np.linalg.inv(V)
                for z in heights:
                    a, b, c = [I[i, 0] * X + I[i, 1] * Y + I[i, 2] * z + I[i, 3] for i in range(3)]
                    if g.tag == 'box':
                        hx, hy, hz = np.array(g.find('size').text.split(), float) / 2
                        hit = (abs(a) <= hx + RES/2) & (abs(b) <= hy + RES/2) & (abs(c) <= hz + 1e-4)
                    else:
                        r, h = float(g.find('radius').text), float(g.find('length').text) / 2
                        hit = (np.hypot(a, b) <= r + RES/2) & (abs(c) <= h)
                    occ[hit] = True

    walk(ET.parse(SDF).getroot(), np.eye(4))
    x0, x1, y0, y1 = DOOR
    occ |= (X >= x0 - RES/2) & (X <= x1 + RES/2) & (Y >= y0) & (Y <= y1)
    return occ, np.array(meshes)


def obstacles():
    occ, _ = geometry('collision', np.linspace(.01, .21, 9))
    # mesh models, as keep-out shapes
    for x, y, r in ((.8835,-.5762,.36),(1.8835,1.9119,.30),(-4.6935,4.8944,.30),(4.8835,2.9258,.95)):
        occ |= np.hypot(X - x, Y - y) <= r
    occ |= (abs(X + 2.657) < .78) & (abs(Y - 2.424) < .48)
    return occ


def plan(dist, start, goal):
    cell = lambda p: (int((p[1] - Y0) / RES), int((p[0] - X0) / RES))
    s, g = cell(start), cell(goal)
    cost = 1 + 3 * np.clip((.45 - dist) / .27, 0, 1)
    best, prev, queue = {s: 0.}, {}, [(0., 0., s)]
    while queue:
        _, c, u = heapq.heappop(queue)
        if u == g:
            break
        if c > best[u]:
            continue
        for i, j in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
            v = (u[0] + i, u[1] + j)
            if not (0 <= v[0] < NY and 0 <= v[1] < NX):
                continue
            if dist[v] < .18 and dist[v] <= dist[u]:
                continue
            n = c + (i*i + j*j) ** .5 * cost[v]
            if n < best.get(v, 1e18):
                best[v], prev[v] = n, u
                heapq.heappush(queue, (n + ((v[0]-g[0])**2 + (v[1]-g[1])**2) ** .5, n, v))
    else:
        raise ValueError(f'no path from {start} to {goal}')
    path = [g]
    while path[-1] != s:
        path.append(prev[path[-1]])
    return np.array([(X0 + (j+.5)*RES, Y0 + (i+.5)*RES) for i, j in path[::-1]])
