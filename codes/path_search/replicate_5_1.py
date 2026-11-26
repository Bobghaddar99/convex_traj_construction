"""
Replication of Section 5.1 of the OTPWR paper
(Luo et al., Machines 2024, 12, 668 — "Optimality Comparison of Path Searching").

Random 10 x 10 m grid maps with randomly distributed obstacles; four searches:
    Dijkstra (reference), Euclidean A*, Manhattan A*, Diagonal-distance A* (OTPWR)
Metrics (as in Tables 1 & 2): path length (m), search time (s), expanded nodes.

The search algorithms are NOT implemented here. They come from the open-source
`pathfinding` library (PyPI: pathfinding==1.0.23, https://github.com/brean/python-pathfinding):
    - pathfinding.finder.dijkstra.DijkstraFinder   (A* with h = 0)
    - pathfinding.finder.a_star.AStarFinder        (A*, f = g + h, Eq. 6)
    - pathfinding.core.heuristic.{euclidean, manhattan, octile}
`octile` is exactly the paper's diagonal heuristic, Eq. (8), with D_straight = 1 and
D_diagonal = sqrt(2):  h = (dx+dy) + (sqrt2 - 2)*min(dx,dy).
The tie-breaking factor of Eq. (9), h' = (1+p) h, is applied by scaling the library heuristic.

Usage:
    pip install pathfinding==1.0.23 numpy matplotlib
    python replicate_5_1.py                       # 2 maps, seeds 1 and 30
    python replicate_5_1.py --seeds 7 42 --density 0.3 --p 0.001
"""
import argparse
import csv
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.core.heuristic import euclidean, manhattan, octile
from pathfinding.finder.a_star import AStarFinder
from pathfinding.finder.dijkstra import DijkstraFinder

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# --------------------------------------------------------------------------- map
def random_grid_map(size, density, start, goal, rng, diag):
    """Random occupancy grid (1 = obstacle). Re-draw until start->goal is reachable."""
    while True:
        occ = (rng.random((size, size)) < density).astype(int)
        occ[start[1], start[0]] = 0
        occ[goal[1], goal[0]] = 0
        path, _, _ = run(DijkstraFinder(diagonal_movement=diag), occ, start, goal)
        if path:
            return occ


def make_grid(occ):
    # pathfinding's Grid: value > 0 is walkable, <= 0 is an obstacle; indexed [y][x]
    return Grid(matrix=(1 - occ).tolist())


def run(finder, occ, start, goal):
    grid = make_grid(occ)
    s, g = grid.node(*start), grid.node(*goal)
    t0 = time.perf_counter()
    path, runs = finder.find_path(s, g, grid)
    dt = time.perf_counter() - t0
    return [(n.x, n.y) for n in path], runs, dt


def path_length(path, res):
    return sum(math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(path, path[1:])) * res


# ---------------------------------------------------------------- algorithms
def algorithms(diag, p):
    scale = lambda h: (lambda dx, dy: (1.0 + p) * h(dx, dy))  # Eq. (9)
    return {
        "Dijkstra (Reference)": lambda: DijkstraFinder(diagonal_movement=diag),
        "Euclidean Distance":   lambda: AStarFinder(heuristic=scale(euclidean), diagonal_movement=diag),
        "Manhattan Distance":   lambda: AStarFinder(heuristic=scale(manhattan), diagonal_movement=diag),
        "Diagonal Distance":    lambda: AStarFinder(heuristic=scale(octile), diagonal_movement=diag),
    }


STYLE = {  # marker styles described in the paper for Figs. 7 & 8
    "Dijkstra (Reference)": dict(marker="o", mfc="none", title="Dijkstra Algorithm Path Search"),
    "Diagonal Distance":    dict(marker="x", mfc="k",    title="Diagonal Distance A* Algorithm Path Search"),
    "Euclidean Distance":   dict(marker="*", mfc="k",    title="Euclidean Distance A* Algorithm Path Search"),
    "Manhattan Distance":   dict(marker="s", mfc="none", title="Manhattan Distance A* Algorithm Path Search"),
}
PLOT_ORDER = ["Dijkstra (Reference)", "Diagonal Distance", "Euclidean Distance", "Manhattan Distance"]


def plot_map(occ, results, start, goal, res, map_id, fname):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    oy, ox = np.nonzero(occ)
    for ax, name, tag in zip(axes.flat, PLOT_ORDER, "abcd"):
        st = STYLE[name]
        ax.scatter(ox * res, oy * res, s=140, c="tab:green", label="Obstacles")
        ax.scatter(start[0] * res, start[1] * res, s=250, marker="*", c="tab:blue", label="starting point", zorder=3)
        ax.scatter(goal[0] * res, goal[1] * res, s=250, marker="*", c="tab:red", label="Target point", zorder=3)
        px, py = zip(*results[name]["path"])
        ax.plot(np.array(px) * res, np.array(py) * res, "k" + st["marker"], mfc=st["mfc"], ms=8, label="Path point")
        n = occ.shape[0]
        ax.set_xlim(-0.5 * res, (n - 0.5) * res); ax.set_ylim(-0.5 * res, (n - 0.5) * res)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
        ax.set_title(st["title"], fontsize=10); ax.set_xlabel(f"({tag})")
        ax.legend(loc="upper left", fontsize=7)
    fig.suptitle(f"Grid Map {map_id}: comparison of path search results")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 30], help="one random map per seed")
    ap.add_argument("--size", type=int, default=10, help="cells per side (10 x 10 m at 1 m resolution)")
    ap.add_argument("--res", type=float, default=1.0, help="grid resolution (m/cell)")
    ap.add_argument("--density", type=float, default=0.25, help="obstacle probability per cell")
    ap.add_argument("--p", type=float, default=0.001, help="tie-breaking coefficient p of Eq. (9)")
    ap.add_argument("--repeats", type=int, default=200, help="timing repetitions (median reported)")
    ap.add_argument("--diagonal", default="only_when_no_obstacle",
                    choices=["always", "only_when_no_obstacle", "if_at_most_one_obstacle"],
                    help="8-connected expansion rule (Fig. 2b); default forbids corner cutting")
    a = ap.parse_args()

    diag = getattr(DiagonalMovement, a.diagonal)
    start, goal = (0, 0), (a.size - 1, a.size - 1)  # bottom-left -> top-right, as in the paper
    algos = algorithms(diag, a.p)
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    for map_id, seed in enumerate(a.seeds, 1):
        occ = random_grid_map(a.size, a.density, start, goal, np.random.default_rng(seed), diag)
        np.savetxt(os.path.join(OUT_DIR, f"grid_map_{map_id}.txt"), np.flipud(occ), fmt="%d")
        results = {}
        for name, make in algos.items():
            path, runs, _ = run(make(), occ, start, goal)
            times = [run(make(), occ, start, goal)[2] for _ in range(a.repeats)]
            results[name] = dict(path=path, length=path_length(path, a.res),
                                 time=float(np.median(times)), nodes=runs)
        plot_map(occ, results, start, goal, a.res, map_id, os.path.join(OUT_DIR, f"grid_map_{map_id}.png"))

        print(f"\nTable {map_id}. Grid Map {map_id} (seed={seed}, density={a.density}, "
              f"{a.size}x{a.size} cells @ {a.res} m)")
        print(f"{'Category':<16}" + "".join(f"{n:>24}" for n in algos))
        print(f"{'Path length (m)':<16}" + "".join(f"{r['length']:>24.4f}" for r in results.values()))
        print(f"{'Search time (s)':<16}" + "".join(f"{r['time']:>24.6f}" for r in results.values()))
        print(f"{'nodes (count)':<16}" + "".join(f"{r['nodes']:>24d}" for r in results.values()))
        for name, r in results.items():
            rows.append([map_id, seed, name, f"{r['length']:.4f}", f"{r['time']:.6f}", r["nodes"]])

    with open(os.path.join(OUT_DIR, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["map", "seed", "algorithm", "path_length_m", "search_time_s", "expanded_nodes"])
        w.writerows(rows)
    print(f"\nFigures, maps and results.csv written to {OUT_DIR}")


if __name__ == "__main__":
    main()
