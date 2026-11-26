"""
OTPWR Diagonal-distance A* (Section 3 of Luo et al., Machines 2024, 12, 668).

Search engine: AStarFinder from the open-source `pathfinding` library
(PyPI pathfinding==1.0.23, https://github.com/brean/python-pathfinding) — not re-implemented.
What this file adds is the paper's heuristic, plugged into that engine:

    Eq. (8)  h(n)  = D_straight*(dx + dy) + (D_diagonal - 2*D_straight)*min(dx, dy)
    Eq. (9)  h'(n) = (1 + p) * h(n)        p: small positive tie-breaking coefficient

Outputs (results/):
    diagonal_astar_map{1,2}.png  expanded (closed) / frontier (open) nodes + path, p = 0 vs p > 0
    diagonal_astar_p_sweep.png   effect of p over many random maps (Eq. 9 discussion)
    diagonal_astar_p_sweep.csv

Usage:
    python diagonal_astar.py                  # maps from seeds 1 and 30 (same as replicate_5_1.py)
    python diagonal_astar.py --p 0.001 --n-maps 500
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
from pathfinding.core.heuristic import octile
from pathfinding.finder.a_star import AStarFinder
from pathfinding.finder.dijkstra import DijkstraFinder

from replicate_5_1 import OUT_DIR, path_length, random_grid_map

D_STRAIGHT = 1.0
D_DIAGONAL = math.sqrt(2.0)


def diagonal_heuristic(dx, dy, d_straight=D_STRAIGHT, d_diagonal=D_DIAGONAL):
    """Eq. (8). dx, dy are |x - x_goal|, |y - y_goal| (the library passes absolute values)."""
    return d_straight * (dx + dy) + (d_diagonal - 2.0 * d_straight) * min(dx, dy)


def otpwr_heuristic(p):
    """Eq. (9): h' = (1 + p) h."""
    return lambda dx, dy: (1.0 + p) * diagonal_heuristic(dx, dy)


def diagonal_astar(occ, start, goal, p, diag):
    """Run the library A* with the OTPWR heuristic. Returns path, expanded count, time, grid state."""
    grid = Grid(matrix=(1 - occ).tolist())
    finder = AStarFinder(heuristic=otpwr_heuristic(p), diagonal_movement=diag)
    t0 = time.perf_counter()
    path, runs = finder.find_path(grid.node(*start), grid.node(*goal), grid)
    dt = time.perf_counter() - t0
    closed = [(n.x, n.y) for row in grid.nodes for n in row if n.closed]
    opened = [(n.x, n.y) for row in grid.nodes for n in row if n.opened and not n.closed]
    return dict(path=[(n.x, n.y) for n in path], nodes=runs, time=dt, closed=closed, open=opened)


def dijkstra_length(occ, start, goal, diag):
    grid = Grid(matrix=(1 - occ).tolist())
    path, _ = DijkstraFinder(diagonal_movement=diag).find_path(grid.node(*start), grid.node(*goal), grid)
    return path_length([(n.x, n.y) for n in path], 1.0)


def sanity_check():
    """Eq. (8) with D_s = 1, D_d = sqrt(2) must equal the library's octile distance."""
    for dx in range(15):
        for dy in range(15):
            assert abs(diagonal_heuristic(dx, dy) - octile(dx, dy)) < 1e-12
    print("Eq. (8) == pathfinding.core.heuristic.octile  ✓")


def plot_search(occ, runs, start, goal, map_id, ref_len, fname):
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 6))
    n = occ.shape[0]
    oy, ox = np.nonzero(occ)
    for ax, (p, r) in zip(np.atleast_1d(axes), runs.items()):
        if r["closed"]:
            cx, cy = zip(*r["closed"])
            ax.scatter(cx, cy, s=420, marker="s", c="#f2c14e", alpha=0.55, label=f"Expanded / closed ({r['nodes']})")
        if r["open"]:
            qx, qy = zip(*r["open"])
            ax.scatter(qx, qy, s=420, marker="s", facecolors="none", edgecolors="#f2a03d", label="Frontier / open")
        ax.scatter(ox, oy, s=160, c="tab:green", label="Obstacles")
        px, py = zip(*r["path"])
        ax.plot(px, py, "k-", lw=1, alpha=0.6)
        ax.plot(px, py, "kx", ms=9, mew=2, label="Path point")
        ax.scatter(*start, s=300, marker="*", c="tab:blue", label="starting point", zorder=4)
        ax.scatter(*goal, s=300, marker="*", c="tab:red", label="Target point", zorder=4)
        L = path_length(r["path"], 1.0)
        tag = "optimal" if abs(L - ref_len) < 1e-9 else f"Dijkstra {ref_len:.4f}"
        ax.set_title(f"Diagonal A*, p = {p}\nlength {L:.4f} m ({tag}), {r['nodes']} expanded nodes", fontsize=10)
        ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(-0.5, n - 0.5); ax.set_aspect("equal")
        ax.set_xticks(range(n)); ax.set_yticks(range(n)); ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=7)
    fig.suptitle(f"Grid Map {map_id}: OTPWR diagonal-distance A* search")
    fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)


def p_sweep(ps, n_maps, size, density, diag, start, goal):
    maps = [random_grid_map(size, density, start, goal, np.random.default_rng(s), diag) for s in range(1, n_maps + 1)]
    refs = [dijkstra_length(m, start, goal, diag) for m in maps]
    rows = []
    for p in ps:
        lens, nodes = [], []
        for m in maps:
            r = diagonal_astar(m, start, goal, p, diag)
            lens.append(path_length(r["path"], 1.0)); nodes.append(r["nodes"])
        lens, nodes, ref = np.array(lens), np.array(nodes), np.array(refs)
        rows.append(dict(p=p, optimal_pct=100 * np.mean(lens <= ref + 1e-9),
                         mean_excess_m=float(np.mean(lens - ref)), mean_nodes=float(nodes.mean())))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 30])
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--density", type=float, default=0.25)
    ap.add_argument("--p", type=float, default=0.001)
    ap.add_argument("--n-maps", type=int, default=500)
    ap.add_argument("--diagonal", default="only_when_no_obstacle",
                    choices=["always", "only_when_no_obstacle", "if_at_most_one_obstacle"])
    a = ap.parse_args()
    diag = getattr(DiagonalMovement, a.diagonal)
    start, goal = (0, 0), (a.size - 1, a.size - 1)
    os.makedirs(OUT_DIR, exist_ok=True)

    sanity_check()

    # --- per-map search, p = 0 (plain Eq. 8) vs p > 0 (Eq. 9)
    for map_id, seed in enumerate(a.seeds, 1):
        occ = random_grid_map(a.size, a.density, start, goal, np.random.default_rng(seed), diag)
        ref = dijkstra_length(occ, start, goal, diag)
        runs = {p: diagonal_astar(occ, start, goal, p, diag) for p in (0.0, a.p)}
        print(f"\nGrid Map {map_id} (seed {seed}), Dijkstra length {ref:.4f} m")
        for p, r in runs.items():
            t = np.median([diagonal_astar(occ, start, goal, p, diag)["time"] for _ in range(200)])
            print(f"  p = {p:<6}  length {path_length(r['path'], 1.0):.4f} m   expanded {r['nodes']:3d}   "
                  f"open left {len(r['open']):3d}   time {t:.6f} s")
            print(f"            path {r['path']}")
        plot_search(occ, runs, start, goal, map_id, ref, os.path.join(OUT_DIR, f"diagonal_astar_map{map_id}.png"))

    # --- effect of p (Eq. 9): "small positive" helps, "too large" breaks optimality
    ps = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    rows = p_sweep(ps, a.n_maps, a.size, a.density, diag, start, goal)
    print(f"\nEffect of p over {a.n_maps} random maps:")
    print(f"  {'p':>6} {'optimal %':>10} {'mean excess (m)':>16} {'mean expanded':>14}")
    for r in rows:
        print(f"  {r['p']:>6} {r['optimal_pct']:>10.1f} {r['mean_excess_m']:>16.4f} {r['mean_nodes']:>14.2f}")
    with open(os.path.join(OUT_DIR, "diagonal_astar_p_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    fig, ax1 = plt.subplots(figsize=(7, 4))
    x = range(len(ps))
    ax1.plot(x, [r["mean_nodes"] for r in rows], "o-", c="tab:orange", label="mean expanded nodes")
    ax1.set_ylabel("mean expanded nodes"); ax1.set_xticks(list(x)); ax1.set_xticklabels([str(p) for p in ps])
    ax1.set_xlabel("p in h' = (1 + p) h   (Eq. 9)")
    ax2 = ax1.twinx()
    ax2.plot(x, [r["optimal_pct"] for r in rows], "s--", c="tab:blue", label="optimal paths (%)")
    ax2.set_ylabel("optimal paths (%)"); ax2.set_ylim(0, 105)
    fig.legend(loc="lower left", bbox_to_anchor=(0.12, 0.15), fontsize=8)
    ax1.set_title(f"Diagonal A*: effect of tie-breaking coefficient p ({a.n_maps} random maps)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "diagonal_astar_p_sweep.png"), dpi=150); plt.close(fig)
    print(f"\nOutputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
