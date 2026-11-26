"""
Run the four searches on the paper's OWN Grid Map 1 and Grid Map 2 (Figs. 7 & 8 of
Luo et al., Machines 2024, 12, 668), and compare against Tables 1 & 2.

paper_maps.json holds the obstacle cells digitised from the paper's figures (green markers
detected by colour in the PDF render; Map 1: 32 obstacles, Map 2: 40; panels (a) and (b)
agree exactly, max lattice-rounding error 0.007 cell). Cells are (x, y), 1 m resolution;
obstacle centre at (x + 0.5, y + 0.5) m. Start = cell (0, 0), goal = cell (8, 8).

The paper's path lengths are reproduced only with unrestricted 8-connectivity
(DiagonalMovement.always — the paper's Dijkstra path in Fig. 7a passes diagonally between
two obstacles at (5,4)/(4,5)), so that is used here.

Searches come from the `pathfinding` library (see replicate_5_1.py).
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from pathfinding.core.diagonal_movement import DiagonalMovement

from replicate_5_1 import OUT_DIR, PLOT_ORDER, STYLE, algorithms, path_length, run

HERE = os.path.dirname(os.path.abspath(__file__))
START, GOAL = (0, 0), (8, 8)
P = 0.001
REPEATS = 200

PAPER = {  # Tables 1 & 2: (path length m, search time s, nodes)
    "1": {"Dijkstra (Reference)": (12.4853, 0.036507, 64), "Euclidean Distance": (12.4853, 0.021923, 42),
          "Manhattan Distance": (12.4853, 0.018392, 31), "Diagonal Distance": (12.4853, 0.020577, 34)},
    "2": {"Dijkstra (Reference)": (13.8995, 0.05446, 51), "Euclidean Distance": (13.8995, 0.031168, 32),
          "Manhattan Distance": (14.4853, 0.028745, 27), "Diagonal Distance": (13.8995, 0.029058, 29)},
}


def load_maps():
    with open(os.path.join(HERE, "paper_maps.json")) as f:
        cells = json.load(f)
    maps = {}
    for k, cs in cells.items():
        occ = np.zeros((10, 10), int)
        for x, y in cs:
            occ[y, x] = 1
        maps[k] = occ
    return maps


def plot(occ, results, map_id, fname):
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    oy, ox = np.nonzero(occ)
    for ax, name, tag in zip(axes.flat, PLOT_ORDER, "abcd"):
        st = STYLE[name]
        ax.scatter(ox + 0.5, oy + 0.5, s=160, c="mediumseagreen", label="Obstacles")
        ax.scatter(START[0] + 0.5, START[1] + 0.5, s=200, marker="*", facecolors="none",
                   edgecolors="tab:blue", label="starting point", zorder=3)
        ax.scatter(GOAL[0] + 0.5, GOAL[1] + 0.5, s=200, marker="*", facecolors="none",
                   edgecolors="tab:red", label="Target point", zorder=3)
        px, py = zip(*results[name]["path"][1:-1])
        ax.plot(np.array(px) + 0.5, np.array(py) + 0.5, "k" + st["marker"], mfc=st["mfc"], ms=8, label="Path point")
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_aspect("equal")
        ax.set_xticks(range(11)); ax.set_yticks(range(11)); ax.grid(True, alpha=0.3)
        ax.set_xlabel(f"X(m)\n({tag})"); ax.set_ylabel("Y(m)")
        r = results[name]
        ax.set_title(f"{st['title']}\n{r['length']:.4f} m, {r['nodes']} nodes", fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
    fig.suptitle(f"Paper's Grid Map {map_id} (digitised from Fig. {6 + int(map_id)})")
    fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig)


def main():
    algos = algorithms(DiagonalMovement.always, P)
    os.makedirs(OUT_DIR, exist_ok=True)
    for map_id, occ in load_maps().items():
        results = {}
        for name, make in algos.items():
            path, runs, _ = run(make(), occ, START, GOAL)
            t = float(np.median([run(make(), occ, START, GOAL)[2] for _ in range(REPEATS)]))
            results[name] = dict(path=path, length=path_length(path, 1.0), nodes=runs, time=t)
        plot(occ, results, map_id, os.path.join(OUT_DIR, f"paper_map_{map_id}.png"))

        print(f"\nPaper's Grid Map {map_id}  (paper value -> ours)")
        print(f"{'':<22}{'Path length (m)':>26}{'Search time (s)':>26}{'Expanded nodes':>18}")
        for name, r in results.items():
            pl, pt, pn = PAPER[map_id][name]
            ok = "✓" if abs(pl - r["length"]) < 1e-3 else "✗"
            print(f"{name:<22}{pl:>11.4f} -> {r['length']:.4f} {ok}{pt:>14.6f} -> {r['time']:.6f}"
                  f"{pn:>8d} -> {r['nodes']:d}")
        for name, r in results.items():
            print(f"  {name:<22} path cells: {r['path']}")
    print(f"\nFigures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
