"""
Validate dwa_matlab_port.py against the original MATLAB code run in GNU Octave.

    octave --no-gui --quiet --eval "cd matlab; dwa_fig9('../results/dwa_fig9_octave_reach0.25.csv', [0.1 0.2 0.1 3.0], 0.5, 0.25)"
    python dwa_matlab_port.py --reach 0.25 --out results/dwa_fig9_port_reach0.25.csv
    python compare_dwa_port.py results/dwa_fig9_octave_reach0.25.csv results/dwa_fig9_port_reach0.25.csv
"""
import sys

import numpy as np


def main(a, b):
    A = np.loadtxt(a, delimiter=",", ndmin=2)
    B = np.loadtxt(b, delimiter=",", ndmin=2)
    n = min(len(A), len(B))
    cols = ["t", "x", "y", "yaw", "v", "w", "target"]
    print(f"steps: octave {len(A)}, port {len(B)}")
    d = np.abs(A[:n] - B[:n])
    for i, c in enumerate(cols):
        print(f"  max |diff| {c:>6}: {d[:, i].max():.3e}")
    bad = np.nonzero(np.abs(A[:n, 4:6] - B[:n, 4:6]).max(axis=1) > 1e-6)[0]
    print("first diverging step:", "none" if bad.size == 0 else int(bad[0]) + 1)
    return bad.size == 0 and len(A) == len(B)


if __name__ == "__main__":
    sys.exit(0 if main(*sys.argv[1:3]) else 1)
