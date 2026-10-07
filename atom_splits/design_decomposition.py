#!/usr/bin/env python
"""How random are the 462 pairs?  Decompose the pair-to-pair variance of a statistic over the
eigenspaces of the design (the Johnson scheme J(12,6) folded by complementation).

Every pair value is a symmetric function of a balanced 6|6 partition of the 12 atoms.  Such a
function decomposes into atom-interaction orders 0, 2, 4, 6 (odd orders cancel: an atom is
always on exactly one side), with eigenspace dimensions 1, 54, 275, 132.  If the pair values
were exchangeable noise given the design, the variance would spread in proportion to the
dimensions (12% / 60% / 29%).  Variance concentrated in the order-2 space means the pair value
is (close to) a quadratic form in which atoms sit together -- a smooth, low-dimensional
function of the partition, not noise.

    ~/.conda/envs/test/bin/python scripts/atom_splits/design_decomposition.py
-> results/atom_splits/design_decomposition.tsv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results/atom_splits"


def scheme_spaces(P, half=6):
    S = [set(p[:half]) for p in P]
    n = len(S)
    C = np.array([[max(len(S[i] & S[j]), half - len(S[i] & S[j])) for j in range(n)] for i in range(n)])
    M = sum(c * (C == c) for c in range(half // 2, half + 1)) + 0.37 * (C == half - 2)   # generic scheme element
    w, V = np.linalg.eigh(M)
    key = np.round(w, 6)
    spaces = {}
    for k in sorted(set(key)):
        spaces[V[:, key == k].shape[1]] = V[:, key == k]
    return spaces          # dim -> basis; dims 1 (order 0), 54 (2), 275 (4), 132 (6) for G = 12


def main():
    traits = sorted(p.name.split(".")[0] for p in RES.glob("*.pairs.npz"))
    rows = []
    spaces = None
    for t in traits:
        z = np.load(RES / f"{t}.pairs.npz", allow_pickle=True)
        Mb, P, kg = z["Mb"], z["pairs"], int(z["k_g"])
        K = Mb.shape[1] - 1
        if spaces is None:
            spaces = scheme_spaces(P)
        for name, k in (("M(K)", K), ("M(k_g)", max(kg, 1)), ("M(3)", 3), ("M(1)", 1)):
            y = Mb[:, k]
            tot = np.sum((y - y.mean()) ** 2)
            r = dict(trait=t, stat=name, k=k, sd=y.std(ddof=1))
            for dim, B in spaces.items():
                if dim > 1:
                    r[f"order{ {54: 2, 275: 4, 132: 6}[dim] }"] = float(np.sum((B.T @ y) ** 2) / tot)
            rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(RES / "design_decomposition.tsv", sep="\t", index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(D.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nnoise-like expectation: order2 0.117, order4 0.597, order6 0.286")


if __name__ == "__main__":
    main()
