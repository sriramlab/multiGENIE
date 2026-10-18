#!/usr/bin/env python3
"""Re-cut saved atom pairs: `<res>/<trait>.pairs.npz` holds every held-out curve, its jackknife and
its ordering, so the rule is re-applied with no refit.

    python atom_splits/recut_pairs.py --trait ldl --res <results dir> [--gamma 0.05] [--out <dir>]

Writes `<out>/<trait>.summary.tsv`: the cut of the averaged curve of all pairs (k_g, tau, se, z, the
consensus set and its lead) and, for comparison, the spread over single pairs."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import multi_gxe as mg
from multi_gxe.stopping import Curve

ap = argparse.ArgumentParser()
ap.add_argument("--trait", required=True)
ap.add_argument("--res", required=True, help="directory holding <trait>.pairs.npz")
ap.add_argument("--gamma", type=float, default=mg.compsplit.GAMMA)
ap.add_argument("--out", default=None, help="default: --res")
a = ap.parse_args()

z = np.load(Path(a.res) / f"{a.trait}.pairs.npz", allow_pickle=True)
C, J, paths, pair_of, labels = z["C"], z["J"], z["paths"], z["pair_of"], list(z["labels"])


def cut(idx):
    """The rule on the average of the directions in `idx`."""
    M = mg.average_curves([Curve(c=C[d], j=J[d]) for d in idx])
    k = mg.cut_depth(M.c, M.j, a.gamma)
    se = M.se_at(k) if k > 0 else 0.0
    return dict(M=M, k=k, se=se, named=bool(k > 0), tau=float(M.c[k]) if k > 0 else 0.0,
                z=(M.c[k] / se) if se > 0 else np.nan,
                set=mg.consensus([list(paths[d]) for d in idx], k, "freq") if k > 0 else [])


all_r = cut(range(len(C)))
per = [cut(np.flatnonzero(pair_of == p)) for p in np.unique(pair_of)]
K = C.shape[1] - 1
row = dict(trait=a.trait, B=len(per), K=K, gamma=a.gamma, named=int(all_r["named"]), k_g=all_r["k"],
           tau=all_r["tau"], se=all_r["se"], z=all_r["z"],
           tau_full=float(all_r["M"].c[K]), se_full=all_r["M"].se_at(K),
           sd_pairs_kg=float(np.std([r["M"].c[all_r["k"]] for r in per])),
           pair_named_frac=float(np.mean([r["named"] for r in per])),
           pair_k_median=float(np.median([r["k"] for r in per])),
           pair_tau_mean=float(np.mean([r["tau"] for r in per])),
           consensus=";".join(labels[b] for b in all_r["set"]),
           lead=labels[all_r["set"][0]] if all_r["set"] else "")
out = Path(a.out or a.res)
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame([row]).to_csv(out / f"{a.trait}.summary.tsv", sep="\t", index=False)
print(f"{a.trait} gamma {a.gamma}: k_g={row['k_g']}/{K} tau={row['tau']:.4f} (se {row['se']:.4f}, "
      f"SD over pairs {row['sd_pairs_kg']:.4f}); single pairs report {row['pair_named_frac']:.2f} "
      f"with k median {row['pair_k_median']:.0f}; lead {row['lead']}")
