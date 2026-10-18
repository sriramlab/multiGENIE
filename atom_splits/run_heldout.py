#!/usr/bin/env python3
"""The reported estimate tau-hat of a trait: the held-out GxE heritability of its consensus set
(atomsplit/heldout.py), from the trait's block tensor and its saved pairs, plus n_eff over the
held-out components and the bootstrap p-value of the trait-level test (for multiplicity control
across traits).  Slices of the fitted systems only; no genotype pass.

    python atom_splits/run_heldout.py --tensor <trait>.atoms.G12.npz --pairs <res>/<trait>.pairs.npz \
        [--q <trait>.q.npz] [--drop-cells drop_cells.tsv --trait <name>] [--gamma 0.05] [--m 3] --out <dir>
-> <out>/<trait>.heldout.json  (tau, se, k_g, S_hat, comps by environment, n_eff, p0)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import multi_gxe as mg                                                    # noqa: E402
from atomsplit.tensor import AtomTensor                                  # noqa: E402
from atomsplit.pairs import read_drop_cells                              # noqa: E402
from atomsplit.heldout import heldout_tau, n_eff                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", required=True); ap.add_argument("--pairs", required=True); ap.add_argument("--q", default=None,
                    help="phenotype terms of a trait on a shared tensor (as written by the phenotype-only pass)")
    ap.add_argument("--drop-cells", default=None); ap.add_argument("--trait", default=None)
    ap.add_argument("--gamma", type=float, default=mg.compsplit.GAMMA); ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T = AtomTensor.load(a.tensor, mmap=True)
    if a.q:
        q = np.load(a.q, allow_pickle=True)
        assert [str(s) for s in q["env_names"]] == T.env_names and int(q["M"]) == T.M and np.array_equal(q["atom_sizes"], T.atom_sizes)
        T.qg_full, T.qg_blk, T.qdiag = np.asarray(q["qg_full"]), np.asarray(q["qg_blk"]), np.asarray(q["qdiag"])
    z = dict(np.load(a.pairs, allow_pickle=True)); trait = a.trait or Path(a.pairs).name.split(".")[0]
    drop = read_drop_cells(a.drop_cells).get(trait, []) if a.drop_cells else None
    c, j = np.asarray(z["C"], float).mean(0), np.asarray(z["J"], float).mean(0)
    k = int(mg.cut_depth(c, j, a.gamma))
    h = heldout_tau(T, z, k_g=k, m=a.m, drop=drop, log=print)
    labels = [str(x) for x in z["labels"]]
    res = dict(trait=trait, gamma=a.gamma, k_g=k, curve_at_cut=float(c[k]), tau=h["tau"], se=h["se"], m=h["m"], n_subsets=h["n_subsets"],
               S_hat=h["S_hat"], comps={e: float(v) for e, v in zip(labels, h["comps"]) if e in h["S_hat"]},
               n_eff=n_eff([v for e, v in zip(labels, h["comps"]) if e in h["S_hat"]]), p0=mg.tail_pvalue(c, j))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"{trait}.heldout.json").write_text(json.dumps(res, indent=1))
    print(f"{trait}: k_g={k}  tau-hat={res['tau']:.4f} +- {res['se']:.4f}  (curve at the cut {res['curve_at_cut']:.4f})  "
          f"n_eff={res['n_eff']:.2f}  p0={res['p0']:.2g}")


if __name__ == "__main__":
    main()
