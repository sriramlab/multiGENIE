#!/usr/bin/env python
"""Replication across atom partitions: the 462-pair answer of results/atom_splits (seed 1,
atoms refine the paper's folds, probe seed per trait) against results/atom_splits<tag>
(a fresh unanchored partition and a shifted probe seed).  Two independent partitions of
the same people are the honest replicate of "many pairs from one pass"; within one
partition the pairs share the atoms and the probes.

    ~/.conda/envs/test/bin/python scripts/atom_splits/compare_seeds.py --tag .seed2
-> results/atom_splits/compare_seeds<tag>.tsv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def sets(row):
    cons = [s for s in str(row.consensus).split(";") if s and s != "nan"]
    return set(cons)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=".seed2")
    a = ap.parse_args()
    r1, r2 = ROOT / "results/atom_splits", ROOT / f"results/atom_splits{a.tag}"
    rows = []
    for p in sorted(r2.glob("*.summary.tsv")):
        t = p.name.split(".")[0]
        if not (r1 / p.name).exists():
            continue
        A = pd.read_csv(r1 / p.name, sep="\t").iloc[0]
        B = pd.read_csv(p, sep="\t").iloc[0]
        ca = pd.read_csv(r1 / f"{t}.clusters.tsv", sep="\t")
        cb = pd.read_csv(r2 / f"{t}.clusters.tsv", sep="\t")
        top3a = set(ca.sort_values("mean_rank").envs[:3])
        top3b = set(cb.sort_values("mean_rank").envs[:3])
        sa, sb = sets(A), sets(B)
        # curves: SD over pairs is the within-partition spread; |tau1 - tau2| is the between-partition difference
        rows.append(dict(trait=t, k1=int(A.k_g), k2=int(B.k_g), tau1=A.tau, tau2=B.tau, se1=A.se, se2=B.se,
                         dtau=B.tau - A.tau, dtau_over_se=(B.tau - A.tau) / np.hypot(A.se, B.se),
                         full1=A.tau_full, full2=B.tau_full, dfull=B.tau_full - A.tau_full,
                         sd_pairs1=A.sd_pairs_full, sd_pairs2=B.sd_pairs_full,
                         lead1=ca.sort_values("mean_rank").envs.iloc[0], lead2=cb.sort_values("mean_rank").envs.iloc[0],
                         lead_same=int(ca.sort_values("mean_rank").envs.iloc[0] == cb.sort_values("mean_rank").envs.iloc[0]),
                         top3_jac=len(top3a & top3b) / len(top3a | top3b),
                         set_jac=len(sa & sb) / len(sa | sb) if (sa | sb) else np.nan,
                         rank_corr=float(np.corrcoef(ca.mean_rank, cb.mean_rank)[0, 1])))
    D = pd.DataFrame(rows)
    if D.empty:
        raise SystemExit(f"no traits in both {r1} and {r2}")
    out = r1 / f"compare_seeds{a.tag}.tsv"
    D.to_csv(out, sep="\t", index=False, float_format="%.5g")
    pd.set_option("display.width", 250)
    print(D.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("wrote", out)


if __name__ == "__main__":
    main()
