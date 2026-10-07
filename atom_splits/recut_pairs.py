#!/usr/bin/env python3
"""Re-apply the paper's rule at another plateau threshold to saved atom pairs (<res>/<trait>.pairs.npz holds every
held-out curve, jackknife and ordering, so no refit is needed).
    python recut_pairs.py --trait ldl --res results/atom_splits_raw84 --z-p 1.0 [--out results/atom_splits_raw84_zp10]
Writes <out>/<trait>.summary.tsv (tau, se, z, k_g, named, sd over single pairs, lead, consensus) like run_pairs.py."""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/atom_splits")); sys.path.insert(0, "/u/home/z/zhengton/project-sriram/multi_gxe_repo")
import multi_gxe as mg
from multi_gxe.stopping import Curve, plateau
import os
sys.path.insert(0, "/u/home/z/zhengton/project-sriram/multi_gxe_repo/examples")
if os.environ.get("RULE", "plateau") == "twostep":
    from twostep_rule import step1_depth
ap = argparse.ArgumentParser(); ap.add_argument("--trait", required=True); ap.add_argument("--res", required=True)
ap.add_argument("--z-p", type=float, required=True); ap.add_argument("--z-name", type=float, default=1.645); ap.add_argument("--out", default=None)
a = ap.parse_args()
z = np.load(Path(a.res) / f"{a.trait}.pairs.npz", allow_pickle=True)
C, J, paths, pair_of, labels = z["C"], z["J"], z["paths"], z["pair_of"], list(z["labels"])
def rule(idx):
    M = mg.average_curves([Curve(c=C[d], j=J[d]) for d in idx])
    if os.environ.get("RULE", "plateau") == "twostep":
        k = step1_depth(M.c, M.j); se = M.se_at(k) if k > 0 else 0.0; named = bool(k > 0)
    else:
        k = int(plateau(a.z_p)(M)); se = M.se_at(k) if k > 0 else 0.0
        named = bool(k > 0 and se > 0 and M.c[k] / se >= a.z_name)
    return dict(M=M, k=k, se=se, named=named, tau=float(M.c[k]) if named else 0.0, z=(M.c[k] / se) if se > 0 else np.nan,
                set=mg.consensus([list(paths[d]) for d in idx], k, "freq") if named else [])
all_r = rule(list(range(len(C))))
per = [rule(list(np.where(pair_of == p)[0])) for p in np.unique(pair_of)]
taus = np.array([r["tau"] for r in per]); ks = np.array([r["k"] for r in per]); nm = np.array([r["named"] for r in per])
row = dict(trait=a.trait, B=len(per), K=C.shape[1] - 1, z_p=a.z_p, named=int(all_r["named"]), k_g=all_r["k"], tau=all_r["tau"], se=all_r["se"], z=all_r["z"],
           tau_full=float(all_r["M"].c[-1]), se_full=all_r["M"].se_at(C.shape[1] - 1), sd_pairs_kg=float(taus[nm].std()) if nm.sum() > 1 else np.nan,
           pair_named_frac=float(nm.mean()), pair_k_median=float(np.median(ks)), pair_tau_mean=float(taus[nm].mean()) if nm.any() else 0.0,
           consensus=";".join(labels[i] for i in all_r["set"]), lead=labels[all_r["set"][0]] if all_r["set"] else "")
out = Path(a.out or a.res); out.mkdir(parents=True, exist_ok=True)
pd.DataFrame([row]).to_csv(out / f"{a.trait}.summary.tsv", sep="\t", index=False, float_format="%.6g")
print(f"{a.trait} z_p {a.z_p}: named={row['named']} k_g={row['k_g']}/{row['K']} tau={row['tau']:.4f} (se {row['se']:.4f}, SD over pairs {row['sd_pairs_kg']:.4f}) "
      f"pairs named {row['pair_named_frac']:.2f}, pair k median {row['pair_k_median']:.0f}; lead {row['lead']}")
