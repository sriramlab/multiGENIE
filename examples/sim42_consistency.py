#!/usr/bin/env python3
"""Re-analyse the simulated replicates with the SAME environment set and clustering as the real data:
the 42 lifestyle environments (the 5 others -- statin, insomnia, nap, igf1, crp -- sliced out of the
47-environment fold exports) and the pool42 frozen clustering (K = 25), through the public v5 package
with drop_degenerate_nxe, exactly as the 42-trait real run.

Arms: null47, null47_halfn (100 matched null), sparse47_single (tea), sparse47_cluster / _member (smoking
block), dense47 (4 causal: ls05, ls11, ls17 inside the 42 and crp OUTSIDE -> 3 recoverable).
sparse47_new (crp) has its causal environment outside the 42 and is skipped.
-> results/pool87_crossfit/manuscript/sim42_units.tsv, sim42_grid.tsv (z_p grid x z_report list)
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
ROOT = Path(__file__).resolve().parents[2]
REPO = Path.home() / "project-sriram/multi_gxe_repo"; sys.path.insert(0, str(REPO))
import multi_gxe as mg                                   # noqa
from multi_gxe.stopping import plateau                   # noqa
from multi_gxe.evaluator import Evaluator                # noqa
SCR = Path("/u/scratch/z/zhengton/multi_gxe/cluster_greedy_v3")
OUT = ROOT / "results/pool87_crossfit/manuscript"; OUT.mkdir(parents=True, exist_ok=True)
DROP = ["statin", "insomnia", "nap", "igf1", "crp"]
ARMS = ["null47", "null47_halfn", "sparse47_single", "sparse47_cluster", "sparse47_member", "dense47"]
blocks = [list(map(int, c)) for c in json.loads((ROOT / "results/cluster_greedy_v3/sep6/clusters_pool42.json").read_text())["clusters"]]
truth47 = pd.read_csv(ROOT / "results/cluster_greedy_v3/blocks_pool47.tsv", sep="\t")
ZP = [0.5, 0.75, 1.0, 1.28, 1.645]; ZR = {"z2.33": 2.33, "p0.05": norm.isf(0.05), "bonf42": norm.isf(0.05 / 42)}

def causal_blocks42(arm):
    if arm.startswith("null"): return set(), 0
    envs = [e for row in truth47[truth47[f"causal_{arm}"] == 1].env_names for e in row.split(",")]
    idx = {f"ls{i:02d}": i for i in range(42)}
    inside = {b for b, c in enumerate(blocks) if any(idx.get(e) in c for e in envs)}
    outside = sum(1 for e in envs if e not in idx)
    return inside, outside

rows, t0 = [], time.time()
for arm in ARMS:
    tr, n_out = causal_blocks42(arm)
    for d in sorted(SCR.glob(f"{arm}/rep*"), key=lambda p: int(p.name[3:])):
        fa, fb = d / "sim.foldA.normal_eq.json", d / "sim.foldB.normal_eq.json"
        if not (fa.exists() and fb.exists()): continue
        A = mg.drop_degenerate_nxe(mg.drop_envs(mg.load_export(fa), DROP)); B = mg.drop_degenerate_nxe(mg.drop_envs(mg.load_export(fb), DROP))
        assert list(A.env_names) == [f"ls{i:02d}" for i in range(42)]
        evA, evB = Evaluator(A), Evaluator(B)
        base = mg.complementary_split([(A, B)], blocks, gate_rule=plateau(0.75), z_gate=2.33, set_rule=lambda cv: cv.K, evaluators=[(evA, evB)])
        M = base["curve"]; paths = [dd.path for dd in base["directions"]]
        for zp in ZP:
            r = mg.compsplit.select_from_curves([dd.curve for dd in base["directions"]], paths, gate_rule=plateau(zp), z_gate=0.0, set_rule=lambda cv: cv.K)
            kg = r["k"]; se = M.se_at(kg) if kg > 0 else 0.0; z = M.c[kg] / se if se > 0 else -np.inf
            S = set(r["blocks"]) if kg > 0 else set()
            rows.append(dict(arm=arm, rep=int(d.name[3:]), z_p=zp, k_g=kg, tau=float(M.c[kg]), z=z, tau_full=float(M.c[-1]),
                             z_full=M.c[-1] / M.se_at(M.K), n_causal_in=len(tr), n_causal_out=n_out,
                             recall=len(S & tr) / len(tr) if tr else np.nan, fdp=len(S - tr) / len(S) if S else 0.0,
                             exact=float(S == tr) if tr else np.nan, size=len(S)))
    print(f"  {arm} done [{time.time()-t0:.0f}s]", flush=True)
D = pd.DataFrame(rows); D.to_csv(OUT / "sim42_units.tsv", sep="\t", index=False, float_format="%.5g")
D["kind"] = np.select([D.arm.str.startswith("null"), D.arm == "dense47"], ["null", "dense"], "sparse")
out = []
for zp in ZP:
    for name, zr in ZR.items():
        q = D[D.z_p == zp].copy(); q["rep_"] = (q.k_g > 0) & (q.z >= zr)
        r = dict(z_p=zp, z_report=name)
        for kind, g in q.groupby("kind", sort=False):
            r[f"{kind}_n"] = len(g); r[f"{kind}_reported"] = g.rep_.mean()
            if kind != "null":
                gg = g[g.rep_]; r[f"{kind}_recall"] = g.recall.where(g.rep_, 0).mean(); r[f"{kind}_fdp"] = gg.fdp.mean() if len(gg) else np.nan
                r[f"{kind}_exact"] = g.exact.where(g.rep_, 0).mean(); r[f"{kind}_size"] = gg["size"].mean() if len(gg) else 0
        out.append(r)
G = pd.DataFrame(out); G.to_csv(OUT / "sim42_grid.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 250); print(G.round(3).to_string(index=False))
# all-environment Wald (no selection), p<0.05
q = D[D.z_p == 0.75]; print("\nall-42-environment Wald p<0.05:", q.groupby("kind").z_full.apply(lambda v: (v >= norm.isf(0.05)).mean()).round(3).to_dict())
