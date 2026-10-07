#!/usr/bin/env python3
"""Re-analyse the simulated replicates with the SAME environment set and clustering as the 47-environment
real data: all 47 environments of the fold exports and the pool47 clustering (K = 30: the 25 pool42 clusters
plus statin, insomnia, nap, IGF-1 and CRP as singletons), through the public v5 package with
drop_degenerate_nxe, exactly as the 42-trait real run on sep6_pool47.

Arms: null47, null47_halfn (100 matched null), sparse47_single (tea), sparse47_cluster / _member (smoking
block), sparse47_new (crp), dense47 (4 causal: ls05, ls11, ls17, crp -- all inside the pool).
-> <out>/sim47_units.tsv, sim47_grid.tsv (z_p grid x z_report list); paper_sim_calibration.py turns
sim47_units.tsv into the table and figure of the paper.

    python examples/sim47_consistency.py --sims <dir with <arm>/rep<r>/sim.fold{A,B}.normal_eq.json> \
        --clusters clusters_pool47.json --truth blocks_pool47.tsv --out <dir>

The companion sim42_consistency.py slices the 5 extra environments out and uses the pool42 clustering.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
ROOT = Path(os.environ.get("MULTI_GXE_ROOT", Path.cwd()))   # analysis root: data/, results/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import multi_gxe as mg                                   # noqa
from multi_gxe.stopping import plateau                   # noqa
from multi_gxe.evaluator import Evaluator                # noqa
ap = argparse.ArgumentParser()
ap.add_argument("--sims", default="/u/scratch/z/zhengton/multi_gxe/cluster_greedy_v3")
ap.add_argument("--clusters", default=str(ROOT / "results/cluster_greedy_v3/sep6/clusters_pool47.json"))
ap.add_argument("--truth", default=str(ROOT / "results/cluster_greedy_v3/blocks_pool47.tsv"), help="block, causal_<arm> columns")
ap.add_argument("--out", default=str(ROOT / "results/pool87_crossfit/manuscript"))
a = ap.parse_args()
SCR = Path(a.sims)
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
ARMS = ["null47", "null47_halfn", "sparse47_single", "sparse47_cluster", "sparse47_member", "sparse47_new", "dense47"]
CL = json.loads(Path(a.clusters).read_text())
blocks = [list(map(int, c)) for c in CL["clusters"]]
truth47 = pd.read_csv(a.truth, sep="\t").sort_values("block")
ENV47 = [f"ls{i:02d}" for i in range(42)] + ["statin", "insomnia", "nap", "igf1", "crp"]
ZP = [0.5, 0.75, 1.0, 1.28, 1.645]; ZR = {"z2.33": 2.33, "p0.05": norm.isf(0.05), "bonf42": norm.isf(0.05 / 42)}

def causal_blocks(arm):
    if arm.startswith("null"): return set()
    return set(int(b) for b in truth47.loc[truth47[f"causal_{arm}"] == 1, "block"])

rows, t0 = [], time.time()
for arm in ARMS:
    tr = causal_blocks(arm)
    for d in sorted(SCR.glob(f"{arm}/rep*"), key=lambda p: int(p.name[3:])):
        fa, fb = d / "sim.foldA.normal_eq.json", d / "sim.foldB.normal_eq.json"
        if not (fa.exists() and fb.exists()): continue
        A = mg.drop_degenerate_nxe(mg.load_export(fa)); B = mg.drop_degenerate_nxe(mg.load_export(fb))
        assert list(A.env_names) == ENV47, list(A.env_names)
        evA, evB = Evaluator(A), Evaluator(B)
        base = mg.complementary_split([(A, B)], blocks, evaluators=[(evA, evB)])   # grows both orderings; re-cut per z_p below
        M = base["curve"]; paths = [dd.path for dd in base["directions"]]
        for zp in ZP:
            r = mg.compsplit.select_from_curves([dd.curve for dd in base["directions"]], paths, gate_rule=plateau(zp), z_gate=0.0, set_rule=lambda cv: cv.K)
            kg = r["k"]; se = M.se_at(kg) if kg > 0 else 0.0; z = M.c[kg] / se if se > 0 else -np.inf
            S = set(r["blocks"]) if kg > 0 else set()
            rows.append(dict(arm=arm, rep=int(d.name[3:]), z_p=zp, k_g=kg, tau=float(M.c[kg]), z=z, tau_full=float(M.c[-1]),
                             z_full=M.c[-1] / M.se_at(M.K), n_causal_in=len(tr), n_causal_out=0,
                             recall=len(S & tr) / len(tr) if tr else np.nan, fdp=len(S - tr) / len(S) if S else 0.0,
                             exact=float(S == tr) if tr else np.nan, size=len(S)))
    print(f"  {arm} done [{time.time()-t0:.0f}s]", flush=True)
D = pd.DataFrame(rows); D.to_csv(OUT / "sim47_units.tsv", sep="\t", index=False, float_format="%.5g")
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
G = pd.DataFrame(out); G.to_csv(OUT / "sim47_grid.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 250); print(G.round(3).to_string(index=False))
q = D[D.z_p == 0.5]; print("\nall-47-environment Wald p<0.05:", q.groupby("kind").z_full.apply(lambda v: (v >= norm.isf(0.05)).mean()).round(3).to_dict())
