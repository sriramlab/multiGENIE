#!/usr/bin/env python3
"""Simulation table and calibration figure of the paper (Table "Simulations, 47 environments" and the
figure "Calibration and power") from the per-replicate table written by sim47_consistency.py.

    python examples/paper_sim_calibration.py --units sim47_units.tsv --out <dir>

Every replicate yields one paired Wald test of its cumulative h2_GxE: the estimate is M(k_g) at the cut, or
the full-depth M(K) when the cut falls at zero, with its block-jackknife SE; one-sided p < 0.05.
  fig_sim_calibration.{pdf,png}   (a) fraction of null replicates with p < 0.05, (b) of signal replicates,
                                  against the plateau threshold z_p; dotted = the same test on the unselected
                                  all-environment fit; 95% binomial bars
  sim47_paired_wald.tsv           the plotted fractions
  sim47_arms.tsv                  per arm at the adopted rule (z_p = 0.5, reported at p < 0.05): reported,
                                  Wilson interval, recall, precision, exact set
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

ap = argparse.ArgumentParser()
ap.add_argument("--units", required=True, help="sim47_units.tsv from sim47_consistency.py")
ap.add_argument("--out", required=True)
ap.add_argument("--tag", default="sim47")
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
ZP0 = 0.5                                   # the adopted plateau threshold
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
                     "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42})
GREY, BLACK = "0.55", "0.1"

U = pd.read_csv(a.units, sep="\t")
ZP = sorted(U.z_p.unique())
U["kind"] = np.select([U.arm.str.startswith("null"), U.arm == "dense47"], ["null", "dense"], "sparse")
U["z_paired"] = np.where(U.k_g > 0, U.z.astype(float), U.z_full.astype(float)); U["sig"] = norm.sf(U.z_paired) < 0.05
W = U.groupby(["z_p", "kind"]).sig.agg(["mean", "size"]).reset_index()
ref = U[U.z_p == ZP0].groupby("kind").z_full.apply(lambda v: (v >= norm.isf(0.05)).mean())

fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.6))


def band(a_, x, frac, n, **kw):
    f, m = np.asarray(frac, float), np.asarray(n, float); h = 1.96 * np.sqrt(f * (1 - f) / m)
    a_.errorbar(x, f, yerr=[f - np.clip(f - h, 0, 1), np.clip(f + h, 0, 1) - f], ms=3.5, lw=1, capsize=2, **kw)


def pick(kind):
    w = W[W.kind == kind].sort_values("z_p"); return w.z_p.to_numpy(), w["mean"].to_numpy(), w["size"].to_numpy()


x, f, n = pick("null"); band(ax[0], x, f, n, fmt="o-", color=BLACK)
x, f, n = pick("sparse"); band(ax[1], x - 0.012, f, n, fmt="o-", color=BLACK, label="1 causal cluster")
x, f, n = pick("dense"); band(ax[1], x + 0.012, f, n, fmt="s--", color=GREY, label="4 causal clusters")
ax[0].axhline(0.05, color=GREY, lw=0.6, ls=":"); ax[0].axhline(ref["null"], color=BLACK, lw=0.8, ls=":")
ax[1].axhline(ref["sparse"], color=BLACK, lw=0.8, ls=":"); ax[1].axhline(ref["dense"], color=GREY, lw=0.8, ls=":")
ax[1].plot([], [], color=BLACK, lw=0.8, ls=":", label="All environments, no selection")
for a_, t, letter in zip(ax, ["False-positive rate", "Power"], "ab"):
    a_.set_xlabel("Plateau threshold"); a_.set_title(f"{letter}  {t}", fontsize=9, loc="left")
    a_.set_xticks(ZP); a_.set_xticklabels([f"{z:g}" for z in ZP]); a_.axvline(ZP0, color=GREY, lw=0.6, ls=":")
ax[0].set_ylim(-0.005, 0.15); ax[1].set_ylim(0, 1.02); ax[1].legend(fontsize=6.5, loc="lower left")
fig.tight_layout(w_pad=1.5)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_sim_calibration.{ext}", dpi=300, bbox_inches="tight")
W.to_csv(OUT / f"{a.tag}_paired_wald.tsv", sep="\t", index=False, float_format="%.3f")
print(W.pivot(index="z_p", columns="kind", values="mean").round(3).to_string())

# per-arm table at the adopted rule
z = norm.isf(0.05); u = U[U.z_p == ZP0].copy(); u["rep"] = (u.k_g > 0) & (u.z >= z)


def wilson(x, n, zz=1.96):
    p_ = x / n; d = 1 + zz * zz / n; c = (p_ + zz * zz / (2 * n)) / d
    h = zz * np.sqrt(p_ * (1 - p_) / n + zz * zz / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


rows = []
for arm, g in list(u.groupby("arm", sort=False)) + [("sparse all", u[u.kind == "sparse"])]:
    x = int(g.rep.sum()); n = len(g); lo, hi = wilson(x, n); gg = g[g.rep]
    rows.append(dict(arm=arm, n=n, reported=x, P=x / n, lo=lo, hi=hi, recall=g.recall.where(g.rep, 0).mean(),
                     precision=1 - gg.fdp.mean() if len(gg) else np.nan, exact=g.exact.where(g.rep, 0).mean()))
T = pd.DataFrame(rows); T.to_csv(OUT / f"{a.tag}_arms.tsv", sep="\t", index=False, float_format="%.3f")
print(T.round(3).to_string(index=False))
