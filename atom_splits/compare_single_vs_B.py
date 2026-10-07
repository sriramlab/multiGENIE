#!/usr/bin/env python
"""Single split versus many pairs, all traits with atom-split results.

Two questions, answered from saved curves only (no engine runs):

  1. ESTIMATES.  How does the paper's rule on all 462 atom pairs compare with a single run --
     the paper's split, the independent engine re-splits of stability_selection (splits
     1, 6-10), and each of the 462 atom pairs taken alone?
  2. STABILITY.  Does averaging B pairs make tau, the plateau depth k_g and the leading
     clusters more reproducible, for every trait?  Disjoint groups of B pairs, the paper's
     rule on each group (same construction as stability_vs_B.py, which this re-uses), plus
     the fixed-depth value M(K) so the effect of the cut is separated from the estimate.

    ~/.conda/envs/test/bin/python scripts/atom_splits/compare_single_vs_B.py

Reads   results/atom_splits/<trait>.pairs.npz, <trait>.summary.tsv
        results/stability_selection/cache/native.<trait>.split<s>.npz   (engine re-splits)
Writes  results/atom_splits/compare_single_vs_B.tsv             one row per trait
        results/atom_splits/compare_single_vs_B.byB.tsv         trait x B (disjoint groups)
        results/atom_splits/compare_single_vs_B.replicates.tsv  every single run (engine + atom pairs)
        figures/atom_splits/compare_single_vs_B_{estimates,stability}.{png,pdf}

What the between-group spread does and does not measure: all 462 pairs re-use the same people,
the same 12 atoms and one probe draw, so it is the noise of WHICH HALVES were drawn, nothing
else.  The sampling error of tau (genome jackknife SE, atom delete-one SE) is not reduced by B.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from atomsplit.tensor import REPO                           # noqa: E402

sys.path.insert(0, str(REPO))
import multi_gxe as mg                                      # noqa: E402
from multi_gxe.stopping import Curve, plateau               # noqa: E402
from stability_vs_B import jaccard, jaccard_excess          # noqa: E402

ROOT = HERE.parents[1]
Z_P, Z_NAME = 0.5, 1.645
ENGINE_SPLITS = (1, 6, 7, 8, 9, 10)      # independent partitions (2-5 duplicate split 1)
BS = (1, 2, 3, 5, 10, 20, 50, 100)


def rule(C, J, paths, idx):
    """The paper's decision on the directions `idx`, plus the fixed-depth value M(K)."""
    M = mg.average_curves([Curve(c=C[d], j=J[d]) for d in idx])
    K = M.K
    k = int(plateau(Z_P)(M))
    se = M.se_at(k) if k > 0 else 0.0
    named = bool(k > 0 and se > 0 and M.c[k] / se >= Z_NAME)
    P = [list(paths[d]) for d in idx]
    return dict(k=k, named=named, tau=float(M.c[k]) if named else 0.0, se=float(se),
                tau_full=float(M.c[K]), se_full=float(M.se_at(K)),
                lead=int(mg.consensus(P, 1, "freq")[0]),
                set=frozenset(mg.consensus(P, k, "freq")) if named else frozenset(),
                top3=frozenset(mg.consensus(P, 3, "freq")), top5=frozenset(mg.consensus(P, 5, "freq")))


def engine_runs(cache: Path, trait: str, K: int):
    out = []
    for s in ENGINE_SPLITS:
        p = cache / f"native.{trait}.split{s}.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        if len(z["c0"]) != K + 1:
            print(f"[warn] {trait} split {s}: engine cache has K={len(z['c0']) - 1}, atoms K={K}; skipped")
            continue
        r = rule(np.stack([z["c0"], z["c1"]]), np.stack([z["j0"], z["j1"]]), np.stack([z["p0"], z["p1"]]), [0, 1])
        r["split"] = s
        out.append(r)
    return out


def engine_full_n(trait: str, scratch: Path, clusters: Path, drop_cells: Path):
    """Unselected full-depth fit of the engine's FULL-sample pool47 export (own probe draw),
    for comparison with the all-atoms assembly: the two differ only by the probes."""
    import json
    from atomsplit.pairs import prepare_export, read_drop_cells
    from multi_gxe.evaluator import Evaluator
    f = scratch / trait / f"{trait}.pool47.normal_eq.json"
    if not f.exists():
        return float("nan"), float("nan")
    blocks0 = [list(map(int, c)) for c in json.loads(clusters.read_text())["clusters"]]
    ex = mg.load_export(f)
    e, _ = prepare_export(ex, read_drop_cells(str(drop_cells)).get(trait, []), blocks0, list(ex.env_names))
    r = Evaluator(e).joint()
    return float(r["cum_h2"]), float(r["cum_se"])


def sd(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1)) if len(x) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results/atom_splits"))
    ap.add_argument("--cache", default=str(ROOT / "results/stability_selection/cache"))
    ap.add_argument("--fig", default=str(ROOT / "figures/atom_splits"))
    ap.add_argument("--engine-full", default="/u/scratch/z/zhengton/multi_gxe/engine_pool47",
                    help="engine full-sample pool47 exports ('' to skip)")
    ap.add_argument("--clusters", default=str(ROOT / "results/cluster_greedy_v3/sep6/clusters_pool47.json"))
    ap.add_argument("--drop-cells", default=str(ROOT / "data/cluster_greedy_v3/drop_cells_pool47.tsv"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=20, help="random re-groupings averaged per B")
    a = ap.parse_args()
    res, cache = Path(a.results), Path(a.cache)
    traits = sorted(p.name.split(".")[0] for p in res.glob("*.pairs.npz"))
    rows, byB, reps, keep = [], [], [], {}
    for t in traits:
        z = np.load(res / f"{t}.pairs.npz", allow_pickle=True)
        S = pd.read_csv(res / f"{t}.summary.tsv", sep="\t").iloc[0]
        C, J, paths, pair_of, labels = z["C"], z["J"], z["paths"], z["pair_of"], list(z["labels"])
        K = paths.shape[1]
        pairs = np.unique(pair_of)
        dirs_of = {p: np.flatnonzero(pair_of == p) for p in pairs}
        single = [rule(C, J, paths, dirs_of[p]) for p in pairs]             # each atom pair alone
        full = rule(C, J, paths, np.arange(len(pair_of)))                    # B = 462
        eng = engine_runs(cache, t, K)
        efn = engine_full_n(t, Path(a.engine_full), Path(a.clusters), Path(a.drop_cells)) if a.engine_full \
            else (float("nan"), float("nan"))
        for r in eng:
            reps.append(dict(trait=t, source="engine_split", id=r["split"], named=int(r["named"]), k=r["k"],
                             tau=r["tau"], se=r["se"], tau_full=r["tau_full"], se_full=r["se_full"],
                             lead=labels[r["lead"]]))
        for p, r in zip(pairs, single):
            reps.append(dict(trait=t, source="atom_pair", id=int(p), named=int(r["named"]), k=r["k"],
                             tau=r["tau"], se=r["se"], tau_full=r["tau_full"], se_full=r["se_full"],
                             lead=labels[r["lead"]]))

        # disjoint groups of B pairs; averaged over `reps` random re-groupings so the
        # between-group SD at large B (few groups) is not one noisy draw
        rng = np.random.default_rng(a.seed)
        for B in BS:
            acc = []
            for _ in range(a.reps if B > 1 else 1):
                perm = rng.permutation(pairs)
                groups = [perm[i * B:(i + 1) * B] for i in range(len(pairs) // B)]
                rr = [rule(C, J, paths, np.concatenate([dirs_of[p] for p in g])) for g in groups]
                ks = np.array([r["k"] for r in rr])
                acc.append(dict(n_groups=len(rr), named=np.mean([r["named"] for r in rr]),
                                k_mean=ks.mean(), k_sd=sd(ks), k_min=ks.min(), k_max=ks.max(),
                                k_at_Bfull=np.mean(ks == full["k"]),
                                tau_mean=np.mean([r["tau"] for r in rr]), tau_sd=sd([r["tau"] for r in rr]),
                                full_sd=sd([r["tau_full"] for r in rr]),
                                se_cut=np.mean([r["se"] for r in rr]), se_full=np.mean([r["se_full"] for r in rr]),
                                lead_agree=np.mean([r["lead"] == full["lead"] for r in rr]),
                                set_jac_excess=jaccard_excess([r["set"] for r in rr if r["named"]], K),
                                top3_jac=jaccard([r["top3"] for r in rr]), top5_jac=jaccard([r["top5"] for r in rr]),
                                top3_vs_full=np.mean([len(r["top3"] & full["top3"]) / 3 for r in rr]),
                                top5_vs_full=np.mean([len(r["top5"] & full["top5"]) / 5 for r in rr])))
            d = {k: float(np.nanmean([x[k] for x in acc])) for k in acc[0]}
            d["k_min"], d["k_max"] = int(min(x["k_min"] for x in acc)), int(max(x["k_max"] for x in acc))
            byB.append(dict(trait=t, B=B, **d))
        b1 = byB[-len(BS)]
        for r in byB[-len(BS):]:
            # independent pairs would give 1/sqrt(B); drawing B of 462 without replacement gives the fpc on top
            fpc = np.sqrt((len(pairs) - r["B"]) / (len(pairs) - 1))
            r["tau_sd_rel"] = r["tau_sd"] / b1["tau_sd"]
            r["full_sd_rel"] = r["full_sd"] / b1["full_sd"]
            r["iid_rel"] = fpc / np.sqrt(r["B"])

        st = np.array([r["tau"] for r in single])
        sf = np.array([r["tau_full"] for r in single])
        sk = np.array([r["k"] for r in single])
        sn = np.array([r["named"] for r in single])
        et = np.array([r["tau"] for r in eng]) if eng else np.array([])
        ef = np.array([r["tau_full"] for r in eng]) if eng else np.array([])
        ek = [r["k"] for r in eng]
        rows.append(dict(
            trait=t, K=K,
            # the paper's split (engine exports) and the same split from the atoms (pair 0)
            paper_k=int(S.paper_k), paper_tau=S.paper_tau, paper_se=S.paper_se, paper_tau_full=S.paper_tau_full,
            pair0_k=single[0]["k"], pair0_tau=single[0]["tau"], pair0_tau_full=single[0]["tau_full"],
            # engine re-splits: independent partitions of the people, one run each
            eng_n=len(eng), eng_named=int(sum(r["named"] for r in eng)), eng_k=",".join(map(str, ek)),
            eng_tau_mean=et.mean() if len(et) else np.nan, eng_tau_sd=sd(et), eng_tau_min=et.min() if len(et) else np.nan,
            eng_tau_max=et.max() if len(et) else np.nan, eng_full_mean=ef.mean() if len(ef) else np.nan,
            eng_full_sd=sd(ef),
            eng_lead_same=int(len({r["lead"] for r in eng}) == 1) if eng else -1,
            # the 462 atom pairs one at a time (B = 1)
            b1_named=sn.mean(), b1_k_q25=np.percentile(sk, 25), b1_k_med=np.median(sk), b1_k_q75=np.percentile(sk, 75),
            b1_k_min=sk.min(), b1_k_max=sk.max(), b1_tau_mean=st.mean(), b1_tau_sd=sd(st), b1_tau_q05=np.percentile(st, 5),
            b1_tau_q95=np.percentile(st, 95), b1_full_mean=sf.mean(), b1_full_sd=sd(sf),
            b1_se_cut=np.mean([r["se"] for r in single]), b1_se_full=np.mean([r["se_full"] for r in single]),
            b1_lead_agree=np.mean([r["lead"] == full["lead"] for r in single]),
            # all 462 pairs
            B462_k=full["k"], B462_tau=full["tau"], B462_se=full["se"], B462_z=full["tau"] / full["se"],
            B462_full=full["tau_full"], B462_se_full=full["se_full"], lead=labels[full["lead"]],
            top3=";".join(labels[b] for b in sorted(full["top3"])),
            # sampling error B cannot remove: unselected full-N fit, genome jackknife and atom delete-one
            engine_fullN_tau=efn[0], engine_fullN_se=efn[1],
            fullN_tau=S.tau_full_allatoms, fullN_se_genome=S.se_full_genome, fullN_se_atoms=S.se_full_atoms,
            # where the paper's single run falls in the distribution of single pairs
            paper_pctile=float((st < S.paper_tau).mean()), paper_full_pctile=float((sf < S.paper_tau_full).mean()),
            shift_vs_paper=full["tau"] - S.paper_tau, shift_vs_paper_full=full["tau_full"] - S.paper_tau_full,
            split_sd_over_se=sd(sf) / np.mean([r["se_full"] for r in single])))
        keep[t] = dict(single_tau=st, single_full=sf, single_k=sk, eng=eng, full=full, S=S, K=K)

    T = pd.DataFrame(rows)
    Bt = pd.DataFrame(byB)
    R = pd.DataFrame(reps)
    T.to_csv(res / "compare_single_vs_B.tsv", sep="\t", index=False, float_format="%.6g")
    Bt.to_csv(res / "compare_single_vs_B.byB.tsv", sep="\t", index=False, float_format="%.6g")
    R.to_csv(res / "compare_single_vs_B.replicates.tsv", sep="\t", index=False, float_format="%.6g")

    pd.set_option("display.width", 250)
    f4 = lambda x: f"{x:.4f}"
    print("\n== 1. estimates: single run vs all 462 pairs (tau at the rule's own cut) ==")
    print(T[["trait", "paper_k", "paper_tau", "paper_se", "pair0_k", "pair0_tau", "eng_n", "eng_k", "eng_tau_mean",
             "eng_tau_sd", "b1_k_med", "b1_tau_mean", "b1_tau_sd", "B462_k", "B462_tau", "B462_se",
             "paper_pctile"]].to_string(index=False, float_format=f4))
    print("\n== 1b. fixed depth M(K): no cut, so B = 462 is exactly the mean of the single pairs ==")
    print(T[["trait", "paper_tau_full", "pair0_tau_full", "eng_full_mean", "eng_full_sd", "b1_full_mean", "b1_full_sd",
             "b1_se_full", "split_sd_over_se", "B462_full", "B462_se_full", "fullN_tau", "engine_fullN_tau", "fullN_se_genome",
             "fullN_se_atoms", "paper_full_pctile"]].to_string(index=False, float_format=f4))
    print("\n== 2. stability between disjoint groups of B pairs ==")
    for name, col, fmt in (("SD of tau at the cut", "tau_sd", "{:.4f}"), ("SD of M(K)", "full_sd", "{:.4f}"),
                           ("SD of M(K) relative to B = 1", "full_sd_rel", "{:.2f}"),
                           ("SD of k_g", "k_sd", "{:.1f}"), ("share of groups with k_g = the B=462 depth", "k_at_Bfull", "{:.2f}"),
                           ("leading cluster = the B=462 leading cluster", "lead_agree", "{:.2f}"),
                           ("top-3 set, Jaccard between groups", "top3_jac", "{:.2f}"),
                           ("top-5 set, Jaccard between groups", "top5_jac", "{:.2f}"),
                           ("reported set, Jaccard in excess of random sets", "set_jac_excess", "{:.2f}")):
        W = Bt.pivot(index="trait", columns="B", values=col)
        print(f"\n  {name}:\n  " + W.to_string(float_format=lambda x: fmt.format(x)).replace("\n", "\n  "))
    iid = Bt.drop_duplicates("B").set_index("B").iid_rel
    print("\n  1/sqrt(B) with the finite-population factor: " + "  ".join(f"B={b}: {v:.2f}" for b, v in iid.items()))
    plot(T, Bt, keep, Path(a.fig))


def plot(T, Bt, keep, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
                         "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    GREY, BLACK, LIGHT, BLUE = "0.45", "0.1", "0.82", "#2a6fb0"
    out.mkdir(parents=True, exist_ok=True)
    order = list(T.sort_values("B462_tau", ascending=False).trait)
    rng = np.random.default_rng(1)

    # ---- figure 1: the estimates
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 0.52 * len(order) + 1.9), sharey=True)
    for col, (key, skey, fkey, sekey, pkey, title) in enumerate((
            ("single_tau", "tau", "B462_tau", "B462_se", "paper_tau", "at the rule's own cut, $\\hat\\tau = M(k_g)$"),
            ("single_full", "tau_full", "B462_full", "B462_se_full", "paper_tau_full", "at full depth, $M(K)$ (no cut)"))):
        for i, t in enumerate(order):
            k, r = keep[t], T[T.trait == t].iloc[0]
            x = k[key]
            ax[col].plot(x, i + rng.uniform(-0.26, 0.26, len(x)), ".", color=LIGHT, ms=2.5, zorder=1, rasterized=True)
            ax[col].plot([r[fkey] - 1.96 * r[sekey], r[fkey] + 1.96 * r[sekey]], [i, i], "-", color=BLACK, lw=1.4, zorder=3)
            ax[col].plot(r[fkey], i, "o", color=BLACK, ms=6, zorder=4)
            ex = [e[skey] for e in k["eng"] if e["split"] != 1]
            ax[col].plot(ex, [i + 0.34] * len(ex), "o", mfc="white", mec=BLUE, mew=1.2, ms=5.5, zorder=5)
            ax[col].plot(r[pkey], i + 0.34, "D", color=BLUE, ms=5.5, zorder=6)
        ax[col].axvline(0, color=GREY, lw=0.6, ls=":")
        ax[col].set_xlabel("held-out interaction heritability")
        ax[col].set_title(title, fontsize=9.5, loc="left")
        ax[col].set_yticks(range(len(order)))
        ax[col].set_yticklabels(order)
        ax[col].set_ylim(len(order) - 0.4, -0.7)
    h = [Line2D([], [], marker=".", color=LIGHT, ls="", ms=7, label="each of the 462 atom pairs alone ($B=1$)"),
         Line2D([], [], marker="D", color=BLUE, ls="", ms=5.5, label="the paper's split (engine fit)"),
         Line2D([], [], marker="o", mfc="white", mec=BLUE, mew=1.2, ls="", ms=5.5, label="independent engine re-splits"),
         Line2D([], [], marker="o", color=BLACK, ls="-", ms=6, label="all 462 pairs averaged, 95% jackknife interval")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=2)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"compare_single_vs_B_estimates.{ext}", bbox_inches="tight", dpi=170)
    plt.close(fig)

    # ---- figure 2: stability against B, one column per trait (small multiples)
    nT = len(order)
    fig, ax = plt.subplots(4, nT, figsize=(1.75 * nT + 0.8, 8.2), sharex=True, sharey="row")
    Bs = sorted(Bt.B.unique())
    x = np.arange(len(Bs))
    for c, t in enumerate(order):
        d = Bt[Bt.trait == t].sort_values("B")
        r = T[T.trait == t].iloc[0]
        a0, a1, a2, a3 = ax[:, c]
        a0.plot(x, d.full_sd_rel, "o-", color=BLACK, ms=3.5, lw=1.0)
        a0.plot(x, d.iid_rel, "--", color=GREY, lw=0.9)
        a0.set_yscale("log")
        a0.set_title(t, fontsize=9.5)
        a1.vlines(x, d.k_min, d.k_max, color=LIGHT, lw=1.6)
        a1.vlines(x, np.maximum(d.k_mean - d.k_sd, 0), np.minimum(d.k_mean + d.k_sd, r.K), color=GREY, lw=3.2)
        a1.plot(x, d.k_mean, "o", color=BLACK, ms=3.5)
        a1.axhline(r.B462_k, color=BLUE, lw=0.8, ls=":")
        a2.plot(x, d.lead_agree, "o-", color=BLACK, ms=3.5, lw=1.0)
        a2.plot(x, d.top3_vs_full, "s-", color=GREY, ms=3.5, lw=1.0, mfc="white")
        a3.plot(x, d.top3_jac, "o-", color=BLACK, ms=3.5, lw=1.0)
        a3.plot(x, d.top5_jac, "s-", color=GREY, ms=3.5, lw=1.0, mfc="white")
        a3.set_xticks(x)
        a3.set_xticklabels(Bs, fontsize=7, rotation=60)
        for a_ in (a2, a3):
            a_.set_ylim(0, 1.06)
    ax[0, 0].set_ylabel("SD of $M(K)$ between groups\nrelative to $B=1$")
    ax[1, 0].set_ylabel("plateau depth $k_g$\n(mean, $\\pm$SD, range)")
    ax[2, 0].set_ylabel("agreement with the\n462-pair answer")
    ax[3, 0].set_ylabel("agreement between\ngroups (Jaccard)")
    ax[0, -1].legend(handles=[Line2D([], [], marker="o", color=BLACK, ms=3.5, lw=1, label="observed"),
                              Line2D([], [], color=GREY, ls="--", lw=0.9, label="$1/\\sqrt{B}$")],
                     fontsize=7, loc="lower left")
    ax[1, -1].legend(handles=[Line2D([], [], color=BLUE, ls=":", lw=0.8, label="depth at $B=462$")], fontsize=7, loc="lower right")
    ax[2, -1].legend(handles=[Line2D([], [], marker="o", color=BLACK, ms=3.5, lw=1, label="leading cluster"),
                              Line2D([], [], marker="s", color=GREY, mfc="white", ms=3.5, lw=1, label="top 3")],
                     fontsize=7, loc="lower right")
    ax[3, -1].legend(handles=[Line2D([], [], marker="o", color=BLACK, ms=3.5, lw=1, label="top 3"),
                              Line2D([], [], marker="s", color=GREY, mfc="white", ms=3.5, lw=1, label="top 5")],
                     fontsize=7, loc="lower right")
    fig.supxlabel("pairs averaged, $B$ (disjoint groups of the 462 pairs)", fontsize=9.5)
    fig.tight_layout(w_pad=0.6, h_pad=0.8)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"compare_single_vs_B_stability.{ext}", bbox_inches="tight", dpi=170)
    plt.close(fig)
    print("\nwrote", out / "compare_single_vs_B_estimates.png", "and", out / "compare_single_vs_B_stability.png")


if __name__ == "__main__":
    main()
