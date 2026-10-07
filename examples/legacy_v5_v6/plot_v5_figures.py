#!/usr/bin/env python3
"""Figures for the v5 estimator (RESULTS.md section 0, docs/versions/v5_complementary_split.md).

    python examples/plot_v5_figures.py --exports <dir> --clusters clusters.json \
        --env-names env_names.tsv --traits traits.txt

Writes, under figures/:
  v5_rule_on_curve.png       three traits: both directions, the averaged curve, the two cuts
  v5_null_power.png          null false positives and signal-arm identification, per configuration
  v5_pool42_forest.png       per named trait: tau with 95% CI, tau_selected, tau_full, environments
  v5_curve_grid_pool42.pdf   all traits' averaged curves with both cuts
Reads results/compsplit_null_power.tsv; re-runs `complementary_split` on the exports for the curves.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import multi_gxe as mg  # noqa: E402

# reference categorical slots 1-3 (validated all-pairs, light surface) + ink / neutrals
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 9, "axes.titlesize": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "legend.frameon": False,
})
FIG = REPO / "figures"


def short(label, n=34):
    return label if len(label) <= n else label[: n - 1] + "…"


def run_real(a):
    blocks = [list(map(int, c)) for c in json.loads(Path(a.clusters).read_text())["clusters"]]
    lab = {}
    for line in Path(a.env_names).read_text().splitlines()[1:]:
        k, v = line.split("\t")[:2]
        lab[k] = v
    traits = [t.strip() for t in Path(a.traits).read_text().splitlines()
              if t.strip() and not t.lstrip().startswith("#")]
    res = {}
    for t in traits:
        fa = Path(a.exports) / t / f"{t}.foldA.normal_eq.json"
        fb = Path(a.exports) / t / f"{t}.foldB.normal_eq.json"
        if not (fa.exists() and fb.exists()):
            continue
        A, B = mg.drop_degenerate_nxe(mg.load_export(fa)), mg.drop_degenerate_nxe(mg.load_export(fb))
        r = mg.complementary_split([(A, B)], blocks, gate_rule=mg.plateau(0.75), z_gate=2.33,
                                   set_rule=mg.window(3, 1.645))          # the v5 rule, no longer the default
        names = A.env_names
        # one short name per block: its shortest member label (+n for multi-environment blocks)
        def blab(b):
            ls = [lab.get(names[e], names[e]) for e in blocks[b]]
            return min(ls, key=len) + (f" (+{len(ls) - 1})" if len(ls) > 1 else "")
        r["env_labels"] = [blab(b) for b in r["blocks"]]
        r["k_window"] = int(mg.window(3, 1.645)(r["curve"]))
        res[t] = r
        print(f"  {t}", flush=True)
    return res


def draw_curve(ax, t, r, detail=True):
    M = r["curve"]
    K = M.K
    x = np.arange(K + 1)
    se = np.array([M.se_at(k) for k in x])
    if detail:
        for d, col, nm in ((r["directions"][0], BLUE, "A→B"), (r["directions"][1], ORANGE, "B→A")):
            ax.plot(x, d.curve.c, color=col, lw=1.0, alpha=0.55, label=f"{nm} held-out curve")
    ax.fill_between(x, M.c - se, M.c + se, color=INK, alpha=0.08, lw=0)
    ax.plot(x, M.c, color=INK, lw=2.0, label="average M(k) ± 1 SE")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.axhline(M.c[K], color=MUTED, lw=0.8, ls=":")
    kp, kw = r["k"], min(r["k_window"], r["k"])
    if r["names"]:
        same = kp == kw
        ax.plot([kp], [M.c[kp]], "o", ms=13 if same else 8, mfc=ORANGE, mec=SURFACE, mew=1.5, zorder=5)
        ax.plot([kw], [M.c[kw]], "D", ms=6, mfc=AQUA, mec=SURFACE, mew=1.2, zorder=6)
        if detail:
            kw_args = dict(textcoords="offset points", fontsize=8, color=INK,
                           arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
            if same:
                ax.annotate(f"plateau = window, k={kp}\ntotal τ = named share = {M.c[kp]:.3f}",
                            (kp, M.c[kp]), xytext=(28, -42), ha="left", **kw_args)
            else:
                right = kp > K / 2
                ax.annotate(f"plateau k={kp}\ntotal τ = {M.c[kp]:.3f}", (kp, M.c[kp]),
                            xytext=(-12 if right else 12, 26), ha="right" if right else "left", **kw_args)
                ax.annotate(f"window k={kw}\nnamed share {M.c[kw]:.3f}", (kw, M.c[kw]),
                            xytext=(22, -36), ha="left", **kw_args)
    ax.set_title(f"{t}" + ("" if r["names"] else "  (not named)"), loc="left",
                 fontweight="bold" if r["names"] else "normal", color=INK if r["names"] else MUTED)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_xlim(0, K)


def fig_rule_on_curve(res):
    pick = [t for t in ("testosterone", "bmi", "ldl") if t in res]
    fig, axes = plt.subplots(1, len(pick), figsize=(4.2 * len(pick), 3.6), sharey=False)
    for ax, t in zip(np.atleast_1d(axes), pick):
        draw_curve(ax, t, res[t])
        ax.set_xlabel("k = top blocks of the greedy order")
        r = res[t]
        env = "; ".join(r["env_labels"]) or "none"
        ax.text(0.98, 0.04, "named: " + short(env, 48), transform=ax.transAxes, ha="right",
                fontsize=7.5, color=INK2)
    np.atleast_1d(axes)[0].set_ylabel("cumulative h2_GxE, out of fold")
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], ls="", marker="o", mfc=ORANGE, mec=SURFACE, ms=8),
          plt.Line2D([], [], ls="", marker="D", mfc=AQUA, mec=SURFACE, ms=7),
          plt.Line2D([], [], color=MUTED, ls=":")]
    l += ["plateau(0.75) cut → trait named, total τ", "window(3,1.645) cut → environments", "full fit M(K)"]
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("v5: one averaged held-out curve, two cuts (42 environments)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(FIG / "v5_rule_on_curve.png", dpi=200)
    plt.close(fig)


def wilson(x, n, z=1.96):
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fig_null_power():
    lines = (REPO / "results/compsplit_null_power.tsv").read_text().splitlines()
    head = lines[0].split("\t")
    rows = [dict(zip(head, l.split("\t"))) for l in lines[1:]]
    configs = list(dict.fromkeys(r["config"] for r in rows))
    short_cfg = {c: c.replace("v5 default: ", "v5 default\n").replace("gate ", "").replace(", set ", " / ")
                 .replace("one_se-style max within 2.33 SE", "within 2.33 SE of max").replace(" z>=2.33", "")
                 for c in configs}
    panels = [("null", "P_named", "Null: trait named", True),
              ("null", "P_any_env", "Null: any env. named", True),
              ("sparse(1 causal)", "P_named", "1 causal: power", False),
              ("sparse(1 causal)", "precision", "1 causal: precision", False),
              ("sparse(1 causal)", "exact", "1 causal: exact set", False),
              ("dense(4 causal)", "P_named", "4 causal: power", False),
              ("dense(4 causal)", "recall", "4 causal: recall", False)]
    fig, axes = plt.subplots(1, len(panels), figsize=(16.5, 3.9), sharey=True)
    ypos = np.arange(len(configs))[::-1]
    for ax, (arm, col, title, is_null) in zip(axes, panels):
        for y, c in zip(ypos, configs):
            r = next(r for r in rows if r["config"] == c and r["arm"] == arm)
            v, n = float(r[col]), int(r["replicates"])
            default = c.startswith("v5 default")
            color = BLUE if default else MUTED
            if col in ("P_named", "P_any_env", "exact"):
                lo, hi = wilson(round(v * n), n)
                ax.plot([lo, hi], [y, y], color=color, lw=1.6, solid_capstyle="round")
            ax.plot([v], [y], "o", ms=8 if default else 6, mfc=color, mec=SURFACE, mew=1.2, zorder=4)
            ax.text(v, y + 0.28, f"{v:.2f}", ha="center", fontsize=7, color=INK if default else INK2)
        if is_null:
            ax.axvline(0.05, color=ORANGE, lw=1, ls="--")
            ax.text(0.052, ypos[-1] - 0.55, "0.05", color=ORANGE, fontsize=7)
            ax.set_xlim(-0.005, 0.12)
        else:
            ax.set_xlim(0, 1.05)
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_ylim(-0.8, len(configs) - 0.3)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([short_cfg[c] for c in configs], fontsize=7.5)
    fig.suptitle("v5 on simulation arms: null judged on false positives only, signal on power and identification "
                 "(bars = 95% Wilson CI; null n=100, 1 causal n=200, 4 causal n=20)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "v5_null_power.png", dpi=200)
    plt.close(fig)


def fig_forest(res):
    named = sorted([t for t, r in res.items() if r["names"]], key=lambda t: res[t]["tau"])
    fig = plt.figure(figsize=(12, 0.34 * len(named) + 1.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.02)
    ax, tx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    tx.axis("off")
    tx.set_ylim(-0.6, len(named) - 0.4)
    for y, t in enumerate(named):
        r = res[t]
        ax.plot([r["tau"] - 1.96 * r["se"], r["tau"] + 1.96 * r["se"]], [y, y], color=BLUE, lw=1.6,
                solid_capstyle="round")
        ax.plot([r["tau_selected"]], [y], "D", ms=6, mfc=AQUA, mec=SURFACE, mew=1, zorder=4)
        ax.plot([r["tau_full"]], [y], "|", ms=11, mew=1.6, color=MUTED, zorder=3)
        ax.plot([r["tau"]], [y], "o", ms=7, mfc=BLUE, mec=SURFACE, mew=1.2, zorder=5)
        env = "; ".join(r["env_labels"]) or "— no single block stands out (diffuse)"
        tx.text(0.02, y, short(env, 78), va="center", fontsize=7.5, color=INK2 if r["blocks"] else MUTED)
    ax.set_ylim(-0.6, len(named) - 0.4)
    ax.set_yticks(range(len(named)))
    ax.set_yticklabels(named)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("h2_GxE, out of fold")
    ax.grid(axis="x", color=GRID, lw=0.5)
    h = [plt.Line2D([], [], color=BLUE, marker="o", mfc=BLUE, mec=SURFACE, ms=7, lw=1.6),
         plt.Line2D([], [], ls="", marker="D", mfc=AQUA, mec=SURFACE, ms=6),
         plt.Line2D([], [], ls="", marker="|", color=MUTED, ms=11, mew=1.6)]
    ax.legend(h, ["total τ = M(k_plateau), 95% CI", "τ_selected = M(k_window)", "unselected fit M(K)"],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8)
    n_all = len(res)
    frac = sum(res[t]["tau"] for t in named) / sum(r["tau_full"] for r in res.values())
    fig.suptitle(f"v5 on 42 environments: {len(named)}/{n_all} traits named, {100 * frac:.1f}% of the unselected fit",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    tx.text(0.02, len(named) - 0.4, "named environments (consensus order; +n = other block members)",
            va="bottom", fontsize=8, fontweight="bold", color=INK)
    fig.subplots_adjust(left=0.16, right=0.99, top=1 - 1.05 / fig.get_figheight(), bottom=0.5 / fig.get_figheight())
    fig.savefig(FIG / "v5_pool42_forest.png", dpi=200)
    plt.close(fig)


def fig_grid(res):
    ts = sorted(res)
    ncol = 6
    nrow = int(np.ceil(len(ts) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.0 * nrow))
    for ax, t in zip(axes.ravel(), ts):
        draw_curve(ax, t, res[t], detail=False)
        ax.tick_params(labelsize=6)
        ax.title.set_fontsize(8)
    for ax in axes.ravel()[len(ts):]:
        ax.axis("off")
    fig.suptitle("v5 averaged held-out curves, 42 environments (● plateau cut = total, ◆ window cut = environments; "
                 "bold = named; dotted = unselected fit)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / "v5_curve_grid_pool42.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", required=True)
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--env-names", required=True)
    ap.add_argument("--traits", required=True)
    a = ap.parse_args()
    fig_null_power()
    res = run_real(a)
    fig_rule_on_curve(res)
    fig_forest(res)
    fig_grid(res)
    print("wrote", sorted(p.name for p in FIG.glob("v5_*")))


if __name__ == "__main__":
    main()
