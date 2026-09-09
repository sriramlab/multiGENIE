#!/usr/bin/env python3
"""End to end on a synthetic export: cluster, grow, cut, report.

    python examples/quickstart.py

No genotypes and no engine run: `multi_gxe.simulate` builds a normal-equation
system in the real schema, and every stage downstream is the production code.

READ THE NUMBERS AS MECHANICS, NOT AS EVIDENCE.  The toy imposes h2_GxE component
by component, so correlated environments do not split a shared signal, and its
signal sits in two blocks with a quiet tail.  Which stopping rule wins here is a
property of that setup.  The measured comparison on real exports and on simulation
arms built from real fits is in `docs/versions/v4_crossfit.md`.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import multi_gxe as mg


def main() -> None:
    # ---- a design: 13 environments, one near-duplicate triple and one pair ----
    corr, truth_blocks = mg.simulate.make_corr(seed=1)
    L = corr.shape[0]

    # stage 1: the blocks come from E alone, and never see the phenotype
    blocks = mg.cluster_from_corr(corr, cut=0.8)
    print(f"L = {L} environments -> K = {len(blocks)} blocks")
    print(f"  blocks    {blocks}")
    print(f"  truth     {truth_blocks}   (recovered exactly: "
          f"{blocks == truth_blocks})\n")

    # two independent half-sample exports around one truth
    h2 = mg.simulate.sparse_truth(L, tau=0.06, causal=[0, 5])
    A, B = mg.simulate.simulate_fold_pair(corr, h2, se=0.003, seed=11)
    print(f"truth: h2_GxE = 0.06 on env_0 and env_5, 0 elsewhere "
          f"(total {h2.sum():.2f})\n")

    # stage 2: full-depth greedy on fold A, no stopping test
    grown = mg.grow_path(mg.Evaluator(A), blocks)
    print("grow on A (full depth, no gate)")
    print(f"  path      {grown['path']}")
    print(f"  solves    {grown['n_theta_solve']} ordering, "
          f"{grown['n_gate_eval']} jackknifed\n")

    # stages 3 and 4: cut the curve drawn on fold B, both directions, MIN
    for name, rule in (("plateau z=1.645", mg.plateau(1.645)),
                       ("1-SE n_se=2", mg.one_se(2.0)),
                       ("window m=3", mg.window(3, 1.645)),
                       ("gate alpha=0.05", mg.gate(0.05, 2))):
        r = mg.crossfit(A, B, blocks, rule=rule)
        named = ", ".join(f"block {b} = {blocks[b]}" for b in r["blocks"]) or "-"
        print(f"{name:<18} tau = {r['tau']:.4f} +- {r['se']:.4f}  "
              f"z = {r['z']:>5.2f}   k = {r['k']} (A {r['k_A']} / B {r['k_B']})"
              f"   names: {r['names']}")
        print(f"{'':<18} {named}")

    r = mg.crossfit(A, B, blocks, rule=mg.plateau(1.645))
    print(f"\nfull-depth out-of-fold total (the denominator) = {r['tau_full']:.4f}")
    print(f"the plateau rule keeps {100 * r['tau'] / r['tau_full']:.0f}% of it "
          f"on {r['k']} of {len(blocks)} blocks")


if __name__ == "__main__":
    main()
