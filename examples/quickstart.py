#!/usr/bin/env python3
"""The method of the paper on one simulated split, end to end, with no genotype file.

    python examples/quickstart.py

`multi_gxe.simulate` draws two half-sample normal-equation exports around one truth (the real schema,
so the same code runs on engine exports).  The rule of the paper is `mg.complementary_split` with no
arguments: forward selection on the gain in the GxE heritability tau(S) on one half, the ordering
scored on the other half, both directions averaged, the cut k_g by the sequential tail tests at
gamma = 0.01, a trait reported iff k_g > 0, and the reported environments the k_g with the highest
selection frequency across the orderings.  No clustering: every environment is its own unit."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import multi_gxe as mg


def main() -> None:
    # ---- a design: 12 environments, no strong correlations (the pool is curated beforehand) ----
    corr, _ = mg.simulate.make_corr(block_sizes=(1,) * 12, background=0.05, seed=4)
    L = corr.shape[0]
    h2 = mg.simulate.sparse_truth(L, tau=0.03, causal=[0, 5])
    A, B = mg.simulate.simulate_fold_pair(corr, h2, se=0.003, seed=20)       # two independent halves, one truth
    print(f"L = {L} environments; truth: h2_GxE = 0.03 on env_0 and env_5, 0 elsewhere (total {h2.sum():.2f})\n")

    # ---- the rule of the paper: one call ----
    r = mg.complementary_split([(A, B)])
    M = r["curve"]                                                           # the averaged held-out curve c-bar(k)
    print(f"reported = {r['names']}   k_g = {r['k']}   tau = {r['tau']:.4f} +- {r['se']:.4f}   "
          f"full-pool value c(L) = {r['tau_full']:.4f}")
    print(f"reported set (selection frequency in the top k_g, ties by mean position): {r['blocks']}")
    for d in r["directions"]:
        print(f"  ordering grown on one half: {list(d.path)}")
    print("\naveraged held-out curve c-bar(k) with its paired-jackknife SE of c(k) - c(k-1):")
    for k in range(1, L + 1):
        print(f"  k={k:2d}  c={M.c[k]:.4f}  {M.c[k] - M.c[k - 1]:+.4f} (se {M.se(k, k - 1):.4f})"
              + ("   <- cut" if k == r["k"] else ""))

    # ---- the same curve under the other rules of the record (all pure functions of the curve) ----
    print("\nthe same averaged curve under every rule in stopping.PANEL (k):")
    for name, rule in mg.stopping.PANEL.items():
        try:
            print(f"  {name:<24} k = {rule(M)}")
        except ValueError as e:                                              # `gate` needs the in-fold round p-values
            print(f"  {name:<24} ({e})")

    # ---- why the split: the in-sample curve of the ordering grown on A, scored on A ----
    evA = mg.Evaluator(A)
    pathA = r["directions"][0].path
    c_in, j_in = mg.curve(evA, [[int(b)] for b in pathA])
    cv_in = mg.Curve(c=np.asarray(c_in), j=np.asarray(j_in))
    print(f"\nno complementary split (ordering scored on the half that grew it): "
          f"k = {mg.tail_test(0.01)(cv_in)}, in-sample c(k_g) = {c_in[r['k']]:.4f} against the held-out {r['tau']:.4f}; "
          f"on a null curve the in-sample version is reported nearly always.")


if __name__ == "__main__":
    main()
