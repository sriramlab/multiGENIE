"""Complementary-split selection: average the held-out curves, then cut once.

THE ALGORITHM, for B complementary pairs (A_s, B_s):

    for each pair s and each direction (grow on one half, score on the other):
        grow the greedy path to full depth on the selection half, by the gain in tau(S)
        fit its nested prefixes on the held-out half           -> c(k), k = 0..L
    c_bar(k) = mean of the 2B curves at each depth             (jackknife: block-wise mean)
    k_g      = tail_test(gamma)(c_bar): the first depth past which no later depth is
               significantly higher, family-wise error at most gamma            gamma = 0.05
    the trait is REPORTED iff k_g > 0                          (H_0, the trait-level test)
    tau      = c_bar(k_g)                                      with its paired jackknife SE
    the reported environments are the k_g with the highest selection frequency in the
    top k_g of the 2B orderings, ties by mean position

WHAT THE NUMBERS ARE, precisely.  `c_bar(k)` is the out-of-fold GxE heritability of the top k
of a greedy ordering, averaged over 2B orderings each scored on individuals they were not grown
on.  It is NOT the held-out value of the consensus set: that set uses every selection half, so
no half is left to score it.  `tau_full = c_bar(L)` is the full-pool fit averaged over the halves.

Averaging before cutting is what distinguishes this from cutting each half's curve and combining
the cuts: a minimum over halves deletes a trait whenever one half's cut is empty by chance.

CALIBRATION.  The level of the tail tests rests on a Gaussian multiplier approximation, so gamma
is set on matched null replicates (the whole decision is a pure function of the curves, so a
replicate needs only those).  Averaged curves of one replicate co-move, so pairing curves across
replicates is not a valid null.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .crossfit import direction
from .evaluator import Evaluator
from .normal_eq import Export
from .stopping import Curve, StopRule, tail_test

GAMMA = 0.05                # level of the sequential tail tests, eq. (cut)
GATE_RULE = tail_test(GAMMA)
Z_GATE = None               # a trait is reported iff k_g > 0 (H_0 rejected); no separate Wald gate


def average_curves(curves: Sequence[Curve]) -> Curve:
    """Mean of the curves at each depth, with the block-wise mean jackknife.

    Every curve must have the same length K + 1 and the same genome jackknife blocks
    (true for halves fitted on the same genotypes), so the block-wise mean is the
    delete-one estimate of the average.
    """
    if not curves:
        raise ValueError("no curves to average")
    K1 = {len(cv.c) for cv in curves}
    if len(K1) != 1:
        raise ValueError(f"curves have different depths: {sorted(K1)}")
    return Curve(c=np.mean([cv.c for cv in curves], axis=0),
                 j=np.mean([cv.j for cv in curves], axis=0))


def consensus(paths: Sequence[Sequence[int]], k: int, how: str = "freq") -> List[int]:
    """The top `k` blocks over several complete orderings.

    `freq`      frequency of being in an ordering's top k; ties by mean full-path position
    `meanrank`  mean of min(position, k + 1); ties by mean full-path position
    Positions are 1-based.  Returns block indices in consensus order.
    """
    paths = [list(map(int, p)) for p in paths]
    K = len(paths[0])
    if any(sorted(p) != list(range(K)) for p in paths):
        raise ValueError("every path must be a complete ordering of the same K blocks")
    k = max(0, min(int(k), K))
    R = np.empty((len(paths), K))
    for i, p in enumerate(paths):
        R[i, p] = np.arange(1, K + 1)
    mean_full = R.mean(axis=0)
    if how == "freq":
        primary = -(R <= k).mean(axis=0)
    elif how == "meanrank":
        primary = np.minimum(R, k + 1).mean(axis=0)
    else:
        raise ValueError("how must be 'freq' or 'meanrank'")
    order = np.lexsort((np.arange(K), mean_full, primary))
    return [int(b) for b in order[:k]]


def select_from_curves(curves: Sequence[Curve], paths: Sequence[Sequence[int]], *,
                       gate_rule: StopRule = GATE_RULE, z_gate: Optional[float] = Z_GATE,
                       rank: str = "freq") -> Dict[str, Any]:
    """The whole decision, as a pure function of the 2B held-out curves and their paths.

    `curves[i]` must be the held-out curve of `paths[i]`.  This is what a calibration
    run calls inside every null or permutation replicate.
    """
    if len(curves) != len(paths):
        raise ValueError("one path per curve")
    M = average_curves(curves)
    K = M.K
    k_g = int(gate_rule(M))
    se = M.se_at(k_g) if k_g > 0 else 0.0
    z = float(M.c[k_g] / se) if se > 0 else float("nan")
    names = bool(k_g > 0) if z_gate is None else bool(k_g > 0 and se > 0 and z >= z_gate)
    blocks = consensus(paths, k_g, rank) if names else []
    return dict(names=names, tau=float(M.c[k_g]) if names else 0.0, se=float(se), z=z,
                k=k_g, blocks=blocks, tau_full=float(M.c[K]), curve=M)


def complementary_split(pairs: Sequence[Tuple[Export, Export]],
                        blocks: Optional[Sequence[Sequence[int]]] = None, *,
                        gate_rule: StopRule = GATE_RULE, z_gate: Optional[float] = Z_GATE,
                        rank: str = "freq",
                        order: str = "dtau",
                        evaluators: Optional[Sequence[Tuple[Evaluator, Evaluator]]] = None
                        ) -> Dict[str, Any]:
    """Run the algorithm on B complementary pairs of exports.

    `pairs[s] = (export_A_s, export_B_s)`, the two halves of the s-th split.  B = 1
    is one split.  `blocks` groups environments into units of selection; the paper uses
    none (the pool is curated so that no two environments are strongly correlated), which
    is the default: every environment is its own block.  `evaluators[s]` optionally reuses memoised evaluators for pair s.

    Returns `select_from_curves` plus the per-direction paths and curves, so any other
    rule can be re-applied without regrowing.
    """
    if not pairs:
        raise ValueError("need at least one complementary pair")
    if blocks is None:                      # v7: each environment is its own unit (K = L); no clustering
        blocks = [[i] for i in range(pairs[0][0].n_env)]
    dirs = []
    for s, (A, B) in enumerate(pairs):
        evA, evB = evaluators[s] if evaluators is not None else (Evaluator(A), Evaluator(B))
        kw = dict(order=order)
        dirs.append(direction(A, B, blocks, ev_grow=evA, ev_eval=evB, **kw))
        dirs.append(direction(B, A, blocks, ev_grow=evB, ev_eval=evA, **kw))
    out = select_from_curves([d.curve for d in dirs], [d.path for d in dirs],
                             gate_rule=gate_rule, z_gate=z_gate, rank=rank)
    out.update(B=len(pairs), directions=dirs)
    return out
