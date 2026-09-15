"""Complementary-split selection: average the held-out curves, then cut once.

The v4 estimator (`crossfit.py`) cuts each direction on its own curve and then
combines the two cuts.  Under `min` a trait disappears whenever ONE half's cut comes
out empty -- bmi on pool87 split 1 went to zero with the two halves at 0.044 +- 0.035
and 0.101 +- 0.030.  This module averages BEFORE cutting instead.

THE ALGORITHM, for B complementary pairs (A_s, B_s):

    for each pair s and each direction (grow on one half, score on the other):
        grow the theta-ordered greedy path to full depth on the selection half
        fit its nested top-k models on the held-out half      -> C(k), k = 0..K
    M(k)  = mean of the 2B curves at each depth k             (jackknife: block-wise mean)
    k_g   = gate_rule(M);  the trait is NAMED iff k_g > 0 and M[k_g] / se >= z_gate
    tau   = M[k_g]                                            the reported total h2_GxE
    k_s   = min(set_rule(M), k_g)
    rank the blocks over the 2B selection halves by frequency in each half's top k_s
    (ties: mean position over the full paths), or by mean truncated rank
    blocks = the top k_s of that consensus                   the named environments
    tau_selected = M[k_s]                                     held-out value of a selection that size

WHAT THE NUMBERS ARE, precisely.  `M(k)` is the out-of-fold cumulative h2_GxE of the
top k of a greedy order, averaged over 2B orders that are each scored on data they were
not grown on.  It is NOT the held-out value of the consensus set: that set uses every
selection half, so no half is left to score it.  `tau_full = M[K]` is the unselected fit
averaged over the halves.

THE DEFAULTS were chosen on cached curves of real engine fits (see
`docs/versions/v5_complementary_split.md`): the gate `plateau(0.75)` at z >= 2.33 names
the most traits and recovers the most signal at a null rate of 1/100; the set rule
`window(3, 1.645)` gives the purest, most reproducible environment sets.

CALIBRATION.  Averaging removes the squaring that made `min` self-calibrating, and
averaged curves of one replicate co-move, so replicate-crossing pairs are NOT a valid
null for this estimator.  Calibrate it on matched null replicates, or by re-running
the whole pipeline inside permutation replicates -- `select_from_curves` makes that a
pure function of the permuted curves.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .crossfit import direction
from .evaluator import Evaluator
from .normal_eq import Export
from .stopping import Curve, StopRule, plateau, window

GATE_RULE = plateau(0.75)
Z_GATE = 2.33
SET_RULE = window(3, 1.645)


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
                       gate_rule: StopRule = GATE_RULE, z_gate: float = Z_GATE,
                       set_rule: StopRule = SET_RULE, rank: str = "freq") -> Dict[str, Any]:
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
    names = bool(k_g > 0 and se > 0 and z >= z_gate)
    k_s = min(int(set_rule(M)), k_g) if names else 0
    blocks = consensus(paths, k_s, rank) if names else []
    return dict(names=names, tau=float(M.c[k_g]) if names else 0.0, se=float(se), z=z,
                k=k_g, blocks=blocks, k_set=k_s,
                tau_selected=float(M.c[k_s]) if names else 0.0,
                tau_full=float(M.c[K]), curve=M)


def complementary_split(pairs: Sequence[Tuple[Export, Export]],
                        blocks: Sequence[Sequence[int]], *,
                        gate_rule: StopRule = GATE_RULE, z_gate: float = Z_GATE,
                        set_rule: StopRule = SET_RULE, rank: str = "freq",
                        choose=None, corr: Optional[np.ndarray] = None,
                        order: str = "theta",
                        evaluators: Optional[Sequence[Tuple[Evaluator, Evaluator]]] = None
                        ) -> Dict[str, Any]:
    """Run the algorithm on B complementary pairs of exports.

    `pairs[s] = (export_A_s, export_B_s)`, the two halves of the s-th split.  B = 1
    is one split.  `evaluators[s]` optionally reuses memoised evaluators for pair s.

    Returns `select_from_curves` plus the per-direction paths, curves and
    representatives, so any other gate or set rule can be re-applied without
    regrowing.
    """
    if not pairs:
        raise ValueError("need at least one complementary pair")
    dirs = []
    for s, (A, B) in enumerate(pairs):
        evA, evB = evaluators[s] if evaluators is not None else (Evaluator(A), Evaluator(B))
        kw = dict(choose=choose, corr=corr, order=order)
        dirs.append(direction(A, B, blocks, ev_grow=evA, ev_eval=evB, **kw))
        dirs.append(direction(B, A, blocks, ev_grow=evB, ev_eval=evA, **kw))
    out = select_from_curves([d.curve for d in dirs], [d.path for d in dirs],
                             gate_rule=gate_rule, z_gate=z_gate, set_rule=set_rule, rank=rank)
    out.update(B=len(pairs), directions=dirs)
    return out
