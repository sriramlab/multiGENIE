"""Common-ordering variant of the complementary split: ONE ordering, then the same cut.

OPTIONAL.  Nothing else in the package imports this module; deleting it (and
`tests/test_commonpath.py`) restores the package exactly.

`complementary_split` averages 2B held-out curves that each run along their OWN greedy
ordering, so `M(k)` is "the value of a top-k", not the value of the set that is reported
(the consensus top k_g), and the cut lands anywhere on a flat curve.  Here the 2B
orderings are merged FIRST and every direction is scored along the merged ordering:

    grow the 2B full-depth paths exactly as `complementary_split` does
    order = consensus(paths, K, rank)          one ordering; mean rank over the 2B paths
    for each direction: V_d(k) = the joint fit of order[:k] on THAT direction's held-out half
    V(k)  = mean of the 2B curves              (jackknife: block-wise mean)
    then `select_from_curves` UNCHANGED: k_g = gate_rule(V), named iff V[k_g]/se >= z_gate,
    tau = V[k_g], blocks = order[:k_g]

So tau IS the held-out value of the reported set, refitted on each evaluating half.

CAVEAT.  The merged ordering has seen every half, so a direction's evaluating half
contributed to the ordering it is scored on.  The growth of each path is still blind to
its own evaluating half, and the cut and the test are unchanged, but the null rate has
to be measured, not assumed (`scripts/stability_selection` in the analysis repo does).

The return value has the keys of `complementary_split`; `directions[i].path` is the
common ordering and `directions[i].curve` the common-ordering curve of direction i, in
the same order (pair s: grown on A scored on B, then grown on B scored on A).  The
original per-direction results are kept under `own_directions`.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .compsplit import GATE_RULE, SET_RULE, Z_GATE, complementary_split, consensus, select_from_curves
from .evaluator import Evaluator, curve
from .normal_eq import Export
from .stopping import Curve, StopRule

RANK = "meanrank"


def common_order(paths: Sequence[Sequence[int]], rank: str = RANK) -> list:
    """The merged full ordering of several complete orderings (ties: block index)."""
    return consensus(paths, len(paths[0]), rank)


def common_path_split(pairs: Sequence[Tuple[Export, Export]], blocks: Sequence[Sequence[int]], *,
                      gate_rule: StopRule = GATE_RULE, z_gate: float = Z_GATE,
                      set_rule: StopRule = SET_RULE, rank: str = "freq", order_rank: str = RANK,
                      evaluators: Optional[Sequence[Tuple[Evaluator, Evaluator]]] = None,
                      **grow_kwargs) -> Dict[str, Any]:
    """`complementary_split` with the 2B curves drawn along one merged ordering.

    Same arguments as `complementary_split`; `order_rank` is how the orderings are merged.
    """
    if evaluators is None:
        evaluators = [(Evaluator(A), Evaluator(B)) for A, B in pairs]
    own = complementary_split(pairs, blocks, gate_rule=gate_rule, z_gate=z_gate, set_rule=set_rule,
                              rank=rank, evaluators=evaluators, **grow_kwargs)
    blocks = [list(b) for b in blocks]
    order = common_order([d.path for d in own["directions"]], order_rank)
    evals = [ev for evA, evB in evaluators for ev in (evB, evA)]      # direction i is scored on evals[i]
    dirs = []
    for d, ev in zip(own["directions"], evals):
        c, j = curve(ev, [blocks[b] for b in order])
        dirs.append(replace(d, path=[int(b) for b in order], tau_full=float(c[-1]),
                            curve=Curve(c=c, j=j, path=np.asarray(order, dtype=int))))
    out = select_from_curves([d.curve for d in dirs], [d.path for d in dirs],
                             gate_rule=gate_rule, z_gate=z_gate, set_rule=set_rule, rank=rank)
    out.update(B=len(pairs), directions=dirs, order=[int(b) for b in order], own_directions=own["directions"],
               own=dict((k, own[k]) for k in ("names", "tau", "se", "z", "k", "blocks", "tau_full")))
    return out
