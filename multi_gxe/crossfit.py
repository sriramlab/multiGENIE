"""The estimator: grow on one half, score on the other, combine the two directions.

THE PROBLEM IT SOLVES.  A selected set and the number attached to it cannot come
from the same data.  Grown and scored on one sample, the cumulative curve rises well
above the unselected joint fit before coming back down -- under a pure null it peaks
at more than twice the standard error of the joint fit, in every replicate.  Any
rule that cuts such a curve reports its own selection noise.

THE FIX.  Split the individuals into two halves, A and B, respecting family and
ancestry structure so the halves are genuinely independent.  Run the engine once per
half, so each has its own exported system.  Then:

    direction A->B :  choose representatives on A, grow the path on A (full depth,
                      no gate), draw the curve on B, cut it, report B's number
    direction B->A :  the same the other way

Each direction is fully out of fold: everything that touches the selection -- the
representatives, the path, the cut -- comes from one half, and the number comes from
the other.  Combine the two with `combine=`.

THREE THINGS YOU CAN CHANGE, and nothing else has to move:

    choose=    which environment represents a block      multi_gxe/represent.py
    rule=      where to stop along the grown path        multi_gxe/stopping.py
    combine=   what gets reported from the two halves    multi_gxe/combine.py

WHAT YOU GIVE UP.  Two engine fits on half samples instead of one on the whole, and
every estimate is a half-sample estimate.  The two directions may name different
sets; `shared` says how far they agree.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import combine as combiners
from . import represent
from .evaluator import Evaluator, curve
from .grow import grow_path, path_curve
from .normal_eq import Export
from .stopping import Curve, StopRule, plateau, report


@dataclass
class Direction:
    """One direction: everything chosen on one export, the number from the other."""
    k: int
    tau: float
    se: float
    z: float
    blocks: List[int]
    path: List[int]
    tau_full: float
    reps: Optional[List[int]] = None
    curve: Curve = field(repr=False, default=None)


def direction(export_grow: Export, export_eval: Export,
              blocks: Sequence[Sequence[int]], *,
              rule: Optional[StopRule] = None,
              choose=None, corr: Optional[np.ndarray] = None,
              order: str = "dtau",
              ev_grow: Optional[Evaluator] = None,
              ev_eval: Optional[Evaluator] = None) -> Direction:
    """Grow on `export_grow`, cut the curve drawn on `export_eval`.

    `choose` picks one representative environment per block and is evaluated
    HERE, against the growth half, so a data-driven chooser cannot leak the
    evaluating half into the selection.  `None` scores the whole block.

    The grow runs to FULL DEPTH with no stopping test of its own.  All the
    stopping happens on the held-out half, where it is honest.
    """
    rule = rule or plateau()
    evg = ev_grow or Evaluator(export_grow)
    eve = ev_eval or Evaluator(export_eval)

    blocks = [list(b) for b in blocks]
    reps = represent.resolve(
        choose, blocks,
        represent.RepContext(blocks=blocks, corr=corr, ev=evg, export=export_grow))

    grown = grow_path(evg, blocks, order=order, reps=reps)
    path = grown["path"]
    c, j = curve(eve, [blocks[b] for b in path])
    c_in, j_in = path_curve(grown)
    cv = Curve(c=c, j=j, path=np.asarray(path, dtype=int),
               p=np.asarray([r["p"] for r in grown["rounds"]], dtype=float),
               z=np.asarray([r["z"] for r in grown["rounds"]], dtype=float),
               c_in=c_in, j_in=j_in)
    rec = report(cv, rule(cv))
    return Direction(k=rec["k"], tau=rec["tau"], se=rec["se"], z=rec["z"],
                     blocks=rec["blocks"], path=[int(b) for b in path],
                     tau_full=float(c[-1]), reps=reps, curve=cv)


def recut(d: Direction, rule: StopRule) -> Direction:
    """The same grown path, cut by a different rule.  No refitting, no regrowing.

    The grow depends on the chooser and the growth half; the cut depends only on
    the curve the grow produced.  So a calibration sweep over many rules pays for
    ONE traversal per (replicate, chooser) and re-cuts it for the rest -- which is
    the same argument that makes the exhaustive grow affordable in the first place.
    """
    if d.curve is None:
        raise ValueError("this Direction carries no curve to re-cut")
    rec = report(d.curve, rule(d.curve))
    return Direction(k=rec["k"], tau=rec["tau"], se=rec["se"], z=rec["z"],
                     blocks=rec["blocks"], path=d.path, tau_full=d.tau_full,
                     reps=d.reps, curve=d.curve)


def crossfit(export_A: Export, export_B: Export,
             blocks: Sequence[Sequence[int]], *,
             rule: Optional[StopRule] = None,
             choose=None, corr: Optional[np.ndarray] = None,
             combine="min", order: str = "dtau",
             ev_A: Optional[Evaluator] = None,
             ev_B: Optional[Evaluator] = None) -> Dict[str, Any]:
    """Both directions, combined.  This is the algorithm.

    `combine` is a name (`min`, `avg`, `A`, `B`, `agreed`) or any callable
    `(dir_A, dir_B) -> dict`; see `multi_gxe/combine.py`.

    Returns the reported estimate `tau`, its jackknife standard error and z, the
    named blocks, whether the trait was named at all, the per-direction cut sizes,
    how many blocks the two directions agree on, and `tau_full` -- the average
    full-depth out-of-fold total, which is the natural denominator for "how much
    of the available signal did selection keep".

    Pass `ev_A` / `ev_B` to reuse memoised evaluators across configurations.  The
    memo is keyed on the admitted SET, so two rules that visit the same prefixes
    re-solve nothing -- which is what makes sweeping knobs interactive.
    """
    rule = rule or plateau()
    fn = combiners.resolve(combine)
    kw = dict(rule=rule, choose=choose, corr=corr, order=order)
    a = direction(export_A, export_B, blocks, ev_grow=ev_A, ev_eval=ev_B, **kw)
    b = direction(export_B, export_A, blocks, ev_grow=ev_B, ev_eval=ev_A, **kw)

    out = dict(fn(a, b))
    out.update(combine=combiners.name_of(fn), represent=represent.name_of(choose),
               k_A=int(a.k), k_B=int(b.k),
               shared=len(set(a.blocks) & set(b.blocks)),
               tau_A=float(a.tau), tau_B=float(b.tau),
               tau_full=float(0.5 * (a.tau_full + b.tau_full)),
               direction_A=a, direction_B=b)
    return out
