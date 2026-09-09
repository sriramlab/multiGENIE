"""The estimator: grow on one half, score on the other, combine the two directions.

THE PROBLEM IT SOLVES.  A selected set and the number attached to it cannot come
from the same data.  Grown and scored on one sample, the cumulative curve rises well
above the unselected joint fit before coming back down -- under a pure null it peaks
at more than twice the standard error of the joint fit, in every replicate.  Any
rule that cuts such a curve reports its own selection noise.

THE FIX.  Split the individuals into two halves, A and B, respecting family and
ancestry structure so the halves are genuinely independent.  Run the engine once per
half, so each has its own exported system.  Then:

    direction A->B :  grow the path on A (full depth, no gate)
                      draw the curve on B, cut it, report B's number
    direction B->A :  the same the other way

Each direction is fully out of fold: the set comes from one half, the number from
the other.  Combine by MIN.

WHY MIN.  The minimum of the two estimates requires BOTH halves to name something
before anything is reported, so a rule whose per-direction null rate is `alpha`
gives a combined null rate of `alpha^2`.  That is where the calibration comes from
-- not from a permutation array.  AVG is the alternative and is not the same claim:
it recovers more, but most of the extra comes from traits only one half could see.

WHAT YOU GIVE UP.  Two engine fits on half samples instead of one on the whole, and
every estimate is a half-sample estimate.  The two directions may name different
sets; the reported set is the one belonging to the reported (minimum) estimate, and
`shared` says how much the two agree.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .evaluator import Evaluator, curve
from .grow import grow_path
from .normal_eq import Export
from .stopping import Curve, StopRule, plateau, report


@dataclass
class Direction:
    """One direction: grown on one export, scored on the other."""
    k: int
    tau: float
    se: float
    z: float
    blocks: List[int]
    path: List[int]
    tau_full: float
    curve: Curve = field(repr=False, default=None)


def direction(export_grow: Export, export_eval: Export,
              blocks: Sequence[Sequence[int]], *,
              rule: Optional[StopRule] = None, order: str = "theta",
              reps: Optional[Sequence[int]] = None,
              ev_grow: Optional[Evaluator] = None,
              ev_eval: Optional[Evaluator] = None) -> Direction:
    """Grow on `export_grow`, cut the curve drawn on `export_eval`.

    The grow runs to FULL DEPTH with no stopping test of its own.  All the
    stopping happens on the held-out half, where it is honest.
    """
    rule = rule or plateau()
    evg = ev_grow or Evaluator(export_grow)
    eve = ev_eval or Evaluator(export_eval)

    grown = grow_path(evg, blocks, order=order, reps=reps)
    path = grown["path"]
    c, j = curve(eve, [blocks[b] for b in path])
    cv = Curve(c=c, j=j, path=np.asarray(path, dtype=int),
               p=np.asarray([r["p"] for r in grown["rounds"]], dtype=float),
               z=np.asarray([r["z"] for r in grown["rounds"]], dtype=float),
               c_in=np.asarray([0.0] + [r["cum"] for r in grown["rounds"]], dtype=float))
    rec = report(cv, rule(cv))
    return Direction(k=rec["k"], tau=rec["tau"], se=rec["se"], z=rec["z"],
                     blocks=rec["blocks"], path=[int(b) for b in path],
                     tau_full=float(c[-1]), curve=cv)


def crossfit(export_A: Export, export_B: Export,
             blocks: Sequence[Sequence[int]], *,
             rule: Optional[StopRule] = None, combine: str = "min",
             order: str = "theta",
             reps: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    """Both directions, combined.  This is the algorithm.

    `combine`:
      * `min` -- report the smaller estimate and its set; names something only if
        BOTH directions do, which squares the null rate.  The default.
      * `avg` -- report the average of the two estimates; names something if
        EITHER direction does, so the null rate is not squared.
      * `A` / `B` -- one direction only, for diagnostics.

    Returns the estimate `tau`, its jackknife SE and z, the reported blocks, the
    per-direction cut sizes, how many blocks the two directions agree on, and
    `tau_full`, the average full-depth out-of-fold total, which is the natural
    denominator for "how much of the available signal did selection keep".
    """
    if combine not in ("min", "avg", "A", "B"):
        raise ValueError(f"combine must be min/avg/A/B; got {combine!r}")
    rule = rule or plateau()
    a = direction(export_A, export_B, blocks, rule=rule, order=order, reps=reps)
    b = direction(export_B, export_A, blocks, rule=rule, order=order, reps=reps)

    if combine == "A":
        pick, names = a, a.k > 0
    elif combine == "B":
        pick, names = b, b.k > 0
    elif combine == "min":
        pick = a if a.tau <= b.tau else b
        names = (a.k > 0) and (b.k > 0)
    else:
        pick = a if a.tau <= b.tau else b     # set reporting still needs one path
        names = (a.k > 0) or (b.k > 0)

    tau = 0.5 * (a.tau + b.tau) if combine == "avg" else pick.tau
    se = (0.5 * float(np.hypot(a.se, b.se))) if combine == "avg" else pick.se
    return dict(tau=float(tau), se=float(se),
                z=float(tau / se) if se > 0 else np.nan,
                k=int(pick.k), blocks=list(pick.blocks),
                names=bool(names), combine=combine,
                k_A=int(a.k), k_B=int(b.k),
                shared=len(set(a.blocks) & set(b.blocks)),
                tau_A=float(a.tau), tau_B=float(b.tau),
                tau_full=float(0.5 * (a.tau_full + b.tau_full)),
                direction_A=a, direction_B=b)
