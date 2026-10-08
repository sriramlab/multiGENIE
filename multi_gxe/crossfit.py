"""One direction of a complementary split: grow on one half, score on the other.

    direction A->B :  grow the path on A (full depth, no stopping test), score every
                      prefix on B -> the held-out curve of A's ordering.

Everything that chooses -- the ordering -- comes from one half, and the number comes
from the other.  `compsplit.complementary_split` runs both directions of every pair and
averages the held-out curves before the cut.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .evaluator import Evaluator, curve
from .grow import grow_path, path_curve
from .normal_eq import Export
from .stopping import Curve, StopRule, tail_test, report


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
    curve: Curve = field(repr=False, default=None)


def direction(export_grow: Export, export_eval: Export,
              blocks: Optional[Sequence[Sequence[int]]] = None, *,
              rule: Optional[StopRule] = None,
              order: str = "dtau",
              ev_grow: Optional[Evaluator] = None,
              ev_eval: Optional[Evaluator] = None) -> Direction:
    """Grow on `export_grow`, cut the curve drawn on `export_eval`.

    `blocks` groups environments into units of selection; by default every environment is
    its own unit.  The grow runs to FULL DEPTH with no stopping test of its own.  All the
    stopping happens on the held-out half, where it is honest.
    """
    rule = rule or tail_test()
    evg = ev_grow or Evaluator(export_grow)
    eve = ev_eval or Evaluator(export_eval)

    blocks = [[i] for i in range(export_grow.n_env)] if blocks is None else [list(b) for b in blocks]
    grown = grow_path(evg, blocks, order=order)
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
                     tau_full=float(c[-1]), curve=cv)
