"""Stage 1b — pick ONE candidate environment per block, before the path is grown.

A block of near-duplicate environments has to be scored somehow.  Two choices, and
the difference is a modelling decision, not a detail:

* **score the whole block** (`none()`, the reference).  The round statistic is the
  block's SUMMED G x E with a jackknife of the sum, so within-block correlation is
  absorbed.  Nothing is thrown away.
* **score one representative** (everything else here).  The round statistic is a
  single environment's G x E.  The block is still FITTED whole once admitted, so the
  reported number covers every member; only the ranking is done on one column.

A chooser is a function `(blocks, ctx) -> list[int] | None`, one representative per
block, each a member of its own block.  Write your own and pass it as `choose=`.

**THE ONE RULE: choose on the growth half only.**  `ctx.ev` is the growing fold's
evaluator and `ctx.export` its system.  A chooser that peeks at the evaluating half
puts the held-out data into the selection, and the estimate stops being out of fold.
`crossfit` calls the chooser once per direction for exactly this reason, so a
data-driven chooser sees A when growing on A and B when growing on B.

Measured: on the real pool a Gram-matrix representative induces the same partition as
the correlation one at K = 30, and the selected sets do not move.  That is why the
default is `none()` -- but it is a knob, and this module is where you turn it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np


@dataclass
class RepContext:
    """What a chooser may look at."""
    blocks: List[List[int]]
    corr: Optional[np.ndarray] = None      # the environment correlation matrix
    ev: Optional[object] = None            # Evaluator on the GROWTH half
    export: Optional[object] = None        # its Export


Chooser = Callable[[Sequence[Sequence[int]], RepContext], Optional[List[int]]]


def _named(fn: Chooser, name: str) -> Chooser:
    fn.rep_name = name          # type: ignore[attr-defined]
    return fn


def name_of(choose: Optional[Chooser]) -> str:
    return getattr(choose, "rep_name", "custom") if choose is not None else "none"


def none() -> Chooser:
    """Score the whole block.  The reference behaviour, and the default."""
    return _named(lambda blocks, ctx: None, "none")


def first() -> Chooser:
    """The lowest-numbered member.  Deterministic and blind, useful as a control."""
    return _named(lambda blocks, ctx: [int(min(b)) for b in blocks], "first")


def centroid() -> Chooser:
    """The member most correlated with its own block -- a medoid on `abs(r)`.

    Needs `ctx.corr`.  This is a property of the design only: it never sees the
    phenotype, so it can be frozen with the partition and reused everywhere.
    """
    def choose(blocks, ctx):
        if ctx.corr is None:
            raise ValueError("centroid() needs the environment correlation matrix")
        a = np.abs(np.asarray(ctx.corr, dtype=float))
        out = []
        for b in blocks:
            b = list(b)
            out.append(int(b[0]) if len(b) == 1
                       else int(b[int(np.argmax(a[np.ix_(b, b)].sum(axis=1)))]))
        return out
    return _named(choose, "centroid")


def marginal() -> Chooser:
    """The member with the largest MARGINAL h2_GxE on the growth half.

    One solve per environment in a multi-environment block, and none for a
    singleton.  This one is data-driven, so it is the case the growth-half rule
    exists for: it must never be computed on the half the curve is drawn on.

    Marginal means fitted alone, not conditional on anything.  Two near-duplicates
    have near-identical marginals by construction, so the choice between them is
    close to a coin flip -- which is the reason the block, not the environment, is
    the reporting unit in the first place.
    """
    def choose(blocks, ctx):
        if ctx.ev is None:
            raise ValueError("marginal() needs the growth-half Evaluator")
        out = []
        for b in blocks:
            b = list(b)
            if len(b) == 1:
                out.append(int(b[0]))
                continue
            out.append(int(b[int(np.argmax([ctx.ev.theta([], [e]) for e in b]))]))
        return out
    return _named(choose, "marginal")


def fixed(reps: Sequence[int]) -> Chooser:
    """Representatives you picked yourself, in block order."""
    reps = [int(r) for r in reps]
    return _named(lambda blocks, ctx: list(reps), "fixed")


def resolve(choose, blocks, ctx) -> Optional[List[int]]:
    """Run a chooser and check its answer belongs to the blocks it was given."""
    if choose is None:
        return None
    reps = choose(blocks, ctx)
    if reps is None:
        return None
    reps = [int(r) for r in reps]
    if len(reps) != len(blocks):
        raise ValueError(f"chooser returned {len(reps)} representatives "
                         f"for {len(blocks)} blocks")
    for r, b in zip(reps, blocks):
        if r not in list(b):
            raise ValueError(f"representative {r} is not a member of block {list(b)}")
    return reps


PANEL = {
    "none (whole block)": none(),
    "centroid": centroid(),
    "marginal (growth half)": marginal(),
    "first": first(),
}
