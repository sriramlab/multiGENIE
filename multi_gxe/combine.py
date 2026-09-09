"""Stage 4b — how the two directions are turned into one reported result.

A combiner is a function `(a, b) -> dict`, where `a` and `b` are the two `Direction`
objects: `a` grown on A and scored on B, `b` the other way.  It decides three
separate things, and they are worth keeping separate in your head:

    names   did this trait produce a result at all
    tau     what number is reported, with its standard error
    blocks  which set is named

The default, `minimum()`, is what makes the procedure self-calibrating: a trait is
named only when BOTH disjoint halves name something, so a per-direction null rate
`alpha` becomes `alpha^2`.  Any combiner that names on EITHER direction gives that
up, and has to earn its null rate some other way.

Write your own and pass it as `combine=`.  It receives everything each direction
knows -- `k`, `tau`, `se`, `blocks`, `path`, `tau_full`, and the whole `curve` -- so a
combiner can re-cut, re-weight, or refuse to report.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

Combiner = Callable[[object, object], Dict]


def _named(fn: Combiner, name: str) -> Combiner:
    fn.combine_name = name          # type: ignore[attr-defined]
    return fn


def name_of(combine) -> str:
    return getattr(combine, "combine_name", "custom")


def _result(pick, tau, se, names) -> Dict:
    return dict(tau=float(tau), se=float(se),
                z=float(tau / se) if se > 0 else np.nan,
                k=int(pick.k), blocks=list(pick.blocks), names=bool(names))


def minimum() -> Combiner:
    """Report the SMALLER of the two estimates, and its set.  Names only if both do.

    The default.  Conservative on purpose: the two halves are disjoint, so
    requiring both to name something squares the null rate, and the smaller of two
    unbiased half-sample estimates is a lower bound on what the trait carries.
    """
    def fn(a, b):
        pick = a if a.tau <= b.tau else b
        return _result(pick, pick.tau, pick.se, (a.k > 0) and (b.k > 0))
    return _named(fn, "min")


def average() -> Combiner:
    """Report the mean of the two estimates.  Names if EITHER direction does.

    Recovers more than `minimum` -- and most of the extra comes from traits only
    one half could see, which is exactly what the minimum is there to drop.  The
    standard error is the half-sum in quadrature, which assumes the two halves are
    independent; they are, by construction of the split.
    """
    def fn(a, b):
        pick = a if a.tau <= b.tau else b            # the set still needs one path
        return _result(pick, 0.5 * (a.tau + b.tau),
                       0.5 * float(np.hypot(a.se, b.se)), (a.k > 0) or (b.k > 0))
    return _named(fn, "avg")


def single(which: str = "A") -> Combiner:
    """One direction only.  A diagnostic: it has the per-direction null rate, not its square."""
    if which not in ("A", "B"):
        raise ValueError("single() takes 'A' or 'B'")

    def fn(a, b):
        pick = a if which == "A" else b
        return _result(pick, pick.tau, pick.se, pick.k > 0)
    return _named(fn, f"single_{which}")


def agreed() -> Combiner:
    """Report only the blocks BOTH directions named, with the minimum's number.

    The old hard intersection, kept because it is the conservative extreme and
    people ask for it.  Measured: it discards most admitted blocks and its
    reported set uses both halves, so the set is not out of fold even though the
    number is.  Prefer `minimum()`.
    """
    def fn(a, b):
        pick = a if a.tau <= b.tau else b
        shared = [x for x in pick.blocks if x in set(a.blocks) & set(b.blocks)]
        out = _result(pick, pick.tau, pick.se, bool(shared))
        out["blocks"] = shared
        out["k"] = len(shared)
        return out
    return _named(fn, "agreed")


BUILT_IN = {
    "min": minimum, "minimum": minimum,
    "avg": average, "average": average, "mean": average,
    "A": lambda: single("A"), "B": lambda: single("B"),
    "agreed": agreed, "intersection": agreed,
}


def resolve(combine) -> Combiner:
    """Accept a name or a callable."""
    if combine is None:
        return minimum()
    if callable(combine):
        return combine
    try:
        return BUILT_IN[combine]()
    except KeyError:
        raise ValueError(f"unknown combiner {combine!r}; "
                         f"known: {sorted(set(BUILT_IN))}, or pass a callable") from None


PANEL = {
    "min (default)": minimum(),
    "avg": average(),
    "A only": single("A"),
    "B only": single("B"),
    "agreed (old intersection)": agreed(),
}
