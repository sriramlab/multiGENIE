"""Core set and diffuse remainder: what of a complementary-split result is reproducible.

OPTIONAL and ADDITIVE.  Nothing in the package imports this module and it changes no
other file: `complementary_split` and `select_from_curves` return exactly what they did.
To roll back, delete this file and `tests/test_coreset.py`.

WHY.  On native 47-environment fits over six independent partitions (whr, creatinine,
testosterone) the trait-level number reproduces and the named SET does not: the plateau
depth k_g lands at 1-3 or at 19-30 depending on the split, and as B grows the averaged
curve keeps rising so k_g goes to K.  The signal is one reproducible leading cluster plus
a tail spread over most clusters; the tail's SIZE is estimable (about 0.03, z ~ 2.5 pooled)
and its MEMBERSHIP is not (top-5 overlap 0.3 between partitions).  So the report is split
into the part that can be named and the part that can only be measured:

    f_C(q)     share of the 2B orderings with cluster C among their first q
    core       { C : f_C(q) >= pi }                  q FIXED BY DESIGN, not read off a curve
    tau_total  M[K]            the unselected fit averaged over the held-out halves
    tau_core   held-out value of the core, averaged over the 2B evaluating halves
    remainder  tau_total - tau_core, with the PAIRED jackknife SE

This is Shah & Samworth (2013) complementary-pairs stability selection with "the first q
of a forward path" as the base selector (Meinshausen & Buhlmann 2010).  With pi = 1 the
core is the intersection of the 2B top-q sets; under the null the two directions of a
split order like independent permutations, so the expected core size is about q^2 / K
at B = 1 and falls geometrically in B.

`tau_total` is M[K], not M[k_g]: it involves no selection at all, so it cannot move with
the cut.  The trait-level decision (is the trait named, tau = M[k_g]) stays with
`select_from_curves` and is returned untouched under `base`.

HONESTY OF `tau_core`.  The core uses every selection half, so no half is fully held out
from it (the same caveat as `compsplit`'s consensus set).  When the core is a common
prefix of every ordering -- the usual case: one leading cluster first everywhere -- then
tau_core = M[|core|] exactly, each term of which IS out-of-fold.  `prefix` says whether
that holds.  Otherwise the core is scored on each evaluating half directly
(`core_values`), which conditions on that half having ranked the core in its own top q:
a mild upward bias that vanishes for clusters selected with probability near one.

THE SE is the genome block jackknife of the B-averaged curve.  It does not include the
between-split variance; at B > 1 compare it with the spread over splits.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .compsplit import average_curves, complementary_split
from .evaluator import Evaluator
from .normal_eq import Export, jk_se
from .stopping import Curve

Q = 3
PI = 1.0


def topq_frequency(paths: Sequence[Sequence[int]], q: int) -> np.ndarray:
    """f_C(q): the share of the orderings that have block C among their first `q`."""
    paths = [list(map(int, p)) for p in paths]
    if not paths:
        raise ValueError("no paths")
    K = len(paths[0])
    if any(sorted(p) != list(range(K)) for p in paths):
        raise ValueError("every path must be a complete ordering of the same K blocks")
    q = max(0, min(int(q), K))
    f = np.zeros(K)
    for p in paths:
        f[p[:q]] += 1
    return f / len(paths)


def core_set(paths: Sequence[Sequence[int]], q: int = Q, pi: float = PI) -> List[int]:
    """{C : f_C(q) >= pi}, ordered by frequency, ties by mean position over the full paths."""
    f = topq_frequency(paths, q)
    K = len(f)
    R = np.empty((len(paths), K))
    for i, p in enumerate(paths):
        R[i, list(map(int, p))] = np.arange(1, K + 1)
    order = np.lexsort((np.arange(K), R.mean(axis=0), -f))
    return [int(b) for b in order if f[b] >= pi - 1e-9]


def is_common_prefix(paths: Sequence[Sequence[int]], core: Sequence[int]) -> bool:
    """True iff the first |core| blocks of EVERY ordering are exactly the core (in any order)."""
    m, want = len(core), set(map(int, core))
    return all(set(map(int, p[:m])) == want for p in paths)


def core_report(curves: Sequence[Curve], paths: Sequence[Sequence[int]], *,
                q: int = Q, pi: float = PI,
                core_values: Optional[Sequence[Tuple[float, np.ndarray]]] = None
                ) -> Dict[str, Any]:
    """Core, total and remainder as a pure function of the 2B held-out curves and paths.

    `core_values[i] = (value, jackknife vector)` is the core scored on the evaluating half
    of direction i (`core_split` supplies it).  Without it the report still gives the core
    and the total, and gives tau_core / remainder only when the core is a common prefix.
    """
    if len(curves) != len(paths):
        raise ValueError("one path per curve")
    M = average_curves(curves)
    K = M.K
    f = topq_frequency(paths, q)
    core = core_set(paths, q, pi)
    m = len(core)
    prefix = is_common_prefix(paths, core)

    if core_values is not None:
        if len(core_values) != len(curves):
            raise ValueError("one core value per direction")
        tau_core = float(np.mean([v for v, _ in core_values]))
        j_core = np.mean([np.asarray(j, dtype=float) for _, j in core_values], axis=0)
    elif prefix:
        tau_core, j_core = float(M.c[m]), M.j[m]
    else:
        tau_core, j_core = float("nan"), None

    out = dict(q=int(q), pi=float(pi), n_orderings=len(paths), core=core, f=f,
               prefix=bool(prefix), tau_total=float(M.c[K]), se_total=float(M.se_at(K)),
               tau_core=tau_core, se_core=float("nan"), remainder=float("nan"),
               se_remainder=float("nan"), z_remainder=float("nan"), curve=M)
    if j_core is not None:
        se_r = float(jk_se(M.j[K] - j_core))
        rem = float(M.c[K]) - tau_core
        out.update(se_core=float(jk_se(j_core)) if m else 0.0, remainder=rem, se_remainder=se_r,
                   z_remainder=rem / se_r if se_r > 0 else float("nan"))
    return out


def score_core(ev: Evaluator, blocks: Sequence[Sequence[int]], core: Sequence[int]
               ) -> Tuple[float, np.ndarray]:
    """The joint fit of the core's environments on one evaluating half: (value, jackknife)."""
    envs: List[int] = []
    for b in core:
        for e in blocks[int(b)]:
            if int(e) not in envs:
                envs.append(int(e))
    r = ev.cumulative(envs)
    return float(r["cum_h2"]), np.asarray(r["jack_cum"], dtype=float)


def core_split(pairs: Sequence[Tuple[Export, Export]], blocks: Sequence[Sequence[int]], *,
               q: int = Q, pi: float = PI,
               evaluators: Optional[Sequence[Tuple[Evaluator, Evaluator]]] = None,
               **compsplit_kwargs) -> Dict[str, Any]:
    """`complementary_split` plus the core report, the core scored on every held-out half.

    `compsplit_kwargs` go to `complementary_split` unchanged and its result is returned
    under `base`, so the paper's decision and this report come from the same grown paths.
    """
    blocks = [list(b) for b in blocks]
    if evaluators is None:
        evaluators = [(Evaluator(A), Evaluator(B)) for A, B in pairs]
    base = complementary_split(pairs, blocks, evaluators=evaluators, **compsplit_kwargs)
    dirs = base["directions"]
    paths = [d.path for d in dirs]
    core = core_set(paths, q, pi)
    # directions come in the order (grow A_s, score B_s), (grow B_s, score A_s)
    ev_eval = [ev for evA, evB in evaluators for ev in (evB, evA)]
    values = [score_core(ev, blocks, core) for ev in ev_eval]
    out = core_report([d.curve for d in dirs], paths, q=q, pi=pi, core_values=values)
    # the trait-level decision is the unchanged rule's: no core is REPORTED for an unnamed trait
    out.update(B=len(pairs), base=base, named=bool(base["names"]),
               reported_core=list(out["core"]) if base["names"] else [])
    return out
