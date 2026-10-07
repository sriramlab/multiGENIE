"""Stage 3 — the cut: how many leading blocks of a grown path to keep.

A stopping rule is a pure function `Curve -> int`.  It may read the curve, its
jackknife, the in-fold round statistics and the path; it may not re-order the path
or refit anything -- that would be changing the grow, not the cut.  Because the grow
is exhaustive, one traversal scores every rule here at every setting.

WHAT MAKES A RULE HONEST.  The curve a rule cuts must come from data the path was
NOT grown on.  Grown and scored on the same half, the cumulative is monotone by
construction, rises well above the unselected joint fit, and comes back down; only
`k = K` is unbiased.  Every rule below is written to be read on an out-of-fold
curve.  See `crossfit.py` for how the two halves are put together.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

from .normal_eq import jk_se


@dataclass
class Curve:
    """What a stopping rule sees.

    c      (K+1,)     out-of-fold cumulative h2_GxE over the prefixes
    j      (K+1, J)   its delete-one jackknife
    path   (K,)       block index admitted at each round
    p      (K,)       in-fold round p-value, a MINIMUM over live candidates
    z      (K,)       in-fold round z
    c_in   (K+1,)     in-fold cumulative, monotone and biased upward
    j_in   (K+1, J)   its jackknife
    """
    c: np.ndarray
    j: np.ndarray
    path: Optional[np.ndarray] = None
    p: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None
    c_in: Optional[np.ndarray] = None
    j_in: Optional[np.ndarray] = None

    @property
    def K(self) -> int:
        return len(self.c) - 1

    def se(self, a: int, b: int) -> float:
        """PAIRED jackknife SE of `c[a] - c[b]`.

        Paired, never the sum of two marginal SEs: the two prefixes are fits on
        the same individuals and their errors are strongly positively correlated,
        so treating them as independent inflates the SE several-fold and stops
        the walk far too early.
        """
        return jk_se(self.j[a] - self.j[b])

    def se_at(self, k: int) -> float:
        return jk_se(self.j[k])

    def in_fold(self) -> "Curve":
        """The same path scored on the half that GREW it.

        Every rule here reads a `Curve`, so handing it this one answers "what
        would this rule report with no split at all" -- which is the comparison
        that prices the split.  Never report from it: the set and the number
        would come from the same individuals.
        """
        if self.c_in is None or self.j_in is None:
            raise ValueError("this Curve carries no in-fold jackknife")
        return Curve(c=self.c_in, j=self.j_in, path=self.path, p=self.p, z=self.z)


StopRule = Callable[[Curve], int]


def plateau(z_thr: float = 1.645) -> StopRule:
    """The MINIMAL prefix that is not significantly below the full-depth value.

    `k* = min { k : (c[K] - c[k]) / se(c[K] - c[k]) < z_thr }`.

    **`z_thr` is a ONE-SIDED normal quantile**, because only a shortfall below the
    plateau is evidence against stopping.  `1.645` is the one-sided 95th percentile,
    the same number as a two-sided 90% interval, and gives a per-direction rate of
    0.05; `1.28` is the one-sided 90th, a two-sided 80% interval, and gives 0.10.
    Under the MIN combiner the reported null rate is the SQUARE of that, so 0.0025
    and 0.010 respectively -- which is why 1.28 is the better operating point here
    and would not be in a single-direction analysis.

    Self-calibrating, and that is the whole point.  Under the null the out-of-fold
    curve is flat at zero, so the test at `k = 0` already fails and the rule
    returns the EMPTY SET.  Combined by MIN over the two fold directions
    (`crossfit.crossfit`), naming anything requires both disjoint halves to name
    something, so the null rate is `alpha(z_thr)^2` with no permutation array and
    no calibration run.  A trait that returns nothing is the rule working.

    Its weakness is measured: the entry test references the whole full-depth
    total, whose SE on a half sample is large, so on a sparse signal the noise
    blocks needed to reach `c[K]` swamp the few that carry it.  Power on the
    sparse simulation arms is 0.13 against 0.99 for the raw gate.
    """
    def rule(cv: Curve) -> int:
        K = cv.K
        for k in range(K + 1):
            s = cv.se(K, k)
            if s <= 0 or (cv.c[K] - cv.c[k]) / s < z_thr:
                return k
        return K
    return rule


def tail_test(gamma: float = 0.01, n_draw: int = 10000, seed: int = 20261005) -> StopRule:
    """THE RULE OF THE PAPER (v7): sequential tail tests on the averaged held-out curve.

    For each depth k = 0, 1, ... test H_k: "no depth past k is higher than k" with the
    statistic `T_k = max_{k' > k} (c[k'] - c[k]) / se(c[k'] - c[k])` (paired jackknife SE),
    reject when `T_k >= q_k(gamma)`, and cut at the first k not rejected:
    `k_g = min { k : T_k < q_k(gamma) }`, `k_g = K` if every H_k is rejected.

    `q_k(gamma)` is the upper-gamma quantile of T_k under the boundary of H_k, drawn by a
    Gaussian multiplier bootstrap over the delete-one-block replicates: with
    `d_j(k, k') = j[k', j] - j[k, j]` centred over blocks and xi ~ N(0, I_J),
    `T*_k = max_{k'} sqrt((J-1)/J) * sum_j xi_j d_j(k, k') / se(k, k')`, each term of unit
    variance given the data.  One draw of xi (fixed seed) serves every k and every curve.

    The tests are nested (H_k true => H_{k+1} true) and run in a fixed order, so the
    family-wise error is at most gamma with no multiplicity correction (Maurer 1995).
    H_0 is the trait-level test: a trait is reported iff k_g > 0.  Because each test
    compares the tail with depth k and not with the full-pool endpoint, the cut does not
    depend on how many uninformative environments the pool carries.  The level is
    calibrated on null simulations (the multiplier approximation is anti-conservative:
    nominal 0.05 realized 0.10 on 50 null replicates; gamma = 0.01 realizes 0.04).
    """
    _xi = {}

    def xi(J: int) -> np.ndarray:
        if J not in _xi:
            _xi[J] = np.random.default_rng(seed).standard_normal((n_draw, J))
        return _xi[J]

    def rule(cv: Curve) -> int:
        c, Jm = np.asarray(cv.c, float), np.asarray(cv.j, float)
        K = len(c) - 1
        J = Jm.shape[1]
        X = xi(J)
        for k in range(K):
            D = Jm[k + 1:] - Jm[k]                                   # (K-k, J) delete-one differences
            se = np.sqrt((J - 1) / J * ((D - D.mean(1, keepdims=True)) ** 2).sum(1))
            dm = c[k + 1:] - c[k]
            ok = se > 0
            if not ok.any():
                return k
            T = np.max(dm[ok] / se[ok])
            W = np.sqrt((J - 1) / J) * (X @ (D[ok] - D[ok].mean(1, keepdims=True)).T)
            if T < np.quantile(np.max(W / se[ok], axis=1), 1 - gamma):
                return k
        return K
    rule.gamma = gamma
    return rule


def one_se(n_se: float = 2.0) -> StopRule:
    """Smallest prefix within `n_se` paired SEs of the curve's MAXIMUM.

    The 1-SE rule of cross-validated model selection, with the curve maximum as
    the reference instead of the endpoint.  That is what fixes the plateau rule's
    weakness: the maximum is where the signal actually is, so the comparison does
    not have to cross the noise blocks in the tail.

    Measured the best real-data recovery at zero false-positive rate on the
    simulation arms (`n_se = 2`); `n_se = 1.5` buys more power at a false-positive
    rate just over 0.05.  `n_se` is the knob.
    """
    def rule(cv: Curve) -> int:
        kmax = int(np.argmax(cv.c))
        for k in range(cv.K + 1):
            if cv.c[k] >= cv.c[kmax] - n_se * cv.se(kmax, k):
                return k
        return kmax
    return rule


def window(m: int = 3, z_thr: float = 1.645) -> StopRule:
    """Stop when the NEXT `m` blocks add nothing significant.

    A local version of `plateau`: the reference is `c[k+m]` rather than `c[K]`, so
    the test is one short increment instead of the whole remaining tail.  Highest
    power of the rules measured at a false-positive rate of zero.
    """
    def rule(cv: Curve) -> int:
        K = cv.K
        for k in range(K + 1):
            hi = min(k + m, K)
            s = cv.se(hi, k)
            if s <= 0 or (cv.c[hi] - cv.c[k]) / s < z_thr:
                return k
        return K
    return rule


def gate(alpha: float = 0.05, patience: int = 2) -> StopRule:
    """The in-fold gate: walk while the round p-value passes, trim the failing tail.

    This is the original rule.  It is the most powerful of the family and the
    least calibrated: the round p is a minimum over live candidates with no
    multiplicity correction, so the per-block false-positive proportion stays
    nominal while the family-wise rate climbs with the number of blocks
    (0.39 at K = 10, 0.60 at K = 25, 0.74 at K = 30).  Kept as the reference
    anyone new is measured against.
    """
    def rule(cv: Curve) -> int:
        if cv.p is None:
            raise ValueError("gate needs the in-fold round p-values")
        consec, last_ok = 0, 0
        for i in range(cv.K):
            ok = cv.p[i] <= alpha
            consec = 0 if ok else consec + 1
            if ok:
                last_ok = i + 1
            if consec >= patience:
                break
        return last_ok
    return rule


def argmax() -> StopRule:
    """The curve's own maximum.  ORACLE and biased upward -- an upper bound, not a rule."""
    def rule(cv: Curve) -> int:
        return int(np.argmax(cv.c))
    return rule


def fixed_k(k: int) -> StopRule:
    def rule(cv: Curve) -> int:
        return min(int(k), cv.K)
    return rule


def nothing() -> StopRule:
    def rule(cv: Curve) -> int:
        return 0
    return rule


def everything() -> StopRule:
    def rule(cv: Curve) -> int:
        return cv.K
    return rule


def report(cv: Curve, k: int) -> dict:
    """What a cut is worth: the estimate, its jackknife SE and z."""
    k = max(0, min(int(k), cv.K))
    tau = float(cv.c[k])
    se = cv.se_at(k) if k > 0 else 0.0
    return dict(k=k, tau=tau, se=se,
                z=(tau / se if se > 0 else np.nan),
                blocks=([] if cv.path is None else [int(b) for b in cv.path[:k]]))


PANEL = {
    "tail test gamma=0.01": tail_test(0.01),
    "plateau z=1.645": plateau(1.645),
    "plateau z=1.28": plateau(1.28),
    "1-SE n_se=2": one_se(2.0),
    "window m=3": window(3, 1.645),
    "gate alpha=0.05": gate(0.05, 2),
    "argmax (oracle)": argmax(),
}
