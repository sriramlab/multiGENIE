"""The cut: how many leading environments of a grown ordering to keep.

A stopping rule is a pure function `Curve -> int`.  It may read the held-out cumulative curve,
its delete-one jackknife and the ordering; it may not re-order or refit anything, which would be
changing the selection rather than the cut.  The rule of the method is `tail_test`.

WHAT MAKES A CUT HONEST.  The curve a rule cuts must come from individuals the ordering was NOT
grown on.  Grown and scored on the same half the cumulative is monotone by construction, rises
above the full-pool fit and comes back down; only `k = L` is unbiased there.  See `crossfit.py`
and `compsplit.py` for how the two halves are put together.
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


def tail_test(gamma: float = 0.05, n_draw: int = 10000, seed: int = 20261005) -> StopRule:
    """THE RULE OF THE PAPER: sequential tail tests on the averaged held-out curve.

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
    H_0 is the trait-level test: a trait is reported iff k_g > 0.  The guarantee concerns the
    depth at which the AVERAGED curve stops rising, not the number of causal environments: when
    orderings disagree on which environment comes next, the averaged curve can rise past the last
    causal one.  Because each test compares the tail with depth k and not with the full-pool
    endpoint, the cut does not depend on how many uninformative environments the pool carries.

    Calibration (500 null replicates, B = 50 pairs): realized level 5.8% at gamma = 0.05 (95%
    Wilson interval 4.1-8.2%), the default; the multiplier approximation is anti-conservative
    further in the tail (2.8% at gamma = 0.01), where the empirical / bootstrap quantile ratio is
    1.08-1.14 instead of 1.05.  With the causal environments forced first (partial nulls H_1-H_10,
    370 replicates) the rejection rate was 4.6% at gamma = 0.05.
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


def cut_depth(c, j, gamma: float = 0.05) -> int:
    """`tail_test(gamma)` applied to a curve given as raw arrays: `c` (K+1,) and its delete-one
    jackknife `j` (K+1, J).  The convenience form used by scripts that hold saved curves."""
    return int(tail_test(gamma)(Curve(c=np.asarray(c, float), j=np.asarray(j, float))))


def tail_pvalue(c, j, k: int = 0, n_draw: int = 10000, seed: int = 20261005) -> float:
    """Bootstrap p-value of the tail test at depth k (k = 0: the trait-level test of GxE),
    `p_k = Pr*(T*_k >= T_k)` from the same multiplier draws as `tail_test`, floored at 1 / n_draw.
    At k = 0 this is the trait's statistic for the test of GxE: rank it against the same statistic
    on null phenotypes (`nulltest.mc_pvalue`) before multiplicity control across traits
    (`nulltest.benjamini_hochberg`); on its own it inherits the anti-conservatism of the
    multiplier bootstrap on the averaged curve."""
    c, Jm = np.asarray(c, float), np.asarray(j, float)
    Jn = Jm.shape[1]
    X = np.random.default_rng(seed).standard_normal((n_draw, Jn))
    D = Jm[k + 1:] - Jm[k]
    se = np.sqrt((Jn - 1) / Jn * ((D - D.mean(1, keepdims=True)) ** 2).sum(1))
    ok = se > 0
    if not ok.any():
        return 1.0
    T = np.max((c[k + 1:] - c[k])[ok] / se[ok])
    W = np.sqrt((Jn - 1) / Jn) * (X @ (D[ok] - D[ok].mean(1, keepdims=True)).T)
    M = np.max(W / se[ok], axis=1)
    return max(float((M >= T).mean()), 1.0 / n_draw)


def report(cv: Curve, k: int) -> dict:
    """What a cut is worth: the estimate, its jackknife SE and z."""
    k = max(0, min(int(k), cv.K))
    tau = float(cv.c[k])
    se = cv.se_at(k) if k > 0 else 0.0
    return dict(k=k, tau=tau, se=se,
                z=(tau / se if se > 0 else np.nan),
                blocks=([] if cv.path is None else [int(b) for b in cv.path[:k]]))
