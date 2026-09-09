"""The solve layer: every number the search asks for, memoised.

Two operations, deliberately separate because they cost different amounts:

    theta(prev, block)      ONE full-sample solve.        The ordering key.
    evaluate(prev, block)   that plus J delete-one solves. The test.

Both are keyed on the SET `prev`, not on the order it was admitted in, because the
fit depends only on which components are in the system.  A greedy path revisits the
same subsets constantly, so the memo is what makes a full-depth traversal cheap.

The J jackknife solves have identical shape, so they go to LAPACK as one batched
call instead of a Python loop.  `tests/test_evaluator.py` asserts that this
reproduces the unbatched result exactly; it is a speedup, not a different statistic.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import norm

from .normal_eq import Export, jk_se, select_indices


def _component_h2_stack(x_l: np.ndarray, y_r: np.ndarray) -> np.ndarray:
    """`component_h2` for a stack of systems: x_l (..., D, D), y_r (..., D)."""
    # b as an explicit column: numpy >= 2 reads a (..., M) rhs as a stack of
    # MATRICES, not of vectors, and would broadcast the wrong way.
    sigma = np.linalg.solve(x_l, y_r[..., None])[..., 0]
    n = x_l.shape[-1] - 1
    adj = sigma.copy()
    adj[..., :n] *= x_l[..., :n, -1]
    adj[..., -1] *= x_l[..., -1, -1]
    total = adj.sum(-1, keepdims=True)
    if np.any(np.abs(total) < 1e-12):
        raise ValueError("near-zero total in a restricted solution")
    return adj[..., :-1] / total


class Evaluator:
    """Memoised access to one export's systems."""

    def __init__(self, export: Export, batched: bool = True):
        self.export = export
        self.batched = bool(batched)
        self._theta: Dict[Tuple, float] = {}
        self._eval: Dict[Tuple, Dict[str, Any]] = {}
        self.n_theta = 0            # counters: the cost side of any ablation
        self.n_eval = 0
        if self.batched:
            self._jx = np.stack([j.x_l for j in export.jackknife])
            self._jy = np.stack([j.y_r for j in export.jackknife])

    @staticmethod
    def _key(prev: Sequence[int], block: Sequence[int]) -> Tuple:
        return (frozenset(prev), tuple(sorted(block)))

    def theta(self, prev: Sequence[int], block: Sequence[int]) -> float:
        """Summed G x E of `block` in the joint fit on `prev + block`."""
        key = self._key(prev, block)
        if key in self._theta:
            return self._theta[key]
        subset = list(prev) + list(block)
        idx = np.asarray(select_indices(self.export, subset), dtype=int)
        f = self.export.full
        comp = _component_h2_stack(f.x_l[np.ix_(idx, idx)], f.y_r[idx])
        val = float(comp[1 + len(prev):1 + len(subset)].sum())
        self._theta[key] = val
        self.n_theta += 1
        return val

    def evaluate(self, prev: Sequence[int], block: Sequence[int]) -> Dict[str, Any]:
        """`theta` plus its jackknife: SE of the SUM, z, one-sided p, cumulative.

        Taking the jackknife SE of the block's SUMMED G x E is what makes a
        multi-environment block one legitimate test: within-block correlation is
        absorbed rather than modelled, and a singleton block reduces exactly to
        the single-environment test.

        `p = 1` whenever the estimate is <= 0.  A variance component is
        non-negative in truth, so a negative estimate is noise, not evidence.
        """
        key = self._key(prev, block)
        if key in self._eval:
            return self._eval[key]

        subset = list(prev) + list(block)
        n_prev, n_all = len(prev), len(subset)
        idx = np.asarray(select_indices(self.export, subset), dtype=int)
        gxe, new = slice(1, 1 + n_all), slice(1 + n_prev, 1 + n_all)

        f = self.export.full
        comp = _component_h2_stack(f.x_l[np.ix_(idx, idx)], f.y_r[idx])
        block_h2, cum_h2 = float(comp[new].sum()), float(comp[gxe].sum())

        if self.batched:
            jc_all = _component_h2_stack(self._jx[:, idx][:, :, idx], self._jy[:, idx])
            jack_block = jc_all[:, new].sum(1).tolist()
            jack_cum = jc_all[:, gxe].sum(1).tolist()
        else:
            jack_block, jack_cum = [], []
            for js in self.export.jackknife:
                c = _component_h2_stack(js.x_l[np.ix_(idx, idx)], js.y_r[idx])
                jack_block.append(float(c[new].sum()))
                jack_cum.append(float(c[gxe].sum()))

        se = jk_se(jack_block)
        if se == 0.0:
            z = np.inf if block_h2 != 0.0 else 0.0
        else:
            z = block_h2 / se
        p = 1.0 if block_h2 <= 0.0 else float(norm.sf(z))

        rec = dict(block_h2=block_h2, block_se=se, z=float(z), p=p,
                   cum_h2=cum_h2, cum_se=jk_se(jack_cum),
                   jack_block=jack_block, jack_cum=jack_cum)
        self._eval[key] = rec
        self.n_eval += 1
        self._theta.setdefault(key, block_h2)   # a gate result pins theta for free
        return rec

    def cumulative(self, envs: Sequence[int]) -> Dict[str, Any]:
        """The joint fit on `envs` alone, which is `evaluate([], envs)`."""
        envs = list(envs)
        if not envs:
            return dict(block_h2=0.0, block_se=0.0, z=0.0, p=1.0,
                        cum_h2=0.0, cum_se=0.0,
                        jack_block=[0.0] * self.export.n_jack,
                        jack_cum=[0.0] * self.export.n_jack)
        return self.evaluate([], envs)

    def joint(self) -> Dict[str, Any]:
        """The unselected fit over every environment in the export."""
        return self.cumulative(range(self.export.n_env))


def curve(ev: Evaluator, path: Sequence[Sequence[int]]) -> tuple:
    """Cumulative h2_GxE over the prefixes of `path`, including the empty one.

    `path` is a sequence of blocks (each a list of environment indices).  Returns
    `(c, j)` with `c[k]` the joint fit on the first `k` blocks and `j[k]` its
    delete-one vector.  Nothing is refitted: every prefix is a solve of the
    exported system.

    The evaluating export need not be the one the path was grown on.  That is
    exactly how the cross-fitted estimator is built: grow on one half, draw this
    curve on the other.
    """
    envs: List[int] = []
    c = [0.0]
    j = [np.zeros(ev.export.n_jack)]
    for block in path:
        for e in block:
            if e not in envs:
                envs.append(int(e))
        r = ev.cumulative(envs)
        c.append(float(r["cum_h2"]))
        j.append(np.asarray(r["jack_cum"], dtype=float))
    return np.asarray(c), np.vstack(j)
