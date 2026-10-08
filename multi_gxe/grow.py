"""Stage 2 — grow a nested path over the blocks, to full depth, with no stopping test.

Each round scores every block that is not yet admitted, jointly with everything that
is, and admits the winner.  It runs to exhaustion: the path is a permutation of the
blocks, and `path[:k]` is a nested sequence of sub-models.

**The grow is exhaustive so that the cut is free.**  One traversal scores every
stopping rule, at any threshold, forever.  That is what makes rule design cheap
enough to do honestly: the expensive object is the engine fit, the traversal is a
few hundred solves of a small system, and comparing twenty stopping rules over it
costs microseconds.

ORDERING KEY.  `theta` admits the block with the largest summed G x E; `p` admits the
smallest one-sided p-value.  They disagree on few traits, but `theta` needs one solve
per candidate where `p` needs the whole jackknife, which is a ~24x cost difference and
also a reproducibility win: the p ordering re-ranks on a statistic whose denominator
is itself noisy, so two runs of the same fit can order the tail differently.
`theta` is the default for that reason, not because it selects better.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .evaluator import Evaluator

ORDERINGS = ("dtau", "theta", "p")   # dtau = gain tau(S+l) - tau(S) (Methods eq:greedy, 2026-10-06 default)


def envs_of(blocks: Sequence[Sequence[int]], path: Sequence[int]) -> List[int]:
    """The environments a prefix of `path` fits over, in admission order."""
    out: List[int] = []
    for b in path:
        for e in blocks[b]:
            if e not in out:
                out.append(int(e))
    return out


def grow_path(ev: Evaluator, blocks: Sequence[Sequence[int]], *,
              order: str = "dtau",
              max_rounds: Optional[int] = None) -> Dict[str, Any]:
    """Full-depth greedy traversal.  Returns the path and one record per round.

    Each round record carries `block_h2`, `block_se`, `z`, `p`, the cumulative
    `cum`/`cum_se`, the delete-one vector `jack_cum`, and `n_cand`, the number of
    candidates the round chose between.  **`p` is the minimum over `n_cand` live
    candidates and so is not U(0,1)** -- any rule that treats it as one test is
    anti-conservative by roughly that factor.
    """
    if order not in ORDERINGS:
        raise ValueError(f"order must be one of {ORDERINGS}; got {order!r}")
    blocks = [list(b) for b in blocks]
    def scored(b: int) -> List[int]:
        return blocks[b]

    live = list(range(len(blocks)))
    limit = len(blocks) if max_rounds is None else min(int(max_rounds), len(blocks))
    path: List[int] = []
    rounds: List[Dict[str, Any]] = []

    while live and len(path) < limit:
        prev = envs_of(blocks, path)
        if order == "dtau":
            base = ev.tau(prev)
            key = {b: ev.tau(prev + scored(b)) - base for b in live}
            win = max(live, key=lambda b: (key[b], -b))
            rec = ev.evaluate(prev, scored(win))
        elif order == "theta":
            key = {b: ev.theta(prev, scored(b)) for b in live}
            win = max(live, key=lambda b: (key[b], -b))
            rec = ev.evaluate(prev, scored(win))
        else:
            evals = {b: ev.evaluate(prev, scored(b)) for b in live}
            win = min(live, key=lambda b: (evals[b]["p"], -evals[b]["block_h2"]))
            rec = evals[win]
        cum = ev.cumulative(envs_of(blocks, path + [win]))
        rounds.append(dict(block=int(win), envs=list(blocks[win]),
                           block_h2=rec["block_h2"], block_se=rec["block_se"],
                           z=rec["z"], p=rec["p"],
                           cum=cum["cum_h2"], cum_se=cum["cum_se"],
                           jack_cum=list(cum["jack_cum"]), n_cand=len(live)))
        path.append(win)
        live.remove(win)

    return dict(path=path, rounds=rounds, blocks=blocks, order=order,
                n_block=len(blocks), n_env=ev.export.n_env,
                n_theta_solve=ev.n_theta, n_gate_eval=ev.n_eval)


def path_curve(result: Dict[str, Any]) -> tuple:
    """`(c, j)` along a grown path, read straight off the round records.

    This is the IN-FOLD curve: the same data grew the path and drew the curve, so
    it is monotone by construction and biased upward.  For an honest number draw
    the curve on data the path did not see -- `evaluator.curve` on the other fold.
    """
    c = [0.0] + [r["cum"] for r in result["rounds"]]
    j = [np.zeros(len(result["rounds"][0]["jack_cum"]))] if result["rounds"] else [np.zeros(1)]
    j += [np.asarray(r["jack_cum"], dtype=float) for r in result["rounds"]]
    return np.asarray(c), np.vstack(j)
