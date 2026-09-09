"""Stage 1 — group the environments into blocks, from `E` alone.

WHY THE BLOCK IS THE UNIT.  The seed instability of environment-by-environment
forward selection is not estimation noise, it is a tie-break inside a set of
near-duplicates.  Where three columns sit at |r| >= 0.94 with indistinguishable
marginal signal, whichever wins round 1 buries the other two, and two runs of the
same fit share nothing.  Admitting the whole group removes the coin flip.  It also
fixes the yardstick: per-environment false-discovery rates charge a rule for naming
an r = 0.98 copy of a causal environment, which is an artefact of the pool, not an
error of the rule.

The partition is computed from the environment matrix only.  It never sees the
phenotype, so it is a property of the design and can be frozen on disk and reused
across every trait and every replicate.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

CUT = 0.8            # merge while average within-block |r| >= CUT
LINKAGE = "average"


def cluster_from_corr(corr: np.ndarray, cut: float = CUT,
                      method: str = LINKAGE) -> List[List[int]]:
    """Blocks of environment indices, average linkage on `1 - |r|` cut at `1 - cut`.

    Average linkage means `cut` is an average within-block |r|, not a minimum.
    Returns a deterministic order: blocks sorted by their first member.
    """
    a = np.abs(np.asarray(corr, dtype=float)).copy()
    np.fill_diagonal(a, 1.0)
    a = np.clip(a, 0.0, 1.0)
    if a.shape[0] < 2:
        return [[0]] if a.shape[0] == 1 else []
    d = squareform(np.clip(1.0 - a, 0.0, None), checks=False)
    labels = fcluster(linkage(d, method=method), t=1.0 - cut, criterion="distance")
    groups = [sorted(np.nonzero(labels == c)[0].tolist())
              for c in sorted(set(labels.tolist()))]
    return sorted(groups, key=lambda g: g[0])


def pairwise_corr(x: np.ndarray) -> np.ndarray:
    """Pairwise-complete correlation of the columns of `x`, NaN = missing."""
    x = np.asarray(x, dtype=float)
    m = x.shape[1]
    out = np.eye(m)
    for i in range(m):
        for j in range(i + 1, m):
            ok = np.isfinite(x[:, i]) & np.isfinite(x[:, j])
            if ok.sum() > 2:
                a, b = x[ok, i], x[ok, j]
                sa, sb = a.std(), b.std()
                out[i, j] = out[j, i] = (0.0 if sa == 0 or sb == 0 else
                                         float(((a - a.mean()) * (b - b.mean())).mean()
                                               / (sa * sb)))
    return out


def corr_from_env_file(path: str | Path,
                       names: Sequence[str] | None = None) -> np.ndarray:
    """Correlation matrix from a whitespace `.env` file (FID IID then columns).

    Missing is coded `-9` and is masked PAIRWISE.  Leaving it in is a real trap:
    the missingness pattern is shared across columns, so every pair inherits it
    and the whole matrix reads as strongly correlated.
    """
    with open(path) as fh:
        cols = fh.readline().split()[2:]
    x = np.loadtxt(path, skiprows=1, usecols=range(2, 2 + len(cols)))
    x[x == -9] = np.nan
    if names is not None:
        x = x[:, [cols.index(n) for n in names]]
    return pairwise_corr(x)


def representatives(corr: np.ndarray, blocks: Sequence[Sequence[int]]) -> List[int]:
    """One environment per block: the member most correlated with its own block.

    Used only where a block is SCORED by a single column; the fit still carries
    every member.  Measured inert on the real pool (the partition and the selected
    sets do not move), so it is off by default.
    """
    a = np.abs(np.asarray(corr, dtype=float))
    out = []
    for b in blocks:
        b = list(b)
        if len(b) == 1:
            out.append(int(b[0]))
            continue
        sub = a[np.ix_(b, b)].sum(axis=1)
        out.append(int(b[int(np.argmax(sub))]))
    return out
