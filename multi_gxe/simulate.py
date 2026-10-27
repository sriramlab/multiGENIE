"""A synthetic export, so the algorithm runs end to end without a genotype file.

WHAT IT IS.  A normal-equation system in the real schema, built from a chosen truth:
`L` environments with a chosen correlation structure, a chosen per-environment
h2_GxE, and estimation noise with a chosen standard error.  Everything downstream --
clustering, the greedy grow, every stopping rule, the cross-fitted estimator -- runs
on it unchanged, because it only ever sees the system.

WHAT IT IS NOT.  There are no SNPs, no individuals and no engine fit here.  The
estimation error is DRAWN, not earned, and h2_GxE is imposed component by component,
so correlated environments do not split a shared signal the way they do in a real
joint fit.  Use it for mechanics, for tests and for teaching the interface.  Any
number you intend to report belongs to real exports.

The one structural feature it does reproduce faithfully is the one that decides
whether a rule is calibrated: the round statistic is a MAXIMUM (or a minimum
p-value) over live candidates whose errors are correlated exactly as the
environments are, and the pool shrinks by one each round.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .normal_eq import Export, System

N_INDIV = 200_000
N_JACK = 100


def make_corr(block_sizes: Sequence[int] = (3, 2, 1, 1, 1, 1, 1, 1, 1, 1),
              rho: float = 0.95, background: float = 0.05,
              seed: int = 0) -> Tuple[np.ndarray, List[List[int]]]:
    """A correlation matrix with near-duplicate groups, and the true grouping.

    Within a group the correlation is `rho`; across groups it is `background`
    times a random sign.  With `rho` well above the 0.8 clustering cut and
    `background` well below it, `cluster.cluster_from_corr` recovers the groups
    exactly, which is what the tests assert.
    """
    rng = np.random.default_rng(seed)
    L = int(sum(block_sizes))
    corr = np.eye(L)
    blocks, start = [], 0
    for size in block_sizes:
        idx = list(range(start, start + size))
        blocks.append(idx)
        for i in idx:
            for j in idx:
                if i != j:
                    corr[i, j] = rho
        start += size
    for i in range(L):
        for j in range(i + 1, L):
            if corr[i, j] == 0.0:
                v = background * rng.normal()
                corr[i, j] = corr[j, i] = v
    # nearest PSD, cheaply: clip the spectrum and renormalise the diagonal
    w, V = np.linalg.eigh(corr)
    corr = V @ np.diag(np.clip(w, 1e-6, None)) @ V.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    return corr, blocks


def _psd_from_corr(corr: np.ndarray, has_nxe: bool, ridge: float, trace_base: float = 0.0) -> np.ndarray:
    """Cross-component structure for the trace matrix `tr(K_i K_j)`.

    G is taken as weakly correlated with everything; the G x E components inherit
    the environment correlation; each N x E component tracks its own G x E term.

    `trace_base` (2026-10-27) lifts every off-diagonal entry to `trace_base + (1 - trace_base) * v`.
    Genomic kernels share the genome, so tr(K_i K_j) is of the order of tr(K_i) even for unrelated
    environments; with the default 0 a sub-model that omits a causal component pushes the omitted
    variance into the noise term and the retained GxE term downwards, which biases MARGINAL
    (one-environment) fits.  0.25 reproduces the real behaviour; the default keeps the earlier toys.
    """
    L = corr.shape[0]
    n = 1 + L + (L if has_nxe else 0)
    V = np.eye(n)
    V[1:1 + L, 1:1 + L] = corr
    V[0, 1:1 + L] = V[1:1 + L, 0] = 0.10
    if has_nxe:
        V[1 + L:, 1 + L:] = corr
        V[1:1 + L, 1 + L:] = V[1 + L:, 1:1 + L] = 0.30 * corr
        V[0, 1 + L:] = V[1 + L:, 0] = 0.05
    if trace_base:
        off = ~np.eye(n, dtype=bool); V[off] = trace_base + (1.0 - trace_base) * V[off]
    w, U = np.linalg.eigh(V)
    V = U @ np.diag(np.clip(w, 1e-8, None)) @ U.T
    return V + ridge * np.eye(n)


def simulate_export(corr: np.ndarray, h2_gxe: Sequence[float], *,
                    h2_g: float = 0.30, h2_nxe: float = 0.0,
                    se: float = 0.004, n_jack: int = N_JACK,
                    n_indiv: int = N_INDIV, model: str = "G+GxE+NxE",
                    seed: int = 0, env_names: Optional[Sequence[str]] = None,
                    ridge: float = 0.35, trace_base: float = 0.0) -> Export:
    """One export drawn around a truth.

    `h2_gxe` is the per-environment truth, one entry per column of `corr`; `se`
    sets the scale of the estimation error.  The full-sample estimate carries an
    error of that size and the delete-one blocks carry the spread that measures
    it, so the reported SE is neither free nor fabricated at report time.  The
    SE a SUB-MODEL reports is larger, because a subset absorbs the error of the
    correlated components it left out -- that inflation is the real behaviour and
    is why the realised SE of a one-environment fit exceeds `se`.

    `h2_nxe` is 0 by default.  A non-zero N x E truth leaks into whichever
    correlated G x E terms a subset retains, which is faithful but makes the toy
    hard to read; turn it on deliberately.
    """
    corr = np.asarray(corr, dtype=float)
    L = corr.shape[0]
    h2_gxe = np.asarray(h2_gxe, dtype=float)
    if h2_gxe.shape != (L,):
        raise ValueError(f"h2_gxe must have {L} entries; got {h2_gxe.shape}")
    if model not in ("G+GxE", "G+GxE+NxE"):
        raise ValueError(f"unsupported model {model!r}")
    has_nxe = model == "G+GxE+NxE"
    rng = np.random.default_rng(seed)

    sigma = np.concatenate([[h2_g], h2_gxe,
                            (np.full(L, h2_nxe) if has_nxe else np.empty(0))])
    sigma = np.concatenate([sigma, [max(1.0 - sigma.sum(), 0.05)]])
    n = sigma.size - 1

    # trace matrix: tr(K_i) = N for every kernel, so component_h2 is sigma / sum
    V = _psd_from_corr(corr, has_nxe, ridge, trace_base)
    scale = float(n_indiv) * 4.0
    x_l = np.empty((n + 1, n + 1))
    x_l[:n, :n] = scale * V
    x_l[:n, n] = x_l[n, :n] = float(n_indiv)
    x_l[n, n] = float(n_indiv)

    # errors: a common full-sample draw, plus per-block spread that the jackknife
    # measures.  jk_se over J blocks of a spread with sd s / sqrt(J-1) is ~s.
    C = V[:n, :n]
    C = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    Lc = np.linalg.cholesky(C + 1e-9 * np.eye(n))
    err_full = se * (Lc @ rng.standard_normal(n))
    spread = (se / np.sqrt(n_jack - 1)) * (Lc @ rng.standard_normal((n, n_jack))).T

    sigma_full = sigma.copy()
    sigma_full[:n] += err_full
    full = System(x_l=x_l, y_r=x_l @ sigma_full)

    jack = []
    for b in range(n_jack):
        s_b = sigma_full.copy()
        s_b[:n] += spread[b]
        xb = x_l * ((n_jack - 1) / n_jack)
        xb[:n, n] = xb[n, :n] = float(n_indiv) * (n_jack - 1) / n_jack
        jack.append(System(x_l=xb, y_r=xb @ s_b))

    names = list(env_names) if env_names is not None else [f"env_{i}" for i in range(L)]
    return Export(model=model, n_env=L, env_names=names, full=full, jackknife=jack)


def simulate_fold_pair(corr: np.ndarray, h2_gxe: Sequence[float], *,
                       se: float = 0.006, seed: int = 0,
                       **kw) -> Tuple[Export, Export]:
    """Two independent half-sample exports sharing one truth.

    The cross-fitted estimator needs exactly this: same truth, same environments,
    independent estimation error, and the larger standard error a half sample
    buys you.
    """
    a = simulate_export(corr, h2_gxe, se=se, seed=seed, **kw)
    b = simulate_export(corr, h2_gxe, se=se, seed=seed + 10_000, **kw)
    return a, b


def sparse_truth(L: int, tau: float = 0.05, causal: Sequence[int] = (0,)) -> np.ndarray:
    """`tau` on the named environments, zero everywhere else."""
    v = np.zeros(L)
    for i in causal:
        v[i] = tau
    return v
