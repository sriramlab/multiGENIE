"""The trait-level test of H_0 (no GxE) by simulation, and multiplicity across traits.

The multiplier bootstrap of `stopping.tail_test` holds the orderings fixed. For one direction (an ordering grown on
one half and scored on the other) that is accurate, but the averaged curve combines orderings grown on individuals who
score other orderings, and the bootstrap p-value of H_0 on the average is anti-conservative (5.8% at 0.05 on 500 null
replicates). EXPOSE therefore tests H_0 against null phenotypes simulated from the trait's own fitted model without GxE:

    y*_i = sum_{m in C} x_im beta_m + e*_i,     e*_i ~ N(0, sigma_i^2 - h2),

where C is a set of `n_causal` SNPs drawn afresh for every null phenotype, x the standardized genotypes, beta ~ N(0, 1)
scaled so that the genetic term has variance h2 (the trait's additive heritability), and sigma_i^2 the trait's residual
variance as a function of its environments (`residual_variance_model`), so that the total variance follows the fitted
pattern. Each null phenotype is analysed exactly as the trait; with p*_1..p*_R the bootstrap p-values of the nulls,

    p_MC = (1 + #{p*_r <= p_0}) / (R + 1)                                                       (`mc_pvalue`)

is a valid p-value whenever H_0 and the fitted null model hold. Traits are then reported at a Benjamini-Hochberg false
discovery rate (`benjamini_hochberg`), valid under positive dependence of the traits' tests. R = 100 nulls, extended
to 300 for traits with fewer than 10 null p-values as small as p_0 (`needs_more_nulls`, Besag & Clifford 1991).

Drawing the causal set afresh per null is essential: with one fixed set per trait the null p-values are too concentrated
and the test is liberal (7.0% at 0.05 end to end; 5.2% with fresh sets).

The genotype pass and the assembly of the null phenotypes' curves live in atom_splits/nulltest/ (they need the atom
tensor of the trait); everything here is numpy only.
"""
from __future__ import annotations
import zlib
from typing import Callable, Sequence
import numpy as np

SEED_BASE = 20261018


def residual_variance_model(y: np.ndarray, E: np.ndarray, iters: int = 25, tol: float = 1e-8):
    """Gamma GLM with log link of y^2 on [1, standardized E] (IRLS): E[y_i^2] = exp(z_i' eta).

    y: the residualized, standardized phenotype (N,); E: the environments (N, L).
    Returns (v, eta): v the fitted variance pattern scaled to mean 1, eta the coefficients."""
    y = np.asarray(y, float); E = np.asarray(E, float)
    sd = E.std(0); Es = (E - E.mean(0)) / np.where(sd > 0, sd, 1.0)
    Z = np.hstack([np.ones((len(y), 1)), Es]); r2 = y ** 2
    eta = np.zeros(Z.shape[1]); eta[0] = np.log(r2.mean()); ZtZi = np.linalg.pinv(Z.T @ Z)
    for _ in range(iters):
        lin = Z @ eta; mu = np.exp(lin); new = ZtZi @ (Z.T @ (lin + (r2 - mu) / mu))
        done = np.max(np.abs(new - eta)) < tol; eta = new
        if done:
            break
    mu = np.exp(Z @ eta)
    return mu / mu.mean(), eta


def null_noise_sd(v: np.ndarray, h2: float, var_total: float = 1.0, floor: float = 0.05) -> np.ndarray:
    """Per-individual noise SD so that genetic (variance h2, flat) + noise has total variance v * var_total.

    The noise variance is v * var_total - h2, floored at `floor` times its mean, then rescaled to mean
    var_total - h2 (the deconvolution: scaling the noise alone by v would dilute the fitted pattern)."""
    v = np.asarray(v, float); mean_noise = max(var_total - h2, 1e-6)
    nv = np.maximum(v * var_total - h2, floor * mean_noise); nv *= mean_noise / nv.mean()
    return np.sqrt(nv)


def _rngs(trait: str, reps: Sequence[int]):
    crc = zlib.crc32(trait.encode())
    return [np.random.default_rng([SEED_BASE, crc, int(r)]) for r in reps]


def null_phenotypes(read_block: Callable[[int, int], np.ndarray], n_snps: int, n: int, h2: float, v: np.ndarray,
                    reps: Sequence[int], trait: str = "", n_causal: int = 5000, snp_block: int = 1024) -> np.ndarray:
    """Null phenotypes (n, len(reps)), float32, drawn from the trait's model without GxE.

    read_block(a, b) -> standardized genotypes of SNPs a..b-1 for the analysed individuals, (n, b - a).
    Replicate r uses numpy default_rng([20261018, crc32(trait), r]): its causal set (sorted choice of n_causal of
    n_snps), then its effects, then its noise, so a replicate is the same whatever batch it is drawn in. All
    replicates share one streaming pass over the genotypes."""
    reps = list(reps); R = len(reps); rngs = _rngs(trait, reps); n_causal = min(n_causal, n_snps)
    sets = [(np.sort(g.choice(n_snps, n_causal, replace=False)), g.standard_normal(n_causal)) for g in rngs]
    G = np.zeros((n, R))
    for a0 in range(0, n_snps, snp_block):
        b0 = min(a0 + snp_block, n_snps); Bb = np.zeros((b0 - a0, R), dtype=np.float32); hit = False
        for c, (idx, beta) in enumerate(sets):
            lo, hi = np.searchsorted(idx, [a0, b0])
            if hi > lo:
                Bb[idx[lo:hi] - a0, c] = beta[lo:hi]; hit = True
        if hit:
            G += np.asarray(read_block(a0, b0), np.float32) @ Bb
    sv = null_noise_sd(v, h2); Y = np.empty((n, R), dtype=np.float32)
    for c, g in enumerate(rngs):
        gr = G[:, c] - G[:, c].mean(); gr = gr * np.sqrt(h2 / gr.var()) if h2 > 0 and gr.var() > 0 else 0.0 * gr
        Y[:, c] = gr + sv * g.standard_normal(n)
    return Y


def mc_pvalue(p0, p_null) -> np.ndarray | float:
    """p_MC = (1 + #{p*_r <= p0}) / (R + 1): p0 (scalar or array) ranked against the null p-values p_null."""
    pn = np.sort(np.asarray(p_null, float)); p0a = np.asarray(p0, float)
    out = (1 + np.searchsorted(pn, p0a, side="right")) / (len(pn) + 1)
    return float(out) if out.ndim == 0 else out


def needs_more_nulls(p0: float, p_null, min_exceed: int = 10) -> bool:
    """Besag-Clifford: extend the null set (100 -> 300) while fewer than `min_exceed` nulls are as small as p0."""
    return int((np.asarray(p_null, float) <= p0).sum()) < min_exceed


def benjamini_hochberg(p) -> np.ndarray:
    """BH q-values (step-up), in the input order."""
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    out = np.empty(m); out[o] = np.minimum(q, 1.0)
    return out
