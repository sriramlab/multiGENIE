"""Dense N x N reference for validation V1 (small N only).

`fixed_projector_system` computes, with explicit kernels and the SAME probes, what the
atom assembler is supposed to reproduce for a sub-sample mask h:

    A_r = D_h V K_r V D_h,   T[r,s] = (1/B) sum_p (A_r z_p)'(A_s z_p),   q[r] = (V D_h y~)' K_r (V D_h y~)
    delete-one j: K_r built from the SNPs outside block j, normalised by 1/(M - M_j)
    Fix 2: T[noise,noise] = T[noise,G] = NC_h = sum_{n in h} (1 - h_nn)

`exact_traces` gives tr(A_r A_s) without probes.  `engine_half_system` is the engine's OWN
convention for a half (W, SNP statistics and y~ recomputed on the half's rows) -- what a
standalone the engine run on that half computes, up to its fresh probes -- for validation V3.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .bedio import MISSING, standardize_block
from .model import Design
from .stream import jk_bounds


def standardized_matrix(codes: np.ndarray, snp_global: Optional[np.ndarray] = None) -> np.ndarray:
    """(N, M) float64 standardised genotypes, the engine imputation and statistics."""
    codes = np.array(codes, dtype=np.int8)
    N, M = codes.shape
    if snp_global is None:
        snp_global = np.arange(M)
    return standardize_block(codes, snp_global, np.arange(N), np.float64)


def snp_kernels(X: np.ndarray, design: Design, snps: Optional[np.ndarray] = None,
                norm: Optional[float] = None):
    """The L+1 SNP kernels as N x N matrices from the columns `snps` of X."""
    Xs = X if snps is None else X[:, snps]
    norm = Xs.shape[1] if norm is None else norm
    K = Xs @ Xs.T / norm
    e = design.snp_env()
    return [K * np.outer(e[:, r], e[:, r]) for r in range(design.P_snp)]


def diag_kernels(design: Design):
    d = design.diag_env()
    return [d[:, s].copy() for s in range(design.P_diag)]          # as diagonals


def _apply(design: Design, Ks, ds, mask: np.ndarray, Zin: np.ndarray):
    """A_r z for every kernel: D_h V K_r V (D_h z).  Returns (P, N, B)."""
    Dz = Zin * mask[:, None]
    VDz = design.project(Dz)
    out = []
    for K in Ks:
        out.append(design.project(K @ VDz) * mask[:, None])
    for d in ds:
        out.append(design.project(d[:, None] * VDz) * mask[:, None])
    return np.stack(out)


def _fix2(T: np.ndarray, nc: float, P: int) -> None:
    T[..., P - 1, P - 1] = nc
    T[..., P - 1, 0] = nc
    T[..., 0, P - 1] = nc


def fixed_projector_system(design: Design, X: np.ndarray, mask: np.ndarray, Z: np.ndarray,
                           n_jk: int, fix_noise: bool = True):
    """(T, q, T_j, q_j) for the fixed-projector model restricted to `mask`."""
    mask = np.asarray(mask, dtype=bool)
    N, M = X.shape
    P, Ps = design.P_all, design.P_snp
    B = Z.shape[1]
    bounds = jk_bounds(M, n_jk)
    ds = diag_kernels(design)
    nc = float(np.sum((1.0 - design.lev)[mask]))

    def system(snps, norm):
        Ks = snp_kernels(X, design, snps, norm)
        A = _apply(design, Ks, ds, mask, Z)                        # (P, N, B)
        T = np.einsum("rnb,snb->rs", A, A) / B
        w = design.project(design.ytil * mask)
        q = np.array([w @ K @ w for K in Ks] + [np.sum(d * w * w) for d in ds])
        if fix_noise:
            _fix2(T, nc, P)
        return T, q

    T, q = system(None, M)
    Tj = np.empty((n_jk, P, P))
    qj = np.empty((n_jk, P))
    for j in range(n_jk):
        keep = np.r_[0:bounds[j], bounds[j + 1]:M]
        Tj[j], qj[j] = system(keep, M - (bounds[j + 1] - bounds[j]))
    return T, q, Tj, qj


def exact_traces(design: Design, X: np.ndarray, mask: np.ndarray, fix_noise: bool = True) -> np.ndarray:
    """tr(A_r A_s) with A_r = D_h V K_r V D_h formed explicitly (N small)."""
    mask = np.asarray(mask, dtype=bool)
    Dm = mask.astype(float)
    Vm = np.eye(design.N) - design.W @ design.Q @ design.W.T
    Ks = snp_kernels(X, design)
    ds = diag_kernels(design)
    A = []
    for K in Ks:
        A.append(Dm[:, None] * (Vm @ K @ Vm) * Dm[None, :])
    for d in ds:
        A.append(Dm[:, None] * (Vm @ (d[:, None] * Vm)) * Dm[None, :])
    P = len(A)
    T = np.array([[np.sum(A[r] * A[s].T) for s in range(P)] for r in range(P)])
    if fix_noise:
        _fix2(T, float(np.sum((1.0 - design.lev)[mask])), P)
    return T


def engine_half_system(design: Design, codes: np.ndarray, rows: np.ndarray, Z_half: np.ndarray,
                       n_jk: int):
    """The engine's own half-sample convention: build W, Q, y~ and the SNP statistics on
    `rows` only, then the full-sample system of that restricted design (mask = all)."""
    rows = np.asarray(rows)
    sub = design.restricted(rows, G=1, atom=np.zeros(len(rows), dtype=int))
    Xh = standardized_matrix(codes[rows])
    return fixed_projector_system(sub, Xh, np.ones(len(rows), dtype=bool), Z_half, n_jk)
