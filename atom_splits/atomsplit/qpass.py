"""Phenotype terms ONLY (qg_full, qg_blk, qdiag) for P phenotypes in ONE genotype pass.

`atom_pass` builds the whole tensor; a phenotype that shares a tensor's rows, covariates, environments and atoms
needs only its q terms, which has so far been obtained by running the full two-pass `atom_pass` with one
dummy probe and discarding the rest (pass 2, the probe columns of pass 1 and S_full are wasted work).  Here, per SNP
block j and phenotype p,

    S_jp   = X_j' Cin_p,   Cin_p = [ e_r o V (D_g y~_p) ]_{r, g}            (N x Ps G)
    qg_blk[p, j, r, g, h] = sum_{m in block j} S_jp[m, r, g] S_jp[m, r, h]
    qg_full[p] = sum_j qg_blk[p, j],      qdiag[p, s, g, h] = sum_n d_s(n) v_pg(n) v_ph(n),  v_pg = V D_g y~_p

with y~_p exactly `Design._prepare`'s standardised residual of phenotype p.  The arithmetic per phenotype is that of
the y~ column of `atom_pass` (same projector, same cast to `dtype`, same SNP standardisation via bedio), so for
P = 1 the result equals `atom_pass(...).q*` up to floating-point summation order.  The genotype read and
standardisation are shared by all P phenotypes; the GEMM grows linearly in P.

method="factored" (default) uses V = I - W Q W' to split the GEMM:
    X'(e_r o V D_g y~) = X'(e_r o D_g y~)  -  [X'(e_r o W)] beta_g,      beta_g = Q W' D_g y~
The first term touches only atom g's rows, so all G atoms together cost ONE N-row product per kernel instead of G;
the second term's X'(e_r o W) is phenotype-free and shared by the whole batch (its cost equals n_cov phenotype
columns per kernel, so batches of >= ~n_cov phenotypes amortise it).  method="direct" forms Cin_p explicitly as
atom_pass does.  Both give the same q up to floating-point rounding.

`snp_kernels` (optional) restricts the SNP kernels r to a subset (e.g. the additive kernel plus the GxE kernels of
an analysed pool); the other rows of qg are filled with NaN so that any accidental use downstream fails loudly.
"""
from __future__ import annotations

import time
from typing import Callable, Optional, Sequence

import numpy as np

from .bedio import standardize_block
from .model import Design
from .stream import jk_bounds, _fmt


def standardized_residuals(design: Design, Y: np.ndarray) -> np.ndarray:
    """(N, P) y~ for each column of Y, as `Design._prepare` computes it for a single phenotype."""
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    if Y.shape[0] != design.N:
        raise ValueError(f"Y has {Y.shape[0]} rows, design has {design.N}")
    if not np.isfinite(Y).all():
        raise ValueError("Y has non-finite entries: every phenotype must be observed on every analysed row")
    out = np.empty_like(Y)
    W, Q = design.W, design.Q
    for p in range(Y.shape[1]):
        y = Y[:, p]
        yt = y - W @ (Q @ (W.T @ y))
        yt = yt - yt.mean()
        sd = np.sqrt(np.sum(yt ** 2) / (design.N - 1))
        out[:, p] = yt / sd if sd > 1e-15 else yt
    return out


def q_pass_multi(design: Design, Y: np.ndarray, reader, *, n_jk: int = 100, snp_block: int = 2048,
                 dtype=np.float32, max_snps: Optional[int] = None,
                 snp_kernels: Optional[Sequence[int]] = None, method: str = "factored",
                 log: Optional[Callable[[str], None]] = print) -> dict:
    """q terms of every column of Y.  `reader.read(a, b)` -> int8 (N, b-a) codes (bedio readers).

    Returns dict(qg_full (P, Ps, G, G), qg_blk (P, J, Ps, G, G), qdiag (P, Pd, G, G), jk_bounds, M, nc,
    atom_sizes, snp_kernels, seconds)."""
    log = log or (lambda s: None)
    t0 = time.time()
    N, G = design.N, design.G
    Ps, Pd = design.P_snp, design.P_diag
    sel = np.arange(Ps) if snp_kernels is None else np.asarray(sorted(set(int(r) for r in snp_kernels)))
    if sel.min() < 0 or sel.max() >= Ps:
        raise ValueError(f"snp_kernels must lie in 0..{Ps - 1}")
    R = len(sel)
    M = int(reader.n_snps if max_snps is None else min(max_snps, reader.n_snps))
    if reader.n_rows != N:
        raise ValueError(f"reader has {reader.n_rows} rows, design has {N}")
    bounds = jk_bounds(M, n_jk)
    if np.any(np.diff(bounds) <= 0):
        raise ValueError(f"M = {M} too small for {n_jk} jackknife blocks")

    Yt = standardized_residuals(design, Y)                                         # (N, P) float64
    P = Yt.shape[1]
    W = design.W.astype(dtype)
    Q = design.Q.astype(dtype)
    rows = [design.rows_of(g) for g in range(G)]
    if method not in ("direct", "factored"):
        raise ValueError("method must be 'direct' or 'factored'")
    # c[:, p, g] = V (D_g y~_p), cast and projected as atom_pass does for its y~ column (needed for qdiag either way)
    Yc = Yt.astype(dtype)
    c = np.empty((N, P, G), dtype=dtype)
    for g in range(G):
        tmp = np.zeros_like(Yc)
        tmp[rows[g]] = Yc[rows[g]]
        tmp -= W @ (Q @ (W.T @ tmp))
        c[:, :, g] = tmp
    e_snp = design.snp_env().astype(dtype)[:, sel]                                 # (N, R)
    nC = W.shape[1]
    if method == "direct":
        Cin = (e_snp[:, None, :, None] * c[:, :, None, :]).reshape(N, P * R * G)   # (N, P R G)
        mem = Cin.nbytes
    else:
        # rows sorted by atom so that each atom's rows are one contiguous slice of every genotype block
        perm = np.concatenate(rows); off = np.concatenate([[0], np.cumsum([len(r) for r in rows])])
        Ey = (e_snp[perm][:, None, :] * Yc[perm][:, :, None]).reshape(N, P * R)       # (N, P R): e_r o y~_p, atom-sorted
        EW = (e_snp[perm][:, :, None] * W[perm][:, None, :]).reshape(N, R * nC)       # (N, R C): e_r o W, atom-sorted
        beta = np.stack([(design.Q @ (design.W[rows[g]].T @ Yt[rows[g]])) for g in range(G)], axis=-1)   # (C, P, G)
        mem = Ey.nbytes + EW.nbytes
    log(f"[q_pass_multi] N={N:,} M={M:,} P={P} SNP kernels {R}/{Ps} G={G} J={n_jk} method={method}; "
        f"inputs {mem / 2 ** 30:.2f} GB; setup {_fmt(time.time() - t0)}")

    qg_blk = np.zeros((P, n_jk, R, G, G))
    sample_local = np.arange(N)
    t1 = time.time(); t_read = t_x = 0.0
    for j in range(n_jk):
        for a in range(bounds[j], bounds[j + 1], snp_block):
            b = min(a + snp_block, bounds[j + 1]); m = b - a
            tr = time.time()
            X = standardize_block(reader.read(a, b), np.arange(a, b), sample_local, dtype)
            t_read += time.time() - tr
            tr = time.time()
            if method == "direct":
                S = (X.T @ Cin).reshape(m, P, R, G).astype(np.float64)
            else:
                Xs = X[perm]
                A = (Xs.T @ EW).reshape(m, R, nC).astype(np.float64)                  # X'(e_r o W)
                S = np.empty((m, P, R, G))
                for g in range(G):
                    sl = slice(off[g], off[g + 1])
                    Dg = (Xs[sl].T @ Ey[sl]).reshape(m, P, R).astype(np.float64)       # X'(e_r o D_g y~)
                    S[..., g] = Dg - np.einsum("mrc,cp->mpr", A, beta[:, :, g])
            qg_blk[:, j] += np.einsum("mprg,mprh->prgh", S, S)
            t_x += time.time() - tr
        if j % max(1, n_jk // 10) == 0 or j == n_jk - 1:
            log(f"[q_pass_multi] block {j + 1}/{n_jk}  read+std {_fmt(t_read)}  X'Cin+q {_fmt(t_x)}  "
                f"elapsed {_fmt(time.time() - t1)}")

    if R < Ps:                                                                    # NaN-fill kernels not computed
        full = np.full((P, n_jk, Ps, G, G), np.nan)
        full[:, :, sel] = qg_blk
        qg_blk = full
    t_pass = time.time() - t1
    qg_full = qg_blk.sum(axis=1)
    # qdiag[p, s, g, h] = sum_n d_s(n) v_pg(n) v_ph(n): one BLAS product (Pd x N) @ (N x G^2) per phenotype
    d = design.diag_env().astype(np.float64)                                       # (N, Pd)
    qdiag = np.empty((P, Pd, G, G))
    for p in range(P):
        v = c[:, p, :].astype(np.float64)                                          # (N, G)
        qdiag[p] = (d.T @ (v[:, :, None] * v[:, None, :]).reshape(N, G * G)).reshape(Pd, G, G)
    return dict(qg_full=qg_full, qg_blk=qg_blk, qdiag=qdiag, jk_bounds=bounds, M=M, nc=design.nc_per_atom(),
                atom_sizes=design.atom_sizes, snp_kernels=sel,
                seconds=dict(total=time.time() - t0, genotype_pass=t_pass, read=t_read, gemm=t_x))
