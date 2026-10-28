"""Stage B: two passes over the genotypes -> AtomTensor.

Pass 1   for every SNP block j (sub-blocks of `snp_block` SNPs):
             S_j = X_j' Cin                     Cin = [ e_r o V(D_g z_p) | e_r o V(D_g y~) ]   (N x (B+1) K_snp)
             U  += X_j S_j[probe columns]        (N x B K_snp)
             qg_blk[j] from the y~ columns of S_j
         then U <- V (e_r o U / M) and S_full from [U | diagonal sketches] per output atom.
Pass 2   for every j: recompute U_j = V (e_r o X_j S_j / M) and the Grams <U_j, U>, <U_j, U_j>
         per output atom.  S_j is re-used from memory when `keep_sketch` fits.

The diagonal-kernel sketches V(d_s o V D_g z_p) are dense but cheap: they are rebuilt per
output atom from c = V D_g z and a (C x B x K_diag) coefficient table instead of stored.

Cost model: the X-step (N x m x B K_snp) dominates, twice; everything else is a few
percent.  Timings are logged per stage so the profile (plan Step 0 item 9) is measured,
not assumed.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np

from .bedio import standardize_block
from .model import Design
from .tensor import AtomTensor


def jk_bounds(M: int, J: int) -> np.ndarray:
    """The engine's block scheme: bounds[j] = j * M // J."""
    return np.array([(j * M) // J for j in range(J + 1)], dtype=np.int64)


def _fmt(sec: float) -> str:
    return f"{sec:7.1f}s"


def atom_pass(design: Design, reader, *, n_probes: int = 10, seed: int = 0, n_jk: int = 100,
              snp_block: int = 2048, dtype=np.float32, keep_sketch: bool = True,
              cache_uall: bool = True, max_snps: Optional[int] = None, gram_dtype=np.float32,
              log: Optional[Callable[[str], None]] = print) -> AtomTensor:
    """Run both passes.

    `reader.read(a, b)` -> int8 (N, b-a) codes (bedio.BedBlockReader / ArrayBlockReader);
    decoding, standardisation and GEMMs in numpy.

    `dtype` is the pipeline precision (genotypes, sketches, GEMMs); `gram_dtype` is the
    STORAGE precision of the per-block Grams Gb1/Gb2 (float32 halves the tensor file and
    perturbs the delete-one systems at ~1e-7 of |T|, i.e. ~1e-5 of the jackknife deltas)."""
    log = log or (lambda s: None)
    t0 = time.time()
    N, L, G = design.N, design.L, design.G
    B, B1 = int(n_probes), int(n_probes) + 1
    Ps, Pd, Pa = design.P_snp, design.P_diag, design.P_all
    Ks, Kd, Ka = Ps * G, Pd * G, Pa * G
    M = int(reader.n_snps if max_snps is None else min(max_snps, reader.n_snps))
    if reader.n_rows != N:
        raise ValueError(f"reader has {reader.n_rows} rows, design has {N}")
    bounds = jk_bounds(M, n_jk)
    J = n_jk
    m_blk = np.diff(bounds)
    if np.any(m_blk <= 0):
        raise ValueError(f"M = {M} too small for {J} jackknife blocks")
    W = design.W.astype(dtype)
    Q = design.Q.astype(dtype)
    rows = [design.rows_of(g) for g in range(G)]
    sample_local = np.arange(N)

    def project_inplace(X2):
        X2 -= W @ (Q @ (W.T @ X2))
        return X2

    # ---- inputs: c[:, p, g] = V (D_g [z_p | y~]) -----------------------------------
    Z = design.probes(seed, B).astype(dtype)
    Zy = np.concatenate([Z, design.ytil.astype(dtype)[:, None]], axis=1)          # (N, B1)
    c = np.empty((N, B1, G), dtype=dtype)
    for g in range(G):
        tmp = np.zeros_like(Zy)
        tmp[rows[g]] = Zy[rows[g]]
        c[:, :, g] = project_inplace(tmp)
    e_snp = design.snp_env().astype(dtype)                                           # (N, Ps)
    d_diag = design.diag_env().astype(dtype)                                         # (N, Pd)
    Cin = (e_snp[:, None, :, None] * c[:, :, None, :]).reshape(N, B1 * Ks)         # (N, B1 Ks)
    # diagonal-sketch coefficients beta[:, p, s*G+g] = Q W' (d_s o c_{g,p})
    beta = np.empty((W.shape[1], B, Pd, G), dtype=dtype)
    Cp = c[:, :B, :]
    for s in range(Pd):
        Dc = (d_diag[:, s, None, None] * Cp).reshape(N, B * G)
        beta[:, :, s, :] = (Q @ (W.T @ Dc)).reshape(W.shape[1], B, G)
    beta = beta.reshape(W.shape[1], B * Kd)
    gb = lambda a: a.nbytes / 2 ** 30
    log(f"[atom_pass] N={N:,} M={M:,} L={L} G={G} B={B} J={J} dtype={np.dtype(dtype).name}; "
        f"Cin {gb(Cin):.2f} GB, U {N * B * Ks * np.dtype(dtype).itemsize / 2**30:.2f} GB, "
        f"sketch store {M * B1 * Ks * np.dtype(dtype).itemsize / 2**30:.2f} GB "
        f"({'kept' if keep_sketch else 'recomputed in pass 2'}); setup {_fmt(time.time() - t0)}")

    def diag_slice(r):
        """(n, B, K_diag) sketches of the diagonal kernels on rows r."""
        n = len(r)
        out = (d_diag[r][:, None, :, None] * c[r, :B, :][:, :, None, :]).reshape(n, B * Kd)
        out -= W[r] @ beta
        return out.reshape(n, B, Kd)

    # ---- pass 1 ----------------------------------------------------------------
    t1 = time.time()
    t_read = t_xt = t_x = 0.0
    U2 = np.zeros((N, B * Ks), dtype=dtype)
    qg_blk = np.zeros((J, Ps, G, G))
    Sstore = np.empty((M, B1 * Ks), dtype=dtype) if keep_sketch else None
    for j in range(J):
        for a in range(bounds[j], bounds[j + 1], snp_block):
            b = min(a + snp_block, bounds[j + 1])
            tr = time.time()
            X = standardize_block(reader.read(a, b), np.arange(a, b), sample_local, dtype)
            t_read += time.time() - tr
            tr = time.time()
            S = X.T @ Cin                                                     # (m, B1 Ks)
            t_xt += time.time() - tr
            if keep_sketch:
                Sstore[a:b] = S
            S3 = S.reshape(b - a, B1, Ks)
            tr = time.time()
            U2 += X @ np.ascontiguousarray(S3[:, :B, :]).reshape(b - a, B * Ks)
            t_x += time.time() - tr
            Sy = S3[:, B, :].reshape(b - a, Ps, G).astype(np.float64)
            qg_blk[j] += np.einsum("mrg,mrh->rgh", Sy, Sy)
        if log and (j % max(1, J // 10) == 0 or j == J - 1):
            log(f"[pass 1] block {j + 1}/{J}  read+std {_fmt(t_read)}  X'Cin {_fmt(t_xt)}  "
                f"X S {_fmt(t_x)}  elapsed {_fmt(time.time() - t1)}")
    U = U2.reshape(N, B, Ps, G)
    U *= e_snp[:, None, :, None]
    U *= np.dtype(dtype).type(1.0 / M)
    project_inplace(U2)
    U3 = U2.reshape(N, B, Ks)
    qg_full = qg_blk.sum(axis=0)

    # q for the diagonal kernels: v_g = c[:, B, g]
    v = c[:, B, :].astype(np.float64)                                              # (N, G)
    qdiag = np.einsum("ns,ng,nh->sgh", d_diag.astype(np.float64), v, v)          # (Pd, G, G)

    # S_full per output atom, and the cached [U | diag] slices
    tg = time.time()
    S_full = np.zeros((G, Ka, Ka))
    uall_cache = [] if cache_uall else None
    for g in range(G):
        r = rows[g]
        F = np.concatenate([U3[r], diag_slice(r)], axis=2)                       # (n, B, Ka)
        Ft = np.ascontiguousarray(F.transpose(1, 0, 2))                           # (B, n, Ka)
        S_full[g] = np.matmul(Ft.transpose(0, 2, 1), Ft).sum(axis=0)
        if cache_uall:
            uall_cache.append(Ft)
    log(f"[pass 1] done {_fmt(time.time() - t1)}; S_full grams {_fmt(time.time() - tg)}")

    # ---- pass 2 ----------------------------------------------------------------
    t2 = time.time()
    Gb1 = np.zeros((J, G, Ks, Ka), dtype=gram_dtype)
    Gb2 = np.zeros((J, G, Ks, Ks), dtype=gram_dtype)
    t_read = t_x = t_gram = 0.0
    Uj2 = np.empty((N, B * Ks), dtype=dtype)
    for j in range(J):
        Uj2[:] = 0
        for a in range(bounds[j], bounds[j + 1], snp_block):
            b = min(a + snp_block, bounds[j + 1])
            tr = time.time()
            X = standardize_block(reader.read(a, b), np.arange(a, b), sample_local, dtype)
            t_read += time.time() - tr
            if keep_sketch:
                S3 = Sstore[a:b].reshape(b - a, B1, Ks)
            else:
                S3 = (X.T @ Cin).reshape(b - a, B1, Ks)
            tr = time.time()
            Uj2 += X @ np.ascontiguousarray(S3[:, :B, :]).reshape(b - a, B * Ks)
            t_x += time.time() - tr
        Uj = Uj2.reshape(N, B, Ps, G)
        Uj *= e_snp[:, None, :, None]
        Uj *= np.dtype(dtype).type(1.0 / M)
        project_inplace(Uj2)
        Uj3 = Uj2.reshape(N, B, Ks)
        tr = time.time()
        for g in range(G):
            r = rows[g]
            A = np.ascontiguousarray(Uj3[r].transpose(1, 0, 2))                  # (B, n, Ks)
            if cache_uall:
                Ft = uall_cache[g]
            else:
                F = np.concatenate([U3[r], diag_slice(r)], axis=2)
                Ft = np.ascontiguousarray(F.transpose(1, 0, 2))
            At = A.transpose(0, 2, 1)
            Gb1[j, g] = np.matmul(At, Ft).sum(axis=0)
            Gb2[j, g] = np.matmul(At, A).sum(axis=0)
        t_gram += time.time() - tr
        if log and (j % max(1, J // 10) == 0 or j == J - 1):
            log(f"[pass 2] block {j + 1}/{J}  read+std {_fmt(t_read)}  X S {_fmt(t_x)}  "
                f"grams {_fmt(t_gram)}  elapsed {_fmt(time.time() - t2)}")
    log(f"[pass 2] done {_fmt(time.time() - t2)}; total {_fmt(time.time() - t0)}")

    meta = dict(backend="numpy", trait=design.trait, seed=int(seed), dtype=np.dtype(dtype).name,
                gram_dtype=np.dtype(gram_dtype).name, N=N, M=M,
                n_cov=int(design.n_cov), used_pinv=bool(design.used_pinv),
                mask_hash=design.mask_hash(), snp_block=int(snp_block),
                seconds=dict(total=time.time() - t0, pass1=t2 - t1, pass2=time.time() - t2),
                fam_index_first=int(design.fam_index[0]), fam_index_last=int(design.fam_index[-1]))
    return AtomTensor(L=L, G=G, B=B, M=M, env_names=list(design.env_names),
                      atom_sizes=design.atom_sizes, nc=design.nc_per_atom(), jk_bounds=bounds,
                      S_full=S_full, Gb1=Gb1, Gb2=Gb2, qg_full=qg_full, qg_blk=qg_blk,
                      qdiag=qdiag, meta=meta)
