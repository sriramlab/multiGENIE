"""The per-atom Gram tensors and the assembler: any union of atoms -> an `Export`.

Column convention.  A "sketch column" is one (kernel r, input atom g) pair, indexed
`r * G + g`.  SNP kernels are r = 0..L (K_snp = (L+1) G columns), diagonal kernels
r = L+1..2L+1 (K_all = (2L+2) G columns in total).  For probe p the sketch is

    u_{g,r,p} = V K_r V (D_g z_p)            an N-vector, dense

and the tensors are same-probe Grams restricted to the rows of OUTPUT atom g'':

    S_full[g'', (g,r), (g',s)]    = sum_p < u_{g,r,p}      , u_{g',s,p} >_{g''}
    Gb1[j, g'', (g,r), (g',s)]    = sum_p < u^{(j)}_{g,r,p}, u_{g',s,p} >_{g''}      r SNP, s any
    Gb2[j, g'', (g,r), (g',s)]    = sum_p < u^{(j)}_{g,r,p}, u^{(j)}_{g',s,p} >_{g''} r, s SNP

where u^{(j)} is the contribution of SNP block j (unnormalised by anything but 1/M).
Quadratic forms, with v_g = V D_g y~ and w_{g,r} = X' (e_r o v_g):

    qg_full[r, g, g']  = < w_{g,r}, w_{g',r} >           qg_blk[j] the same over block j only
    qdiag[s, g, g']    = sum_n d_s[n] v_g[n] v_{g'}[n]

Assembling half h (a set of atoms, mask m):

    T_h[r,s]      = (1/B) sum_{g'' in h} sum_{g,g' in h} S_full[g'', (g,r), (g',s)]
    T_h^{-j}      = alpha_r alpha_s (1/B) [ S - Gb1[j] - Gb1[j]' + Gb2[j] ]_h,   alpha = M/(M - M_j) on SNP kernels
    q_h[r]        = (1/M) sum_{g,g' in h} qg_full[r,g,g'],   q_h^{-j}[r] = alpha_r (1/M) sum (qg_full - qg_blk[j])
    q_h[s]        = sum_{g,g' in h} qdiag[s,g,g']                          (diagonal kernels, every j)
    Fix 2         T[noise,noise] = T[noise,G] = T[G,noise] = NC_h = sum_{g in h} NC_g

which is line for line `the engine/csrc/se/loo_traces.cpp` and `io/normal_eq_export.py` with the
half's restricted kernels in place of the full ones.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .model import kernel_names

REPO = Path(__file__).resolve().parents[2]          # the repository root, which holds the multi_gxe package


def _mg():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import multi_gxe as mg
    return mg


@dataclass
class AtomTensor:
    L: int
    G: int
    B: int
    M: int
    env_names: List[str]
    atom_sizes: np.ndarray            # (G,)
    nc: np.ndarray                    # (G,)  tr(D_g V D_g)
    jk_bounds: np.ndarray             # (J+1,)
    S_full: np.ndarray                # (G, K_all, K_all) f64
    Gb1: np.ndarray                   # (J, G, K_snp, K_all) f32
    Gb2: np.ndarray                   # (J, G, K_snp, K_snp) f32
    qg_full: np.ndarray               # (P_snp, G, G) f64
    qg_blk: np.ndarray                # (J, P_snp, G, G) f64
    qdiag: np.ndarray                 # (P_diag, G, G) f64
    meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------ shapes
    @property
    def J(self) -> int:
        return int(len(self.jk_bounds) - 1)

    @property
    def P_snp(self) -> int:
        return self.L + 1

    @property
    def P_all(self) -> int:
        return 2 * self.L + 2

    @property
    def K_snp(self) -> int:
        return self.P_snp * self.G

    @property
    def K_all(self) -> int:
        return self.P_all * self.G

    @property
    def m_blk(self) -> np.ndarray:
        return np.diff(self.jk_bounds)

    @property
    def N(self) -> int:
        return int(self.atom_sizes.sum())

    def kernel_names(self) -> List[str]:
        return kernel_names(self.L)

    # ------------------------------------------------------------ assembly
    def _cols(self, atoms: Sequence[int], P: int) -> np.ndarray:
        h = np.asarray(sorted(set(int(a) for a in atoms)))
        return (np.arange(P)[:, None] * self.G + h[None, :]).ravel(), h

    def systems(self, atoms: Sequence[int], *, fix_noise: bool = True):
        """(T, q, T_j (J,P,P), q_j (J,P)) for the union of `atoms`."""
        cols_all, h = self._cols(atoms, self.P_all)
        cols_snp, _ = self._cols(atoms, self.P_snp)
        nh, P, Ps, J, B, M = len(h), self.P_all, self.P_snp, self.J, self.B, self.M
        if nh == 0:
            raise ValueError("a half needs at least one atom")

        # full system: sum over output atoms and over input atom pairs
        S = np.zeros((P * nh, P * nh))
        for g in h:
            S += self.S_full[g][cols_all[:, None], cols_all[None, :]]
        S = S.reshape(P, nh, P, nh).sum(axis=(1, 3))

        G1 = np.zeros((J, Ps * nh, P * nh))
        G2 = np.zeros((J, Ps * nh, Ps * nh))
        for g in h:
            G1 += self.Gb1[:, g][:, cols_snp[:, None], cols_all[None, :]]
            G2 += self.Gb2[:, g][:, cols_snp[:, None], cols_snp[None, :]]
        G1 = G1.reshape(J, Ps, nh, P, nh).sum(axis=(2, 4))
        G2 = G2.reshape(J, Ps, nh, Ps, nh).sum(axis=(2, 4))

        Tj = np.broadcast_to(S, (J, P, P)).copy()
        Tj[:, :Ps, :] -= G1
        Tj[:, :, :Ps] -= G1.transpose(0, 2, 1)
        Tj[:, :Ps, :Ps] += G2
        alpha = M / (M - self.m_blk.astype(float))
        Tj[:, :Ps, :] *= alpha[:, None, None]
        Tj[:, :, :Ps] *= alpha[:, None, None]
        T = S / B
        Tj /= B

        qs_full = self.qg_full[:, h][:, :, h].sum(axis=(1, 2))                     # (Ps,)
        qs_blk = self.qg_blk[:, :, h][:, :, :, h].sum(axis=(2, 3))                 # (J, Ps)
        qd = self.qdiag[:, h][:, :, h].sum(axis=(1, 2))                            # (P_diag,)
        q = np.concatenate([qs_full / M, qd])
        qj = np.concatenate([(qs_full[None, :] - qs_blk) / M * alpha[:, None],
                             np.broadcast_to(qd, (J, len(qd)))], axis=1)

        if fix_noise:
            nc = float(self.nc[h].sum())
            for X in (T, Tj):
                X[..., P - 1, P - 1] = nc
                X[..., P - 1, 0] = nc
                X[..., 0, P - 1] = nc
        return T, q, Tj, qj

    def assemble(self, atoms: Sequence[int], *, fix_noise: bool = True):
        """A `multi_gxe.Export` for the union of `atoms` (model G+GxE+NxE)."""
        mg = _mg()
        T, q, Tj, qj = self.systems(atoms, fix_noise=fix_noise)
        return mg.Export(model="G+GxE+NxE", n_env=self.L, env_names=list(self.env_names),
                         full=mg.System(x_l=T, y_r=q),
                         jackknife=[mg.System(x_l=Tj[j], y_r=qj[j]) for j in range(self.J)])

    def export_json(self, atoms: Sequence[int], path: str | Path) -> str:
        """Write the union's system in the engine's normal_eq.json schema."""
        T, q, Tj, qj = self.systems(atoms)
        payload = {
            "schema_version": 1, "model": "G+GxE+NxE", "gen_nbin": 1, "n_env": self.L,
            "env_names": list(self.env_names), "producer": "atomsplit",
            "kernel_names": self.kernel_names(), "atoms": [int(a) for a in atoms],
            "full": {"x_l": T.tolist(), "y_r": q.tolist()},
            "jackknife": [{"x_l": Tj[j].tolist(), "y_r": qj[j].tolist()} for j in range(self.J)],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload))
        return str(path)

    # ------------------------------------------------------------ storage
    def save(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, L=self.L, G=self.G, B=self.B, M=self.M, env_names=np.array(self.env_names),
                 atom_sizes=self.atom_sizes, nc=self.nc, jk_bounds=self.jk_bounds,
                 S_full=self.S_full, Gb1=self.Gb1, Gb2=self.Gb2, qg_full=self.qg_full,
                 qg_blk=self.qg_blk, qdiag=self.qdiag, meta=json.dumps(self.meta))
        return str(path)

    @classmethod
    def load(cls, path: str | Path, mmap: bool = False) -> "AtomTensor":
        z = np.load(path, mmap_mode="r" if mmap else None, allow_pickle=False)
        return cls(L=int(z["L"]), G=int(z["G"]), B=int(z["B"]), M=int(z["M"]),
                   env_names=[str(s) for s in z["env_names"]], atom_sizes=np.asarray(z["atom_sizes"]),
                   nc=np.asarray(z["nc"]), jk_bounds=np.asarray(z["jk_bounds"]),
                   S_full=z["S_full"], Gb1=z["Gb1"], Gb2=z["Gb2"], qg_full=np.asarray(z["qg_full"]),
                   qg_blk=np.asarray(z["qg_blk"]), qdiag=np.asarray(z["qdiag"]),
                   meta=json.loads(str(z["meta"])))

    def nbytes(self) -> int:
        return sum(int(getattr(self, k).nbytes) for k in
                   ("S_full", "Gb1", "Gb2", "qg_full", "qg_blk", "qdiag"))
