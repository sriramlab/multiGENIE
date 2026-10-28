"""The fixed preprocessing every half shares, mirroring the engine's preprocessing.

    W  = [1, covariates, E]     non-constant columns centred      (prepare(), 126-171)
    Q  = (W'W)^{-1}             pinv when cond(W'W) >= 1e6          (prepare(), 183-228)
    V x = x - W Q W' x          the projector, applied to y, probes and kernel outputs
    y~ = V y, re-centred, / sqrt(sum^2 / (N-1))                     (prepare(), 289-300)

Kernel layout (the export's order, `methods/gxe.py` + `io/normal_eq_export.py`):

    0            G        K = X X' / M
    1 .. L       GxE_i    diag(E_i) K diag(E_i)
    L+1 .. 2L    NxE_i    diag(E_i^2)
    2L+1         noise    I

"SNP kernels" are the first L+1 (they move with the jackknife block); the last L+1 are
"diagonal kernels" (they do not).  For the atom construction every kernel is fitted as
V K V restricted to the half: D_h V K V D_h.

The one deliberate departure from a half-sample engine run: W, the SNP statistics, E and
the scaling of y are those of the trait's FULL analysed sample, not of the half
(plan, "Hard constraint"; validation V3 measures the effect).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

NA = -9.0
NA_TOL = 1e-6
COND_PINV = 1e6          # prepare(): pinv fallback threshold
COND_ERROR = 1e12


def kernel_names(L: int) -> List[str]:
    return (["h2g"] + [f"h2gxe_e{i}" for i in range(L)]
            + [f"h2nxe_e{i}" for i in range(L)] + ["noise"])


def _is_missing_str(v: str) -> bool:
    if v in ("NA", "na", "NaN", "nan", ".", ""):
        return True
    try:
        f = float(v)
    except ValueError:
        return True
    return f == -9 or np.isnan(f)


@dataclass
class Design:
    """Analysed rows (in .fam order) with everything the atom pass needs."""
    y: np.ndarray                    # (N,) raw phenotype
    E: np.ndarray                    # (N, L) environment as given to the engine
    C: np.ndarray                    # (N, c) covariates WITHOUT the intercept
    atom: np.ndarray                 # (N,) int in 0..G-1
    G: int
    ids: Optional[np.ndarray] = None         # IIDs
    fam_index: Optional[np.ndarray] = None   # row indices into the .fam (for the BED reader)
    env_names: Optional[List[str]] = None
    trait: str = ""
    # derived in __post_init__
    W: np.ndarray = field(init=False, repr=False)
    Q: np.ndarray = field(init=False, repr=False)
    ytil: np.ndarray = field(init=False, repr=False)
    lev: np.ndarray = field(init=False, repr=False)
    used_pinv: bool = field(init=False, default=False)

    def __post_init__(self):
        self.y = np.asarray(self.y, dtype=np.float64).reshape(-1)
        self.E = np.asarray(self.E, dtype=np.float64)
        if self.E.ndim == 1:
            self.E = self.E[:, None]
        self.C = np.asarray(self.C, dtype=np.float64)
        if self.C.ndim == 1:
            self.C = self.C[:, None]
        self.atom = np.asarray(self.atom, dtype=np.int64)
        N = self.y.size
        for name, arr in (("E", self.E), ("C", self.C), ("atom", self.atom)):
            if arr.shape[0] != N:
                raise ValueError(f"{name} has {arr.shape[0]} rows for N = {N}")
        if self.atom.min() < 0 or self.atom.max() >= self.G:
            raise ValueError("atom labels must lie in 0..G-1")
        if self.env_names is None:
            self.env_names = [f"env_{i}" for i in range(self.L)]
        if self.fam_index is None:
            self.fam_index = np.arange(N)
        if self.ids is None:
            self.ids = np.asarray([str(i) for i in self.fam_index])
        self._prepare()

    # ----------------------------------------------------------- dimensions
    @property
    def N(self) -> int:
        return int(self.y.size)

    @property
    def L(self) -> int:
        return int(self.E.shape[1])

    @property
    def n_cov(self) -> int:
        return int(self.W.shape[1])

    @property
    def atom_sizes(self) -> np.ndarray:
        return np.bincount(self.atom, minlength=self.G)

    def rows_of(self, g: int) -> np.ndarray:
        return np.flatnonzero(self.atom == g)

    # ------------------------------------------------------------ prepare()
    def _prepare(self) -> None:
        W = np.hstack([np.ones((self.N, 1)), self.C, self.E])
        col_sd = W.std(axis=0, ddof=0)
        nonconst = col_sd > 0
        W = W.copy()
        if nonconst.any() and (~nonconst).any():
            W[:, nonconst] -= W[:, nonconst].mean(axis=0, keepdims=True)
        elif nonconst.all():
            W -= W.mean(axis=0, keepdims=True)
        WtW = W.T @ W
        cond = float(np.linalg.cond(WtW))
        if cond >= COND_ERROR:
            raise ValueError(f"W'W numerically rank-deficient (cond = {cond:.3e}); "
                             "a covariate or environment column is a linear combination of others")
        if cond >= COND_PINV:
            Q = np.linalg.pinv(WtW, rcond=1e-10)
            self.used_pinv = True
        else:
            Q = np.linalg.solve(WtW, np.eye(WtW.shape[0]))
        self.W, self.Q = W, Q
        yt = self.y - W @ (Q @ (W.T @ self.y))
        yt = yt - yt.mean()
        sd = np.sqrt(np.sum(yt ** 2) / (self.N - 1))
        if sd > 1e-15:
            yt = yt / sd
        self.ytil = yt
        self.lev = np.einsum("nc,nc->n", W @ Q, W)     # diag(W Q W')

    def project(self, X: np.ndarray) -> np.ndarray:
        """V X = X - W Q W' X, for (N,) or (N, k)."""
        return X - self.W @ (self.Q @ (self.W.T @ X))

    def project_inplace(self, X: np.ndarray) -> np.ndarray:
        """Same as `project`, writing into `X` (which must be 2-D and writeable)."""
        X -= (self.W @ (self.Q @ (self.W.T @ X))).astype(X.dtype, copy=False)
        return X

    # -------------------------------------------------------------- kernels
    @property
    def P_snp(self) -> int:
        return self.L + 1

    @property
    def P_diag(self) -> int:
        return self.L + 1

    @property
    def P_all(self) -> int:
        return 2 * self.L + 2

    def snp_env(self) -> np.ndarray:
        """(N, L+1): the e_r folded into each SNP kernel -- ones for G, E_i for GxE_i."""
        return np.hstack([np.ones((self.N, 1)), self.E])

    def diag_env(self) -> np.ndarray:
        """(N, L+1): the diagonal d_s of each diagonal kernel -- E_i^2 for NxE_i, ones for noise."""
        return np.hstack([self.E ** 2, np.ones((self.N, 1))])

    def kernel_names(self) -> List[str]:
        return kernel_names(self.L)

    def nc_per_atom(self) -> np.ndarray:
        """tr(D_g V D_g) = sum_{n in g} (1 - h_nn); sums to N - C over all atoms."""
        return np.bincount(self.atom, weights=1.0 - self.lev, minlength=self.G)

    # --------------------------------------------------------------- probes
    def probes(self, seed: int, n_probes: int) -> np.ndarray:
        """the engine's draw: default_rng(seed).standard_normal((N, B), dtype=float32)."""
        rng = np.random.default_rng(int(seed))
        return rng.standard_normal((self.N, int(n_probes)), dtype=np.float32)

    def mask_hash(self) -> str:
        return hashlib.sha1(np.asarray(self.fam_index, dtype=np.int64).tobytes()).hexdigest()[:12]

    # ------------------------------------------------------------ variants
    def with_permuted_env(self, seed: int) -> "Design":
        """E rows permuted WITHIN each atom (one permutation for all L columns), the null
        of validation V5.  Every half stays a union of permuted atoms."""
        rng = np.random.default_rng(int(seed))
        perm = np.arange(self.N)
        for g in range(self.G):
            rows = self.rows_of(g)
            perm[rows] = rows[rng.permutation(len(rows))]
        return Design(y=self.y, E=self.E[perm], C=self.C, atom=self.atom, G=self.G,
                      ids=self.ids, fam_index=self.fam_index, env_names=list(self.env_names),
                      trait=self.trait + f"_perm{seed}")

    def restricted(self, rows: np.ndarray, G: Optional[int] = None,
                   atom: Optional[np.ndarray] = None) -> "Design":
        """A Design over a subset of rows with ITS OWN W, Q, y~ -- the engine's half-sample
        convention, for validation V3."""
        rows = np.asarray(rows)
        return Design(y=self.y[rows], E=self.E[rows], C=self.C[rows],
                      atom=(self.atom[rows] if atom is None else atom), G=(self.G if G is None else G),
                      ids=self.ids[rows], fam_index=self.fam_index[rows],
                      env_names=list(self.env_names), trait=self.trait)

    # --------------------------------------------------------------- files
    @classmethod
    def from_files(cls, geno_prefix: str, pheno_path: str, covar_path: Optional[str],
                   env_path: str, atoms_path: str, *, pheno_col: Optional[str] = None,
                   trait: str = "", log=print) -> "Design":
        """The analysed sample as the engine would build it: rows of the
        .fam with an observed phenotype, a complete environment row and a covariate row
        (covariates mean-imputed per column), in .fam order -- plus an atom label."""
        import pandas as pd
        from .bedio import read_fam_iids
        fam = read_fam_iids(geno_prefix)
        n_fam = len(fam)

        ph = pd.read_csv(pheno_path, sep=r"\s+", dtype=str)
        col = pheno_col or ph.columns[2]
        ph_ids = ph.iloc[:, 1].to_numpy(str)
        ph_val = np.array([np.nan if _is_missing_str(v) else float(v) for v in ph[col].to_numpy(str)])

        env = pd.read_csv(env_path, sep=r"\s+", dtype={0: str, 1: str})
        env_names = list(env.columns[2:])
        env_ids = env.iloc[:, 1].to_numpy(str)
        E_all = np.array(env.iloc[:, 2:].to_numpy(float), copy=True)
        env_ok = np.isfinite(E_all).all(1) & ~(np.abs(E_all - NA) < NA_TOL).any(1)

        if covar_path:
            cv = pd.read_csv(covar_path, sep=r"\s+", dtype={0: str, 1: str})
            cv_ids = cv.iloc[:, 1].to_numpy(str)
            C_all = np.array(cv.iloc[:, 2:].apply(pd.to_numeric, errors="coerce").to_numpy(float), copy=True)
            C_all[np.abs(C_all - NA) < NA_TOL] = np.nan
            for j in range(C_all.shape[1]):          # the engine: mean-impute per column
                m = np.isnan(C_all[:, j])
                if m.any():
                    C_all[m, j] = np.nanmean(C_all[:, j])
        else:
            cv_ids, C_all = fam, np.zeros((n_fam, 0))

        at = pd.read_csv(atoms_path, sep="\t", dtype={"FID": str, "IID": str})
        at_ids = at["IID"].to_numpy(str)
        at_lab = at["atom"].to_numpy(int)
        G = int(at_lab.max()) + 1

        def pos_map(ids):
            return {s: k for k, s in enumerate(ids)}
        p_ph, p_env, p_cv, p_at = pos_map(ph_ids), pos_map(env_ids), pos_map(cv_ids), pos_map(at_ids)

        rows, y, E, C, atom, ids = [], [], [], [], [], []
        n_no_pheno = n_no_env = n_no_cov = n_no_atom = 0
        for r, s in enumerate(fam):
            i = p_ph.get(s)
            if i is None or np.isnan(ph_val[i]):
                n_no_pheno += 1
                continue
            j = p_env.get(s)
            if j is None or not env_ok[j]:
                n_no_env += 1
                continue
            k = p_cv.get(s)
            if k is None:
                n_no_cov += 1
                continue
            a = p_at.get(s)
            if a is None:
                n_no_atom += 1
                continue
            rows.append(r); y.append(ph_val[i]); E.append(E_all[j]); C.append(C_all[k])
            atom.append(at_lab[a]); ids.append(s)
        log(f"[design] {trait or Path(pheno_path).stem}: {len(rows):,} analysed rows of {n_fam:,} "
            f"(no phenotype {n_no_pheno:,}, env incomplete {n_no_env:,}, no covariates {n_no_cov:,}, "
            f"no atom label {n_no_atom:,}); G = {G}")
        if n_no_atom:
            log(f"[design] WARNING: {n_no_atom:,} rows with phenotype and environment carry no atom label "
                "and are dropped -- the atom file should cover the env-complete sample")
        return cls(y=np.array(y), E=np.array(E), C=np.array(C), atom=np.array(atom), G=G,
                   ids=np.array(ids), fam_index=np.array(rows), env_names=env_names,
                   trait=trait or Path(pheno_path).stem)
