"""The exported normal-equation system, and the one identity the whole method rests on.

A method-of-moments variance-component fit solves

    X sigma = y,     X[i][j] = tr(K_i K_j),   X[i][C] = tr(K_i),   X[C][C] = N
                     y[i]    = y' K_i y,      y[C]    = y' y

for kernels `K_i`.  Every entry depends only on the two components it indexes and
never on which other components are in the model, so **the normal equations of any
sub-model are exactly the corresponding principal submatrix of the full system.**
Fitting a subset of environments is therefore a slice and a solve, not another pass
over the genotypes: one fit on the cluster buys every subset.

The engine (GENIE or the engine) writes that system once for the full sample and once per
jackknife block, so delete-one standard errors survive the slicing too.
See `docs/export_format.md` for the JSON schema.

Component order in the exported system, for `L` environments:

    [ G , GxE_0 .. GxE_{L-1} , NxE_0 .. NxE_{L-1} , noise ]      G+GxE+NxE
    [ G , GxE_0 .. GxE_{L-1} , noise ]                           G+GxE

Selecting environment `k` keeps BOTH its kernels: dropping the N x E term of an
unselected environment lets omitted heteroscedasticity leak onto the correlated
environments that were selected.
"""
from __future__ import annotations

import json, os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

MODELS = ("G+GxE", "G+GxE+NxE")


@dataclass
class System:
    """One linear system: `x_l @ sigma = y_r`."""
    x_l: np.ndarray
    y_r: np.ndarray


@dataclass
class Export:
    """A full-sample system plus its per-jackknife-block systems."""
    model: str
    n_env: int
    env_names: List[str]
    full: System
    jackknife: List[System]
    # Which environments still carry an N x E column.  `None` means all of them,
    # which is every export as the engine writes it.  `drop_degenerate_nxe` sets it when
    # a column has been removed for being collinear with the noise kernel, and the
    # G x E side is deliberately left untouched, so the two sides can differ.
    nxe_keep: tuple | None = None

    @property
    def has_nxe(self) -> bool:
        return self.model == "G+GxE+NxE"

    @property
    def nxe_index(self) -> tuple:
        """Environment indices carrying an N x E column, in column order."""
        if not self.has_nxe:
            return ()
        return tuple(range(self.n_env)) if self.nxe_keep is None else self.nxe_keep

    @property
    def dim(self) -> int:
        return 1 + self.n_env + len(self.nxe_index) + 1

    @property
    def n_jack(self) -> int:
        return len(self.jackknife)


# --------------------------------------------------------------------- loading
def _system(raw: Dict[str, Any], dim: int, label: str) -> System:
    x_l = np.asarray(raw["x_l"], dtype=float)
    y_r = np.asarray(raw["y_r"], dtype=float)
    if x_l.shape != (dim, dim) or y_r.shape != (dim,):
        raise ValueError(f"{label}: expected {dim}x{dim} / {dim}, "
                         f"got {x_l.shape} / {y_r.shape}")
    return System(x_l=x_l, y_r=y_r)


def load_export(source: str | Path | Dict[str, Any],
                drop: Sequence[str] | None = None) -> Export:
    """Read an export from a JSON file (or an already-parsed dict).

    `gen_nbin` must be 1: the index arithmetic below assumes a single additive
    component, and a partitioned annotation would shift every G x E position.

    `drop` removes named environments by slicing their rows and columns out of
    every system.  That is the L - d fit itself, not a correction applied to the
    L fit, so a column can leave the pool without re-running the engine.
    """
    payload = (source if isinstance(source, dict)
               else json.loads(Path(source).read_text()))
    for key in ("model", "gen_nbin", "n_env", "full", "jackknife"):
        if key not in payload:
            raise ValueError(f"export is missing required key {key!r}")

    model = payload["model"]
    if model not in MODELS:
        raise ValueError(f"unsupported model {model!r}; expected one of {MODELS}")
    if int(payload["gen_nbin"]) != 1:
        raise ValueError("this selector requires gen_nbin == 1 (single annotation)")

    n_env = int(payload["n_env"])
    if n_env <= 0:
        raise ValueError(f"n_env must be positive; got {n_env}")
    dim = 1 + n_env + (n_env if model == "G+GxE+NxE" else 0) + 1

    names = payload.get("env_names") or [f"env_{i}" for i in range(n_env)]
    if len(names) != n_env:
        raise ValueError(f"env_names has {len(names)} entries for n_env={n_env}")

    jack = [_system(j, dim, f"jackknife[{i}]")
            for i, j in enumerate(payload["jackknife"])]
    if not jack:
        raise ValueError("export carries no jackknife systems")

    export = Export(model=model, n_env=n_env, env_names=list(names),
                    full=_system(payload["full"], dim, "full"), jackknife=jack)
    return export if not drop else drop_envs(export, drop)


def drop_envs(export: Export, names: Sequence[str]) -> Export:
    """The same export with `names` sliced out of every system."""
    drop = set(names)
    keep = [i for i, n in enumerate(export.env_names) if n not in drop]
    if len(keep) == export.n_env:
        return export
    if not keep:
        raise ValueError("dropping every environment leaves nothing to select over")
    idx = select_indices(export, keep)
    old = set(export.nxe_index)
    new_keep = tuple(j for j, i in enumerate(keep) if i in old)
    return Export(model=export.model, n_env=len(keep),
                  env_names=[export.env_names[i] for i in keep],
                  full=slice_system(export.full, idx),
                  jackknife=[slice_system(j, idx) for j in export.jackknife],
                  nxe_keep=None if len(new_keep) == len(keep) else new_keep)


# --------------------------------------------------------------------- slicing
def select_indices(export: Export, subset: Sequence[int]) -> List[int]:
    """Rows/columns of the full system that make up the sub-model on `subset`.

    Order matters and is used downstream: the G x E components come out in exactly
    the order given, so a block admitted last occupies the trailing positions.
    """
    idx = [0]
    idx.extend(1 + int(k) for k in subset)
    if export.has_nxe:
        pos = {e: j for j, e in enumerate(export.nxe_index)}
        idx.extend(1 + export.n_env + pos[int(k)]
                   for k in subset if int(k) in pos)
    idx.append(export.dim - 1)
    return idx


def slice_system(system: System, indices: Sequence[int]) -> System:
    ix = np.asarray(indices, dtype=int)
    return System(x_l=system.x_l[np.ix_(ix, ix)], y_r=system.y_r[ix])


def solve(system: System) -> np.ndarray:
    """`sigma` for one system.  Raises rather than returning a pivoted answer.

    A near-singular slice means two near-duplicate environments are in the same
    fit; silently least-squaring that away would hide a modelling decision.
    """
    try:
        return np.linalg.solve(system.x_l, system.y_r)
    except np.linalg.LinAlgError as exc:
        raise ValueError("restricted normal equation is singular") from exc


def component_h2(sigma: np.ndarray, system: System) -> np.ndarray:
    """Variance components as fractions of the total: `sigma_i tr(K_i) / sum`.

    The noise term is scaled by `N` and enters the denominator only, so the
    returned vector covers the non-noise components and sums to the model's total
    heritability.  This is GENIE's convention; the engine prints a different
    normaliser, so its `h2_*` lines and these numbers differ by construction.
    """
    sigma = np.asarray(sigma, dtype=float)
    n = sigma.size - 1
    adjusted = sigma.copy()
    adjusted[:n] *= system.x_l[:n, -1]          # tr(K_i)
    adjusted[-1] *= system.x_l[-1, -1]          # N
    total = float(adjusted.sum())
    if abs(total) < 1e-12:
        raise ValueError("restricted solution has a near-zero total; cannot normalise")
    return adjusted[:-1] / total


def fit_subset(export: Export, subset: Sequence[int]) -> np.ndarray:
    """`component_h2` of the sub-model on `subset`, full sample."""
    sub = slice_system(export.full, select_indices(export, subset))
    return component_h2(solve(sub), sub)


def jk_se(values: Sequence[float]) -> float:
    """Delete-one jackknife standard error, `sqrt((n-1)/n * sum (x - xbar)^2)`."""
    v = np.asarray(values, dtype=float)
    n = v.size
    if n == 0:
        raise ValueError("no jackknife blocks")
    return float(np.sqrt((n - 1) / n * np.sum((v - v.mean()) ** 2)))


# ------------------------------------------------- degenerate N x E columns
NXE_IDENTITY_COS = 0.99


def nxe_identity_cos(export: Export) -> np.ndarray:
    """Angle of each N x E kernel to the identity, in the trace inner product.

    `x_l[i, j] = tr(K_i K_j)`, and the noise kernel is the identity, so the last
    column holds `tr(K_i)` and the cosine is `tr(K_i) / sqrt(tr(K_i^2) tr(I))`.

    For a standardised BINARY environment with frequency p this is near one by
    construction: `diag(e^2) = a^2 I + (b^2 - a^2) diag(x)` with
    `b^2 - a^2 = (1 - 2p) / (p(1-p))`, so the identifying part vanishes as p
    approaches one half and a balanced binary environment gives `K_nxe = I`
    exactly.  Such a column is the noise kernel under another name.
    """
    if not export.has_nxe:
        return np.zeros(0)
    T = export.full.x_l
    noise = export.dim - 1
    cols = [1 + export.n_env + j for j in range(len(export.nxe_index))]
    return np.array([T[c, noise] / np.sqrt(T[c, c] * T[noise, noise])
                     for c in cols])


def drop_degenerate_nxe(export: Export,
                        threshold: float = NXE_IDENTITY_COS) -> Export:
    """Remove N x E columns collinear with the noise kernel; keep their G x E side.

    A column above `threshold` is not identified against the noise term, so the
    solver splits weight between the two arbitrarily.  On the 42-trait pool that
    single column drove sum h2_NxE to a worst case of -5.27 with 52% of fits
    negative; removing it leaves +0.083 with 8% negative and a 13-fold smaller
    spread, while sum h2_GxE moves by at most 0.0056 on any trait.

    The G x E column is deliberately kept: the environment is still a legitimate
    candidate for selection, and only its noise-variance term was degenerate.
    """
    if not export.has_nxe:
        return export
    if os.environ.get("NXE_DROP", "0") != "1":      # 2026-10-06 (Methods A(S)): every selected environment keeps its NxE term;
        return export                                # the identifiability drop is applied only when NXE_DROP=1 is set
    cos = nxe_identity_cos(export)
    keep_pos = [j for j, c in enumerate(cos) if c <= threshold]
    if len(keep_pos) == len(cos):
        return export
    if not keep_pos:
        raise ValueError("every N x E column is collinear with the noise kernel")
    head = list(range(1 + export.n_env))
    cols = [1 + export.n_env + j for j in keep_pos]
    idx = head + cols + [export.dim - 1]
    return Export(model=export.model, n_env=export.n_env,
                  env_names=list(export.env_names),
                  full=slice_system(export.full, idx),
                  jackknife=[slice_system(j, idx) for j in export.jackknife],
                  nxe_keep=tuple(export.nxe_index[j] for j in keep_pos))
