"""The trait-level test of H_0 (no GxE with any environment of the set), on the full sample.

For each environment l of the set, fit its marginal model by slicing the full-sample normal equations
(`normal_eq.select_indices(export, [l])`: G, GxE_l, NxE_l and noise), and form the one-sided Wald statistic

    z_l = h2_GxE_l / se_l,      se_l the delete-one SNP-block jackknife SE,      p_l = Pr(N(0, 1) >= z_l).

The L marginal p-values are combined by the Cauchy combination test (ACAT; Liu & Xie 2020, JASA 115:393),

    T = (1/L) sum_l tan((1/2 - p_l) pi),      p_H0 = 1/2 - arctan(T) / pi,

whose tail is that of a standard Cauchy whatever the dependence between the z_l. No sample split, no averaging over
pairs and no simulated nulls: one slice and J + 1 small solves per environment. Traits are then reported at a
Benjamini-Hochberg false discovery rate across traits (`nulltest.benjamini_hochberg`); the depth k_g and the estimate
of a reported trait come from the held-out curves as before (`compsplit.select_from_curves(..., h0_reject=...)`).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy.stats import norm

from .evaluator import _component_h2_stack
from .normal_eq import Export, select_indices


def acat(p, weights: Optional[Sequence[float]] = None) -> float:
    """Cauchy combination of p-values (equal weights by default), with the exact-tail forms for tiny p and large T."""
    p = np.clip(np.asarray(p, float), 1e-300, 1 - 1e-15)
    w = np.full(p.size, 1.0 / p.size) if weights is None else np.asarray(weights, float) / np.sum(weights)
    t = np.where(p < 1e-15, 1.0 / (p * np.pi), np.tan((0.5 - p) * np.pi))
    T = float(np.sum(w * t))
    return 1.0 / (T * np.pi) if T > 1e15 else 0.5 - np.arctan(T) / np.pi


def marginal_wald(export: Export, envs: Optional[Sequence[int]] = None) -> Dict[str, np.ndarray]:
    """Per environment: marginal h2_GxE, its jackknife SE, z and one-sided p (full sample)."""
    envs = list(range(export.n_env)) if envs is None else [int(e) for e in envs]
    jx = np.stack([j.x_l for j in export.jackknife]); jy = np.stack([j.y_r for j in export.jackknife])
    J = jx.shape[0]
    h2, se = np.empty(len(envs)), np.empty(len(envs))
    for i, l in enumerate(envs):
        ix = np.asarray(select_indices(export, [l]), dtype=int)
        h2[i] = _component_h2_stack(export.full.x_l[np.ix_(ix, ix)][None], export.full.y_r[ix][None])[0, 1]
        v = _component_h2_stack(jx[:, ix][:, :, ix], jy[:, ix])[:, 1]
        se[i] = np.sqrt((J - 1) / J * np.sum((v - v.mean()) ** 2))
    z = np.where(se > 0, h2 / np.where(se > 0, se, 1.0), np.nan)
    return dict(envs=np.asarray(envs), h2=h2, se=se, z=z, p=norm.sf(z))


def marginal_h0(export: Export, envs: Optional[Sequence[int]] = None) -> Dict[str, object]:
    """The trait-level H_0 test: ACAT of the marginal one-sided Wald p-values over the set (`p`), with the
    per-environment inputs (`env_h2`, `env_se`, `env_z`, `env_p`) and the environment of the smallest p."""
    m = marginal_wald(export, envs)
    ok = np.isfinite(m["p"])
    if not ok.any():
        raise ValueError("no environment with a positive jackknife SE")
    return dict(p=acat(m["p"][ok]), min_p_env=int(m["envs"][ok][np.argmin(m["p"][ok])]),
                env_h2=m["h2"], env_se=m["se"], env_z=m["z"], env_p=m["p"], envs=m["envs"])
