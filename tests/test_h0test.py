import numpy as np
import pytest
import multi_gxe as mg
from multi_gxe import h0test as H


def _export(L=4, J=20, seed=0, signal=None):
    """A random well-conditioned system (G, GxE x L, NxE x L, noise) with J jackknife perturbations."""
    rng = np.random.default_rng(seed); D = 2 * L + 2
    A = rng.standard_normal((D, D)); X = A @ A.T + D * np.eye(D)
    s = np.full(D, 0.01); s[0] = 0.3; s[-1] = 0.6
    if signal is not None:
        s[1 + signal] = 0.2
    y = X @ s
    jk = [mg.System(x_l=X + 0.01 * np.diag(rng.random(D)), y_r=y + 0.05 * rng.standard_normal(D)) for _ in range(J)]
    return mg.Export(model="G+GxE+NxE", n_env=L, env_names=[f"e{i}" for i in range(L)],
                     full=mg.System(x_l=X, y_r=y), jackknife=jk)


def test_acat_basic():
    assert H.acat([0.3]) == pytest.approx(0.3)
    assert H.acat([0.02] * 7) == pytest.approx(0.02)
    rng = np.random.default_rng(1); P = rng.random((100000, 6))
    lev = np.mean([H.acat(p) <= 0.05 for p in P])
    assert abs(lev - 0.05) < 0.004
    assert 0 < H.acat([1e-30, 0.9]) < 1e-29


def test_marginal_wald_is_the_slice():
    ex = _export(signal=2)
    m = H.marginal_wald(ex)
    for l in range(ex.n_env):
        idx = mg.select_indices(ex, [l]); sub = mg.slice_system(ex.full, idx)
        assert m["h2"][l] == pytest.approx(mg.component_h2(mg.solve(sub), sub)[1], rel=1e-12)
        jv = [mg.component_h2(mg.solve(mg.slice_system(j, idx)), mg.slice_system(j, idx))[1] for j in ex.jackknife]
        assert m["se"][l] == pytest.approx(mg.jk_se(jv), rel=1e-10)
    r = H.marginal_h0(ex)
    assert r["min_p_env"] == int(np.nanargmin(m["p"])) and 0 <= r["p"] <= 1


def test_select_from_curves_with_h0_decision():
    from multi_gxe.compsplit import select_from_curves
    rng = np.random.default_rng(2); K, J = 5, 30
    c = np.array([0, .02, .03, .03, .03, .03]); j = c[:, None] + 0.001 * rng.standard_normal((K + 1, J))
    curves = [mg.Curve(c=c, j=j)] * 2; paths = [list(range(K))] * 2
    assert select_from_curves(curves, paths, h0_reject=False)["names"] is False
    r = select_from_curves(curves, paths, h0_reject=True)
    assert r["names"] and r["k"] >= 1
    flat = [mg.Curve(c=np.zeros(K + 1), j=0.001 * rng.standard_normal((K + 1, J)))] * 2
    r = select_from_curves(flat, paths, h0_reject=True)       # H_0 rejected elsewhere: depth at least 1
    assert r["names"] and r["k"] >= 1
