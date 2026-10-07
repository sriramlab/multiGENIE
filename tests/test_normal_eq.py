"""The slicing identity, the h2 normalisation, and the jackknife SE."""
import numpy as np
import pytest

import multi_gxe as mg


def test_load_round_trip(export, export_json):
    e = mg.load_export(export_json)
    assert e.model == export.model and e.n_env == export.n_env
    assert e.dim == 1 + 2 * e.n_env + 1
    assert np.allclose(e.full.x_l, export.full.x_l)
    assert len(e.jackknife) == export.n_jack


def test_load_rejects_partitioned_annotation(export_json):
    import json
    payload = json.loads(export_json.read_text())
    payload["gen_nbin"] = 2
    with pytest.raises(ValueError, match="gen_nbin"):
        mg.load_export(payload)


def test_indices_keep_both_kernels(export):
    idx = mg.select_indices(export, [2, 5])
    L = export.n_env
    assert idx == [0, 1 + 2, 1 + 5, 1 + L + 2, 1 + L + 5, export.dim - 1]


def test_subset_fit_is_a_slice_and_a_solve(export):
    """The point of the whole method: a sub-model is a principal submatrix."""
    subset = [0, 3, 5]
    sub = mg.slice_system(export.full, mg.select_indices(export, subset))
    direct = mg.component_h2(mg.solve(sub), sub)
    assert np.allclose(mg.fit_subset(export, subset), direct, atol=0, rtol=0)


def test_full_fit_recovers_the_truth(design):
    corr, _ = design
    truth = mg.simulate.sparse_truth(corr.shape[0], tau=0.06, causal=[0, 5])
    noiseless = mg.simulate.simulate_export(corr, truth, se=0.0, seed=0)
    comp = mg.fit_subset(noiseless, range(noiseless.n_env))
    assert comp[0] == pytest.approx(0.30, abs=1e-6)                 # h2_G
    assert comp[1:1 + noiseless.n_env] == pytest.approx(truth, abs=1e-6)


def test_drop_envs_equals_never_having_fitted_them(export):
    keep = [i for i in range(export.n_env) if i not in (1, 4)]
    dropped = mg.drop_envs(export, [export.env_names[1], export.env_names[4]])
    assert dropped.n_env == len(keep)
    assert np.allclose(mg.fit_subset(dropped, range(dropped.n_env)),
                       mg.fit_subset(export, keep))


def test_jk_se_matches_the_formula():
    v = np.array([1.0, 2.0, 4.0, 8.0])
    n = v.size
    expected = np.sqrt((n - 1) / n * np.sum((v - v.mean()) ** 2))
    assert mg.jk_se(v) == pytest.approx(expected)


def test_singular_slice_raises_rather_than_least_squaring(export):
    bad = mg.System(x_l=np.zeros((3, 3)), y_r=np.ones(3))
    with pytest.raises(ValueError, match="singular"):
        mg.solve(bad)


# ------------------------------------------------- degenerate N x E columns
def test_nxe_identity_cos_flags_a_pure_identity_column():
    """A column equal to the noise kernel has cosine one; a deviation column does not."""
    import numpy as np
    from multi_gxe import (Export, System, nxe_identity_cos,
                           drop_degenerate_nxe, select_indices, fit_subset)
    L, N = 3, 1000.0
    # trace inner products: kernel 0 is G, then 3 GxE, then 3 NxE, then noise.
    # Every kernel has tr(K_i) = N (the last column) but an ordinary one has
    # tr(K_i^2) = 3N, giving cos = N / sqrt(3N * N) = 0.577.  The identity has
    # tr(K^2) = N, giving cos = 1 exactly.
    D = 1 + L + L + 1
    T = np.eye(D) * (3.0 * N)
    T[-1, -1] = N
    T[:, -1] = T[-1, :] = N            # tr(K_i) = N for every kernel
    # NxE column 1 IS the identity: its row equals the noise row exactly.
    T[1 + L + 1, :] = T[-1, :]
    T[:, 1 + L + 1] = T[:, -1]
    ex = Export(model="G+GxE+NxE", n_env=L, env_names=["a", "b", "c"],
                full=System(x_l=T, y_r=np.ones(D)), jackknife=[])
    cos = nxe_identity_cos(ex)
    assert cos.shape == (L,)
    assert cos[1] == pytest.approx(1.0)
    assert cos[0] < 0.99 and cos[2] < 0.99

    assert drop_degenerate_nxe(ex).nxe_index == (0, 1, 2)   # v7: every NxE term is kept unless NXE_DROP=1
    import os
    os.environ["NXE_DROP"] = "1"
    try:
        pruned = drop_degenerate_nxe(ex)
    finally:
        os.environ.pop("NXE_DROP")
    assert pruned.nxe_index == (0, 2)          # with the drop on: env 1 keeps its GxE, loses its NxE
    assert pruned.n_env == L                   # the GxE side is untouched
    assert pruned.dim == 1 + L + 2 + 1
    assert pruned.full.x_l.shape == (pruned.dim, pruned.dim)


def test_select_indices_skips_a_dropped_nxe_column():
    """A subset containing the degenerate environment still builds a square system."""
    import numpy as np
    from multi_gxe import Export, System, select_indices
    L = 3
    D = 1 + L + 2 + 1
    ex = Export(model="G+GxE+NxE", n_env=L, env_names=["a", "b", "c"],
                full=System(x_l=np.eye(D), y_r=np.ones(D)), jackknife=[],
                nxe_keep=(0, 2))
    # env 1 has no NxE column, so the subset {1} yields G + GxE_1 + noise only
    assert select_indices(ex, [1]) == [0, 2, D - 1]
    # subset {0, 1} picks up NxE for env 0 alone
    assert select_indices(ex, [0, 1]) == [0, 1, 2, 1 + L, D - 1]


def test_drop_degenerate_nxe_is_a_noop_without_nxe():
    import numpy as np
    from multi_gxe import Export, System, drop_degenerate_nxe
    D = 1 + 2 + 1
    ex = Export(model="G+GxE", n_env=2, env_names=["a", "b"],
                full=System(x_l=np.eye(D), y_r=np.ones(D)), jackknife=[])
    assert drop_degenerate_nxe(ex) is ex
