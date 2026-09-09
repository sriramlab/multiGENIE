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
