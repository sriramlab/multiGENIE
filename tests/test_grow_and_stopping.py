"""The traversal is a permutation; the rules are pure functions of a curve."""
import numpy as np
import pytest

import multi_gxe as mg


def test_grow_is_exhaustive_and_nested(export, design):
    _, blocks = design
    got = mg.grow_path(mg.Evaluator(export), blocks)
    assert sorted(got["path"]) == list(range(len(blocks)))
    assert len(got["rounds"]) == len(blocks)
    # nested: each round's environment set contains the previous one
    prev = set()
    for k in range(1, len(blocks) + 1):
        envs = set(mg.envs_of(blocks, got["path"][:k]))
        assert prev < envs
        prev = envs


def test_theta_ordering_is_cheaper_than_p_ordering(export, design):
    _, blocks = design
    ev_t, ev_p = mg.Evaluator(export), mg.Evaluator(export)
    mg.grow_path(ev_t, blocks, order="theta")
    mg.grow_path(ev_p, blocks, order="p")
    assert ev_t.n_eval < ev_p.n_eval


def test_representatives_score_one_column_and_fit_the_block(export, design):
    corr, blocks = design
    reps = mg.representatives(corr, blocks)
    got = mg.grow_path(mg.Evaluator(export), blocks, reps=reps)
    # the fit still carries every member of an admitted block
    assert set(mg.envs_of(blocks, got["path"])) == set(range(export.n_env))
    with pytest.raises(ValueError, match="not a member"):
        mg.grow_path(mg.Evaluator(export), blocks, reps=[99] * len(blocks))


def _flat_curve(K=8, J=100, level=0.0, seed=0):
    rng = np.random.default_rng(seed)
    j = level + 0.001 * rng.standard_normal((K + 1, J))
    return mg.Curve(c=j.mean(axis=1), j=j, path=np.arange(K))


def _rising_curve(K=8, J=100, top=0.10, k_true=2, seed=0):
    rng = np.random.default_rng(seed)
    base = np.array([min(k, k_true) / k_true * top for k in range(K + 1)])
    j = base[:, None] + 0.002 * rng.standard_normal((K + 1, J))
    return mg.Curve(c=j.mean(axis=1), j=j, path=np.arange(K))


def test_plateau_returns_nothing_on_a_flat_curve():
    assert mg.plateau(1.645)(_flat_curve()) == 0


def test_plateau_stops_where_the_curve_stops_rising():
    cv = _rising_curve(k_true=2)
    assert mg.plateau(1.645)(cv) == 2


def test_one_se_is_monotone_in_its_knob():
    cv = _rising_curve(k_true=4)
    ks = [mg.one_se(n)(cv) for n in (0.0, 1.0, 2.0, 5.0)]
    assert ks == sorted(ks, reverse=True)


def test_window_reads_a_local_increment_not_the_tail():
    """A curve that rises late: the tail test enters, the window test does not."""
    K, J = 12, 100
    rng = np.random.default_rng(3)
    base = np.array([0.0] * 9 + [0.03, 0.06, 0.09, 0.12])
    j = base[:, None] + 0.002 * rng.standard_normal((K + 1, J))
    cv = mg.Curve(c=j.mean(axis=1), j=j, path=np.arange(K))
    assert mg.window(2, 1.645)(cv) == 0
    assert mg.plateau(1.645)(cv) > 0


def test_gate_needs_the_in_fold_p_values():
    with pytest.raises(ValueError, match="in-fold"):
        mg.gate()(_flat_curve())


def test_report_is_zero_at_k_zero():
    r = mg.report(_flat_curve(level=0.05), 0)
    assert r["k"] == 0 and r["se"] == 0.0 and r["blocks"] == []
