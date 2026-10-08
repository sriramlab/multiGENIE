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


def _flat_curve(K=8, J=100, level=0.0, seed=0):
    rng = np.random.default_rng(seed)
    j = level + 0.001 * rng.standard_normal((K + 1, J))
    return mg.Curve(c=j.mean(axis=1), j=j, path=np.arange(K))


def _rising_curve(K=8, J=100, top=0.10, k_true=2, seed=0):
    rng = np.random.default_rng(seed)
    base = np.array([min(k, k_true) / k_true * top for k in range(K + 1)])
    j = base[:, None] + 0.002 * rng.standard_normal((K + 1, J))
    return mg.Curve(c=j.mean(axis=1), j=j, path=np.arange(K))


def test_the_tail_test_returns_nothing_on_a_flat_curve():
    assert mg.tail_test(0.01)(_flat_curve()) == 0


def test_the_cut_ignores_null_environments_added_to_the_pool():
    """The tail test compares with depth k, not with the full-pool endpoint, so padding the
    pool with environments that add nothing must not move the cut."""
    short = _rising_curve(K=8, k_true=2)
    long_ = _rising_curve(K=20, k_true=2)
    assert mg.tail_test(0.2)(short) == mg.tail_test(0.2)(long_)


def test_cut_depth_matches_the_rule_on_raw_arrays():
    cv = _rising_curve(k_true=2)
    assert mg.cut_depth(cv.c, cv.j, 0.2) == mg.tail_test(0.2)(cv)


def test_report_is_zero_at_k_zero():
    r = mg.report(_flat_curve(level=0.05), 0)
    assert r["k"] == 0 and r["se"] == 0.0 and r["blocks"] == []
