"""The rule of the paper (v7): sequential tail tests on the averaged held-out curve."""
import numpy as np
import multi_gxe as mg


def _curve(step, K=12, J=100, noise=0.0004, seed=0):
    rng = np.random.default_rng(seed)
    j = np.cumsum(rng.standard_normal((K + 1, J)) * noise, axis=0); j[0] = 0.0
    for k, v in step.items():
        j[k:] += v
    return mg.Curve(c=j.mean(1), j=j)


def test_flat_curve_is_not_reported():
    cv = _curve({})
    assert mg.tail_test(0.01)(cv) == 0


def test_one_step_is_cut_at_one():
    cv = _curve({1: 0.05})
    assert mg.tail_test(0.01)(cv) == 1


def test_two_steps_are_cut_at_two_and_the_cut_ignores_null_depth():
    cv = _curve({1: 0.05, 2: 0.03})
    assert mg.tail_test(0.01)(cv) == 2
    longer = _curve({1: 0.05, 2: 0.03}, K=30)
    assert mg.tail_test(0.01)(longer) == 2          # extra null environments past the signal do not move the cut


def test_level_is_monotone_in_gamma():
    cv = _curve({1: 0.05, 2: 0.004, 3: 0.004})      # weak rises past depth 1
    k = {g: mg.tail_test(g)(cv) for g in (0.001, 0.01, 0.2)}
    assert k[0.001] <= k[0.01] <= k[0.2]


def test_select_from_curves_default_reports_iff_cut_positive():
    cvs = [_curve({1: 0.05}, seed=s) for s in (1, 2)]
    paths = [list(range(12)), [1, 0] + list(range(2, 12))]
    out = mg.compsplit.select_from_curves(cvs, paths)
    assert out["names"] and out["k"] >= 1 and len(out["blocks"]) == out["k"] and out["tau"] > 0
    flat = mg.compsplit.select_from_curves([_curve({}, seed=3), _curve({}, seed=4)], paths)
    assert flat["names"] is False and flat["blocks"] == [] and flat["tau"] == 0.0
