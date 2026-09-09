"""The estimator: out of fold in both directions, and MIN squares the null rate."""
import numpy as np
import pytest

import multi_gxe as mg


@pytest.fixture(scope="module")
def folds():
    corr, blocks = mg.simulate.make_corr(seed=1)
    truth = mg.simulate.sparse_truth(corr.shape[0], tau=0.06, causal=[0, 5])
    A, B = mg.simulate.simulate_fold_pair(corr, truth, se=0.003, seed=11)
    return A, B, blocks, corr


def test_direction_scores_on_the_other_half(folds):
    A, B, blocks, _ = folds
    d = mg.direction(A, B, blocks)
    # the path is grown on A; the curve it is cut on comes from B
    grown_on_A = mg.grow_path(mg.Evaluator(A), blocks)["path"]
    assert d.path == grown_on_A
    c_on_B, _ = mg.curve(mg.Evaluator(B), [blocks[b] for b in d.path])
    assert d.tau == pytest.approx(c_on_B[d.k])


def test_min_needs_both_halves_to_name_something(folds):
    A, B, blocks, _ = folds
    both = mg.crossfit(A, B, blocks, rule=mg.plateau(1.645))
    assert both["names"] == (both["k_A"] > 0 and both["k_B"] > 0)
    assert both["tau"] == min(both["tau_A"], both["tau_B"])

    nothing_one_way = mg.crossfit(A, B, blocks, rule=mg.fixed_k(0))
    assert nothing_one_way["names"] is False
    assert nothing_one_way["tau"] == 0.0


def test_avg_names_if_either_direction_does(folds):
    A, B, blocks, _ = folds
    avg = mg.crossfit(A, B, blocks, rule=mg.plateau(1.645), combine="avg")
    assert avg["tau"] == pytest.approx(0.5 * (avg["tau_A"] + avg["tau_B"]))


def test_finds_the_signal_and_reports_below_the_full_depth_total(folds):
    A, B, blocks, _ = folds
    r = mg.crossfit(A, B, blocks, rule=mg.plateau(1.645))
    assert r["names"]
    named = set()
    for b in r["blocks"]:
        named |= set(blocks[b])
    assert {0, 5} & named                      # a causal environment is in there
    assert r["tau"] <= r["tau_full"] + 1e-12   # a prefix cannot beat the whole path


def test_null_rate_is_small_and_min_beats_single_direction():
    """Under a pure null MIN should name much less often than one direction alone."""
    corr, blocks = mg.simulate.make_corr(seed=1)
    zero = np.zeros(corr.shape[0])
    n_min = n_one = 0
    reps = 40
    for s in range(reps):
        A, B = mg.simulate.simulate_fold_pair(corr, zero, se=0.003, seed=100 * s + 1)
        n_min += mg.crossfit(A, B, blocks, rule=mg.plateau(1.645))["names"]
        n_one += mg.crossfit(A, B, blocks, rule=mg.plateau(1.645),
                             combine="A")["names"]
    assert n_min <= n_one
    assert n_min / reps <= 0.10
