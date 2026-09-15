"""Complementary-split selection: average the held-out curves, cut once, rank the set."""
import numpy as np
import pytest

import multi_gxe as mg
from multi_gxe import compsplit as CS


@pytest.fixture(scope="module")
def folds():
    corr, blocks = mg.simulate.make_corr(seed=1)
    truth = mg.simulate.sparse_truth(corr.shape[0], tau=0.06, causal=[0, 5])
    A, B = mg.simulate.simulate_fold_pair(corr, truth, se=0.003, seed=11)
    return A, B, blocks


def test_consensus_frequency_then_mean_rank():
    paths = [[2, 0, 1, 3], [2, 1, 0, 3]]
    # top-1 of both is block 2
    assert CS.consensus(paths, 1) == [2]
    # top-2: 2 is in both; 0 and 1 are in one each, tied on mean rank (2.5) -> lower id
    assert CS.consensus(paths, 2) == [2, 0]
    assert CS.consensus(paths, 0) == []
    assert CS.consensus(paths, 2, "meanrank") == [2, 0]
    with pytest.raises(ValueError):
        CS.consensus([[0, 1], [0, 0]], 1)


def test_average_curves_is_the_pointwise_and_blockwise_mean():
    a = mg.Curve(c=np.array([0.0, 1.0, 2.0]), j=np.ones((3, 4)))
    b = mg.Curve(c=np.array([0.0, 3.0, 4.0]), j=3 * np.ones((3, 4)))
    m = CS.average_curves([a, b])
    assert np.allclose(m.c, [0.0, 2.0, 3.0])
    assert np.allclose(m.j, 2.0)
    with pytest.raises(ValueError):
        CS.average_curves([a, mg.Curve(c=np.zeros(4), j=np.zeros((4, 4)))])


def test_b1_equals_the_average_of_the_two_v4_directions(folds):
    A, B, blocks = folds
    out = CS.complementary_split([(A, B)], blocks)
    a, b = out["directions"]
    assert a.path == mg.direction(A, B, blocks).path
    M = 0.5 * (a.curve.c + b.curve.c)
    assert np.allclose(out["curve"].c, M)
    assert out["tau_full"] == pytest.approx(M[-1])
    if out["names"]:
        assert out["tau"] == pytest.approx(M[out["k"]])
        assert out["k_set"] <= out["k"]
        assert len(out["blocks"]) == out["k_set"]
        assert out["tau_selected"] == pytest.approx(M[out["k_set"]])


def test_signal_is_named_and_the_consensus_leads_with_causal_blocks(folds):
    # The toy simulator's curves are jagged (a correlated block can pull the joint
    # estimate down), so the default window set rule may legitimately return an empty
    # set here -- "named, no single block stands out".  Fix the set size to test the
    # consensus ranking itself.
    A, B, blocks = folds
    out = CS.complementary_split([(A, B)], blocks, set_rule=mg.fixed_k(2))
    assert out["names"]
    causal_blocks = {i for i, blk in enumerate(blocks) if 0 in blk or 5 in blk}
    assert set(out["blocks"]) == causal_blocks


def test_a_flat_curve_names_nothing():
    K, J = 6, 20
    rng = np.random.default_rng(0)
    curves = [mg.Curve(c=np.concatenate([[0.0], np.cumsum(rng.normal(0, 1e-4, K))]),
                       j=rng.normal(0, 1e-3, (K + 1, J))) for _ in range(2)]
    paths = [list(range(K)), list(reversed(range(K)))]
    out = CS.select_from_curves(curves, paths)
    assert out["names"] is False and out["blocks"] == [] and out["tau"] == 0.0


def test_more_pairs_average_more_curves(folds):
    A, B, blocks = folds
    one = CS.complementary_split([(A, B)], blocks)
    two = CS.complementary_split([(A, B), (B, A)], blocks)
    assert two["B"] == 2 and len(two["directions"]) == 4
    # the same halves listed twice give the same average
    assert np.allclose(one["curve"].c, two["curve"].c)
