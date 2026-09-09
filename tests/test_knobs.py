"""The three knobs: which column represents a block, where to stop, what to report."""
import numpy as np
import pytest

import multi_gxe as mg
from multi_gxe import combine as C
from multi_gxe import represent as R


@pytest.fixture(scope="module")
def folds():
    corr, blocks = mg.simulate.make_corr(seed=1)
    truth = mg.simulate.sparse_truth(corr.shape[0], tau=0.06, causal=[0, 5])
    A, B = mg.simulate.simulate_fold_pair(corr, truth, se=0.003, seed=11)
    return A, B, blocks, corr


def test_every_chooser_returns_one_member_per_block(folds):
    A, _, blocks, corr = folds
    ctx = R.RepContext(blocks=blocks, corr=corr, ev=mg.Evaluator(A), export=A)
    for name, choose in R.PANEL.items():
        reps = R.resolve(choose, blocks, ctx)
        if reps is None:
            assert name.startswith("none")
            continue
        assert len(reps) == len(blocks)
        assert all(r in b for r, b in zip(reps, blocks)), name


def test_chooser_answers_are_validated(folds):
    A, _, blocks, corr = folds
    ctx = R.RepContext(blocks=blocks, corr=corr, ev=mg.Evaluator(A), export=A)
    with pytest.raises(ValueError, match="not a member"):
        R.resolve(R.fixed([99] * len(blocks)), blocks, ctx)
    with pytest.raises(ValueError, match="representatives"):
        R.resolve(R.fixed([0]), blocks, ctx)


def test_centroid_and_marginal_need_what_they_need(folds):
    A, _, blocks, corr = folds
    bare = R.RepContext(blocks=blocks)
    with pytest.raises(ValueError, match="correlation"):
        R.resolve(R.centroid(), blocks, bare)
    with pytest.raises(ValueError, match="Evaluator"):
        R.resolve(R.marginal(), blocks, bare)


def test_marginal_chooser_reads_the_growth_half_only(folds):
    """Each direction chooses on the half it grows on, so the two can differ."""
    A, B, blocks, corr = folds
    a = mg.direction(A, B, blocks, choose=R.marginal())
    b = mg.direction(B, A, blocks, choose=R.marginal())
    from_A = R.resolve(R.marginal(), blocks,
                       R.RepContext(blocks=blocks, ev=mg.Evaluator(A)))
    assert a.reps == from_A
    assert b.reps is not None and len(b.reps) == len(blocks)


def test_representatives_do_not_change_what_is_fitted(folds):
    A, B, blocks, corr = folds
    r = mg.crossfit(A, B, blocks, choose=R.centroid(), corr=corr)
    assert r["represent"] == "centroid"
    # an admitted block still contributes every one of its members to the fit
    envs = mg.envs_of(blocks, r["direction_A"].path)
    assert sorted(envs) == list(range(A.n_env))


def test_scoring_the_whole_block_is_still_the_default(folds):
    A, B, blocks, _ = folds
    r = mg.crossfit(A, B, blocks)
    assert r["represent"] == "none"
    assert r["direction_A"].reps is None


def test_combiners_disagree_in_the_documented_direction(folds):
    A, B, blocks, _ = folds
    rule = mg.plateau(1.645)
    out = {name: mg.crossfit(A, B, blocks, rule=rule, combine=fn)
           for name, fn in C.PANEL.items()}
    m, av = out["min (default)"], out["avg"]
    assert m["tau"] <= av["tau"] + 1e-12          # min is a lower bound on the mean
    assert out["agreed (old intersection)"]["k"] <= m["k"]
    assert set(out["agreed (old intersection)"]["blocks"]) <= set(m["blocks"])


def test_a_combiner_can_be_any_callable(folds):
    A, B, blocks, _ = folds

    def only_when_they_agree_on_the_first_block(a, b):
        ok = bool(a.blocks) and bool(b.blocks) and a.blocks[0] == b.blocks[0]
        pick = a if a.tau <= b.tau else b
        return dict(tau=pick.tau if ok else 0.0, se=pick.se if ok else 0.0,
                    z=np.nan, k=pick.k if ok else 0,
                    blocks=pick.blocks if ok else [], names=ok)

    r = mg.crossfit(A, B, blocks, combine=only_when_they_agree_on_the_first_block)
    assert r["combine"] == "custom"
    assert r["names"] == (r["direction_A"].blocks[:1] == r["direction_B"].blocks[:1]
                          and bool(r["direction_A"].blocks))


def test_unknown_combiner_name_is_refused(folds):
    A, B, blocks, _ = folds
    with pytest.raises(ValueError, match="unknown combiner"):
        mg.crossfit(A, B, blocks, combine="whatever")
