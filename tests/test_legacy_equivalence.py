"""The 4.0 package must reproduce the frozen implementations, number for number.

These are the tests that make the rewrite safe to trust: the published results were
produced by the modules under `legacy/`, so anything the package computes differently
is a regression, not a cleanup.
"""
import numpy as np
import pytest

import multi_gxe as mg


def test_component_h2_matches_legacy(export_json, legacy):
    S, _ = legacy
    old = S.load_export(export_json, None, None, keep_dropped=True)
    new = mg.load_export(export_json)

    for subset in ([0], [0, 5], [1, 2, 7], list(range(new.n_env))):
        idx_old = S.select_indices(old, subset)
        sys_old = S.slice_system(old.full, idx_old)
        h2_old = S.compute_h2_from_solution(
            S.solve_linear_system(sys_old.x_l, sys_old.y_r), sys_old)["component_h2"]
        assert np.allclose(mg.fit_subset(new, subset), h2_old, atol=0, rtol=0)


def test_block_statistic_matches_legacy(export_json, legacy, design):
    """`Evaluator.evaluate` is `cluster_select.block_eval`, batched."""
    S, CS = legacy
    old = S.load_export(export_json, None, None, keep_dropped=True)
    new = mg.load_export(export_json)
    ev = mg.Evaluator(new)

    for prev, block in (([], [0]), ([], [0, 1, 2]), ([0, 1, 2], [3, 4]), ([5], [6])):
        a = CS.block_eval(old, prev, block)
        b = ev.evaluate(prev, block)
        for key in ("block_h2", "block_se", "z", "p", "cum_h2", "cum_se"):
            assert b[key] == pytest.approx(a[key], rel=1e-12, abs=1e-15), key


def test_batched_and_unbatched_agree(export, design):
    _, blocks = design
    fast = mg.Evaluator(export, batched=True).evaluate([0, 1, 2], [3, 4])
    slow = mg.Evaluator(export, batched=False).evaluate([0, 1, 2], [3, 4])
    for key in ("block_h2", "block_se", "z", "p", "cum_h2", "cum_se"):
        assert fast[key] == pytest.approx(slow[key], rel=1e-12)


def test_min_p_grow_matches_legacy_traversal(export_json, legacy, design):
    """`grow_path(order='p')` is `cluster_select.cluster_traverse`."""
    S, CS = legacy
    _, blocks = design
    old = S.load_export(export_json, None, None, keep_dropped=True)
    new = mg.load_export(export_json)

    rec = CS.cluster_traverse(old, blocks)
    got = mg.grow_path(mg.Evaluator(new), blocks, order="p")

    assert got["path"] == [int(c) for c in rec["cluster_index"]]
    assert np.allclose([r["cum"] for r in got["rounds"]], rec["cum"])
    assert np.allclose([r["p"] for r in got["rounds"]], rec["p_raw"])


def test_clustering_matches_legacy(design, legacy):
    _, CS = legacy
    corr, blocks = design
    old = CS.labels_to_clusters(CS.cluster_from_corr(corr, cut=mg.CUT))
    assert mg.cluster_from_corr(corr, cut=mg.CUT) == old == blocks
