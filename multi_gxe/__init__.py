"""multi_gxe -- which of many candidate environments carry gene-environment interaction for a trait.

    import multi_gxe as mg

    A = mg.load_export("trait.foldA.normal_eq.json")      # one engine fit per half
    B = mg.load_export("trait.foldB.normal_eq.json")
    r = mg.complementary_split([(A, B)])                  # the method, no arguments

Four stages, all but the first paid for by one engine fit per half:

    1  fit        all L environments jointly, once per half      (the engine; docs/export_format.md)
    2  grow       forward selection on the gain in tau(S),
                  full depth, on one half                         grow.py
    3  score      the ordering on the other half -> a held-out
                  curve; average the 2B curves of B pairs         crossfit.py, compsplit.py
    4  cut        sequential tail tests on the averaged curve     stopping.py (tail_test)

`tau(S)`, the GxE heritability of any subset, is a slice of the exported normal equations and a
small solve: `docs/algorithm.md` derives the identity and states the statistics as implemented.
`atom_splits/` assembles many complementary pairs from one pass over the genotypes.
"""
from .normal_eq import (Export, System, load_export, drop_envs, select_indices,
                        slice_system, solve, component_h2, fit_subset, jk_se,
                        nxe_identity_cos, drop_degenerate_nxe, NXE_IDENTITY_COS)
from .evaluator import Evaluator, curve
from .grow import grow_path, path_curve, envs_of
from .stopping import Curve, StopRule, tail_test, tail_pvalue, cut_depth, report
from .crossfit import direction, Direction
from .compsplit import (complementary_split, select_from_curves, average_curves,
                        consensus)
from . import simulate, compsplit, nulltest, h0test
from .h0test import marginal_h0, marginal_wald, acat

__version__ = "2.1.0"

__all__ = [
    "marginal_h0", "marginal_wald", "acat",
    "Export", "System", "load_export", "drop_envs", "select_indices",
    "slice_system", "solve", "component_h2", "fit_subset", "jk_se",
    "nxe_identity_cos", "drop_degenerate_nxe", "NXE_IDENTITY_COS",
    "Evaluator", "curve",
    "grow_path", "path_curve", "envs_of",
    "Curve", "StopRule", "tail_test", "tail_pvalue", "cut_depth", "report",
    "direction", "Direction", "simulate",
    "complementary_split", "select_from_curves", "average_curves", "consensus", "compsplit",
    "__version__",
]
