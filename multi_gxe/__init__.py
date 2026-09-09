"""multi_gxe -- which of many candidate environments carry real GxE signal for a trait.

    from multi_gxe import load_export, cluster_from_corr, Evaluator, grow_path
    from multi_gxe import crossfit, plateau, one_se

    export       = load_export("trait.normal_eq.json")
    blocks       = cluster_from_corr(corr, cut=0.8)
    result       = crossfit(export_A, export_B, blocks, rule=one_se(2.0))

Five stages.  Three of them are knobs, and they are the three you are meant to change:

    1   cluster    group the environments from E alone           cluster.py
    1b  choose     one candidate environment per block  <- knob  represent.py
    2   grow       full-depth greedy path over the blocks        grow.py
    3   cut        how many leading blocks to keep      <- knob  stopping.py
    4   report     what the two half-sample directions
                   turn into one result                 <- knob  combine.py

    crossfit(export_A, export_B, blocks, choose=..., rule=..., combine=...)

`docs/algorithm.md` derives the identity that makes stages 2 and 3 cost nothing once
the engine has run once; `CHANGELOG.md` says what each version changed and why.
"""
from .normal_eq import (Export, System, load_export, drop_envs, select_indices,
                        slice_system, solve, component_h2, fit_subset, jk_se,
                        nxe_identity_cos, drop_degenerate_nxe, NXE_IDENTITY_COS)
from .evaluator import Evaluator, curve
from .cluster import (cluster_from_corr, pairwise_corr, corr_from_env_file,
                      representatives, CUT, LINKAGE)
from .grow import grow_path, path_curve, envs_of
from .stopping import (Curve, StopRule, plateau, one_se, window, gate, argmax,
                       fixed_k, nothing, everything, report, PANEL)
from .crossfit import crossfit, direction, recut, Direction
from . import simulate, represent, combine

__version__ = "4.2.0"

__all__ = [
    "Export", "System", "load_export", "drop_envs", "select_indices",
    "slice_system", "solve", "component_h2", "fit_subset", "jk_se",
    "nxe_identity_cos", "drop_degenerate_nxe", "NXE_IDENTITY_COS",
    "Evaluator", "curve",
    "cluster_from_corr", "pairwise_corr", "corr_from_env_file",
    "representatives", "CUT", "LINKAGE",
    "grow_path", "path_curve", "envs_of",
    "Curve", "StopRule", "plateau", "one_se", "window", "gate", "argmax",
    "fixed_k", "nothing", "everything", "report", "PANEL",
    "crossfit", "direction", "recut", "Direction", "simulate", "represent", "combine",
    "__version__",
]
