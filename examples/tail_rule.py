"""Helpers around `multi_gxe.tail_test` (the default cut of
`complementary_split` since v7.  `apply_tail(r)` re-cuts a result dict in place with the tail test
at GAMMA (env, default 0.01) and the selection-frequency consensus; `step1_depth(c, j)` is the cut alone."""
import os
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import multi_gxe as mg
GAMMA = float(os.environ.get("GAMMA", "0.01")); NDRAW = int(os.environ.get("NDRAW", "10000"))
_RULES = {}
def _rule(gamma):
    if gamma not in _RULES: _RULES[gamma] = mg.tail_test(gamma, n_draw=NDRAW)
    return _RULES[gamma]
def step1_depth(c, Jm, gamma=GAMMA):
    return int(_rule(gamma)(mg.Curve(c=np.asarray(c, float), j=np.asarray(Jm, float))))
def apply_tail(r, gamma=GAMMA):
    M = r["curve"]; k = step1_depth(M.c, M.j, gamma)
    paths = [list(dd.path) for dd in r["directions"]]; order = mg.consensus(paths, k, "freq") if k > 0 else []
    r.update(k=k, k_hat=k, k_N=k, k_tilde=k, names=bool(k > 0), tau=float(M.c[k]) if k > 0 else 0.0,
             se=M.se_at(k) if k > 0 else 0.0, blocks=order, named_blocks=order, tail_blocks=[])
    r["z"] = r["tau"] / r["se"] if r["se"] > 0 else float("nan"); r["rule"] = f"tail_test gamma={gamma}"
    return r
