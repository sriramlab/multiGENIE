#!/usr/bin/env python3
"""Null and power of the v5 estimator on cached held-out curves of simulation arms.

    python examples/compsplit_sim_arms.py --curves fits_sim.npz --index index_sim.json \
        --truth blocks_pool47.tsv --out results/compsplit_null_power.tsv

The cache holds, per replicate and direction, the full-depth path, its held-out cumulative
curve and the jackknife (`<key>|AB|path`, `<key>|AB|c`, `<key>|AB|j`, same for BA), from
real engine fits on a simulated pool (L = 47, K = 30).  Every number below goes through
`multi_gxe.select_from_curves`, the same function the real-data run uses.

NULL arms (no environment carries G x E): only false positives are reported --
  P(trait named), P(any environment named).  Matched replicates only: averaged curves of
  one replicate co-move, so crossing replicates is not a valid null for this estimator.
SIGNAL arms: power and identification -- recall (fraction of causal blocks named),
  precision (fraction of named blocks that are causal, over named replicates), exact (named
  set == causal set), top1 (first named block causal), environments named.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import multi_gxe as mg  # noqa: E402

NULL = ("null47", "null47_halfn")
CONFIGS = [  # (label, gate rule, z_gate, set rule)
    ("v5 default: gate plateau(0.75) z>=2.33, set window(3,1.645)", mg.plateau(0.75), 2.33, mg.window(3, 1.645)),
    ("gate plateau(0.75), set window(5,1.28)", mg.plateau(0.75), 2.33, mg.window(5, 1.28)),
    ("gate plateau(0.75), set one_se-style max within 2.33 SE", mg.plateau(0.75), 2.33, None),
    ("gate plateau(0.75), set = gate depth", mg.plateau(0.75), 2.33, mg.everything()),
    ("gate plateau(1.0), set window(3,1.645)", mg.plateau(1.0), 2.33, mg.window(3, 1.645)),
    ("gate window(5,1.645), set window(3,1.645)", mg.window(5, 1.645), 2.33, mg.window(3, 1.645)),
]


def maxse(n):
    """smallest k within n paired SEs of the curve maximum (a one-sided 'indistinguishable')."""
    def rule(cv):
        km = int(np.argmax(cv.c))
        for k in range(km + 1):
            s = cv.se(km, k)
            if s <= 0 or (cv.c[km] - cv.c[k]) / s <= n:
                return k
        return km
    return rule


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(a.curves, allow_pickle=False)
    idx = json.loads(Path(a.index).read_text())
    header = Path(a.truth).read_text().splitlines()
    cols = header[0].split("\t")
    table = [line.split("\t") for line in header[1:]]
    table.sort(key=lambda r: int(r[cols.index("block")]))

    rows = []
    for label, gate, zg, setr in CONFIGS:
        setr = setr or maxse(2.33)
        acc = {}
        for r in idx:
            arm, key = r["unit"], r["key"]
            curves = [mg.Curve(c=z[f"{key}|{d}|c"].astype(float), j=z[f"{key}|{d}|j"].astype(float))
                      for d in ("AB", "BA")]
            paths = [z[f"{key}|{d}|path"].astype(int) for d in ("AB", "BA")]
            out = mg.select_from_curves(curves, paths, gate_rule=gate, z_gate=zg, set_rule=setr)
            S = set(out["blocks"])
            truth = set() if arm in NULL else {i for i, t in enumerate(table)
                                                if t[cols.index(f"causal_{arm}")] == "1"}
            kind = "null" if arm in NULL else ("dense(4 causal)" if len(truth) > 1 else "sparse(1 causal)")
            acc.setdefault(kind, []).append(dict(
                named=out["names"], any_env=bool(S), n_env=len(S),
                recall=len(S & truth) / len(truth) if truth else np.nan,
                precision=len(S & truth) / len(S) if (S and truth) else np.nan,
                exact=float(out["names"] and S == truth) if truth else np.nan,
                top1=float(out["blocks"][0] in truth) if (out["blocks"] and truth) else np.nan,
                ratio=(out["tau"], out["tau_full"])))
        for kind, v in acc.items():
            m = lambda f: float(np.nanmean([x[f] for x in v])) if any(np.isfinite(x[f]) for x in v if not isinstance(x[f], bool)) or f in ("named", "any_env", "n_env") else float("nan")
            row = dict(config=label, arm=kind, replicates=len(v),
                       named=f"{sum(x['named'] for x in v)}/{len(v)}",
                       P_named=m("named"), P_any_env=m("any_env"))
            if kind != "null":
                row.update(recall=m("recall"), precision=m("precision"), exact=m("exact"),
                           top1=m("top1"), envs_named=m("n_env"),
                           tau_over_full=sum(x["ratio"][0] for x in v) / sum(x["ratio"][1] for x in v))
            rows.append(row)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(a.out, "w") as fh:
        fh.write("\t".join(keys) + "\n")
        for r in rows:
            fh.write("\t".join((f"{r[k]:.4f}" if isinstance(r.get(k), float) else str(r.get(k, ""))) for k in keys) + "\n")
    for r in rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
