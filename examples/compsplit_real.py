#!/usr/bin/env python3
"""The v5 estimator on real fold exports: one row per trait.

    python examples/compsplit_real.py --exports <dir> --clusters clusters.json \
        --env-names env_names.tsv --traits traits.txt --out results/compsplit_pool42.tsv

`<dir>/<trait>/<trait>.fold{A,B}.normal_eq.json` are the two halves of one complementary
split (B = 1).  `clusters.json` holds the frozen block partition ({"clusters": [[...]]}).
N x E columns collinear with the noise kernel are dropped first (`drop_degenerate_nxe`).
Blocks are scored whole (`choose=None`).

Columns: named, k (gate depth), tau = M[k] and its jackknife se / z, tau_full = M[K],
k_set and tau_selected = M[k_set], the named environments in consensus order, and the
two halves' own full fits.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import multi_gxe as mg  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", required=True)
    ap.add_argument("--clusters", required=True)
    ap.add_argument("--env-names", default=None, help="tsv with columns env, full_name")
    ap.add_argument("--traits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-degenerate-nxe", action="store_true")
    a = ap.parse_args()

    blocks = [list(map(int, c)) for c in json.loads(Path(a.clusters).read_text())["clusters"]]
    label = {}
    if a.env_names:
        for line in Path(a.env_names).read_text().splitlines()[1:]:
            k, v = line.split("\t")[:2]
            label[k] = v
    traits = [t.strip() for t in Path(a.traits).read_text().splitlines()
              if t.strip() and not t.lstrip().startswith("#")]
    rows, t0 = [], time.time()
    for t in traits:
        fa = Path(a.exports) / t / f"{t}.foldA.normal_eq.json"
        fb = Path(a.exports) / t / f"{t}.foldB.normal_eq.json"
        if not (fa.exists() and fb.exists() and fa.stat().st_size and fb.stat().st_size):
            continue
        A, B = mg.load_export(fa), mg.load_export(fb)
        if not a.keep_degenerate_nxe:
            A, B = mg.drop_degenerate_nxe(A), mg.drop_degenerate_nxe(B)
        r = mg.complementary_split([(A, B)], blocks)
        names = A.env_names
        envs = [" + ".join(label.get(names[e], names[e]) for e in blocks[b]) for b in r["blocks"]]
        dA, dB = r["directions"]
        rows.append(dict(trait=t, named=int(r["names"]), k=r["k"], tau=r["tau"], se=r["se"], z=r["z"],
                         tau_full=r["tau_full"], k_set=r["k_set"], tau_selected=r["tau_selected"],
                         environments="; ".join(envs),
                         full_A_on_B=dA.tau_full, full_B_on_A=dB.tau_full,
                         top5_overlap=len(set(dA.path[:5]) & set(dB.path[:5]))))
        print(f"  {t:<28} named={int(r['names'])} tau={r['tau']:.4f} full={r['tau_full']:.4f} "
              f"k={r['k']:>2} set={r['k_set']} [{time.time() - t0:.0f}s]", flush=True)
    cols = list(rows[0])
    with open(a.out, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    print(f"wrote {a.out} ({len(rows)} traits)")


if __name__ == "__main__":
    main()
