#!/usr/bin/env python3
"""Stage A: G family-intact, PC-balanced atoms on the env-complete sample, refining the
paper's fold file so that atoms 0..G/2-1 = fold A and G/2..G-1 = fold B.

    ~/.conda/envs/test/bin/python scripts/atom_splits/make_atoms.py --G 12 --seed 20260928
        -> data/atom_splits/atoms_pool47.G12.tsv (+ .report.tsv)

Login node, ~5 min (the KING shards are read once).  Same inputs as
cluster_greedy_v3/folds.py: GENO, KIN glob, PCs; `--assume-unrelated` only if no kinship
table in the right ID space exists (records the assumption in the report).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from atomsplit.atoms import make_atoms                                   # noqa: E402

GENO = "/u/project/sriram/ukbiobank/data/geno/cal/filter4_no_mhc/filter4_no_mhc"
KIN = "/u/project/sriram/boyang19/Shared/Kinship/king_all.kin0.*"
PCS = "/u/scratch/z/zhengton/nonlinear_gxe/data/genetic_PCs.txt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=str(ROOT / "data/envs_pool47.env"))
    ap.add_argument("--geno", default=GENO)
    ap.add_argument("--kin", default=KIN)
    ap.add_argument("--pcs", default=PCS)
    ap.add_argument("--G", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--n-pc", type=int, default=4)
    ap.add_argument("--w-pc", type=float, default=1.0)
    ap.add_argument("--kin-threshold", type=float, default=0.0442)
    ap.add_argument("--refine-folds", default=str(ROOT / "data/cluster_greedy_v3/folds_pool47.tsv"),
                    help="fold file to refine ('' for free atoms)")
    ap.add_argument("--assume-unrelated", action="store_true")
    ap.add_argument("--pool", default="pool47")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    out = a.out or str(ROOT / f"data/atom_splits/atoms_{a.pool}.G{a.G}.tsv")
    make_atoms(a.env, a.geno, a.kin, a.pcs, a.G, a.seed, a.refine_folds or None, out,
               n_pc=a.n_pc, w_pc=a.w_pc, kin_threshold=a.kin_threshold,
               assume_unrelated=a.assume_unrelated)


if __name__ == "__main__":
    main()
