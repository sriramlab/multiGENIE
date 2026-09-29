#!/usr/bin/env python3
"""Stage D for one trait: every complementary pair of atom unions through the paper's rule.

    ~/.conda/envs/test/bin/python atom_splits/run_pairs.py --trait bmi \
        --tensor /u/scratch/z/zhengton/multi_gxe/atom_splits/bmi/bmi.atoms.G12.npz

-> results/atom_splits/<trait>.{summary,clusters,curve}.tsv + <trait>.pairs.npz.
Needs ~6 GB RAM for a G = 12 tensor; `--jobs 4` forks workers that share it.
"""
from __future__ import annotations

import os
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("MULTI_GXE_ROOT", Path.cwd()))   # analysis root: data/, results/
sys.path.insert(0, str(HERE))
from atomsplit.tensor import AtomTensor                                   # noqa: E402
from atomsplit.pairs import run_trait, Z_P, Z_NAME                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", required=True)
    ap.add_argument("--tensor", default=None,
                    help="default: /u/scratch/z/zhengton/multi_gxe/atom_splits/<trait>/<trait>.atoms.G12.npz")
    ap.add_argument("--clusters", default=str(ROOT / "results/cluster_greedy_v3/sep6/clusters_pool47.json"))
    ap.add_argument("--drop-cells", default=str(ROOT / "data/cluster_greedy_v3/drop_cells_pool47.tsv"))
    ap.add_argument("--out", default=str(ROOT / "results/atom_splits"))
    ap.add_argument("--n-pairs", type=int, default=None, help="random subset of pairs (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--z-p", type=float, default=Z_P)
    ap.add_argument("--z-name", type=float, default=Z_NAME)
    ap.add_argument("--order", default="theta", choices=["theta", "p"])
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--paper-dir", default="/u/scratch/z/zhengton/multi_gxe/sep6_pool47",
                    help="the paper's fold exports for the V4 comparison ('' to skip)")
    a = ap.parse_args()
    tensor_path = a.tensor or f"/u/scratch/z/zhengton/multi_gxe/atom_splits/{a.trait}/{a.trait}.atoms.G12.npz"
    tensor = AtomTensor.load(tensor_path)
    run_trait(tensor, a.trait, a.clusters, a.drop_cells, a.out, n_pairs=a.n_pairs, seed=a.seed,
              z_p=a.z_p, z_name=a.z_name, order=a.order, jobs=a.jobs, paper_dir=a.paper_dir or None)


if __name__ == "__main__":
    main()
