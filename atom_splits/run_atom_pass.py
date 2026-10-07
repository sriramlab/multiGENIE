#!/usr/bin/env python3
"""Stage B for one trait: two genotype passes -> <out>/<trait>.atoms.G<G>.npz

    ~/.conda/envs/test/bin/python scripts/atom_splits/run_atom_pass.py --trait bmi \
        --atoms data/atom_splits/atoms_pool47.G12.tsv --out /u/scratch/z/zhengton/multi_gxe/atom_splits

Compute node, ~128 GB, 8-12 threads; see run_atom_pass.sh for the SGE wrapper.  The
probe seed is fixed per trait so a rerun reproduces the tensor.  `--permute-env SEED`
permutes the environment rows within atoms (the V5 null); `--max-snps/--max-rows` are for
smoke tests and profiling only and are recorded in the file name and meta.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from atomsplit import Design, atom_pass, bedio                            # noqa: E402

GENO = "/u/project/sriram/ukbiobank/data/geno/cal/filter4_no_mhc/filter4_no_mhc"
COVAR = "/u/scratch/z/zhengton/nonlinear_gxe/data/genetic_PCs.txt"
PHENO_DIR = "/u/project/sriram/alipazok/Data/new_ukkb_phenotypes"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trait", required=True)
    ap.add_argument("--pheno", default=None, help="default: PHENO_DIR/<trait>.pheno")
    ap.add_argument("--geno", default=GENO)
    ap.add_argument("--covar", default=COVAR)
    ap.add_argument("--env", default=str(ROOT / "data/envs_pool47.env"))
    ap.add_argument("--atoms", default=str(ROOT / "data/atom_splits/atoms_pool47.G12.tsv"))
    ap.add_argument("--out", default="/u/scratch/z/zhengton/multi_gxe/atom_splits")
    ap.add_argument("--probes", type=int, default=10)
    ap.add_argument("--jk", type=int, default=100)
    ap.add_argument("--seed", type=int, default=None, help="default: 28000000 + (trait hash mod 1e6)")
    ap.add_argument("--snp-block", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--gram-dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--no-keep-sketch", action="store_true", help="recompute X'Cin in pass 2 (saves ~11 GB)")
    ap.add_argument("--no-cache-uall", action="store_true", help="rebuild [U|diag] slices per block (saves ~10 GB)")
    ap.add_argument("--permute-env", type=int, default=None, help="V5 null: permute E rows within atoms with this seed")
    ap.add_argument("--max-snps", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--backend", default="numpy", choices=["numpy", "engine"],
                    help="engine = the C++ passes C++ passes (build_engine_atoms.sh); numpy = bed_reader + BLAS")
    ap.add_argument("--cengine-so", default=None, help="path of the the C++ passes _cext*.so (default: the C++ passes/src/engine)")
    ap.add_argument("--seed-shift", type=int, default=0, help="added to the per-trait default probe seed (second probe draw)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(a.threads)
    except Exception:
        pass
    pheno = a.pheno or f"{PHENO_DIR}/{a.trait}.pheno"
    t0 = time.time()
    design = Design.from_files(a.geno, pheno, a.covar, a.env, a.atoms, trait=a.trait)
    if a.max_rows:
        design = design.restricted(np.arange(min(a.max_rows, design.N)))
    if a.permute_env is not None:
        design = design.with_permuted_env(a.permute_env)
    seed = a.seed if a.seed is not None else 28000000 + (sum(ord(ch) * (i + 1) for i, ch in enumerate(a.trait)) % 1000000)
    seed += a.seed_shift
    print(f"[run] {a.trait}: N={design.N:,} L={design.L} G={design.G} cov={design.n_cov} seed={seed} "
          f"threads={a.threads} load {time.time() - t0:.0f}s", flush=True)
    if a.backend == "engine":
        from atomsplit.engine import EngineEngine
        reader = EngineEngine(a.geno, design.fam_index, n_threads=a.threads, so_path=a.cengine_so)
    else:
        reader = bedio.BedBlockReader(a.geno, design.fam_index, num_threads=a.threads)
    tensor = atom_pass(design, reader, n_probes=a.probes, seed=seed, n_jk=a.jk, snp_block=a.snp_block,
                       dtype=getattr(np, a.dtype), gram_dtype=getattr(np, a.gram_dtype),
                       keep_sketch=not a.no_keep_sketch, cache_uall=not a.no_cache_uall,
                       max_snps=a.max_snps, backend=a.backend, log=lambda s: print(s, flush=True))
    tensor.meta.update(trait=a.trait, pheno=pheno, backend=a.backend, env=a.env, atoms=a.atoms, geno=a.geno,
                       permute_env=a.permute_env, max_snps=a.max_snps, max_rows=a.max_rows, seed=seed, tag=a.tag)
    tag = a.tag or ("" if a.permute_env is None else f".perm{a.permute_env}")
    if a.max_snps or a.max_rows:
        tag += f".smoke{a.max_rows or ''}x{a.max_snps or ''}"
    out = Path(a.out) / a.trait / f"{a.trait}.atoms.G{design.G}{tag}.npz"
    tmp = out.with_suffix(f".part{os.getpid()}.npz")
    tensor.save(tmp)
    os.replace(tmp, out)
    print(f"[run] wrote {out} ({tensor.nbytes() / 2**30:.2f} GB); total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
