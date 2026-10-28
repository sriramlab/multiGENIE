#!/usr/bin/env python3
"""Validation V1: exact equivalence of the atom assembler with a dense N x N reference.

    ~/.conda/envs/test/bin/python scripts/atom_splits/validate_v1.py [--N 3000 --M 5000 --L 5 --G 6 --J 10]

Simulates genotypes with missingness, L environments (one binary), 2 covariates and a
G+GxE+NxE phenotype; builds the atom tensor with the streaming code (float64 AND float32,
in-memory reader AND a written .bed through bed_reader); and for several unions of atoms
compares T, q and every delete-one-block system with `dense.fixed_projector_system`
computed from explicit kernels and the SAME probes.  Also reports the Hutchinson error
against exact traces and the toy-data magnitude of V3 (GENIE's within-half convention).

Pass criteria: float64 max relative difference < 1e-8 on T, q, T_j, q_j; float32 < 1e-3.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from atomsplit import Design, atom_pass, bedio, dense, toy          # noqa: E402
from atomsplit.tensor import AtomTensor                              # noqa: E402

SCRATCH = Path(__import__("os").environ.get("ATOM_SCRATCH", "/tmp/atom_toy"))      # where the toy .bed is written


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def compare(tensor: AtomTensor, design: Design, X: np.ndarray, Z: np.ndarray, atoms, J, label):
    mask = np.isin(design.atom, atoms)
    T, q, Tj, qj = tensor.systems(atoms)
    T0, q0, Tj0, qj0 = dense.fixed_projector_system(design, X, mask, Z, J)
    r = dict(T=rel(T, T0), q=rel(q, q0), Tj=rel(Tj, Tj0), qj=rel(qj, qj0))
    # the delete-one DIFFERENCES are the fragile part: compare T - T_j too
    r["dT"] = rel(T[None] - Tj, T0[None] - Tj0)
    print(f"  {label:28s} atoms={list(atoms)!s:22s} N_h={int(mask.sum()):5d}  "
          + "  ".join(f"{k} {v:.2e}" for k, v in r.items()))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=3000)
    ap.add_argument("--M", type=int, default=5000)
    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--G", type=int, default=6)
    ap.add_argument("--J", type=int, default=10)
    ap.add_argument("--B", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--skip-bed", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    t = toy.make_toy(N=a.N, M=a.M, L=a.L, G=a.G, seed=a.seed)
    design = toy.design_from_toy(t)
    X = dense.standardized_matrix(t["codes"])
    Z = design.probes(a.seed, a.B)
    print(f"toy: N={a.N} M={a.M} L={a.L} G={a.G} J={a.J} B={a.B}; missing {np.mean(t['codes'] == bedio.MISSING):.4f}; "
          f"cov {design.n_cov}; NC_total {design.nc_per_atom().sum():.3f} = N - C = {a.N - design.n_cov}")

    halves = [tuple(range(a.G)), tuple(range(a.G // 2)), tuple(range(a.G // 2, a.G)), (0,), (1, 3, 4)]
    ok = True
    worst = {}
    for dtype, gdtype, tol in ((np.float64, np.float64, 1e-8), (np.float64, np.float32, 1e-5),
                              (np.float32, np.float32, 1e-3)):
        name = f"{np.dtype(dtype).name}/grams {np.dtype(gdtype).name}"
        print(f"\n== streaming pass in {name} (in-memory reader) ==")
        reader = bedio.ArrayBlockReader(t["codes"])
        tensor = atom_pass(design, reader, n_probes=a.B, seed=a.seed, n_jk=a.J, snp_block=700,
                           dtype=dtype, gram_dtype=gdtype, keep_sketch=(dtype is np.float64),
                           cache_uall=(dtype is np.float32), log=lambda s: None)
        for h in halves:
            r = compare(tensor, design, X, Z, h, a.J, name)
            w = max(r.values())
            worst[name] = max(worst.get(name, 0.0), w)
            if w > tol:
                ok = False
                print(f"  !! exceeds tolerance {tol:g}")
        if dtype is np.float64 and gdtype is np.float64:
            tensor64 = tensor
            # save/load round trip
            p = SCRATCH / "toy_tensor.npz"
            tensor.save(p)
            t2 = AtomTensor.load(p)
            T1, q1, Tj1, qj1 = tensor.systems((0, 2, 5))
            T2, q2, Tj2, qj2 = t2.systems((0, 2, 5))
            assert rel(T1, T2) < 1e-12 and rel(Tj1, Tj2) < 1e-12 and rel(q1, q2) < 1e-12 and rel(qj1, qj2) < 1e-12
            print("  save/load round trip: exact")

    if not a.skip_bed:
        print("\n== .bed round trip through bed_reader (float64) ==")
        files = toy.write_toy(t, SCRATCH, "toy", half_atoms=np.arange(a.G // 2))
        d2 = Design.from_files(files["geno"], files["pheno"], files["covar"], files["env"], files["atoms"],
                               trait="toy", log=lambda s: print("  " + s))
        assert d2.N == design.N and np.allclose(d2.ytil, design.ytil) and np.array_equal(d2.atom, design.atom)
        reader = bedio.BedBlockReader(files["geno"], d2.fam_index, num_threads=2)
        codes_bed = reader.read(0, a.M)
        assert np.array_equal(codes_bed, t["codes"]), "bed round trip differs"
        tensor_bed = atom_pass(d2, reader, n_probes=a.B, seed=a.seed, n_jk=a.J, snp_block=1000,
                               dtype=np.float64, gram_dtype=np.float64, log=lambda s: None)
        for h in halves[:3]:
            r = compare(tensor_bed, design, X, Z, h, a.J, "bed float64")
            if max(r.values()) > 1e-8:
                ok = False
        print(f"  toy files for V2 written under {SCRATCH}")

    print("\n== Hutchinson vs exact traces (all atoms, then fold A) ==")
    for h in (tuple(range(a.G)), tuple(range(a.G // 2))):
        mask = np.isin(design.atom, h)
        Tex = dense.exact_traces(design, X, mask)
        T, _, _, _ = tensor64.systems(h)
        P = design.P_all
        snp = slice(0, design.P_snp)
        rr = np.abs(T[snp, snp] - Tex[snp, snp]) / np.abs(Tex[snp, snp])
        print(f"  atoms {list(h)}: SNP-kernel block max rel err {rr.max():.3f}, median {np.median(rr):.3f} "
              f"(B = {a.B} probes); tr(K_g^2) exact {Tex[0, 0]:.1f} vs {T[0, 0]:.1f}")
        if rr.max() > 0.5:
            ok = False

    print("\n== V3 on toy data: GENIE's within-half convention vs fixed projector (fold A) ==")
    rows = np.flatnonzero(np.isin(design.atom, range(a.G // 2)))
    Zh = Z[rows]
    Th, qh, Tjh, qjh = dense.genie_half_system(design, t["codes"], rows, Zh, a.J)
    Ta, qa, Tja, qja = tensor64.systems(tuple(range(a.G // 2)))
    print(f"  T rel diff {rel(Ta, Th):.2e}   q rel diff {rel(qa, qh):.2e}   (same probes restricted; differences = "
          "within-half W, SNP stats and y scaling)")
    try:
        sys.path.insert(0, str(HERE.parent))
        import multi_gxe as mg
        exA = tensor64.assemble(tuple(range(a.G // 2)))
        exH = mg.Export(model="G+GxE+NxE", n_env=a.L, env_names=design.env_names,
                        full=mg.System(Th, qh), jackknife=[mg.System(Tjh[j], qjh[j]) for j in range(a.J)])
        ja, jh = mg.Evaluator(exA).joint(), mg.Evaluator(exH).joint()
        print(f"  unselected sum h2_GxE on fold A: atom {ja['cum_h2']:.5f} +- {ja['cum_se']:.5f}   "
              f"GENIE-convention {jh['cum_h2']:.5f} +- {jh['cum_se']:.5f}   "
              f"diff / SE = {(ja['cum_h2'] - jh['cum_h2']) / max(jh['cum_se'], 1e-12):.2f}")
    except Exception as e:                                  # multi_gxe not importable here
        print(f"  (multi_gxe comparison skipped: {e})")

    print(f"\nworst relative differences: {worst};  elapsed {time.time() - t0:.1f}s")
    print("V1 PASS" if ok else "V1 FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
