"""Small, fast checks of the atom-split machinery (pytest, ~20 s)."""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from atomsplit import Design, atom_pass, bedio, dense, toy                    # noqa: E402
from atomsplit.atoms import assign_atoms, report_atoms                       # noqa: E402
from atomsplit.pairs import enumerate_pairs                                  # noqa: E402

N, M, L, G, J, B = 400, 600, 3, 4, 5, 6


@pytest.fixture(scope="module")
def setup():
    t = toy.make_toy(N=N, M=M, L=L, G=G, seed=3, miss_rate=0.01)
    d = toy.design_from_toy(t)
    X = dense.standardized_matrix(t["codes"])
    Z = d.probes(3, B)
    tensor = atom_pass(d, bedio.ArrayBlockReader(t["codes"]), n_probes=B, seed=3, n_jk=J,
                       snp_block=97, dtype=np.float64, gram_dtype=np.float64, log=None)
    return t, d, X, Z, tensor


def rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(np.asarray(b))))


@pytest.mark.parametrize("atoms", [(0, 1, 2, 3), (0, 1), (2, 3), (1,), (0, 3)])
def test_assembler_matches_dense(setup, atoms):
    t, d, X, Z, tensor = setup
    T, q, Tj, qj = tensor.systems(atoms)
    T0, q0, Tj0, qj0 = dense.fixed_projector_system(d, X, np.isin(d.atom, atoms), Z, J)
    assert rel(T, T0) < 1e-9 and rel(q, q0) < 1e-9
    assert rel(Tj, Tj0) < 1e-9 and rel(qj, qj0) < 1e-9
    assert rel(T[None] - Tj, T0[None] - Tj0) < 1e-7


def test_nc_and_fix2(setup):
    t, d, X, Z, tensor = setup
    T, q, Tj, qj = tensor.systems(range(G))
    P = d.P_all
    assert abs(T[P - 1, P - 1] - (N - d.n_cov)) < 1e-8
    assert abs(T[P - 1, 0] - (N - d.n_cov)) < 1e-8
    assert np.allclose(Tj[:, P - 1, P - 1], N - d.n_cov)
    assert abs(q[P - 1] - np.sum(d.ytil ** 2)) < 1e-8      # noise q = y~'y~ on the full sample


def test_export_interface(setup):
    t, d, X, Z, tensor = setup
    pytest.importorskip("multi_gxe")
    import multi_gxe as mg
    ex = tensor.assemble((0, 1))
    assert ex.dim == 2 * L + 2 and ex.n_jack == J
    r = mg.Evaluator(ex).joint()
    assert np.isfinite(r["cum_h2"]) and r["cum_se"] > 0


def test_bed_roundtrip(tmp_path):
    t = toy.make_toy(N=50, M=40, L=2, G=2, seed=5, miss_rate=0.05)
    files = toy.write_toy(t, tmp_path, "rt")
    d = Design.from_files(files["geno"], files["pheno"], files["covar"], files["env"], files["atoms"],
                          log=lambda s: None)
    r = bedio.BedBlockReader(files["geno"], d.fam_index)
    assert np.array_equal(r.read(0, 40), t["codes"])
    assert np.allclose(d.y, t["y"]) and np.allclose(d.E, t["E"])


def test_imputation_is_deterministic_and_hwe():
    codes = np.full((2000, 3), bedio.MISSING, dtype=np.int8)
    codes[:1000] = np.array([0, 1, 2], dtype=np.int8)[None, :] * 0 + np.array([[0, 1, 2]], dtype=np.int8)
    codes[:1000, 0] = 1                                    # p = 0.5 for SNP 0
    a = bedio.hwe_impute(codes.copy(), np.arange(3), np.arange(2000))
    b = bedio.hwe_impute(codes.copy(), np.arange(3), np.arange(2000))
    assert np.array_equal(a, b)
    freq = np.bincount(a[1000:, 0], minlength=3) / 1000
    assert abs(freq[1] - 0.5) < 0.06 and abs(freq[0] - 0.25) < 0.06


def test_atoms_family_intact_balanced_refining_folds():
    rng = np.random.default_rng(0)
    n = 5000
    fam = np.repeat(np.arange(2200), rng.multinomial(n - 2200, np.ones(2200) / 2200) + 1)[:n]
    fam = fam[:n]
    P = rng.standard_normal((n, 4))
    fold = (fam % 2)                                    # families never straddle
    atom = assign_atoms(fam, P, 6, seed=1, fold=fold)
    rep = report_atoms(atom, fam, P, fold)
    assert rep["families_straddling_atoms"] == 0
    assert rep["atoms_inconsistent_with_fold"] == 0
    sizes = np.bincount(atom, minlength=6)
    assert sizes.max() - sizes.min() < 0.05 * n / 6 + 10
    assert rep["max_PC_z_atom_vs_rest"] < 1.0


def test_enumerate_pairs():
    ps = enumerate_pairs(6)
    assert len(ps) == 10 and ps[0] == ((0, 1, 2), (3, 4, 5))
    assert all(sorted(A + B) == list(range(6)) for A, B in ps)
    assert len(enumerate_pairs(8, n_pairs=7, seed=1)) == 7


def test_heldout_tau_reproduces_curve_points(tmp_path):
    """With m = G/2 blocks held out, one ordering survives per subset -- the one grown on the
    complement -- and the held-out refit of its consensus is that direction's curve point c(k)."""
    import json
    sys.path.insert(0, str(HERE.parent))
    from atomsplit.pairs import run_trait
    from atomsplit.heldout import heldout_tau, n_eff
    t = toy.make_toy(N=600, M=600, L=5, G=G, seed=4, h2_gxe=(0.05, 0.03))   # run_trait summarises the top 5: L >= 5
    tensor = atom_pass(toy.design_from_toy(t), bedio.ArrayBlockReader(t["codes"]), n_probes=B, seed=4, n_jk=J, snp_block=97, log=None)
    names = list(tensor.env_names); cj = tmp_path / "toy.clusters.json"
    json.dump(dict(pool="toy", trait="toy", K=len(names), clusters=[[i] for i in range(len(names))], env_names=names), cj.open("w"))
    run_trait(tensor, "toy", str(cj), None, str(tmp_path), n_pairs=None, jobs=1, paper_dir=None, log=lambda s: None)
    z = dict(np.load(tmp_path / "toy.pairs.npz", allow_pickle=True))
    C, paths, pairs, po = np.asarray(z["C"], float), np.asarray(z["paths"]), np.asarray(z["pairs"]), np.asarray(z["pair_of"])
    k = 1
    chk = heldout_tau(tensor, z, k_g=k, m=G // 2)
    grow = [set(pairs[po[d]][: G // 2]) if d % 2 == 0 else set(pairs[po[d]][G // 2:]) for d in range(len(paths))]
    for r in chk["rows"]:
        d = [d for d in range(len(paths)) if not (grow[d] & set(r["subset"]))]
        assert len(d) == 1 and abs(r["tau"] - C[d[0], k]) < 1e-10
    h = heldout_tau(tensor, z, k_g=k)
    assert h["m"] == G // 4 and h["n_subsets"] == G and np.isfinite(h["tau"]) and h["se"] >= 0
    assert abs(sum(h["comps"]) - h["tau"]) < 1e-10 and np.isfinite(n_eff(h["comps"])) or h["tau"] <= 0
