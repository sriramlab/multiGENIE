#!/usr/bin/env python3
"""Many complementary splits from one genotype pass, end to end on simulated data (runs anywhere, ~1 min).

    python examples/atom_splits_example.py            # N=3000, M=5000, L=5, G=6 atoms -> C(6,3)/2 = 10 complementary pairs

1. simulate genotypes, environments, covariates and a G+GxE+NxE phenotype (atomsplit.toy);
2. partition the people into G atoms and run the two streaming passes ONCE (atomsplit.atom_pass) -> an AtomTensor
   holding per-atom Gram tensors of the Hutchinson sketches and the per-atom phenotype terms;
3. assemble ANY half (a union of G/2 atoms) and its complement into multi_gxe.Export objects (no genotype access);
4. run the rule of the paper on every complementary pair, and on the average of all their held-out curves:
   forward selection on the gain in tau, scored on the other half, sequential tail tests at gamma = 0.05,
   consensus set by selection frequency (multi_gxe.complementary_split with no arguments).
The real analysis is the same with G = 12 (462 pairs), one C++ pass per trait (atom_splits/run_atom_pass.sh), and
atom_splits/run_pairs.py / recut_pairs.py / stability_vs_B.py on the saved pairs."""
import sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "atom_splits"))
import multi_gxe as mg
from atomsplit import atom_pass, bedio, toy
from atomsplit.pairs import enumerate_pairs

t0 = time.time(); N, M, L, G = 3000, 5000, 5, 6
t = toy.make_toy(N=N, M=M, L=L, G=G, seed=1, h2_gxe=(0.05, 0.03))        # GxE on environments 0 and 1
design = toy.design_from_toy(t)
print(f"toy: N={N} M={M} L={L} G={G}; true h2_GxE 0.05 on e0, 0.03 on e1, 0 elsewhere")
tensor = atom_pass(design, bedio.ArrayBlockReader(t["codes"]), n_probes=10, seed=1, n_jk=20, snp_block=700, log=lambda s: None)
print(f"two passes over the genotypes: {time.time() - t0:.0f} s -> tensor with {tensor.G} atoms, {tensor.J} jackknife blocks")

pairs = enumerate_pairs(G)                                                # every complementary pair of 3-atom halves
print(f"{len(pairs)} complementary pairs assembled from the tensor, no further genotype access")
exports = [(tensor.assemble(A), tensor.assemble(B)) for A, B in pairs]
names = exports[0][0].env_names
single = mg.complementary_split([exports[0]])                              # one split: the usual analysis
print(f"\npair 0 alone:   reported={single['names']} k_g={single['k']} tau={single['tau']:.4f} +- {single['se']:.4f} "
      f"set={[names[b] for b in single['blocks']]}")
res = mg.complementary_split(exports)                                      # B = 10 pairs: 20 held-out curves averaged
print(f"all {len(pairs)} pairs: reported={res['names']} k_g={res['k']} tau={res['tau']:.4f} +- {res['se']:.4f} "
      f"set={[names[b] for b in res['blocks']]}  (full-pool value {res['tau_full']:.4f})")
ks = [mg.complementary_split([e])["k"] for e in exports]
print(f"cut of each single pair: {ks}  (the average is the stable one)")
f = np.zeros(L)
for d in res["directions"]:
    f[list(d.path[:res["k"]])] += 1 / len(res["directions"])
print("selection frequency in the top k_g over the 2B orderings:", {n: round(float(v), 2) for n, v in zip(names, f)})

# the reported estimate: tau-hat, the held-out GxE heritability of the consensus set (atomsplit/heldout.py), and the
# bootstrap p-value of the trait-level test (multiplicity control across traits)
import json, tempfile
from atomsplit.pairs import run_trait
from atomsplit.heldout import heldout_tau
tmp = Path(tempfile.mkdtemp()); cj = tmp / "toy.clusters.json"
json.dump(dict(pool="toy", trait="toy", K=L, clusters=[[i] for i in range(L)], env_names=list(names)), cj.open("w"))
run_trait(tensor, "toy", str(cj), None, str(tmp), n_pairs=None, jobs=1, paper_dir=None, log=lambda s: None)
z = dict(np.load(tmp / "toy.pairs.npz", allow_pickle=True)); cbar, jbar = z["C"].mean(0), z["J"].mean(0)
h = heldout_tau(tensor, z, k_g=mg.cut_depth(cbar, jbar))
print(f"held-out tau-hat of the consensus set {h['S_hat']}: {h['tau']:.4f} +- {h['se']:.4f} "
      f"(each refit on G/4 = {h['m']} atom(s), {h['n_subsets']} subsets); p-value of the trait-level test "
      f"{mg.tail_pvalue(cbar, jbar):.2g}")
print("(N = 3000 is a toy: the values are noisy; the mechanics, one pass -> every pair -> one averaged cut, are the point)")
