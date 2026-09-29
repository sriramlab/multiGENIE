# atom_splits: many complementary splits from one genotype pass

The code behind the Methods subsection "Many complementary splits from one pass over the
genotypes" and the Results paragraph on stability against the number of pairs (Figure
`fig:stability_B`).  The people are partitioned once into
G family-intact atoms; every half is a union of G/2 atoms; the normal equations of any
half (and of all its J = 100 delete-one-SNP-block jackknife systems) are assembled from
per-atom Gram tensors computed in two passes over the genotypes.  Downstream the halves
are ordinary `multi_gxe.Export` objects, so the paper's algorithm runs unchanged
(`crossfit.direction`, plateau rule, `select_from_curves`).  With G = 12 one trait
costs about the Mailman work of 12-13 half fits per pass (two passes) and yields 462
complementary pairs instead of one.

Why this addresses the unstable environment sets: the named set of a single split is
dominated by which people fell on which side.
With 462 pairs per trait the selection frequency f_C of every cluster, the pair-to-pair
spread of the held-out curve, and an ESCV-style stability curve are measured instead of
guessed from B = 1-3 splits -- at a fixed cost.

## Layout

| file | stage | what |
|---|---|---|
| `atomsplit/model.py` | -- | `Design`: the fixed preprocessing (W, Q, V, y~) mirroring the engine `prepare()`; kernel layout |
| `atomsplit/bedio.py` | -- | BED blocks exactly as the engine decodes, standardises and imputes them (`bed_reader`, count_A1=False) |
| `atomsplit/atoms.py`, `make_atoms.py` | A | G atoms refining `folds_pool47.tsv` (atoms 0..5 = fold A, 6..11 = fold B) |
| `atomsplit/stream.py`, `run_atom_pass.py`, `run_atom_pass.sh` | B | the two passes -> `<scratch>/atom_splits/<trait>/<trait>.atoms.G12.npz` |
| `atomsplit/tensor.py`, `assemble_export.py` | C | `AtomTensor.assemble(atoms) -> Export`; JSON in the engine's schema |
| `atomsplit/pairs.py`, `run_pairs.py`, `run_pairs.sh`, `plot_atom_pairs.py` | D | all pairs through the paper's rule; f_C, ES(k), cores, pair-0 = paper split |
| `atomsplit/dense.py`, `validate_v1.py`, `tests/` | V1 | dense N x N reference, exact equivalence (runs anywhere, ~5 min) |
| `validate_v2.py` | V2/V3 | against the engine itself on toy data (needs a node where `_cext` loads) |

## Conventions mirrored from the engine (Step 0 of the plan)

* kernels `V K_r V` with `V = I - W(W'W)^{-1}W'`, `W = [1, PCs, E]`; here `V` is the trait's
  FULL-sample projector and a half's kernel is `D_h V K_r V D_h`;
* SNP standardisation `sd = sqrt(2p(1-p))`, monomorphic sd 1, HWE imputation with the engine's hash;
* Hutchinson `T = (1/B) sum_p (K_r z_p)'(K_s z_p)`, Gaussian probes `default_rng(seed).standard_normal((N,B), float32)`;
* Fix 2: `T[noise,noise] = T[noise,G] = NC`, here `NC_h = sum_{n in h}(1 - h_nn)` (= N - C on the full sample);
* jackknife: `alpha = M/(M - M_j)`, `T_j = alpha_r alpha_s <U - U_j, U - U_j>/B`, `q_j = (q - q_j) alpha`.

What differs from a half-sample engine run, by design: W, SNP statistics, E and the y scale are
those of the full analysed sample (plan "Hard constraint").  V1 shows the assembler is exact for
the fixed-projector model; V3 measures the difference to the engine's within-half convention.

## Run (one command per trait list)

```bash
bash atom_splits/submit.sh                                            # traits_priority.txt: whr, creatinine, testosterone, hba1c, hdl, ldl, bp_diastolic
bash atom_splits/submit.sh atom_splits/traits_paper41.txt   # all 41 (~80 node-hours)
~/.conda/envs/test/bin/python atom_splits/collect_atom_results.py     # when the chain has finished
```
`submit.sh` chains three idempotent arrays task by task (`-hold_jid_ad`): `run_atom_pass.sh`
(tensor, C++ backend) -> `run_pairs.sh` (462 pairs through the paper's rule) -> `run_stability.sh`
(`stability_vs_B.py`: disjoint groups of B pairs re-cut with the paper's rule; k_g, tau, leading cluster
and fixed-depth frequency-set agreement per B).  Re-run the same command after a failure; finished
stages are skipped.

## Run, stage by stage

```bash
PY=~/.conda/envs/test/bin/python
# A (login node, ~5 min; reads the KING shards once)
$PY atom_splits/make_atoms.py --G 12 --seed 20260928        # -> data/atom_splits/atoms_pool47.G12.tsv
# B (compute nodes; smoke test first)
qsub -t 18 -v SMOKE=1 atom_splits/run_atom_pass.sh            # bmi, 20k rows x 20k SNPs, J=10
N=$(grep -cvE '^\s*(#|$)' atom_splits/traits_paper41.txt)
qsub -t 1-$N -tc 10 atom_splits/run_atom_pass.sh              # 41 traits, ~4-6 h each on 8 cores
# D
qsub -t 1-$N atom_splits/run_pairs.sh                         # -> results/atom_splits/<trait>.*
$PY atom_splits/plot_atom_pairs.py --trait bmi
# validation
$PY atom_splits/validate_v1.py                                # anywhere
~/.conda/envs/engine/bin/python atom_splits/validate_v2.py      # AVX-512 node
# V5 null: one permutation replicate = one Stage B pass
qsub -t 18 -v PERM=1 atom_splits/run_atom_pass.sh && qsub -t 18 -v TAG=.perm1 atom_splits/run_pairs.sh
```

Outputs of Stage D per trait: `summary.tsv` (named, k_g, tau, se, sd over pairs, k_ES, core
sizes, pair-0 and paper-split values), `clusters.tsv` (f_C at k_g / 1 / 3 / 5, mean rank,
held-out component, membership in the consensus and core sets), `curve.tsv` (Mbar, jackknife
SE, SD over pairs, ES_tau, ES_c per k), `pairs.npz` (every curve, jackknife and ordering).

## the C++ passes backend (`--backend engine`)

The C++ passes are an addition to the the engine engine (v0.10.27, the copy with the normal-equation
export).  The engine itself is not part of this repository; `engine_backend/` holds what was added:
`atom_pass.{h,cpp}` (goes to `csrc/streaming/`) and `engine_0.10.27_atoms.patch` (two bindings
`_cext.atom_pass1` / `_cext.atom_pass2`, one CMake source line, version `0.10.27+atoms.1`).
Apply both to a copy of the engine tree, point `the engine_ATOMS` at it (default
`atom_splits/engine_backend/the C++ passes`) and build with `build_engine_atoms.sh`.  The passes use
the engine's own `BEDReader::read_block` (subset rows, SNP statistics, HWE imputation hash,
standardisation -- exactly the engine's), Eigen GEMMs parallelised over column/row chunks with
OpenMP, the projection and the per-atom Grams in place on the numpy buffers.  Python keeps the
orchestration (`Design`, the input columns, S_full, the diagonal sketches).

Why not Mailman: the engine's Mailman multiplies are tuned for ~10 right-hand columns.  With the atom pass's
5,760-6,336 dense columns its lookup table no longer fits in cache (`compute_segment_size_for_cols`
shrinks the segment to ~3 SNPs), and its per-SNP-per-column cost (7-12 us at N = 107k, single thread)
is no better than dense sgemm (~7-14 us).  The gain of the C++ path is the fused reader, no float32
block materialisation on the Python side, and no Python temporaries.

```bash
qrsh -l h_data=8G,h_rt=2:00:00 -pe shared 4                # compute node (module gcc >= 9.3, engine env)
bash atom_splits/build_engine_atoms.sh                # cmake build + GLIBC ABI gate + load test
conda activate engine                                         # the module is a cpython-3.12 extension: NOT the system python
python atom_splits/validate_backend.py              # toy exactness (engine vs dense; vs numpy too if bed_reader is installed) + timing
qsub -t 18 -v SMOKE=1,BACKEND=engine atom_splits/run_atom_pass.sh
```
`validate_backend.py` is the acceptance test of a build.

## Memory and cost (G = 12, L = 47, B = 10, N ~ 210k, M = 454k)

Pass 1 holds Cin 5.3 GB, U 4.8 GB, the sketch store 11.5 GB (`--no-keep-sketch` to drop) and the
[U|diag] cache 9.7 GB (`--no-cache-uall`); pass 2 adds U_j 4.8 GB and the Grams 4.8 GB.  Tensor
file ~5 GB per trait (float32 Grams).  Flops: 2 x 1.1e15 for the X-steps + ~4e14 for the Grams.

## Real atoms (data/atom_splits/atoms_pool47.G12.tsv, seed 20260928)

219,209 env-complete individuals, 218,646 families (largest 3), atoms of 17,882-18,731 people
(spread 4.6%, so a half of 6 atoms is within ~1% of N/2), no family straddles atoms, every atom
refines its fold, max |z| of any PC (atom vs rest) 0.044.

## Paths

Every script takes its inputs as arguments.  The defaults resolve `data/` and `results/` against
`MULTI_GXE_ROOT` (default: the current directory), scratch output against `MULTI_GXE_SCRATCH`, and
the genotype, kinship, principal-component and phenotype files against the paths of the cluster the
analysis ran on; none of those files is in this repository.  `folds.py` writes the complementary
split of the main analysis, which the atoms refine.

## Status (2026-09-29)

Validation, as quoted in the paper: the assembly reproduces dense-kernel computations on toy data
(`validate_v1.py`, float64 exact to 4e-13; float32 pipeline 3e-6; .bed round trip exact), and all
atoms assembled with the engine's probes reproduce the full-sample engine export (`validate_v2.py`:
T 5e-8, q 1e-6, jackknife 7e-8 relative).  The atoms of the real data are as described above.

BMI is the one trait run at full size: tensor 6,829 s on 8 AVX-512 threads (4.57 GB), 462 pairs
1,792 s.  tau = 0.098 +- 0.013 over the pairs; the pair of the main analysis rebuilt from atoms
gives 0.082 +- 0.016 against 0.085 +- 0.017 from two separate engine fits.  The re-cut of disjoint
groups of B pairs (`stability_vs_B.py`, `plot_stability_vs_B.py`) is the paper's stability figure;
its tables are in `results/paper_pool47/atom_splits/`.  Not run: the other 40 traits, permutation
replicates.
