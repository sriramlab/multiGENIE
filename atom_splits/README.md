# atom_splits: many complementary splits from one genotype pass

**Start with `python examples/atom_splits_example.py`**: simulated data, G = 6 atoms, two passes, all
10 complementary pairs assembled from the tensor and run through the rule of the paper
(`multi_gxe.complementary_split`), single pair against the average.  Runs anywhere in about a minute.

**The v7 (paper) pipeline on real data**, per trait: `run_atom_pass.sh` (tensor, C++ backend, the FULL
49-column environment file for every trait) -> `run_pairs_raw.sh` (462 pairs; every held-out curve,
jackknife and ordering saved; the tensor deleted afterwards with `DELETE_TENSOR=1`) -> `recut_pairs.py`
(the tail test at gamma = 0.01 on the average of the 462 pairs; the default `RULE`) -> `atom_results_tables.py`,
`plot_atom_vs_single.py`, `plot_atom_curve_grid.py`, `stability_vs_B.py` (figures and tables).
**Calibration of the 462-pair layout**: `sim_q_pass.py` (one-probe 12-atom pass of a simulated phenotype
on one real trait's individuals; the tensor's trace terms are phenotype-free and are reused) ->
`run_pairs_sim.py` -> `score_atom_sims.py`.  Realized null level 0/50 at gamma = 0.01 on the averaged curve.

Implements `thoughts/sep_28/atom_splits_plan.md`.  The people are partitioned once into
G family-intact atoms; every half is a union of G/2 atoms; the normal equations of any
half (and of all its J = 100 delete-one-SNP-block jackknife systems) are assembled from
per-atom Gram tensors computed in two passes over the genotypes.  Downstream the halves
are ordinary `multi_gxe.Export` objects, so the paper's algorithm runs unchanged
(`crossfit.direction`, the sequential tail tests `tail_test(0.01)`, `select_from_curves`).  With G = 12 one trait
costs about the Mailman work of 12-13 half fits per pass (two passes) and yields 462
complementary pairs instead of one.

Why this addresses the unstable environment sets: the named set of a single split is
dominated by which people fell on which side (`thoughts/sep_21/README_current_stage.md`).
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
| `atomsplit/pairs.py`, `run_pairs.py`, `run_pairs.sh`, `plot_atom_pairs.py` | D | all pairs; every curve, jackknife and ordering saved (`pairs.npz`), so any rule is re-applied by `recut_pairs.py` without refits; f_C, pair-0 = paper split |
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
bash scripts/atom_splits/submit.sh                                            # traits_priority.txt: whr, creatinine, testosterone, hba1c, hdl, ldl, bp_diastolic
bash scripts/atom_splits/submit.sh scripts/stability_selection/traits_paper41.txt   # all 41 (~80 node-hours)
~/.conda/envs/test/bin/python scripts/atom_splits/collect_atom_results.py     # when the chain has finished
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
$PY scripts/atom_splits/make_atoms.py --G 12 --seed 20260928        # -> data/atom_splits/atoms_pool47.G12.tsv
# B (compute nodes; smoke test first)
qsub -t 18 -v SMOKE=1 scripts/atom_splits/run_atom_pass.sh            # bmi, 20k rows x 20k SNPs, J=10
N=$(grep -cvE '^\s*(#|$)' scripts/stability_selection/traits_paper41.txt)
qsub -t 1-$N -tc 10 scripts/atom_splits/run_atom_pass.sh              # 41 traits, ~4-6 h each on 8 cores
# D
qsub -t 1-$N scripts/atom_splits/run_pairs.sh                         # -> results/atom_splits/<trait>.*
$PY scripts/atom_splits/plot_atom_pairs.py --trait bmi
# validation
$PY scripts/atom_splits/validate_v1.py                                # anywhere
~/.conda/envs/engine/bin/python scripts/atom_splits/validate_v2.py      # AVX-512 node
# V5 null: one permutation replicate = one Stage B pass
qsub -t 18 -v PERM=1 scripts/atom_splits/run_atom_pass.sh && qsub -t 18 -v TAG=.perm1 scripts/atom_splits/run_pairs.sh
```

Outputs of Stage D per trait: `summary.tsv` (named, k_g, tau, se, sd over pairs, k_ES, core
sizes, pair-0 and paper-split values), `clusters.tsv` (f_C at k_g / 1 / 3 / 5, mean rank,
held-out component, membership in the consensus and core sets), `curve.tsv` (Mbar, jackknife
SE, SD over pairs, ES_tau, ES_c per k), `pairs.npz` (every curve, jackknife and ordering).

## the C++ passes backend (`--backend engine`)

`multi_gxe/the C++ passes/` is a copy of `multi_gxe/the engine` (untouched) plus `csrc/streaming/atom_pass.{h,cpp}`
and two bindings, `_cext.atom_pass1` / `_cext.atom_pass2` (version 0.10.27+atoms.1, see
`the C++ passes/ATOMS_VERSION.md`).  They run the two passes in C++: the engine's own `BEDReader::read_block`
(subset rows, SNP statistics, HWE imputation hash, standardisation -- exactly the engine's), Eigen GEMMs
parallelised over column/row chunks with OpenMP, the projection and the per-atom Grams in place on the
numpy buffers.  Python keeps the orchestration (`Design`, the input columns, S_full, the diagonal sketches).

Why not Mailman: the engine's Mailman multiplies are tuned for ~10 right-hand columns.  With the atom pass's
5,760-6,336 dense columns its lookup table no longer fits in cache (`compute_segment_size_for_cols`
shrinks the segment to ~3 SNPs), and its per-SNP-per-column cost (7-12 us at N = 107k, single thread)
is no better than dense sgemm (~7-14 us).  The gain of the C++ path is the fused reader, no float32
block materialisation on the Python side, and no Python temporaries -- expected 1.2-1.5x, to be
MEASURED by `validate_backend.py`, not assumed.

```bash
qrsh -l h_data=8G,h_rt=2:00:00 -pe shared 4                # compute node (module gcc >= 9.3, engine env)
bash scripts/atom_splits/build_engine_atoms.sh                # cmake build + GLIBC ABI gate + load test
conda activate engine                                         # the module is a cpython-3.12 extension: NOT the system python
python scripts/atom_splits/validate_backend.py              # toy exactness (engine vs dense; vs numpy too if bed_reader is installed) + timing
qsub -t 18 -v SMOKE=1,BACKEND=engine scripts/atom_splits/run_atom_pass.sh
```
The C++ was written without a compiler that can build it here (the sandbox has g++ 4.8 and cannot load
the AVX-512 engine), so the first build may need small fixes; `validate_backend.py` is the acceptance test.

## Memory and cost (G = 12, L = 47, B = 10, N ~ 210k, M = 454k)

Pass 1 holds Cin 5.3 GB, U 4.8 GB, the sketch store 11.5 GB (`--no-keep-sketch` to drop) and the
[U|diag] cache 9.7 GB (`--no-cache-uall`); pass 2 adds U_j 4.8 GB and the Grams 4.8 GB.  Tensor
file ~5 GB per trait (float32 Grams).  Flops: 2 x 1.1e15 for the X-steps + ~4e14 for the Grams.

## Real atoms (data/atom_splits/atoms_pool47.G12.tsv, seed 20260928)

219,209 env-complete individuals, 218,646 families (largest 3), atoms of 17,882-18,731 people
(spread 4.6%, so a half of 6 atoms is within ~1% of N/2), no family straddles atoms, every atom
refines its fold, max |z| of any PC (atom vs rest) 0.044.

## Status (2026-09-28)

V1 passes (float64 exact to 4e-13; float32 pipeline 3e-6; .bed round trip exact; toy V3
difference 0.03 SE).  Stage A ran on the real data (above).  Stage B ran on real inputs at
reduced size on the sandbox node (bmi, 10k rows x 20k SNPs, J = 10, 4 threads: 147 s; the
analysed sample of 218,747 rows equals the engine's count); Stage D on that tensor with the
pool47 clusters, tier drops and the paper's exports runs through (the paper pair gives bmi
tau 0.0847, k 30, as in the manuscript).  Measured per 2,000-SNP block at N = 10k, 4 AVX2
threads: X'Cin 1.8 s, X S 2.2 s (x2 passes), Grams 3.1 s -> full scale ~8 h at 4 threads,
~3-4 h on 8 AVX-512 threads.  V2 PASS on n1036 (engine export vs atoms, same probes: T 5e-8, q 1e-6, jackknife 7e-8). the C++ passes backend built; SGE smoke
(bmi 20k x 20k, J=10, 8 threads) 96 s -> ~2-2.5 h per full trait. Not yet run: the full-size bmi tensor, the 41-trait array,
V5 permutations.

## Status (2026-09-30): 8 traits done, single split vs many pairs

Tensors + 462 pairs + stability-vs-B exist for bmi and the 7 priority traits (jobs 14964807-09; whr's
tensor took 15.7 h on its node, the others ~2 h).  The "Not yet run" list above is superseded.

```bash
~/.conda/envs/test/bin/python scripts/atom_splits/collect_atom_results.py      # summary_all.tsv, stability_vs_B_all.tsv
~/.conda/envs/test/bin/python scripts/atom_splits/compare_single_vs_B.py       # ~2 min, saved curves only
```
`compare_single_vs_B.py` puts the 462-pair answer next to every kind of single run (the paper's split, the
engine re-splits in `results/stability_selection/cache`, each atom pair alone, the engine's full-sample fit)
and re-cuts disjoint groups of B pairs -> `results/atom_splits/compare_single_vs_B{,.byB,.replicates}.tsv`,
`figures/atom_splits/compare_single_vs_B_{estimates,stability}.*`.  Findings:
`thoughts/sep_30/README_atom_stability_8traits.md` (tau and the leading cluster stabilise for every trait;
the plateau depth does not, and goes to K for 6 of 8).

### Replicating from a fresh partition (2026-09-30)

The 462 pairs share one atom partition and one probe draw (`design_decomposition.py`: at full depth
97-100% of the pair-to-pair variance is the order-2 atom-interaction space, so the pairs are a
balanced design, not independent splits).  The honest replicate is a second pass from free atoms:

```bash
$PY scripts/atom_splits/make_atoms.py --G 12 --seed 20260930 --refine-folds '' \
    -o data/atom_splits/atoms_pool47.G12.seed2.tsv                            # done 2026-09-30
ATOMS=data/atom_splits/atoms_pool47.G12.seed2.tsv TAG=.seed2 SEED_SHIFT=1 bash scripts/atom_splits/submit.sh
$PY scripts/atom_splits/compare_seeds.py --tag .seed2                        # -> results/atom_splits/compare_seeds.seed2.tsv
```
Outputs go to `<scratch>/atom_splits/<trait>/<trait>.atoms.G12.seed2.npz` and `results/atom_splits.seed2/`.
