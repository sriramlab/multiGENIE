# atom_splits: many complementary splits from one pass over the genotypes

A single complementary split is one draw of the orderings, and on a trait whose held-out curve
rises slowly the cut depends on which rises clear the critical value. Averaging many pairs fixes
that, but fitting each half separately costs one genotype pass per half. This module assembles
every half from one pass instead.

**Start here:** `python examples/atom_splits_example.py` runs the whole idea on simulated data in
about a minute: two passes, `G = 6` atoms, all 10 complementary pairs assembled from the tensor
and run through the method (`multi_gxe.complementary_split`), single pair against the average.

## The construction

Partition the individuals once into `G` atoms that keep families intact and are balanced on size
and genetic ancestry. A half is any union of `G/2` atoms, its complement the other half; `G = 12`
gives `C(12,6)/2 = 462` complementary pairs. Because the randomized trace estimate of a half is a
sum over its individuals, with `D_a` selecting atom `a` and `w_{a,p} = K_p D_a z_r` independent of
the half,

    T_hat[p,p'](H) = sum_{a'' in H} sum_{a,a' in H} Gamma[a''; (a,p), (a',p')],
    Gamma[a''; (a,p), (a',p')] = (1/R) sum_r sum_{n in A_{a''}} w^(r)_{a,p,n} w^(r)_{a',p',n},

so one tensor `Gamma`, computed in two streaming passes, gives the normal equations of every half
by summation; the phenotype terms split the same way, and the delete-one-genome-block systems need
only each block's contribution against the totals, formed in the second pass. Downstream every half
is an ordinary `multi_gxe.Export`, so the method runs unchanged.

What differs from a half-sample engine run, by design: the projection `W`, the SNP statistics, the
environment matrix and the phenotype scale are those of the full analysed sample, not of the half.

## Layout

| file | stage | what |
|---|---|---|
| `atomsplit/model.py` | — | `Design`: the preprocessing (W, Q, V, y~) mirroring the engine; kernel layout |
| `atomsplit/bedio.py` | — | BED blocks decoded, standardised and imputed exactly as the engine does |
| `atomsplit/atoms.py`, `make_atoms.py`, `folds.py` | A | the two halves, and `G` atoms refining them |
| `atomsplit/stream.py`, `run_atom_pass.py`, `run_atom_pass.sh` | B | the two passes → `<trait>.atoms.G12.npz` |
| `atomsplit/tensor.py`, `assemble_export.py` | C | `AtomTensor.assemble(atoms) -> Export`; JSON in the engine's schema |
| `atomsplit/pairs.py`, `run_pairs.py`, `run_pairs_raw.sh` | D | every pair; each curve, jackknife and ordering saved to `<trait>.pairs.npz` |
| `recut_pairs.py`, `stability_vs_B.py`, `atom_results_tables.py` | — | any rule re-applied to the saved curves; stability against the number of pairs; tables |
| `plot_atom_vs_single.py`, `plot_atom_curve_grid.py`, `plot_stability_vs_B.py`, `plot_atom_pairs.py` | — | figures |
| `sim_q_pass.py`, `run_pairs_sim.py`, `score_atom_sims.py` | — | calibration of the many-pair layout |
| `atomsplit/dense.py`, `validate_v1.py`, `validate_v2.py`, `validate_backend.py`, `tests/` | V | exactness against dense kernels, against the engine, and the C++ backend |
| `engine_backend/` | — | the two passes in C++, as a patch to the engine |

## Conventions mirrored from the engine

* kernels `V K_r V` with `V = I - W(W'W)^{-1}W'`, `W = [1, covariates, E]`; `V` is the trait's
  full-sample projector and a half's kernel is `D_h V K_r V D_h`;
* SNP standardisation `sd = sqrt(2p(1-p))`, monomorphic sd 1, the engine's HWE imputation hash;
* Hutchinson traces `T = (1/B) sum_p (K_r z_p)'(K_s z_p)` with Gaussian probes
  `default_rng(seed).standard_normal((N, B), float32)`;
* the noise row and column of `T` carry `NC_h = sum_{n in h} (1 - h_nn)`;
* jackknife `alpha = M/(M - M_j)`, `T_j = alpha_r alpha_s <U - U_j, U - U_j>/B`, `q_j = (q - q_j) alpha`.

## Run it on real data

```bash
PY=python                                       # an interpreter with numpy, scipy and bed_reader
# A: the halves and the atoms (minutes; reads the kinship table and the genetic PCs once)
$PY atom_splits/make_atoms.py --G 12 --seed 20260928 -o data/atoms.G12.tsv
# B: two streaming passes per trait (hours on 8 cores; the C++ backend with --backend engine)
qsub -t 1-<n traits> -v TRAITS_FILE=<list>,ATOMS=data/atoms.G12.tsv,BACKEND=engine atom_splits/run_atom_pass.sh
# D: every pair through the method; the tensor can be deleted afterwards (DELETE_TENSOR=1)
qsub -t 1-<n traits> -v TRAITS_FILE=<list>,POOL=raw84 atom_splits/run_pairs_raw.sh
# anything downstream reads <trait>.pairs.npz only: no refits
$PY atom_splits/recut_pairs.py --trait bmi --res <results dir>        # the cut on the averaged curve
$PY atom_splits/stability_vs_B.py --trait bmi --results <results dir> # disjoint groups of B pairs
$PY atom_splits/atom_results_tables.py                                # tables over all traits
```

The job scripts carry the absolute paths of the machine the paper was run on; read them as a
runbook and adapt the five paths at the top.

**Calibration of the layout.** The trace terms of a tensor are phenotype-free, so one real trait's
tensor serves every simulated phenotype on those individuals: `sim_q_pass.py` computes a replicate's
phenotype terms with a single one-probe pass, `run_pairs_sim.py` assembles the 462 pairs from the
two, and `score_atom_sims.py` applies the rule. `examples/null100/` packages this end to end.

## The C++ backend (`--backend engine`)

`engine_backend/` adds `atom_pass1` / `atom_pass2` to the engine: its own BED reader (subset rows,
SNP statistics, imputation hash, standardisation), Eigen GEMMs parallelised with OpenMP, and the
projection and per-atom Grams written in place on the numpy buffers. Python keeps the orchestration.
Build and check with `build_engine_atoms.sh` and `validate_backend.py`. Mailman is deliberately not
used: at the thousands of right-hand columns an atom pass needs, its lookup table no longer fits in
cache and its per-column cost is no better than dense sgemm.

## Cost (G = 12, L = 49, R = 10 probes, N ≈ 210k, M ≈ 454k)

One trait: about 2.5 h on 8 AVX-512 cores for the two passes, against 34 min for a single
half-sample engine fit, so 462 pairs cost what 4 or 5 separate fits would, instead of 924. Pass 1
holds about 5 GB for the input sketches, 5 GB for the projected sketches and 11 GB for the kept
sketch store (`--no-keep-sketch` drops it); the tensor is about 16 GB per trait at `L = 49` and can
be deleted once `pairs.npz` exists.

## Validation

`validate_v1.py` (runs anywhere, minutes): the assembler against explicit `N x N` kernels on
simulated data, float64 exact to 4e-13, float32 pipeline to 3e-6, and the BED round trip exact.
`validate_v2.py` (needs the engine): one half assembled from atoms against the engine's own export
with the same probes, traces to 5e-8 and phenotype terms to 1e-6. `tests/` runs the assembler,
the atom partition and the pair enumeration in a few seconds.
