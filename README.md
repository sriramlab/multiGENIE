# multi_gxe

Which environments carry gene--environment interaction (GxE) for a trait, and how much GxE
heritability they carry together, from one joint variance-components fit over a pool of
environments.

The method, in one paragraph. All `L` environments are fitted jointly in a model with a
genetic, an interaction (GxE) and a noise-by-environment (NxE) term per environment. Because
the trace terms of the method-of-moments normal equations depend only on pairs of kernels, the
normal equations of any sub-model are a principal submatrix of the exported system, and the GxE
heritability of any subset of environments is a small solve rather than a new genotype pass.
Environments are ordered by forward selection on the gain in that heritability, on one half of
the individuals, and the ordering is scored on the other half, which gives a held-out curve of
GxE heritability against the number of environments admitted. Many complementary pairs of halves
are assembled from one pass over the genotypes, their held-out curves are averaged, and the
averaged curve is cut once by a sequence of tail tests: at each depth `k` the test asks whether
the curve rises significantly anywhere past `k`, and the cut `k_g` is the first depth at which it
does not, with the family-wise error of the sequence held at `gamma`. The cut gives the trait's
GxE heritability with a block-jackknife standard error, a trait is reported iff `k_g > 0`, and
the reported environments are the `k_g` with the highest selection frequency across the
orderings. `gamma` is the only tuning constant and is set on null simulations.

## Install and run

```bash
pip install -e .                               # numpy, scipy; python >= 3.9
python examples/quickstart.py                  # one split, simulated normal equations, ~5 s
python examples/atom_splits_example.py         # 10 complementary pairs from one simulated genotype pass, ~1 min
python -m pytest tests atom_splits/tests -q    # 41 tests (one needs bed_reader)
```

```python
import multi_gxe as mg

A = mg.load_export("trait.foldA.normal_eq.json")      # one engine fit per half (docs/export_format.md)
B = mg.load_export("trait.foldB.normal_eq.json")
r = mg.complementary_split([(A, B)])                  # the method, no arguments
r["names"]      # reported?  (k_g > 0)
r["k"]          # k_g, the cut of the averaged held-out curve
r["tau"], r["se"]   # GxE heritability at the cut, block-jackknife SE
r["blocks"]     # the reported environments (indices into A.env_names), by selection frequency
r["tau_full"]   # the full-pool value c(L), the natural denominator
r["curve"]      # the averaged held-out curve and its delete-one replicates
```

With `B` pairs assembled from atoms (below), pass all of them: `mg.complementary_split(pairs)`.

Quickstart output:

```
L = 12 environments; truth: h2_GxE = 0.03 on env_0 and env_5, 0 elsewhere (total 0.06)

reported = True   k_g = 2   tau = 0.0700 +- 0.0140   full-pool value c(L) = 0.0609
reported set (selection frequency in the top k_g, ties by mean position): [0, 5]
  ordering grown on one half: [0, 5, 7, 9, 2, 1, 6, 4, 10, 3, 8, 11]
  ordering grown on one half: [0, 5, 6, 3, 2, 9, 1, 7, 4, 10, 11, 8]

averaged held-out curve c-bar(k) with its paired-jackknife SE of c(k) - c(k-1):
  k= 1  c=0.0273  +0.0273 (se 0.0036)
  k= 2  c=0.0700  +0.0427 (se 0.0112)   <- cut
  k= 3  c=0.0398  -0.0302 (se 0.0734)
  k= 4  c=0.0573  +0.0175 (se 0.0477)
  k= 5  c=0.0628  +0.0054 (se 0.0061)
  k= 6  c=0.0643  +0.0015 (se 0.0031)
  k= 7  c=0.0667  +0.0024 (se 0.0018)
  k= 8  c=0.0674  +0.0007 (se 0.0020)
  k= 9  c=0.0662  -0.0012 (se 0.0016)
  k=10  c=0.0665  +0.0004 (se 0.0015)
  k=11  c=0.0623  -0.0042 (se 0.0016)
  k=12  c=0.0609  -0.0014 (se 0.0013)

no complementary split (ordering scored on the half that grew it): k = 2, in-sample c(k_g) = 0.0645 against the held-out 0.0700; on a null curve the in-sample version is reported nearly always.
```

## The pieces

| what | where |
|---|---|
| the export: load, slice to a subset, solve, normalise to heritability, delete-one jackknife | `multi_gxe/normal_eq.py` |
| memoised and batched subset solves; `Evaluator.tau(S)`; the held-out curve of an ordering | `multi_gxe/evaluator.py` |
| forward selection on the gain in `tau(S)`, full depth, ties by lowest index | `multi_gxe/grow.py` |
| one direction of a split: grow on one half, score on the other | `multi_gxe/crossfit.py` |
| averaging the `2B` curves block by block; the decision; the consensus set | `multi_gxe/compsplit.py` |
| the cut: the sequential tail tests, `tail_test(gamma)` (or any `Curve -> int`) | `multi_gxe/stopping.py` |
| simulated exports in the real schema, for tests and the quickstart | `multi_gxe/simulate.py` |
| many complementary splits from one genotype pass: atoms, the two passes, the tensor, assembly of any half, all pairs, calibration | `atom_splits/` |
| the scripts behind the paper's figures and tables | `examples/` (`docs/paper.md` maps each item to its script) |

`docs/algorithm.md` states the algorithm in full with the statistics as implemented;
`docs/export_format.md` the input the engine writes.

## Many complementary splits from one genotype pass

A single split is one draw of the orderings, and on a trait whose curve rises slowly the cut
depends on which rises of a noisy curve clear the critical value. The individuals are
partitioned once into `G` family-intact, ancestry-balanced atoms; any union of `G/2` atoms is a
half and its complement the other; two streaming passes over the genotypes produce per-atom
Gram tensors from which the normal equations of every half, and of all its delete-one-block
jackknife systems, are assembled by summation. With `G = 12`, 462 complementary pairs per trait
cost one pass instead of 924 half-sample fits, and `complementary_split(pairs)` averages their
held-out curves before the cut. `atom_splits/README.md` has the layout, the runbook and the
validation (exact against dense kernels; against the engine to `1e-6`).

## Calibration

The level of the tail tests rests on a Gaussian multiplier approximation to the jackknife
distribution of the curve, which is anti-conservative on 100 genome blocks, so `gamma` is set
on null replicates fitted exactly as the real traits (the trace terms are phenotype-free, so
one engine fit per half serves every replicate and a replicate adds only its own phenotype
terms). `examples/plot_sim_gamma.py` draws the realized level and the power against `gamma`;
`examples/rule_lab.py` applies any rule to the cached curves; `atom_splits/sim_q_pass.py` and
`run_pairs_sim.py` calibrate the many-pair layout the same way.

## Requirements

Python >= 3.9, numpy, scipy. The atom passes additionally need a genotype reader
(`bed_reader`, or the C++ backend in `atom_splits/engine_backend/` built against the engine) and
run on a compute node; everything else runs anywhere.

## Citation

Manuscript in preparation. Until then please cite this repository.
