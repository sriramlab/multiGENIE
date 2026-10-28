# multiGENIE: the exposome architecture of gene-environment interaction

Biobanks measure dozens of environments on the same genotyped individuals. For a given trait,
multiGENIE asks **which** of these environments interact with genetic effects (GxE), **how many**
there are, and **how much** GxE heritability they carry together: the trait's *exposome
architecture*.

The Python package is imported as `multi_gxe`.

## The idea

1. **One joint model.** All environments are fitted together in a linear mixed model with a GxE
   and a noise-by-environment (NxE) variance component per environment, so that
   environment-dependent noise is not mistaken for GxE. From one fit, the GxE heritability of
   *any* subset of environments is a small linear solve, not a new pass over the genotypes.
2. **Choose on one half, measure on the other.** Environments are added one at a time, each time
   the one that most increases GxE heritability. A curve built and scored on the same individuals
   keeps rising on noise, so the ordering is grown on one half of the sample and scored on the
   other, giving a *held-out* curve of GxE heritability against the number of environments.
3. **Average many splits.** Hundreds of complementary splits are assembled from a single pass
   over the genotypes and their held-out curves are averaged, which stabilizes both the curve and
   the chosen environments.
4. **Test, then cut.** Whether the trait has any GxE is tested on the full sample: for every
   environment, the GxE heritability of its marginal model (a slice of the same normal equations)
   divided by its jackknife standard error gives a one-sided `t` p-value, and the `L` p-values are
   combined by the Cauchy combination test. Across traits, traits are reported at a
   Benjamini--Hochberg false discovery rate. For a reported trait, sequential tail tests find the
   depth `k_g` at which the averaged curve stops rising.
5. **Report.** The `k_g` environments chosen most often across splits, their GxE heritability
   estimated on individuals not used to choose them, and the effective number of environments
   `n_eff` (1 when one environment carries all of the GxE, `n` when `n` carry equal shares).

## Install

```bash
pip install -e .                 # numpy, scipy; Python >= 3.9
python -m pytest tests -q        # unit tests
```

## Usage

### A first run on simulated data (about 5 seconds)

```bash
python examples/quickstart.py
```

It simulates 12 environments, two of which carry GxE, tests the trait-level hypothesis on the
full-sample system, and runs one complementary split:

```
trait-level test: p_H0 = 8.42e-89 (Cauchy combination of 12 marginal t-tests; smallest p at env_0)
reported = True   k_g = 2   tau = 0.0635 +- 0.0027   full-pool value c(L) = 0.0628
reported set (selection frequency in the top k_g, ties by mean position): [0, 5]
```

### Your own trait

The input is one exported system of normal equations per half of the sample, written by the
variance-components engine (format in `docs/export_format.md`).

```python
import multi_gxe as mg

A = mg.load_export("trait.halfA.normal_eq.json")
B = mg.load_export("trait.halfB.normal_eq.json")
r = mg.complementary_split([(A, B)])        # pass a list of (A, B) pairs to average many splits

r["names"]             # True if the averaged curve rises at all
r["k"]                 # k_g, the depth at which the averaged held-out curve stops rising
r["blocks"]            # the reported environments (indices into A.env_names), by selection frequency
r["tau"], r["se"]      # GxE heritability at the cut and its block-jackknife standard error
r["tau_full"]          # the value with every environment, for comparison
r["curve"]             # the averaged held-out curve (.c) and its jackknife replicates (.j)
```

### Many splits from one genotype pass

`python examples/atom_splits_example.py` (about a minute) runs the construction on simulated
genotypes: the individuals are divided once into 12 groups, two streaming passes produce
per-group tensors, and the normal equations of every half (462 complementary pairs) are assembled
by summation. `atom_splits/README.md` is the runbook for real data.

### Is there GxE? Testing across traits

The trait-level hypothesis is tested on the full-sample export (all individuals), not on the
held-out curves: the marginal GxE test of every environment, combined by the Cauchy combination
test (`multi_gxe.h0test`), then Benjamini--Hochberg across traits.

```python
h0 = mg.marginal_h0(full_export)       # p: the combined p-value; env_z, env_p: the marginal statistics
q = mg.nulltest.benjamini_hochberg([h0_of_trait["p"] for h0_of_trait in per_trait])   # report q < 0.05

r = mg.complementary_split(pairs, h0_reject=bool(q[i] < 0.05))   # depth from k = 1 when H_0 is rejected
```

No simulation and no sample split are needed for this step; its calibration on simulated null
phenotypes, including null phenotypes drawn from each trait's own fitted model, is reported in the
paper. The earlier simulation-based test is kept in `multi_gxe.nulltest` for reference.

## The pieces

| what | where |
|---|---|
| the exported normal equations: load, slice to a subset, solve, jackknife | `multi_gxe/normal_eq.py`, `multi_gxe/evaluator.py` |
| forward selection; one split (grow on one half, score on the other) | `multi_gxe/grow.py`, `multi_gxe/crossfit.py` |
| averaging many splits; the reported environments | `multi_gxe/compsplit.py` |
| the tail tests that set the depth `k_g` | `multi_gxe/stopping.py` |
| the trait-level test of GxE (marginal Wald statistics, Cauchy combination); Benjamini--Hochberg across traits | `multi_gxe/h0test.py`, `multi_gxe/nulltest.py` |
| the held-out GxE heritability of the reported set; `n_eff` | `atom_splits/atomsplit/heldout.py` |
| many complementary splits from one genotype pass | `atom_splits/` |
| simulated inputs for tests and examples | `multi_gxe/simulate.py` |

`docs/algorithm.md` states the algorithm in full.

## Requirements

Python >= 3.9, numpy and scipy. The genotype passes of `atom_splits/` also need a PLINK reader
(`bed_reader`, or the C++ backend in `atom_splits/engine_backend/`) and are meant for a compute node.

## Citation

Manuscript in preparation: *multiGENIE reveals the exposome architecture of gene--environment
interaction at biobank scale*. Until it appears, please cite this repository.
