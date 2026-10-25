# EXPOSE: EXPOsome Selection and Estimation

Biobanks measure dozens of environments on the same genotyped individuals. For a given trait,
EXPOSE asks **which** of these environments interact with genetic effects (GxE), **how many**
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
4. **Test, then cut.** Whether the trait has any GxE is tested against null phenotypes simulated
   from the trait's own fitted model without GxE, and across traits, traits are reported at a
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

It simulates 12 environments, two of which carry GxE, and runs one complementary split:

```
reported = True   k_g = 2   tau = 0.0700 +- 0.0140   full-pool value c(L) = 0.0609
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

A trait's test statistic is compared with the same statistic on null phenotypes simulated from
the trait's own model without GxE (`multi_gxe.nulltest`):

```python
from multi_gxe import nulltest as nt

v, _ = nt.residual_variance_model(y_resid, E)        # residual variance as a function of the environments
Y0 = nt.null_phenotypes(read_block, n_snps, n, h2=h2_hat, v=v, reps=range(100), trait="bmi")
# analyse each column of Y0 exactly as the trait, keeping each null's p-value in p_null

p0 = mg.stopping.tail_pvalue(r["curve"].c, r["curve"].j)           # the trait's own statistic
p_mc = nt.mc_pvalue(p0, p_null)                                      # (1 + #{p_null <= p0}) / (R + 1)
if nt.needs_more_nulls(p0, p_null):                                  # fewer than 10 nulls as small as p0
    ...                                                              # draw 200 more: reps=range(100, 300)

q = nt.benjamini_hochberg(p_mc_over_traits)                          # report the traits with q < 0.05
```

`read_block(a, b)` returns the standardized genotypes of SNPs `a` to `b - 1` for the analysed
individuals; `atom_splits/atomsplit/qpass.py` computes the phenotype terms of many null
phenotypes in one pass over the genotypes.

## The pieces

| what | where |
|---|---|
| the exported normal equations: load, slice to a subset, solve, jackknife | `multi_gxe/normal_eq.py`, `multi_gxe/evaluator.py` |
| forward selection; one split (grow on one half, score on the other) | `multi_gxe/grow.py`, `multi_gxe/crossfit.py` |
| averaging many splits; the reported environments | `multi_gxe/compsplit.py` |
| the tail tests that set the depth `k_g` | `multi_gxe/stopping.py` |
| the test of GxE against simulated null phenotypes; Benjamini--Hochberg across traits | `multi_gxe/nulltest.py` |
| the held-out GxE heritability of the reported set; `n_eff` | `atom_splits/atomsplit/heldout.py` |
| many complementary splits from one genotype pass | `atom_splits/` |
| simulated inputs for tests and examples | `multi_gxe/simulate.py` |

`docs/algorithm.md` states the algorithm in full.

## Requirements

Python >= 3.9, numpy and scipy. The genotype passes of `atom_splits/` also need a PLINK reader
(`bed_reader`, or the C++ backend in `atom_splits/engine_backend/`) and are meant for a compute node.

## Citation

Manuscript in preparation. Until then, please cite this repository.
