# multi_gxe

**Which of many candidate environments carry real gene-by-environment signal for a
trait, and how much — without one model fit per candidate per round, and without
reporting the selection's own noise as signal.**

Given a trait `y`, genotypes `G`, and `L` candidate environments `E_1 … E_L`, fit

```
y = Xb + g + sum_k (g * E_k) + sum_k (e * E_k) + e            [G + GxE + NxE]
```

and ask which environments carry non-zero `h2_GxE`. Two things make that hard.

**Cost.** Textbook forward selection over `L` environments needs `L + (L-1) + … ≈ L²/2`
fits, and one variance-component fit on biobank-scale genotypes takes hours. At
`L = 50` that is about 1,250 fits.

**Honesty.** Selection and estimation cannot use the same data. Grown and scored on
one sample, the cumulative curve rises well above the unselected joint fit before
coming back down — under a pure null it peaks at more than twice the standard error
of the joint fit, in every replicate we have looked at. A rule that cuts such a curve
reports its own noise.

This repository is the algorithm that answers both, its full version history, and the
frozen code behind the numbers each version was chosen on.

**The current version (7.0.0) is the method of the manuscript.** `complementary_split`
with no arguments is its Algorithm 1: forward selection on the gain in the GxE heritability
`tau(S)` of each prefix (every selected environment keeps its NxE term), the ordering scored on
the other half, the 2B held-out curves averaged, the cut `k_g` by sequential tail tests at level
gamma = 0.01 (`mg.tail_test`), a trait reported iff `k_g > 0`, and the reported environments the
`k_g` with the highest selection frequency across the orderings. `docs/paper.md` maps every
equation, table and figure of the paper to the code that produces it; `tail_test(gamma)` is the
only tuning constant and is set on null simulations.

---

## The identity everything rests on

A method-of-moments variance-component fit solves `X sigma = y`, where

```
X[i][j] = tr(K_i K_j)     X[i][C] = tr(K_i)     X[C][C] = N
y[i]    = y' K_i y        y[C]    = y' y
```

Every entry depends only on the two components it indexes and never on which other
components are in the model. **So the normal equations of any sub-model are exactly
the corresponding principal submatrix of the full system.** Fitting a subset of
environments is a slice and a solve, not another pass over the genotypes.

Run the engine once with all `L` environments, export that system for the full sample
and for each jackknife block, and every subset fit — with delete-one standard errors
— becomes free. One fit buys the whole search.

## The five stages, three of which are knobs

| | stage | what | file |
|---|---|---|---|
| 1 | **cluster** | group the environments from `E` alone, average linkage on `1 - abs(r)` cut at 0.8. Never sees the phenotype, so the partition is a property of the design | `multi_gxe/cluster.py` |
| 1b | **choose** | one candidate environment per block, picked on the growth half. `none()` scores the whole block instead | `multi_gxe/represent.py` |
| 2 | **grow** | full-depth greedy path over the blocks, ordered by summed `h2_GxE`. No stopping test | `multi_gxe/grow.py` |
| 3 | **cut** | how many leading blocks to keep. A pure function `Curve -> int` | `multi_gxe/stopping.py` |
| 4 | **report** | what the two half-sample directions turn into one result. A function `(dir_A, dir_B) -> dict` | `multi_gxe/combine.py` |

```python
mg.crossfit(export_A, export_B, blocks, choose=..., rule=..., combine=...)
```

Those three arguments are the whole interface for changing the method. Each accepts a
built-in or any callable of your own, and `notebooks/algo_lab.ipynb` is where you edit
them against real exports and read the difference.

The grow is exhaustive and the cut only truncates it, so **one traversal scores every
stopping rule at every threshold**. That is what makes rule design cheap enough to do
honestly: the expensive object is the engine fit, the traversal is a few hundred
solves of a small system, and comparing twenty rules over it costs microseconds.

Stage 1 removes a coin flip rather than a nuisance. Where three columns sit at
`abs(r) >= 0.94` with indistinguishable marginal signal, whichever wins round 1
buries the other two, and two runs of the same fit share nothing at `k <= 3`.
Admitting the whole block makes the answer reproducible, and it fixes the yardstick:
per-environment false-discovery rates charge a rule for naming an `r = 0.98` copy of
a causal environment, which is a property of the pool, not an error of the rule.

Stage 4 is where the honesty comes from. Split the individuals into halves A and B,
run the engine once per half, grow on one and score on the other, in both directions,
and report the MINIMUM. A trait is named only when both disjoint halves name
something, so a rule with per-direction null rate `alpha` has a combined null rate of
`alpha²` — no permutation array, no calibration run. **A trait returning nothing is
the rule working.**

## Quickstart

```bash
python examples/quickstart.py                 # one split, the rule of the paper
python examples/atom_splits_example.py        # 10 complementary pairs from one genotype pass (simulated data)
```

```
L = 13 environments -> K = 10 blocks
truth: h2_GxE = 0.06 on env_0 and env_5, 0 elsewhere (total 0.12)

grow on A (full depth, no gate)
  path      [0, 2, 4, 6, 8, 7, 3, 9, 1, 5]
  solves    55 ordering, 19 jackknifed

plateau z=1.645    tau = 0.1123 +- 0.0240  z =  4.68   k = 2 (A 2 / B 2)   names: True   (tail_test(0.01) gives the same cut)
                   block 0 = [0, 1, 2], block 2 = [5]

full-depth out-of-fold total (the denominator) = 0.1194
the plateau rule keeps 94% of it on 2 of 10 blocks
```

In code:

```python
import multi_gxe as mg

export_A = mg.load_export("trait.foldA.normal_eq.json")
export_B = mg.load_export("trait.foldB.normal_eq.json")

result = mg.complementary_split([(export_A, export_B)])      # the rule of the paper, no clustering
result["tau"], result["se"], result["blocks"], result["names"], result["k"]
```

No clustering is applied: the pool is curated beforehand so that no two environments are
strongly correlated, and each environment is its own unit of selection (`K = L`). Grouping
correlated environments into blocks (`mg.cluster_from_corr`) remains available as an optional
first stage for a pool that has not been curated:

```python
blocks = mg.cluster_from_corr(mg.corr_from_env_file("envs.env"), cut=0.8)
result = mg.complementary_split([(export_A, export_B)], blocks)
```

The three knobs, and what each may look at:

```python
mg.crossfit(A, B, blocks,
            choose=mg.represent.marginal(),   # (blocks, ctx) -> one env per block
            rule=mg.one_se(2.0),              # Curve -> int
            combine=mg.combine.average())     # (dir_A, dir_B) -> dict
```

Writing your own stopping rule is the intended extension point:

```python
def my_rule(cv):                      # cv.c, cv.j, cv.p, cv.z, cv.se(a, b), cv.K
    for k in range(cv.K + 1):
        if cv.c[cv.K] - cv.c[k] < 1.0 * cv.se(cv.K, k):
            return k
    return cv.K

mg.crossfit(export_A, export_B, blocks, rule=my_rule)
```

`multi_gxe.simulate` builds a normal-equation system in the real schema, so the whole
pipeline runs with no genotype file. It draws its estimation error rather than
earning it: use it for mechanics and tests, never for a reported number.

## Getting an export

The algorithm consumes one JSON file per fit. Either engine can write it, and the
selector does not care which:

* **GENIE** with the normal-equation sidecar (`--export-normal-eq`), the single C++
  change in the fork this work started from.
* **the engine**, which needed no C++ change, because it already materialises both halves
  of the system.

`docs/export_format.md` gives the schema, the index layout, and what to check before
trusting a file.

## Results

`RESULTS.md` is what the algorithm does on real data and which settings are safe to
report. **Current (v7, the paper):** `complementary_split` averages the held-out curves of
every pair and cuts once by the sequential tail tests at gamma = 0.01; a trait is reported iff
the cut is positive. On 50 null replicates it reports 2 (realized level 0.04); with one causal
environment power is 0.97 and the reported set is exactly the causal environment in 92% of
replicates; on 49 environments and one split it reports 21 of 42 traits at a mean GxE
heritability of 0.041, and on the average of 462 complementary pairs assembled from one
genotype pass per trait the cut deepens on the diffuse traits (the manuscript's main results).
The v6 (plateau) numbers remain in Section 0 for comparison.
The tables behind it are in `results/`, the figures in `figures/`, and everything
there is an aggregate over traits, replicates or genome blocks.

## Repository map

```
multi_gxe/          the current algorithm  (v6 = the paper; this is what you should use)
  normal_eq.py        the exported system: load, slice, solve, normalise, jackknife
  evaluator.py        memoised solve layer; the batched delete-one jackknife
  cluster.py          stage 1
  grow.py             stage 2
  stopping.py         stage 3, and the rules measured so far
  crossfit.py         stage 4, the per-direction estimator (v4)
  compsplit.py        average the held-out curves, cut once, consensus set (Algorithm 1)
  represent.py        stage 1b, the representative choosers
  combine.py          stage 4, how the two directions are reported
  simulate.py         synthetic exports, for tests and teaching
atom_splits/        many complementary splits from one pass over the genotypes (the paper's
                    main real-data results, B = 462): atoms, the two passes, assembly, all
                    pairs, the calibration of the layout, validation, and the
                    complementary split itself (folds.py)
notebooks/
  build_algo_lab.py   generator for the lab notebook
  algo_lab.ipynb      edit the three knobs, see the effect on real exports,
                      and five views of the curve the rules act on:
                      two traits, every rule's cut on one curve, in fold
                      against out of fold, a panel per trait, and what the
                      rules buy
results/            aggregate tables: null calibration, power, real recovery
figures/            the figures RESULTS.md refers to
docs/
  algorithm.md        the method in full, including the statistics as implemented
  paper.md            every item of the manuscript: the code and the table behind it
  export_format.md    the JSON schema and how to produce it
  versions/           one document per version: what changed, and what it was measured on
tests/              51 tests, including equivalence against the frozen implementations
examples/           quickstart; the figure and table scripts of the paper (docs/paper.md);
                    compsplit_sim_arms.py, compsplit_real.py, plot_v5_figures.py (v5 results)
legacy/             every earlier version, frozen, kept to reproduce published numbers
CHANGELOG.md        the version history in one page
```

## Versions

Each row is a change that was measured before it was adopted, and the measurement is
in `docs/versions/`. `CHANGELOG.md` is the short form.

| version | change | why it was adopted |
|---|---|---|
| v0 | per-environment forward selection from one exported system | `L²/2` engine fits become one fit and `O(L² J)` small solves |
| v1 | the BLOCK, not the environment, is the unit | reproducibility: chance-corrected Jaccard across engine seeds 0.30 to 0.68 |
| v2 | order by summed `h2_GxE`, not by p-value | 24x fewer jackknifed evaluations, Jaccard 0.68 to 0.86, same power. Backward elimination was tested and is inert |
| v3 | representatives, and the cross-fitted split | the split is what makes the estimate out of fold. Representatives measured inert |
| risk | a risk criterion replaces the p-value gate | at `L = 47`: RMSE 0.0082 against the full fit's 0.0133. At `L = 10` the older gate wins — the crossover is real |
| v4 | grow full depth, cut on the held-out half, combine by MIN | self-calibrating at `alpha²`; the stopping rule became a pluggable pure function, and two rules already beat the first one adopted |
| v5 | complementary split: average the held-out curves, cut once (plateau for the trait, window for the environments), consensus set | MIN deleted traits one half could not see (bmi -> 0); null 1/100 traits and 0/100 environments, one-causal precision 0.97, 42 environments 20/42 traits at 64.5% |
| v6 | plateau 0.5, reported at p < 0.05, set = consensus top k_g; 47 environments; atoms for many pairs | the window cut was unstable across splits and short on diffuse traits; null 5/100, one-causal power 0.85, 24/41 traits; 462 pairs from one genotype pass |
| **v7** | **the rule of the paper: sequential tail tests at gamma = 0.01 (`tail_test`), reported iff k_g > 0, set = selection-frequency consensus; ordering by the gain in tau; every selected environment keeps its NxE term; one fixed pool of 49 for every trait** | **the plateau threshold had to be re-tuned for every pool and tested against the full-pool endpoint; the tail test's level does not depend on the pool, null 2/50 at gamma 0.01, one-causal power 0.97 with the exact set 92%, and it is what the Methods state** |

## What this repository deliberately does not carry

The algorithm, its tests, the atom passes and the scripts behind the paper. The
pipeline that produces exports on a cluster, the simulation harness and the environment
and trait curation for the biobank analysis live in the private working tree. Nothing
here embeds individual-level data, and none of the caches, exports or phenotypes are
in git history.

## Provenance

The normal-equation sidecar and the original per-environment forward selector are
Ryan Wang's work, developed in a private fork of GENIE. Everything from the block
unit onward — clustering, the ordering change, the risk criterion, the cross-fitted
estimator and the stopping-rule framework — was built on top of it, in the Sriram lab
at UCLA. `legacy/` preserves each of those implementations as it was when its numbers
were produced.

## Requirements

Python 3.9 or newer, `numpy` and `scipy`. `pytest` to run the tests.

```bash
python -m pytest tests -q                 # 51 passed
python -m pytest atom_splits/tests -q     # 11 passed (10 and one skipped without bed_reader)
```
