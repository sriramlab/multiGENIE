# The algorithm, in full

This document states the method as implemented, including the statistics, and then
the properties you should know before trusting a result. `README.md` is the summary;
`docs/versions/` says how each piece was arrived at.

---

## 1. The model

For a trait `y`, genotypes `G`, covariates `X` and `L` candidate environments,

```
y = Xb + g + sum_k (g * E_k) + sum_k (e * E_k) + e
```

with `g ~ N(0, sigma2_g K_G)`, each interaction term `g * E_k` its own variance
component, and `e * E_k` a heteroscedastic noise term. `K_G` is the genetic
relationship matrix; the G x E kernel of environment `k` is `K_G` scaled by the outer
product of `E_k`, and the N x E kernel is diagonal in `E_k²`.

Selecting environment `k` keeps **both** of its kernels. Dropping the N x E term of
an unselected environment lets omitted heteroscedasticity leak onto the correlated
environments that were selected, which is a bias in the reported `h2_GxE`, not a
saving.

## 2. The identity

A method-of-moments estimator solves the normal equations `X sigma = y`:

```
X[i][j] = tr(K_i K_j)        (a randomised trace estimate)
X[i][C] = tr(K_i)            X[C][C] = N
y[i]    = y' K_i y           y[C]    = y' y
```

Each entry is a function of the two components it indexes alone. **The normal
equations of any sub-model are therefore the corresponding principal submatrix of the
full system**, and fitting a subset is a slice and a solve.

The exported system has dimension `1 + L + L + 1` under `G+GxE+NxE`
(`1 + L + 1` under `G+GxE`), written for the full sample and for each of the `J`
jackknife blocks, so delete-one standard errors survive the slicing too.

Cost, for `L` environments and `J` blocks: **one engine fit**, plus `O(L² J)` solves
of a system of dimension at most `2L + 2`. Seconds against hours.

## 3. Stage 1 — blocks from `E` alone

Average linkage on `1 - abs(r)`, cut at `abs(r) = 0.8`; average linkage means the cut
is an average within-block correlation, not a minimum. Missing values are `-9` and
are masked **pairwise**: the missingness pattern is shared across columns in real
environment files, so leaving it in makes every pair look strongly correlated.

The partition never sees `y`, so it is a property of the design. It can be computed
once and frozen, and the same blocks then apply to every trait and every replicate.

Why blocks rather than environments:

* **Reproducibility.** Near-duplicate columns make round 1 a tie-break, and the loser
  is then buried. Two runs of the same fit with different trace seeds shared nothing
  at `k <= 3` on the trait where this was first seen.
* **A fair yardstick.** Per-environment false-discovery rates charge a rule for
  naming an `r = 0.98` copy of a causal environment.
* **One legitimate test.** The block statistic is the jackknife of the block's
  *summed* G x E, so within-block correlation is absorbed rather than modelled, and a
  singleton block reduces exactly to the single-environment test.

## 4. Stage 2 — the grow

Each round scores every block not yet admitted, jointly with everything already
admitted, and admits the winner. It runs to exhaustion: the path is a permutation of
the blocks and every prefix is a nested sub-model.

Ordering key, `order=`:

* `theta` (default) — the largest summed `h2_GxE`. One full-sample solve per
  candidate.
* `p` — the smallest one-sided p-value. Needs the whole `J`-solve jackknife per
  candidate.

They disagree on few traits. `theta` was adopted on cost and reproducibility: about
24x fewer jackknifed evaluations, and a stable tail, because the p ordering re-ranks
on a statistic whose denominator is itself noisy.

`reps=` scores a block by one representative column while still fitting the whole
block once admitted. Measured inert on the real pool; off by default.

**The grow carries no stopping test.** Running to exhaustion is what makes the cut
free: one traversal scores every rule, at every threshold, forever.

## 5. Stage 3 — the cut

A stopping rule is a pure function `Curve -> int`. It may read the out-of-fold
cumulative curve `c`, its jackknife `j`, the in-fold round statistics `p` and `z`,
and the path; it may not re-order the path or refit anything, which would be changing
the grow.

Rules implemented, and what each references:

| rule | stops when | reference |
|---|---|---|
| `plateau(z_thr)` | `(c[K] - c[k]) / se < z_thr` | the full-depth endpoint |
| `one_se(n_se)` | `c[k] >= max(c) - n_se * se` | the curve maximum |
| `window(m, z_thr)` | the next `m` blocks add nothing significant | a local increment |
| `gate(alpha, patience)` | the in-fold round p fails `patience` times | one block at a time |
| `argmax()` | at the curve maximum | oracle, biased upward |

`se` is always the **paired** jackknife of the difference, `jk_se(j[a] - j[b])`,
never the sum of two marginal SEs. The two prefixes are fits on the same
individuals, their errors are strongly positively correlated, and treating them as
independent inflates the standard error several-fold and stops the walk far too
early.

## 6. Stage 4 — the estimator

Split the individuals into halves A and B, respecting family and ancestry structure,
and run the engine once per half. Then

```
direction A->B:  grow the path on A (full depth), draw and cut the curve on B
direction B->A:  the same, the other way
tau = min(tau_{A->B}, tau_{B->A})
```

Each direction is fully out of fold: the set comes from one half, the number from the
other. The reported set is the path of whichever direction gave the minimum, and
`shared` reports how many blocks the two directions agree on.

**Why MIN.** Naming anything requires both disjoint halves to name something, so a
per-direction null rate `alpha` becomes `alpha²`. The calibration is structural, not
tuned. `combine="avg"` is the alternative and is a different claim: it recovers more,
but most of the extra comes from traits only one half could see.

**What it costs.** Two engine fits on half samples instead of one on the whole, and
every estimate is a half-sample estimate with the standard error that implies. The
two directions can name different sets.

## 7. The statistics, exactly as implemented

* **`h2` normalisation** (`component_h2`) — component `i` is scaled by `tr(K_i)`, the
  noise by `N`, and everything is divided by the total. Components are variance
  fractions and their non-noise sum is the model's total. This is GENIE's
  convention; the engine prints a different normaliser, so its `h2_*` lines and these
  numbers differ by construction.
* **Standard error** — delete-one jackknife over the exported blocks,
  `sqrt((n-1)/n * sum (x - xbar)²)`. For a block it is the jackknife of the block's
  summed G x E; for a difference of prefixes it is the paired jackknife of the
  difference.
* **p-value** — one-sided normal, `norm.sf(h2 / se)`, with `p := 1` whenever the
  estimate is `<= 0`. A variance component is non-negative in truth, so a negative
  estimate is noise, not evidence of interaction.
* **The round p-value is a minimum over live candidates** and is therefore not
  `U(0,1)`. Any rule that treats it as a single test is anti-conservative by roughly
  the number of live candidates. This is the whole reason the family-wise error rate
  of the raw gate grows with the pool.

## 8. Properties to know before trusting a result

These are properties of the method, verified against the source. None is an open bug.

1. **The restricted solve is not always identical to re-running the engine on the
   subset.** GENIE hardcodes `add_env_to_cov = true`, so one run with all `L`
   environments regresses out every environment main effect as a fixed covariate. The
   slicing identity is exact for the variance components, but the residual they act
   on was built by conditioning on all `L`. the engine exposes this as a parameter
   (`include_env_in_covariates`), and with it off the projector is identical across
   subsets and the restricted solve becomes equal to a genuine subset refit. That
   projection is also what guards against G-E correlation bias, so it is an
   experiment, not a default.
2. **The sample is the complete-case intersection over all `L` environments.** The
   engine drops individuals missing any environment column and the export bakes that
   sample in. Adding a badly-missing environment to the pool shrinks `N` for every
   selection, including subsets that exclude it.
3. **Single annotation only.** `gen_nbin` must be 1; the index arithmetic assumes one
   additive component. Partitioned (MAF/LD, functional) annotations would shift every
   G x E position.
4. **Ill-conditioning is not silenced.** Near-duplicate environments in one fit make
   the slice near-singular. `solve` raises rather than returning a pivoted
   least-squares answer, because an `r = 0.998` pair is a modelling decision, not
   something to patch over. Neither this nor the engine imposes non-negativity on
   `sigma`.
5. **Only `k = K` is unbiased.** The in-fold cumulative curve rises above the
   unselected joint fit and comes back down; under a pure null it peaks at more than
   twice the joint fit's standard error, in every replicate. Cutting an in-fold curve
   reports selection noise. That is what stage 4 exists to prevent.
6. **Export size** is `(J + 1) x (2L + 2)²` doubles at full precision, held in memory
   on both sides. `L = 20, J = 100` is about 40 MB; `L = 200` would be about 4 GB.
7. **A Wald test on the selected set is not a test.** Under a pure null the selected
   subset's standard error is several times smaller than the full fit's while the
   estimate collects the full fit's positive noise, so almost every named set clears
   `p < 0.05`. "Did it name anything" and "was the named set significant" are the
   same number. Report the first.
