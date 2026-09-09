# Changelog

Versions here are versions of the **algorithm**. Each entry says what changed, what
it was measured on, and what was tried and rejected at the same time. The long form
is in `docs/versions/`; the code of every earlier version is frozen under `legacy/`.

Dates are when the change was adopted.

---

## 4.1.0 — 2026-09-09 — the representative and the report become knobs too

**Added**
- `multi_gxe/represent.py` — stage 1b, picking one candidate environment per block
  before the path is grown. A chooser is `(blocks, ctx) -> list[int]`; built in are
  `none` (score the whole block, the default), `centroid`, `marginal`, `first` and
  `fixed`. **`crossfit` runs the chooser once per direction against the growth
  half**, so a data-driven choice cannot leak the evaluating half into the selection.
- `multi_gxe/combine.py` — stage 4 as a function `(dir_A, dir_B) -> dict`, deciding
  separately whether a trait is named, what number is reported and which set. Built
  in are `min`, `avg`, `A`, `B` and `agreed` (the old hard intersection).
- `notebooks/algo_lab.ipynb` and its generator: the three knobs in one editable cell,
  with sweeps, per-trait curves showing where the rule stopped, and stubs for writing
  your own chooser, rule and combiner.
- `crossfit(..., ev_A=, ev_B=)` to reuse memoised evaluators across configurations,
  which is what makes sweeping knobs interactive.

**Measured on the real pool** (42 traits, fold-matched denominator 1.2419)
- The representative choice barely matters: 27.5% of the full fit scoring whole
  blocks, 28.5% with the correlation medoid or the frozen production
  representatives, the same 8 traits named either way.
- The combiner matters a lot: `min` 8 traits and 27.5%, `avg` 24 traits and 47%, a
  single direction 16 traits and 44-50% — but only `min` has the squared null rate.

---

## 4.0.0 — 2026-09-08 — cross-fitted, with a pluggable cut

*Current. `multi_gxe/`, documented in `docs/versions/v4_crossfit.md`.*

**Changed**
- The grow runs to **full depth with no stopping test**; all stopping happens on the
  held-out half, where it is honest.
- The stopping rule is a **pure function `Curve -> int`**, so one traversal scores
  every rule at every threshold.
- Directions are combined by **MIN**, which requires both disjoint halves to name
  something and makes the null rate `alpha²` by construction.

**Removed**
- The hard set intersection between the two directions. Cross-fitting without it
  recovers 52.6% of the fold-matched full fit against 43.9%, and is fully out of fold
  where the intersection's number is not.
- The absolute increment threshold, replaced by a z-form that needs no per-pool
  calibration.

**Measured**
- Real pool at `z_thr = 1.28`: 38.9% of the full fit on 15 traits, nominal null rate
  0.01. At 1.645: 28.5% on 8 traits.
- Rule comparison on cached curves: `one_se(2.0)` reaches 41.3% real recovery and
  power 0.570 at a false-positive rate of 0.00, and `window(3)` reaches power 0.710 —
  both dominate the adopted `plateau(1.645)`, whose power is 0.130.

**Known open**
- The simulation calibration confirming `alpha²` empirically is unfinished.
- `plateau` stays the default because the current results were produced with it, not
  because it is the best rule measured.

---

## risk — 2026-08-29 — a risk criterion instead of a gate (parallel branch)

*`legacy/risk_select/`, documented in `docs/versions/risk_selection.md`.*

**Added**
- `R_hat(S) = D² - Var(D) + Var(tau_S)` over the nested prefixes, with no p-value
  gate anywhere and the empty set scored explicitly.
- The profiled `L x L` G x E system: the additive term, all N x E terms and the noise
  are Schur-complemented out once.

**Measured**
- `L = 47` null: RMSE 0.0082 against the gate's 0.0093 and the unselected full fit's
  0.0133; P(names anything) 0.44 against 0.74.
- `L = 10`: the gate wins on every axis. **The crossover between 10 and 30 blocks is
  a real property, not noise.**
- Under signal the full fit has the lowest median squared error in all four arms.
  Selection buys interpretability, not precision.

---

## 3.0.0 — 2026-09-04 — the sample split

*`legacy/cluster_greedy_v3/`, documented in
`docs/versions/v3_split_and_representatives.md`.*

**Added**
- Independent half-sample folds, respecting family and ancestry structure; grow on
  one, evaluate on the other. Null naming rate 0.74 to 0.085 from the fold structure
  alone.

**Tried and dropped**
- **Representatives** — scoring a block by one column. The induced partition is
  identical to the correlation partition at `K = 30` and the selected sets do not
  move. Kept as an option, off by default.
- **A proportion-of-total stopping statistic** — its denominator is `<= 0` in 42.6%
  of null replicates.

**Traps recorded**
- A `B`-split ensemble costs `B` engine runs; an export is already reduced over
  individuals and cannot be re-cut.
- Scaling `E` over all rows while analysing complete cases inflates the realised
  effect by `1/(1 - p_missing)`; measured 1.3127 against a predicted 1.3127.

---

## 2.0.0 — 2026-08-27 — order by the estimate, not by its p-value

*`legacy/cluster_greedy_v2/`, documented in `docs/versions/v2_theta_ordering.md`.*

**Changed**
- Candidates are ranked by summed `h2_GxE` (one solve each) and only the winner is
  jackknifed: 108.7 jackknifed evaluations per trait down to 4.5, a 24x cut.
- Seed-swap reproducibility (chance-corrected Jaccard) 0.68 to 0.86; family-wise null
  rate and power unchanged.

**Tried and rejected**
- **Backward elimination.** Fires on 1 of 42 traits at `alpha_drop = 0.10` and never
  in about 700 replicates of the `L = 10` simulations. Moves no median.

**Added**
- A memoised solve layer keyed on the admitted *set*, with the delete-one jackknife
  batched into one LAPACK call. Asserted bit-comparable to the unbatched statistic.

---

## 1.0.0 — 2026-08-27 — the block is the unit

*`legacy/cluster_greedy_v1/`, documented in `docs/versions/v1_cluster_greedy.md`.*

**Added**
- Clustering from `E` alone (average linkage on `1 - abs(r)`, cut 0.8), and a round
  statistic that is the jackknife of a block's **summed** G x E, so within-block
  correlation is absorbed rather than modelled. A singleton block reduces exactly to
  the single-environment test.

**Measured**
- Seed-swap Jaccard 0.30 to 0.68. Simulation false-positive proportion 0.045-0.067 at
  nominal 0.05, power 0.87-0.94.

**Known limit, and the reason for everything after it**
- Per-block false-positive proportion stays nominal while the family-wise rate climbs
  0.39 (`K = 10`) to 0.60 (`K = 25`) to 0.74 (`K = 30`). The round test has no
  multiplicity correction, and the gate walks the in-fold curve.

---

## 0.1.0 — the sidecar and forward selection

*`legacy/select_env_subset.py`, documented in
`docs/versions/v0_forward_selection.md`.*

**Added**
- The normal-equation export and the observation that a sub-model's system is a
  principal submatrix of the full one, so `L²/2` engine fits become one fit plus
  `O(L² J)` small solves.
- Forward selection with beam search, an early-stopping threshold with patience,
  ForwardStop applied post hoc, and Sidak or empirical calibration of the round
  p-value.
- The conventions the whole line still uses: `h2` as `sigma tr(K) / total`,
  delete-one jackknife standard errors, and `p := 1` for a non-positive estimate.

**Known limits**
- Not reproducible on a pool with near-duplicates (seed-swap Jaccard 0.30).
- The round test compares a minimum over candidates to a fixed threshold with no
  multiplicity correction.
