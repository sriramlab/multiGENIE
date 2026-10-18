# The algorithm, in full

What the code computes, in the order it computes it, with the statistics exactly as
implemented. Equation names in parentheses are those of the manuscript.

## 1. The model

For a standardized phenotype `y` (N individuals), standardized genotypes `X` (M variants) and a
pool of `L` standardized environments `e_1, ..., e_L`,

    y = X a + sum_l (X . e_l) b_l + sum_l diag(e_l) d_l + eps

with independent random effects `a ~ N(0, s2_g/M I)`, `b_l ~ N(0, s2_gxe,l/M I)`,
`d_l ~ N(0, s2_nxe,l I)`, `eps ~ N(0, s2_e I)`: an additive term, a GxE term and a
noise-by-environment (NxE) term per environment, and a residual. Covariates, including the
environment main effects, are projected out of `y` and of every kernel. The `P = 2L + 2`
variance components are estimated by the method of moments: `T s2 = q` with
`T[p,p'] = tr(K_p K_p')` and `q[p] = y' K_p y` (eq. normal), the traces by a randomized
(Hutchinson) estimator with `R` probes, and delete-one systems over `J` contiguous genome blocks
for the jackknife. This fit is done by the engine (the engine/GENIE) and exported once per half
(`docs/export_format.md`); the package never touches genotypes.

## 2. The identity: every sub-model is a slice of one export

`T[p,p']` and `q[p]` are pairwise quantities, defined by one or two kernels and by nothing else in
the model. The normal equations of a sub-model that uses only the environments in a subset `S`
are therefore a principal submatrix of `T` on the components

    A(S) = {g} u {(gxe,l) : l in S} u {(nxe,l) : l in S} u {e}

and the matching entries of `q` (eq. subset). `s2(S) = T[A(S),A(S)]^-1 q[A(S)]` is exactly the
estimator that fitting the model with only those environments would give. The GxE heritability of
`S` is the share of phenotypic variance carried by its interaction terms in that fit,

    tau(S) = sum_{l in S} s2_gxe,l(S) tr(K_gxe,l) / sum_{p in A(S)} s2_p(S) tr(K_p),   tr(K_e) = N

(eq. tau_subset), with the traces read from the last column of `T`. It is a refit, not a partial
sum of the full-pool estimates. The same slice of the delete-one systems gives the jackknife
replicates `tau_(j)(S)` and `se = sqrt((J-1)/J sum_j (tau_(j) - mean)^2)` (eq. jkse). A subset
costs `J + 1` solves of order at most `2|S| + 2`.

Code: `normal_eq.select_indices`, `slice_system`, `solve`, `component_h2`, `jk_se`;
`Evaluator.tau` (memoised on the set) and `Evaluator.evaluate` (batched jackknife solves).

## 3. Forward selection on the gain

Starting from the empty set, round `t` adds the environment with the largest gain,

    pi_t = argmax_{l not in S_{t-1}} [ tau(S_{t-1} u {l}) - tau(S_{t-1}) ],   S_t = S_{t-1} u {pi_t}

(eq. greedy), ties by lowest index, refitting at every round and always to full depth. Nothing is
tested inside the grow. Every environment is its own unit of selection; the pool is curated
beforehand so that no two environments are strongly correlated. Code: `grow.grow_path`.

## 4. Complementary splits and held-out curves

An ordering grown and scored on the same individuals is biased upward: each round picks the
environment whose estimate happens to be largest on that sample, and a trait without GxE still
shows a rising curve (on null simulations the in-sample curve is reported in every replicate).
The individuals are therefore split into two complementary halves that keep families intact
and are balanced on size and genetic ancestry (`atom_splits/folds.py`). Each half grows its
ordering `pi^(h)`, and the curve of that ordering is scored on the other half,

    c^(h)(k) = tau^(h')( {pi^(h)_1, ..., pi^(h)_k} ),   k = 0, ..., L

(eq. curve), an out-of-fold estimate at every depth, with jackknife replicates from the other
half's delete-one systems. Over `B` pairs the `2B` curves are averaged,

    c_bar(k) = 1/(2B) sum_{s,h} c^(s,h)(k),

and the replicates block by block (every half shares the same genome blocks), which gives the SE
of `c_bar(k)` and of any difference `c_bar(k') - c_bar(k)` as a **paired** jackknife. The
difference is paired because two prefixes are fits on the same individuals with strongly
correlated errors; summing marginal SEs would inflate it several-fold. Code: `crossfit.direction`,
`compsplit.average_curves`.

## 5. The cut: sequential tail tests

For each depth `k = 0, 1, ...` the hypothesis `H_k`: no depth past `k` is higher than `k`, is
tested with the maximum standardized rise past `k`,

    T_k = max_{k' > k} ( c_bar(k') - c_bar(k) ) / se( c_bar(k') - c_bar(k) )

(eq. tailstat), at level `gamma`, in the fixed order `H_0, H_1, ...`; the cut is the first
hypothesis not rejected, `k_g = min{k : T_k < q_k(gamma)}`, with `k_g = L` if every one is
rejected (eq. cut). The hypotheses are nested (`H_k` true implies `H_{k+1}` true), so no
multiplicity correction is needed: the procedure reaches a true hypothesis only after rejecting
the first true one, which happens with probability at most `gamma`, and the family-wise error is
at most `gamma`. `H_0` is the trait-level test; a trait is reported iff `k_g > 0`. The target of
each test is the averaged curve, so the guarantee concerns the depth at which that curve stops
rising, not the number of causal environments: when orderings disagree on which environment comes
next, the averaged curve can rise past the last causal one and the cut legitimately goes deeper.
Because each test compares the tail with depth `k` and not with the full-pool endpoint, adding
uninformative environments to the pool does not move the cut.

The null quantile `q_k(gamma)` is drawn at the boundary of `H_k` by a Gaussian multiplier
bootstrap over the jackknife replicates: with `d_(j)(k,k') = c_bar_(j)(k') - c_bar_(j)(k)`
centred over blocks and `xi ~ N(0, I_J)`,

    T*_k = max_{k' > k} sqrt((J-1)/J) sum_j xi_j d_(j)(k,k') / se( c_bar(k') - c_bar(k) )

(eq. multiplier), each term of unit variance given the data; `q_k(gamma)` is its upper-`gamma`
quantile over `10^4` draws, and one draw of `xi` serves every depth and every trait. The
approximation is anti-conservative in the far tail on `J = 100` blocks: on 500 null replicates the
realized level was 5.8% at the default `gamma = 0.05` (95% Wilson interval 4.1-8.2%) but 2.8% at
`gamma = 0.01`, where the empirical null quantile of `T_0` exceeds the bootstrap one by 8-14%
(5% at 0.05). The bootstrap p-value of the trait-level test, `p_0 = Pr*(T*_0 >= T_0)` from the same
draws, supports multiplicity control across traits. Code: `stopping.tail_test`, `stopping.tail_pvalue`.

**The reported estimate.** Refitting the consensus set on the halves is not held out: the set is
chosen from all `2B` orderings and every individual sits on the growing side of some pair. The
reported `tau-hat` is held out with respect to the choice of the set: for every subset `Q` of
`G/4` blocks, the orderings grown on halves disjoint from `Q` give a consensus `S_Q` at depth `k_g`,
`S_Q` is refitted on `Q` from the summed block tensor, and the values and their delete-one-block
replicates are averaged over the subsets (eq. tau_heldout). Holding out `G/2` blocks instead leaves
one ordering per subset and reproduces the curve point `c(k_g)` exactly, which is the built-in
check. Code: `atom_splits/atomsplit/heldout.py`.

A stopping rule is a pure function `Curve -> int`, so any alternative can be passed as
`gate_rule` and applied to the same saved curves without refitting.

## 6. What is reported

The `2B` orderings differ, so the reported set is a consensus. With `rho^(s,h)_l` the position of
environment `l` in ordering `(s,h)`, its selection frequency in the top `k_g` is
`f_l = 1/(2B) sum_{s,h} 1{rho^(s,h)_l <= k_g}` and its mean position `rho_bar_l`; environments are
ranked by decreasing `f_l`, ties by increasing `rho_bar_l`, and the reported set is the first
`k_g` (eq. consensus). For each trait: reported or not; `tau_hat = c_bar(k_g)` with its SE; the
set; and the full-pool value `c_bar(L)`. Code: `compsplit.consensus`, `compsplit.select_from_curves`.

Four properties to keep in view. `c_bar(k)` is the out-of-fold value of *a* top-`k` selection,
averaged over orderings scored on individuals they were not grown on. The tests concern the
curve, a sequence of refits of subsets: an environment can raise the curve as a proxy for a
modifier outside the pool. The error statement is about depths of the averaged curve; the
consensus maps them to environments, exactly where the orderings agree. No held-out individuals
remain to score the consensus set itself, so `tau_hat` is the value of a selection of size `k_g`,
not of that exact set.

## 7. Many splits from one pass over the genotypes

The trace estimates of a half are sums over its individuals. Partition the individuals once into
`G` family-intact, ancestry-balanced atoms `A_1, ..., A_G`; a half is any union of `G/2` atoms and
its complement the other. With `D_a` the selector of atom `a`, `K_p D_H z_r = sum_{a in H} w_{a,p}`
with `w_{a,p} = K_p D_a z_r` independent of the half, so

    T_hat[p,p'](H) = sum_{a'' in H} sum_{a,a' in H} Gamma[a''; (a,p), (a',p')],
    Gamma[a''; (a,p), (a',p')] = 1/R sum_r sum_{n in A_a''} w^(r)_{a,p,n} w^(r)_{a',p',n}

(eq. blocks): one tensor `Gamma`, computed in one pass, gives the normal equations of every half
by summation; the phenotype terms split the same way; and the delete-one tensors need only the
inner products of each genome block's contribution with the totals, formed in a second pass.
`G = 12` gives `C(12,6)/2 = 462` complementary pairs per trait for the cost of one pass. Code:
`atom_splits/atomsplit/` (`stream.py` the passes, `tensor.py` the assembly), validated exactly
against dense kernels and against the engine's own export.

## 8. The statistics, exactly as implemented

* **Heritability normalisation** (`component_h2`): component `p` is scaled by `tr(K_p)`, the
  residual by `N`, and everything is divided by the total; components are variance fractions.
* **Standard error**: delete-one jackknife over the exported genome blocks,
  `sqrt((J-1)/J sum_j (x_j - x_bar)^2)`; for a difference of prefixes, the paired jackknife of
  the difference.
* **Multiplier draws**: `10^4`, standard normal, seeded (`20261005`), one draw per `J` shared by
  every depth and every trait.
* **Consensus**: selection frequency at `k_g`, ties by mean full-path position, then by index.
* **Nuisance terms**: every selected environment carries its NxE term into the sub-model; the
  additive and residual terms are always present. (An identifiability drop of NxE kernels nearly
  collinear with the identity exists in `normal_eq.drop_degenerate_nxe` but is off; `NXE_DROP=1`
  enables it.)
