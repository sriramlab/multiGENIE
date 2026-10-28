# The export: what GENIE writes, what the selector reads

One JSON file per fit. It carries the method-of-moments normal equations for the full
sample and for every jackknife block, and nothing else — no genotypes, no phenotypes,
no individual-level anything.

```json
{
  "schema_version": 1,
  "model": "G+GxE+NxE",
  "gen_nbin": 1,
  "n_env": 42,
  "env_names": ["age", "smoking", "..."],
  "full":      { "x_l": [[...], ...], "y_r": [...] },
  "jackknife": [ { "x_l": [[...], ...], "y_r": [...] },  ... ]
}
```

| field | meaning |
|---|---|
| `model` | `G+GxE` or `G+GxE+NxE`; sets the system's dimension |
| `gen_nbin` | additive components. **Must be 1** — the index arithmetic assumes it |
| `n_env` | `L`, the number of environment columns in the fit |
| `env_names` | optional, defaults to `env_0 … env_{L-1}`; used for reporting only |
| `full.x_l` | `D x D`, `x_l[i][j] = tr(K_i K_j)`, last row/column the traces and `N` |
| `full.y_r` | length `D`, `y_r[i] = y' K_i y`, last entry `y' y` |
| `jackknife` | one such system per delete-one block, in block order |

## Component order

```
D = 1 + L + L + 1        [ G , GxE_0 … GxE_{L-1} , NxE_0 … NxE_{L-1} , noise ]     G+GxE+NxE
D = 1 + L + 1            [ G , GxE_0 … GxE_{L-1} , noise ]                         G+GxE
```

`select_indices` builds a sub-model's rows and columns in exactly that layout, **in
the order the environments were given**, so a block admitted last occupies the
trailing G x E positions. `evaluator.Evaluator.evaluate` relies on that to read off a
block's own contribution.

## Producing one

**GENIE** needs the normal-equation sidecar, which is the single C++ change in the
fork this work started from:

```bash
GENIE -g geno -p pheno -c cov -e envs.env -m G+GxE+NxE -k 10 -jn 100 -o out \
      --export-normal-eq out.normal_eq.json
```

The export is assembled from quantities GENIE already materialises for its own jackknife,
every leave-one-block-out trace matrix and the per-block right-hand side, and is inert unless
switched on.

The file is validated the same way it was when the exporter was written:
solving the exported full system must reproduce GENIE's own `sigma` to printed
precision, and delete-one solves of the exported blocks must reproduce its jackknife
standard errors.

## Things that bite

* **The sample is baked in.** GENIE drops individuals missing any environment
  column before it builds the system, so the export is a complete-case fit over all
  `L` columns. Adding one badly-missing environment to the pool shrinks `N` for every
  subset, including those that exclude it.
* **Dropping a column does not need a re-run.** `drop_envs` slices that
  environment's rows and columns out of every system, which *is* the `L - d` fit, not
  a correction to the `L` fit.
* **Size.** `(J + 1) x D²` doubles at full precision, read into memory whole.
  `L = 20, J = 100` is about 40 MB; `L = 200` would be about 4 GB.
* **Environment main effects.** GENIE includes the environment main effects among the
  covariates. See `docs/algorithm.md` section 8.
* **`h2` normalisation is the reader's.** The selector recomputes `h2` from `sigma`
  and the traces in the GENIE convention rather than reading GENIE's printed `h2_*` lines.

## A file to test against

`multi_gxe.simulate` writes systems in this schema from a chosen truth, which is how
`tests/` exercises the loader, the slicing identity and the whole pipeline without a
genotype file. See `tests/conftest.py` for the few lines that dump one to disk.
