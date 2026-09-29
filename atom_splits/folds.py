#!/usr/bin/env python3
"""The complementary split: two family-intact, ancestry-balanced halves.

    ~/.conda/envs/test/bin/python atom_splits/folds.py

Writes `data/cluster_greedy_v3/folds_<pool>.tsv` -- `FID IID fold family` -- and a
report.  The split is a property of the DESIGN (genotypes, relatedness, ancestry),
not of any phenotype, so it is computed once and reused by every arm and replicate.

THREE REQUIREMENTS, ALL FROM THE SPEC
-------------------------------------
1  RELATIVES IN THE SAME FOLD.  Cross-fold relatedness leaks genetic signal from the
   growing fold into the evaluating fold and defeats the whole point of the split.
   Families are the connected components of the UKB kinship table above
   `--kin-threshold` (0.0442 = third degree, UKB's own cut), by union-find, and a
   family is assigned as a unit.  The check afterwards is exact: ZERO related pairs
   may straddle the folds.

2  ANCESTRY BALANCED.  Two halves that differ in ancestry are not exchangeable and
   the evaluation curve inherits a shift of its own.  Families are assigned greedily
   largest-first to whichever fold currently minimises

       imbalance = |n_A - n_B| / n  +  w * sum_j |mean_A(PC_j) - mean_B(PC_j)| / sd(PC_j)

   over the first `--n-pc` genetic PCs.  Largest-first matters: the big families are
   the ones that can unbalance the split, and once the small ones are left there is
   enough freedom to correct.

3  THE GENOME-BLOCK JACKKNIFE IS ORTHOGONAL.  Blocks are over SNPs and folds are
   over samples; each fold runs its own 100-block delete-one jackknife.  Nothing
   here touches that.

The assignment is deterministic given (analysed sample, kinship file, PC file,
seed): re-running reproduces the same folds, which is what makes the two the engine runs
of a replicate re-runnable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

import os

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("MULTI_GXE_ROOT", Path.cwd()))   # analysis root: data/, results/
MISSING, MISSING_TOL = -9.0, 1e-6                            # the engine's missing code

GENO = "/u/project/sriram/ukbiobank/data/geno/cal/filter4_no_mhc/filter4_no_mhc"
# The KING kin0 shards, which ARE in the .fam's ID space.  The other relatedness
# table on this system, `data/aux/ukb33127_rel_s488363.dat`, is NOT: zero of its
# 107,156 pairs has EITHER member in the .fam, so it belongs to a different
# application's ID encryption and using it would silently declare everybody
# unrelated.  `families()` therefore checks the overlap and refuses to continue at
# zero rather than returning 219,209 singleton families.
KIN = "/u/project/sriram/boyang19/Shared/Kinship/king_all.kin0.*"
PCS = "/u/scratch/z/zhengton/nonlinear_gxe/data/genetic_PCs.txt"
KIN_THRESHOLD = 0.0442          # UKB's own third-degree cut
N_PC = 4
W_PC = 1.0


# ------------------------------------------------------------------ union-find
class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


# ---------------------------------------------------------------------- inputs
def analysed_sample(env_path, geno_prefix=GENO):
    """The rows the engine would analyse: listwise-complete in the .env, present in the
    .fam.  Returned in FAM order, which is the order every input file uses."""
    import pandas as pd
    df = pd.read_csv(env_path, sep=r"\s+", dtype={0: str, 1: str})
    E = df.iloc[:, 2:].to_numpy(float)
    iid = df.iloc[:, 1].astype(str).to_numpy()
    valid = ~(~np.isfinite(E) | (np.abs(E - MISSING) < MISSING_TOL)).any(axis=1)
    with open(f"{geno_prefix}.fam") as fh:
        fam = np.asarray([line.split()[1] for line in fh], dtype=str)
    pos = {s: k for k, s in enumerate(iid)}
    take = np.array([pos.get(s, -1) for s in fam])
    if (take < 0).any():
        raise SystemExit("[ERR] the .env does not cover the .fam")
    return fam, valid[take], E[take]


def read_pcs(path, ids: Sequence[str], n_pc: int) -> np.ndarray:
    import pandas as pd
    df = pd.read_csv(path, sep=r"\s+", usecols=list(range(2 + n_pc)),
                     dtype={0: str, 1: str})
    df = df.set_index(df.columns[1])
    P = df.reindex(list(ids)).iloc[:, 1:1 + n_pc].to_numpy(float)
    miss = ~np.isfinite(P).all(1)
    if miss.any():
        # a sample with no PCs cannot be balanced on; centre it so it is neutral
        P[miss] = np.nanmean(P[~miss], axis=0)
        print(f"[folds] {int(miss.sum())} samples have no PCs; set to the mean")
    return (P - P.mean(0)) / P.std(0)


def _pair_frames(kin_glob):
    """Yield (ID1, ID2, KINSHIP) frames from one file or a glob of KING shards."""
    import glob as _glob
    import pandas as pd
    files = sorted(_glob.glob(str(kin_glob))) or [str(kin_glob)]
    for f in files:
        head = open(f).readline().split()
        if head and head[0].lstrip("#").upper() in ("FID1", "ID1"):
            df = pd.read_csv(f, sep=r"\s+", dtype=str)
            df.columns = [c.lstrip("#") for c in df.columns]
        else:
            # headerless KING kin0 shard
            df = pd.read_csv(f, sep=r"\s+", header=None, dtype=str,
                             names=["FID1", "ID1", "FID2", "ID2", "NSNP",
                                    "HETHET", "IBS0", "KINSHIP"])
        kcol = "KINSHIP" if "KINSHIP" in df.columns else "Kinship"
        yield df["ID1"].values, df["ID2"].values, df[kcol].astype(float).values


def related_pairs(ids: Sequence[str], kin_glob, threshold: float):
    """(i, j) index pairs above `threshold` with BOTH members in `ids`.

    Vectorised: the shards hold ~1.6 M pairs in total and a Python loop over them
    costs minutes for an answer pandas gives in seconds.
    """
    pos = {s: i for i, s in enumerate(ids)}
    keys = np.array(list(pos.keys()))
    ii, jj = [], []
    n_tot = n_either = 0
    for a, b, k in _pair_frames(kin_glob):
        n_tot += len(a)
        m = k > threshold
        a, b = a[m], b[m]
        ia = np.array([pos.get(x, -1) for x in a])
        ib = np.array([pos.get(x, -1) for x in b])
        n_either += int(((ia >= 0) | (ib >= 0)).sum())
        ok = (ia >= 0) & (ib >= 0)
        ii.append(ia[ok])
        jj.append(ib[ok])
    ii = np.concatenate(ii) if ii else np.array([], int)
    jj = np.concatenate(jj) if jj else np.array([], int)
    if n_tot and n_either == 0:
        raise SystemExit(
            f"[ERR] none of the {n_tot:,} pairs in {kin_glob} has EITHER member in "
            "the analysed sample. That is an ID-SPACE mismatch, not an unrelated "
            "cohort -- using it would silently declare everybody unrelated and "
            "requirement 1 of the split would be unmet. Point --kin at a table in "
            "the .fam's ID space, or pass --assume-unrelated to proceed with the "
            "assumption recorded.")
    print(f"[folds] {n_tot:,} pairs scanned; {len(ii):,} above kinship {threshold} "
          f"with both members in the analysed sample")
    return ii, jj


def families(ids: Sequence[str], kin_glob, threshold: float,
             assume_unrelated: bool = False) -> np.ndarray:
    """Connected components of the kinship graph, restricted to `ids`."""
    if assume_unrelated:
        print("[folds] --assume-unrelated: every sample is its own family. The "
              "PROVENANCE this rests on is filter4.log, which removed 139,026 of "
              "430,299 people via a `to_remove` list; that this list is the "
              "relatedness filter is an ASSUMPTION recorded here, not a check.")
        return np.arange(len(ids))
    ii, jj = related_pairs(ids, kin_glob, threshold)
    dsu = DSU(len(ids))
    for a, b in zip(ii, jj):
        dsu.union(int(a), int(b))
    lab = np.array([dsu.find(i) for i in range(len(ids))])
    _, lab = np.unique(lab, return_inverse=True)
    sizes = np.bincount(lab)
    print(f"[folds] -> {lab.max() + 1:,} families "
          f"({int((sizes > 1).sum()):,} with more than one member, "
          f"largest {sizes.max()})")
    return lab


# ------------------------------------------------------------------ assignment
def assign(fam_lab: np.ndarray, P: np.ndarray, w_pc: float = W_PC,
           seed: int = 0, shuffle_ties: bool = False) -> np.ndarray:
    """Greedy largest-family-first, minimising size + PC imbalance.

    `shuffle_ties=False` (the default) is DETERMINISTIC AND SEED-INDEPENDENT, and that
    is not a quirk to work around -- it is what makes the B = 1 arm reproducible.  The
    `rng` below is consulted only on an EXACT tie in the imbalance score, which for
    continuous PC sums happens once, on the first family, when both folds are empty.
    So the seed picks the global A/B LABEL and nothing else: every seed returns the
    same partition, or its exact complement.

    Several independent splits need `B` genuinely different
    partitions, and `make_splits.sh` asked for them by varying `--seed`.  It got one
    partition five times (measured: pairwise fold agreement is exactly 1.000000 or
    exactly 0.000000 across seeds 20260904 + 1000*b, not one individual moves).
    `shuffle_ties=True` is the fix: permute the families, then STABLE-sort by
    descending size, so the greedy still runs largest-family-first -- same balance
    objective, same relatedness guarantee -- while the order WITHIN each size tier is
    a function of the seed.  The sample is overwhelmingly singleton families, so that
    tier is where the partition actually lives.

    Split 1 must keep `shuffle_ties=False` to reproduce the exports already on disk.
    """
    n = len(fam_lab)
    sizes = np.bincount(fam_lab)
    if shuffle_ties:
        perm = np.random.default_rng(seed).permutation(len(sizes))
        order = perm[np.argsort(-sizes[perm], kind="stable")]
    else:
        order = np.argsort(-sizes, kind="stable")
    members: Dict[int, np.ndarray] = {}
    for f in order:
        members[int(f)] = np.where(fam_lab == f)[0]

    n_f = np.zeros(2)
    s_f = np.zeros((2, P.shape[1]))
    fold = np.zeros(n, dtype=np.int8)
    rng = np.random.default_rng(seed)

    def imbalance(nn, ss):
        tot = nn.sum()
        if tot == 0:
            return 0.0
        m = np.zeros_like(ss)
        for k in (0, 1):
            if nn[k]:
                m[k] = ss[k] / nn[k]
        return abs(nn[0] - nn[1]) / tot + w_pc * float(np.abs(m[0] - m[1]).sum())

    # SEED THE TWO FOLDS BEFORE THE GREEDY -- only on the shuffled path, so the
    # deterministic split 1 stays bit-identical to the exports already on disk.
    # `imbalance` reads an EMPTY fold's PC mean as 0 (the global mean) while a
    # one-member fold's mean is O(1 sd), so "leave the second fold empty" scores
    # better than "start it", and the greedy can fall into an all-one-fold
    # attractor: 2 of 5 shuffled seeds put every individual in one fold before
    # this was added.  Placing the two largest families opposite each other costs
    # nothing and removes the attractor.
    if shuffle_ties and len(order) >= 2:
        for k, f in enumerate(order[:2]):
            idx = members[int(f)]
            fold[idx] = k
            n_f[k] += len(idx)
            s_f[k] += P[idx].sum(0)
        order = order[2:]

    for f in order:
        idx = members[int(f)]
        best, best_score = None, None
        for k in (0, 1):
            nn, ss = n_f.copy(), s_f.copy()
            nn[k] += len(idx)
            ss[k] += P[idx].sum(0)
            sc = imbalance(nn, ss)
            if best_score is None or sc < best_score - 1e-15:
                best, best_score = k, sc
            elif abs(sc - best_score) <= 1e-15 and rng.random() < 0.5:
                best = k
        fold[idx] = best
        n_f[best] += len(idx)
        s_f[best] += P[idx].sum(0)
    return fold


def report(ids, fold, fam_lab, P, kin_glob, threshold,
           assume_unrelated: bool = False) -> Dict[str, object]:
    out: Dict[str, object] = {}
    out["n_A"], out["n_B"] = int((fold == 0).sum()), int((fold == 1).sum())
    if assume_unrelated:
        out["cross_fold_related_pairs"] = 0
        out["relatedness_checked"] = 0
    else:
        ii, jj = related_pairs(ids, kin_glob, threshold)
        out["cross_fold_related_pairs"] = int((fold[ii] != fold[jj]).sum())
        out["relatedness_checked"] = 1
    for j in range(P.shape[1]):
        mA, mB = P[fold == 0, j].mean(), P[fold == 1, j].mean()
        se = np.sqrt(P[fold == 0, j].var(ddof=1) / out["n_A"]
                     + P[fold == 1, j].var(ddof=1) / out["n_B"])
        out[f"PC{j + 1}_diff_sd"] = float(mA - mB)
        out[f"PC{j + 1}_z"] = float((mA - mB) / se) if se > 0 else np.nan
    out["largest_family"] = int(np.bincount(fam_lab).max())
    return out


def main() -> None:
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=str(ROOT / "data/envs_pool47.env"))
    ap.add_argument("--geno", default=GENO)
    ap.add_argument("--kin", default=KIN,
                    help="KING kin0 file or glob, in the .fam's ID space")
    ap.add_argument("--assume-unrelated", action="store_true",
                    help="skip the relatedness step and record the assumption. "
                         "ONLY if no kinship table in the right ID space exists.")
    ap.add_argument("--pcs", default=PCS)
    ap.add_argument("--kin-threshold", type=float, default=KIN_THRESHOLD)
    ap.add_argument("--n-pc", type=int, default=N_PC)
    ap.add_argument("--w-pc", type=float, default=W_PC)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--shuffle-ties", action="store_true",
                    help="let the seed reach the PARTITION, not just the A/B label, "
                         "by shuffling families within each size tier.  REQUIRED for "
                         "any split after the first; see assign()'s docstring.")
    ap.add_argument("--pool", default=None, help="tag for the output filename")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()

    pool = a.pool or Path(a.env).stem.replace("envs_", "")
    out = Path(a.out or ROOT / f"data/cluster_greedy_v3/folds_{pool}.tsv")

    fam_ids, valid, _ = analysed_sample(a.env, a.geno)
    ids = list(np.asarray(fam_ids)[valid])
    print(f"[folds] analysed sample: {len(ids):,} of {len(fam_ids):,} .fam rows")

    fam_lab = families(ids, a.kin, a.kin_threshold, a.assume_unrelated)
    P = read_pcs(a.pcs, ids, a.n_pc)
    fold = assign(fam_lab, P, a.w_pc, a.seed, a.shuffle_ties)
    rep = report(ids, fold, fam_lab, P, a.kin, a.kin_threshold, a.assume_unrelated)

    print(f"[folds] n_A = {rep['n_A']:,}  n_B = {rep['n_B']:,}  "
          f"(|dn| = {abs(rep['n_A'] - rep['n_B']):,}), "
          f"largest family {rep['largest_family']}")
    print(f"[folds] cross-fold related pairs: {rep['cross_fold_related_pairs']} "
          f"(MUST be 0)")
    for j in range(a.n_pc):
        print(f"[folds]   PC{j + 1}: mean diff {rep[f'PC{j + 1}_diff_sd']:+.5f} sd, "
              f"z = {rep[f'PC{j + 1}_z']:+.2f}")
    if rep["cross_fold_related_pairs"]:
        raise SystemExit("[ERR] related pairs straddle the folds; the split is invalid")
    # A greedy that falls into the one-fold attractor still writes a syntactically
    # perfect TSV, and the the engine runs downstream take ~0.6 h each before anyone looks.
    # Fail here instead.
    skew = abs(rep["n_A"] - rep["n_B"]) / max(rep["n_A"] + rep["n_B"], 1)
    if skew > 0.02:
        raise SystemExit(f"[ERR] folds are {skew:.1%} out of balance "
                         f"(n_A = {rep['n_A']:,}, n_B = {rep['n_B']:,}); "
                         f"the split is invalid")

    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(FID=ids, IID=ids, fold=np.where(fold == 0, "A", "B"),
                      family=fam_lab)).to_csv(out, sep="\t", index=False)
    pd.Series(rep).to_csv(out.with_suffix(".report.tsv"), sep="\t", header=False)
    print(f"wrote {out}\nwrote {out.with_suffix('.report.tsv')}")


if __name__ == "__main__":
    main()
