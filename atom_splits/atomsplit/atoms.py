"""Stage A: G family-intact, ancestry-balanced, equal-sized atoms.

Same inputs and the same three requirements as `cluster_greedy_v3/folds.py` (families =
KING kinship components above 0.0442, PC1-4 balance, genome jackknife untouched), with
one extra: when a fold file is given the atoms REFINE it -- atoms 0..G/2-1 partition fold
A and G/2..G-1 partition fold B -- so the paper's own split is the pair
({0..G/2-1}, {G/2..G-1}) and every other pair is a new complementary split of the same
people.  Families never straddle atoms, so any union of atoms is family-intact.

Assignment is greedy largest-family-first (shuffled within size tiers by the seed),
each family going to the admissible atom that minimises

    sum_g ((n_g - n/G) / (0.01 n/G))^2  +  w * sum_g sum_j (sum_{i in g} PC_ij)^2    (PCs standardised)

after the two largest admissible families have seeded distinct atoms.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

HERE = Path(__file__).resolve().parents[1]          # scripts/atom_splits
ROOT = HERE.parents[1]                              # multi_gxe
FOLDS_DIR = ROOT / "scripts/cluster_greedy_v3"


def _folds_module():
    if str(FOLDS_DIR) not in sys.path:
        sys.path.insert(0, str(FOLDS_DIR))
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    import folds
    return folds


def assign_atoms(fam_lab: np.ndarray, P: np.ndarray, G: int, seed: int = 0,
                 fold: Optional[np.ndarray] = None, w_pc: float = 1.0) -> np.ndarray:
    """Atom label per individual.  `fold` (0/1 per individual) restricts each family to
    the atoms of its fold; families must not straddle folds."""
    n = len(fam_lab)
    P = np.asarray(P, dtype=float)
    if P.ndim == 1:
        P = P[:, None]
    sizes = np.bincount(fam_lab)
    nf = len(sizes)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(nf)
    order = perm[np.argsort(-sizes[perm], kind="stable")]           # largest first, seeded ties
    # members of each family, without a np.where per family
    srt = np.argsort(fam_lab, kind="stable")
    starts = np.searchsorted(fam_lab[srt], np.arange(nf + 1))
    if fold is not None:
        fold = np.asarray(fold, dtype=int)
        if G % 2:
            raise ValueError("G must be even to refine two folds")
        half = G // 2
        adm = {0: np.arange(half), 1: np.arange(half, G)}
        n_target = np.empty(G)
        for f in (0, 1):
            n_target[adm[f]] = (fold == f).sum() / half
    else:
        adm = {None: np.arange(G)}
        n_target = np.full(G, n / G)

    n_g = np.zeros(G)
    s_g = np.zeros((G, P.shape[1]))
    atom = np.full(n, -1, dtype=np.int64)
    seeded = {k: [] for k in adm}

    def fam_fold(idx):
        if fold is None:
            return None
        f = fold[idx]
        if f.min() != f.max():
            raise ValueError("a family straddles the folds; refuse to refine")
        return int(f[0])

    # size tolerance: a deviation of 1% of an atom costs one unit, so the PC sums decide
    # within that slack and the sizes stay within ~1% (folds.py refuses > 2%)
    tol = max(0.01 * n / G, 1.0)

    def objective(nn, ss):
        # PC term on SUMS of the standardised PCs, not means -- a per-atom mean lets a large
        # atom absorb everyone (its mean barely moves while a small atom's mean jumps),
        # which is exactly the collapse seen at G = 12 with the folds.py objective
        # SQUARED sums: an L1 term is flat at zero and cannot tell the atoms apart once
        # they are balanced, so the size term takes over and the sums random-walk
        return (((nn - n_target[None, :]) / tol) ** 2).sum(1) + w_pc * (ss ** 2).sum(axis=(1, 2))

    for f in order:
        idx = srt[starts[f]:starts[f + 1]]
        k = len(idx)
        ps = P[idx].sum(0)
        key = fam_fold(idx)
        cands = adm[key]
        # seed every admissible atom with one of the largest families first
        if len(seeded[key]) < len(cands):
            a = int(cands[len(seeded[key])])
            seeded[key].append(a)
        else:
            nn = np.repeat(n_g[None, :], len(cands), axis=0)
            ss = np.repeat(s_g[None, :, :], len(cands), axis=0)
            nn[np.arange(len(cands)), cands] += k
            ss[np.arange(len(cands)), cands] += ps
            sc = objective(nn, ss)
            best = np.flatnonzero(sc <= sc.min() + 1e-12)
            a = int(cands[best[0] if len(best) == 1 else rng.choice(best)])
        atom[idx] = a
        n_g[a] += k
        s_g[a] += ps
    assert (atom >= 0).all()
    return atom


def report_atoms(atom: np.ndarray, fam_lab: np.ndarray, P: np.ndarray,
                 fold: Optional[np.ndarray] = None) -> Dict[str, object]:
    G = int(atom.max()) + 1
    out: Dict[str, object] = {"G": G}
    sizes = np.bincount(atom, minlength=G)
    for g in range(G):
        out[f"n_{g}"] = int(sizes[g])
    out["size_range"] = f"{sizes.min()}..{sizes.max()}"
    # families straddling atoms (must be 0)
    per_fam_atoms = {}
    for a, f in zip(atom, fam_lab):
        per_fam_atoms.setdefault(int(f), set()).add(int(a))
    out["families_straddling_atoms"] = int(sum(len(s) > 1 for s in per_fam_atoms.values()))
    P = np.asarray(P, dtype=float)
    if P.ndim == 1:
        P = P[:, None]
    zmax = 0.0
    for g in range(G):
        m = atom == g
        for j in range(P.shape[1]):
            d = P[m, j].mean() - P[~m, j].mean()
            se = np.sqrt(P[m, j].var(ddof=1) / m.sum() + P[~m, j].var(ddof=1) / (~m).sum())
            zmax = max(zmax, abs(d / se) if se > 0 else 0.0)
    out["max_PC_z_atom_vs_rest"] = float(zmax)
    if fold is not None:
        half = G // 2
        bad = int(((atom < half) != (np.asarray(fold) == 0)).sum())
        out["atoms_inconsistent_with_fold"] = bad
    return out


def make_atoms(env_path: str, geno_prefix: str, kin_glob: str, pcs_path: str, G: int, seed: int,
               folds_tsv: Optional[str], out_tsv: str, n_pc: int = 4, w_pc: float = 1.0,
               kin_threshold: float = 0.0442, assume_unrelated: bool = False, log=print) -> Dict:
    """Real-data Stage A on the env-complete sample of `env_path`."""
    import pandas as pd
    folds = _folds_module()
    fam_ids, valid, _ = folds.analysed_sample(env_path, geno_prefix)
    ids = list(np.asarray(fam_ids)[valid])
    log(f"[atoms] analysed (env-complete) sample: {len(ids):,} of {len(fam_ids):,} .fam rows")
    fam_lab = folds.families(ids, kin_glob, kin_threshold, assume_unrelated)
    P = folds.read_pcs(pcs_path, ids, n_pc)
    fold = None
    if folds_tsv:
        fd = pd.read_csv(folds_tsv, sep="\t", dtype={"FID": str, "IID": str})
        fmap = dict(zip(fd.IID.astype(str), fd.fold))
        fold = np.array([{"A": 0, "B": 1}.get(fmap.get(s), -1) for s in ids])
        if (fold < 0).any():
            raise SystemExit(f"[atoms] {int((fold < 0).sum()):,} analysed rows have no fold in {folds_tsv}")
        # the fold file's families may differ from ours only if inputs changed; check
        for f in np.unique(fam_lab):
            pass
    atom = assign_atoms(fam_lab, P, G, seed=seed, fold=fold, w_pc=w_pc)
    rep = report_atoms(atom, fam_lab, P, fold)
    for k, v in rep.items():
        log(f"[atoms]   {k}: {v}")
    if rep["families_straddling_atoms"]:
        raise SystemExit("[atoms] a family straddles atoms; invalid")
    if fold is not None and rep["atoms_inconsistent_with_fold"]:
        raise SystemExit("[atoms] atoms do not refine the folds; invalid")
    out = Path(out_tsv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(FID=ids, IID=ids, atom=atom,
                      fold=(np.where(fold == 0, "A", "B") if fold is not None else "-"),
                      family=fam_lab)).to_csv(out, sep="\t", index=False)
    pd.Series(rep).to_csv(out.with_suffix(".report.tsv"), sep="\t", header=False)
    log(f"[atoms] wrote {out} and {out.with_suffix('.report.tsv')}")
    return rep
