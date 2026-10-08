"""Stage D: every complementary pair of atom-unions through the paper's rule, then
aggregate over pairs.

Per pair (A, B) and direction: `multi_gxe.crossfit.direction` unchanged (full-depth greedy
on the growth half, held-out curve on the other, the tail-test cut) -- the same call
`stability_selection/robustness_pool47.py` makes on engine exports.  Tier-0/1 cells are
sliced out and degenerate NxE columns dropped exactly as `load_pair` does there.

Aggregation (plan, Stage D.4):
    Mbar(k)      mean of the 2B held-out curves, genome-jackknife averaged block by block
    M_b(k)       pair-level curve (mean of the pair's two directions)
    f_C(k)       share of the 2B orderings with cluster C in their first k
    rho_C        mean position of C over the orderings
    ES_tau(k)    Var_b M_b(k) / Mbar(k)^2                         (estimation stability, Lim & Yu)
    ES_c(k)      mean_d ||comp_d(k) - mean comp(k)||^2 / ||mean comp(k)||^2, comp = per-cluster
                 held-out components of the top-k refit
    depth        the sequential tail tests on Mbar: k_g, reported iff k_g > 0;
                 k_es = argmin_{1<=k<=k_g} ES_tau(k) reported alongside, never used for the call
    core sets    {C : f_C(q) >= pi} for q in (1, 3, 5, k_g), pi in (0.5, 0.75, 0.9, 1)
Pair 0 is the paper's own split when the atoms refine its folds; its result is reported
next to the paper's exports run through the same code (validation V4).
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .tensor import AtomTensor, REPO

GAMMA = 0.01                           # level of the sequential tail tests (multi_gxe.tail_test)
PIS = (0.5, 0.75, 0.9, 1.0)
QS = (1, 3, 5)


def _mg():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import multi_gxe as mg
    return mg


# ------------------------------------------------------------------ pairs
def enumerate_pairs(G: int, n_pairs: Optional[int] = None, seed: int = 0,
                    first: Optional[Sequence[int]] = None) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """All C(G, G/2)/2 complementary pairs (each once, the half containing atom 0 listed
    first), or a seeded random subset of `n_pairs` that always keeps `first`."""
    if G % 2:
        raise ValueError("G must be even")
    half = G // 2
    allp = []
    for comb in itertools.combinations(range(1, G), half - 1):
        A = (0,) + comb
        B = tuple(sorted(set(range(G)) - set(A)))
        allp.append((A, B))
    first = tuple(sorted(first)) if first is not None else tuple(range(half))
    if 0 not in first:
        first = tuple(sorted(set(range(G)) - set(first)))
    allp.sort(key=lambda ab: ab[0] != first)
    if n_pairs is not None and n_pairs < len(allp):
        rng = np.random.default_rng(seed)
        rest = allp[1:]
        pick = rng.choice(len(rest), size=n_pairs - 1, replace=False)
        allp = [allp[0]] + [rest[i] for i in sorted(pick)]
    return allp


def read_drop_cells(path: Optional[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not path or not Path(path).exists():
        return out
    for line in Path(path).read_text().splitlines()[1:]:
        if not line.strip():
            continue
        t, e = line.split("\t")[:2]
        out.setdefault(t, []).append(e)
    return out


def prepare_export(export, drop: Optional[Sequence[str]], blocks0: Sequence[Sequence[int]],
                   names0: Sequence[str]):
    """Mirror of robustness_pool47.load_pair for one half: drop tier cells, re-index the
    clusters by name, drop degenerate NxE columns."""
    mg = _mg()
    ex = export
    if drop:
        d = [x for x in drop if x in ex.env_names]
        if d:
            ex = mg.drop_envs(ex, d)
    new = {n: i for i, n in enumerate(ex.env_names)}
    blocks = [[new[names0[e]] for e in b if names0[e] in new] for b in blocks0]
    blocks = [b for b in blocks if b]
    return mg.drop_degenerate_nxe(ex), blocks


def held_out_components(ev, blocks: Sequence[Sequence[int]], path: Sequence[int]) -> np.ndarray:
    """(K+1, K): held-out per-cluster h2 of the top-k refit along `path` (row k), zero
    for clusters not in the top k."""
    mg = _mg()
    from multi_gxe.evaluator import _component_h2_stack
    from multi_gxe.grow import envs_of
    K = len(blocks)
    comps = np.zeros((K + 1, K))
    env_to_block = {}
    for b, bl in enumerate(blocks):
        for e in bl:
            env_to_block[int(e)] = b
    f = ev.export.full
    for k in range(1, K + 1):
        envs = envs_of(blocks, path[:k])
        idx = np.asarray(mg.select_indices(ev.export, envs), dtype=int)
        comp = _component_h2_stack(f.x_l[np.ix_(idx, idx)], f.y_r[idx])
        for pos, e in enumerate(envs):
            comps[k, env_to_block[int(e)]] += comp[1 + pos]
    return comps


@dataclass
class DirectionRecord:
    pair: int
    grow: Tuple[int, ...]
    path: np.ndarray
    c: np.ndarray
    j: np.ndarray
    k: int
    tau: float
    se: float
    comps: np.ndarray


def run_pair(tensor: AtomTensor, A: Sequence[int], B: Sequence[int], blocks0, names0, drop,
             gamma: float = GAMMA, order: str = "dtau", pair_index: int = 0) -> List[DirectionRecord]:
    mg = _mg()
    from multi_gxe.crossfit import direction
    from multi_gxe.evaluator import Evaluator
    from multi_gxe.stopping import tail_test
    exA, blocksA = prepare_export(tensor.assemble(A), drop, blocks0, names0)
    exB, blocksB = prepare_export(tensor.assemble(B), drop, blocks0, names0)
    if list(exA.env_names) != list(exB.env_names) or blocksA != blocksB:
        raise RuntimeError("the two halves of a pair carry different environments")
    if exA.nxe_index != exB.nxe_index:
        # degenerate-NxE dropping decided per half; force a common column set
        keep = sorted(set(exA.nxe_index) & set(exB.nxe_index))
        exA = _restrict_nxe(mg, exA, keep)
        exB = _restrict_nxe(mg, exB, keep)
    evA, evB = Evaluator(exA), Evaluator(exB)
    out = []
    for grow, (eg, ee, exg, exe) in zip((A, B), ((evA, evB, exA, exB), (evB, evA, exB, exA))):
        d = direction(exg, exe, blocksA, rule=tail_test(gamma), order=order, ev_grow=eg, ev_eval=ee)
        comps = held_out_components(ee, blocksA, d.path)
        out.append(DirectionRecord(pair=pair_index, grow=tuple(int(a) for a in grow),
                                   path=np.asarray(d.path, dtype=int), c=d.curve.c, j=d.curve.j,
                                   k=int(d.k), tau=float(d.tau), se=float(d.se), comps=comps))
    return out


def _restrict_nxe(mg, ex, keep_envs):
    head = list(range(1 + ex.n_env))
    pos = {e: j for j, e in enumerate(ex.nxe_index)}
    cols = [1 + ex.n_env + pos[e] for e in keep_envs]
    idx = head + cols + [ex.dim - 1]
    return mg.Export(model=ex.model, n_env=ex.n_env, env_names=list(ex.env_names),
                     full=mg.slice_system(ex.full, idx),
                     jackknife=[mg.slice_system(j, idx) for j in ex.jackknife],
                     nxe_keep=tuple(keep_envs))


# -------------------------------------------------------------- aggregation
def aggregate(recs: List[DirectionRecord], K: int, gamma: float = GAMMA) -> Dict[str, Any]:
    mg = _mg()
    from multi_gxe.stopping import Curve, tail_test
    D = len(recs)
    Bp = D // 2
    C = np.stack([r.c for r in recs])                    # (D, K+1)
    Jk = np.stack([r.j for r in recs])                   # (D, K+1, J)
    paths = np.stack([r.path for r in recs])             # (D, K)
    comps = np.stack([r.comps for r in recs])            # (D, K+1, K)
    pair_of = np.array([r.pair for r in recs])
    pairs = np.unique(pair_of)

    Mbar = mg.average_curves([Curve(c=r.c, j=r.j) for r in recs])
    k_g = int(tail_test(gamma)(Mbar))
    se = Mbar.se_at(k_g) if k_g > 0 else 0.0
    z = float(Mbar.c[k_g] / se) if se > 0 else float("nan")
    named = bool(k_g > 0)
    se_k = np.array([Mbar.se_at(k) if k > 0 else 0.0 for k in range(K + 1)])

    Mb = np.stack([C[pair_of == p].mean(0) for p in pairs])          # (B, K+1)
    sd_pairs = Mb.std(0, ddof=1) if len(pairs) > 1 else np.zeros(K + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        es_tau = np.where(np.abs(Mbar.c) > 0, Mb.var(0, ddof=1) / Mbar.c ** 2, np.nan) if len(pairs) > 1 \
            else np.full(K + 1, np.nan)
        cbar = comps.mean(0)                                          # (K+1, K)
        es_c = np.array([np.mean(np.sum((comps[:, k] - cbar[k]) ** 2, 1)) / max(np.sum(cbar[k] ** 2), 1e-300)
                         for k in range(K + 1)])
    es_tau[0] = np.nan
    es_c[0] = np.nan
    k_es = int(1 + np.nanargmin(es_tau[1:k_g + 1])) if named and k_g >= 1 and np.isfinite(es_tau[1:k_g + 1]).any() else 0

    R = np.empty((D, K))
    for d in range(D):
        R[d, paths[d]] = np.arange(1, K + 1)
    f = np.stack([(R <= k).mean(0) for k in range(K + 1)])            # (K+1, K)
    mean_rank = R.mean(0)
    consensus = mg.consensus([list(p) for p in paths], k_g, "freq") if named else []
    per_dir_k = np.array([r.k for r in recs])
    cores = {}
    for q in list(QS) + (["k_g"] if named else []):
        qq = k_g if q == "k_g" else q
        for pi in PIS:
            cores[f"q{q}_pi{pi}"] = [int(c) for c in np.flatnonzero(f[qq] >= pi - 1e-12)] if qq > 0 else []
    return dict(K=K, D=D, B=Bp, k_g=k_g, tau=float(Mbar.c[k_g]) if named else 0.0, se=float(se), z=z,
                named=named, tau_full=float(Mbar.c[K]), se_full=float(se_k[K]),
                Mbar=Mbar.c, se_k=se_k, Mb=Mb, sd_pairs=sd_pairs, es_tau=es_tau, es_c=es_c, k_es=k_es,
                f=f, mean_rank=mean_rank, consensus=[int(c) for c in consensus], per_dir_k=per_dir_k,
                comps_mean=cbar, cores=cores, C=C, J=Jk, paths=paths, pair_of=pair_of)


def leave_one_atom_out(tensor: AtomTensor, blocks0, names0, drop) -> Dict[str, Any]:
    """Unselected full-depth fit on all atoms but one, for each atom: an individual-level
    delete-one jackknife of tau_full.  Free with the tensor."""
    mg = _mg()
    from multi_gxe.evaluator import Evaluator
    vals = []
    for g in range(tensor.G):
        ex, blocks = prepare_export(tensor.assemble([a for a in range(tensor.G) if a != g]), drop, blocks0, names0)
        vals.append(Evaluator(ex).joint()["cum_h2"])
    ex, blocks = prepare_export(tensor.assemble(range(tensor.G)), drop, blocks0, names0)
    r = Evaluator(ex).joint()
    vals = np.array(vals)
    G = tensor.G
    return dict(tau_full_all=float(r["cum_h2"]), se_full_genome=float(r["cum_se"]),
                se_full_atoms=float(np.sqrt((G - 1) / G * np.sum((vals - vals.mean()) ** 2))),
                loo_values=vals)


def paper_pair(paper_dir: str, trait: str, blocks0, names0, drop, gamma: float = GAMMA) -> Optional[Dict]:
    """The paper's two engine exports through the same rule (V4 reference)."""
    mg = _mg()
    from multi_gxe.stopping import tail_test
    fa = Path(paper_dir) / trait / f"{trait}.foldA.normal_eq.json"
    fb = Path(paper_dir) / trait / f"{trait}.foldB.normal_eq.json"
    if not (fa.exists() and fb.exists()):
        return None
    A, bl = prepare_export(mg.load_export(fa), drop, blocks0, names0)
    B, _ = prepare_export(mg.load_export(fb), drop, blocks0, names0)
    if A.nxe_index != B.nxe_index:
        keep = sorted(set(A.nxe_index) & set(B.nxe_index))
        A, B = _restrict_nxe(mg, A, keep), _restrict_nxe(mg, B, keep)
    r = mg.complementary_split([(A, B)], bl, gate_rule=tail_test(gamma), z_gate=gamma, set_rule=lambda cv: cv.K)
    return dict(named=bool(r["names"]), k=int(r["k"]), tau=float(r["tau"]), se=float(r["se"]),
                tau_full=float(r["tau_full"]), blocks=[int(b) for b in r["blocks"]],
                n_A=None, n_B=None)


# --------------------------------------------------------------------- driver
def run_trait(tensor: AtomTensor, trait: str, clusters_json: str, drop_cells: Optional[str],
              out_dir: str, *, n_pairs: Optional[int] = None, seed: int = 0, gamma: float = GAMMA,
              order: str = "dtau", jobs: int = 1,
              paper_dir: Optional[str] = None, log=print) -> Dict[str, Any]:
    mg = _mg()
    t0 = time.time()
    blocks0 = [list(map(int, c)) for c in json.loads(Path(clusters_json).read_text())["clusters"]]
    names0 = list(tensor.env_names)
    drop = read_drop_cells(drop_cells).get(trait, [])
    pairs = enumerate_pairs(tensor.G, n_pairs, seed)
    log(f"[pairs] {trait}: G={tensor.G}, {len(pairs)} pairs, K={len(blocks0)} clusters, drop={drop}")

    recs: List[DirectionRecord] = []
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        global _TENSOR
        _TENSOR = tensor
        args = [(i, A, B, blocks0, names0, drop, gamma, order) for i, (A, B) in enumerate(pairs)]
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for i, rs in enumerate(ex.map(_run_pair_global, args, chunksize=4)):
                recs.extend(rs)
                if i % max(1, len(pairs) // 10) == 0:
                    log(f"[pairs] {i + 1}/{len(pairs)}  {time.time() - t0:.0f}s")
    else:
        for i, (A, B) in enumerate(pairs):
            recs.extend(run_pair(tensor, A, B, blocks0, names0, drop, gamma, order, i))
            if i % max(1, len(pairs) // 10) == 0:
                log(f"[pairs] {i + 1}/{len(pairs)}  {time.time() - t0:.0f}s")
    _, blocks = prepare_export(tensor.assemble(pairs[0][0]), drop, blocks0, names0)
    K = len(blocks)
    agg = aggregate(recs, K, gamma)
    loo = leave_one_atom_out(tensor, blocks0, names0, drop)
    pap = paper_pair(paper_dir, trait, blocks0, names0, drop, gamma) if paper_dir else None
    p0 = [r for r in recs if r.pair == 0]
    agg0 = aggregate(p0, K, gamma)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # cluster labels in the (possibly reduced) block indexing
    ex0, _ = prepare_export(tensor.assemble(pairs[0][0]), drop, blocks0, names0)
    labels = [",".join(ex0.env_names[e] for e in b) for b in blocks]
    np.savez(out / f"{trait}.pairs.npz", C=agg["C"], J=agg["J"], paths=agg["paths"], pair_of=agg["pair_of"],
             comps_mean=agg["comps_mean"], Mbar=agg["Mbar"], se_k=agg["se_k"], Mb=agg["Mb"],
             sd_pairs=agg["sd_pairs"], es_tau=agg["es_tau"], es_c=agg["es_c"], f=agg["f"],
             mean_rank=agg["mean_rank"], per_dir_k=agg["per_dir_k"], labels=np.array(labels),
             pairs=np.array([list(A) + list(B) for A, B in pairs]), loo_values=loo["loo_values"],
             k_g=agg["k_g"], k_es=agg["k_es"], tau=agg["tau"], se=agg["se"], z=agg["z"], named=agg["named"])
    with open(out / f"{trait}.curve.tsv", "w") as fh:
        fh.write("k\tMbar\tse_jk\tsd_pairs\tes_tau\tes_c\n")
        for k in range(K + 1):
            fh.write(f"{k}\t{agg['Mbar'][k]:.6g}\t{agg['se_k'][k]:.6g}\t{agg['sd_pairs'][k]:.6g}\t"
                     f"{agg['es_tau'][k]:.6g}\t{agg['es_c'][k]:.6g}\n")
    with open(out / f"{trait}.clusters.tsv", "w") as fh:
        fh.write("block\tenvs\tf_kg\tf1\tf3\tf5\tmean_rank\tcomp_kg\tin_consensus\tcore_q3_pi1\tcore_q5_pi0.75\n")
        for b in range(K):
            fh.write(f"{b}\t{labels[b]}\t{agg['f'][agg['k_g']][b]:.3f}\t{agg['f'][1][b]:.3f}\t{agg['f'][3][b]:.3f}\t"
                     f"{agg['f'][5 if K >= 5 else K][b]:.3f}\t{agg['mean_rank'][b]:.2f}\t"
                     f"{agg['comps_mean'][agg['k_g']][b]:.5g}\t{int(b in agg['consensus'])}\t"
                     f"{int(b in agg['cores']['q3_pi1.0'])}\t{int(b in agg['cores']['q5_pi0.75'])}\n")
    row = dict(trait=trait, G=tensor.G, B=agg["B"], K=K, N=tensor.N, named=int(agg["named"]), k_g=agg["k_g"],
               tau=agg["tau"], se=agg["se"], z=agg["z"], tau_full=agg["tau_full"], se_full=agg["se_full"],
               sd_pairs_kg=float(agg["sd_pairs"][agg["k_g"]]), sd_pairs_full=float(agg["sd_pairs"][K]),
               k_es=agg["k_es"], n_consensus=len(agg["consensus"]),
               core_q3_pi1=len(agg["cores"]["q3_pi1.0"]), core_q5_pi075=len(agg["cores"]["q5_pi0.75"]),
               per_dir_k_min=int(agg["per_dir_k"].min()), per_dir_k_max=int(agg["per_dir_k"].max()),
               se_full_atoms=loo["se_full_atoms"], se_full_genome=loo["se_full_genome"],
               tau_full_allatoms=loo["tau_full_all"],
               pair0_named=int(agg0["named"]), pair0_k=agg0["k_g"], pair0_tau=agg0["tau"], pair0_se=agg0["se"],
               paper_named=(int(pap["named"]) if pap else ""), paper_k=(pap["k"] if pap else ""),
               paper_tau=(pap["tau"] if pap else ""), paper_se=(pap["se"] if pap else ""),
               paper_tau_full=(pap["tau_full"] if pap else ""),
               consensus=";".join(labels[b] for b in agg["consensus"]),
               core_q3_pi1_envs=";".join(labels[b] for b in agg["cores"]["q3_pi1.0"]),
               seconds=round(time.time() - t0, 1))
    with open(out / f"{trait}.summary.tsv", "w") as fh:
        fh.write("\t".join(row.keys()) + "\n")
        fh.write("\t".join(str(v) for v in row.values()) + "\n")
    log(f"[pairs] {trait}: named={row['named']} k_g={row['k_g']} tau={row['tau']:.4f} (se {row['se']:.4f}, "
        f"sd over pairs {row['sd_pairs_kg']:.4f}) tau_full={row['tau_full']:.4f}; pair0 tau={row['pair0_tau']:.4f} "
        f"k={row['pair0_k']}; paper tau={row['paper_tau']} k={row['paper_k']}; {row['seconds']}s")
    return dict(summary=row, agg=agg, loo=loo, paper=pap, labels=labels)


_TENSOR = None


def _run_pair_global(args):
    i, A, B, blocks0, names0, drop, gamma, order = args
    return run_pair(_TENSOR, A, B, blocks0, names0, drop, gamma, order, i)
