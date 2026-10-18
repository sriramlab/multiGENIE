"""Held-out GxE heritability of the reported set (the paper's tau-hat).

The reported set S-hat is chosen from the selection frequencies of all 2B orderings, and every
individual lies on the selection side of some pair, so refitting S-hat on any half is not held out
with respect to the choice of the set (the refit is inflated by the winner's curse).  Here the set
is chosen without the individuals that score it: for every subset Q of m of the G blocks of
individuals (m = G/4: a quarter of the sample; 220 subsets for G = 12), the orderings grown on
halves disjoint from Q (84 of the 924 when all pairs are used; nothing is regrown) are reduced to
their consensus S_Q at the depth k_g (same rule as S-hat: selection frequency in the first k_g),
S_Q is refitted on the individuals of Q from the summed block tensor, and the values are averaged
over the subsets; the delete-one-block jackknife is averaged block by block.

Check built in: with m = G/2 a single ordering survives per subset -- the one grown on Q's
complement -- and its first k_g environments are scored on Q, which is exactly the curve point
c(k_g) of that direction; `heldout_tau(..., m=G // 2)` must reproduce those curve points.

The depth k_g is chosen on all individuals, so tau-hat is held out with respect to the choice of
the set but not of its size.  In reported null replicates of the paper's simulations tau-hat carries
a selection premium (mean 0.015 for a true value of zero), so for traits with small estimates it is
an upper estimate.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .pairs import prepare_export


def _mg():
    import multi_gxe as mg
    return mg


def heldout_tau(tensor, saved: Dict[str, Any], k_g: Optional[int] = None, m: Optional[int] = None,
                drop: Optional[Sequence[str]] = None, log=None) -> Dict[str, Any]:
    """Held-out tau-hat of the consensus set.

    tensor  an AtomTensor with its phenotype terms (the one the pairs were run on)
    saved   the contents of a <trait>.pairs.npz written by atomsplit.pairs.run_trait (needs C, J,
            paths, pair_of, pairs, labels); a dict or an NpzFile
    k_g     the depth; default the cut of the averaged curve at multi_gxe.compsplit.GAMMA
    m       blocks held out per subset; default G // 4
    drop    environments dropped from the design, as for the pairs (prepare_export)
    Returns tau, se, jack (block replicates), comps (per-environment held-out components, averaged
    over subsets, in the order of `labels`), comps_jack, S_hat, k_g, m, n_subsets, and per-subset rows.
    """
    mg = _mg()
    from multi_gxe.evaluator import Evaluator, _component_h2_stack
    from multi_gxe.normal_eq import jk_se, select_indices
    from multi_gxe.stopping import cut_depth
    paths = np.asarray(saved["paths"], int); pair_of = np.asarray(saved["pair_of"], int)
    pairs = np.asarray(saved["pairs"], int); labels = [str(x) for x in saved["labels"]]
    C, J = np.asarray(saved["C"], float), np.asarray(saved["J"], float)
    G = tensor.G; D = len(paths); K = len(labels)
    if k_g is None:
        k_g = int(cut_depth(C.mean(0), J.mean(0), mg.compsplit.GAMMA))
    m = G // 4 if m is None else int(m)
    if k_g == 0:
        return dict(tau=0.0, se=0.0, jack=None, comps=np.zeros(K), comps_jack=None, S_hat=[], k_g=0, m=m, n_subsets=0, rows=[])
    if pairs.shape[1] != G:
        raise ValueError(f"pairs have {pairs.shape[1]} blocks, the tensor {G}")
    if D != 2 * len(pairs):
        raise ValueError(f"{D} orderings for {len(pairs)} pairs")
    # run_trait stores the two directions of pair i consecutively: the even one grows on the first half
    grow = [set(int(a) for a in (pairs[pair_of[d]][: G // 2] if d % 2 == 0 else pairs[pair_of[d]][G // 2:])) for d in range(D)]
    S_hat = [labels[b] for b in mg.consensus(paths, k_g, "freq")[:k_g]]
    names0 = list(tensor.env_names); blocks0 = [[i] for i in range(len(names0))]
    taus, jacks, rows = [], [], []
    comps_sum = np.zeros(K); cjack_sum = None; used = 0
    for Qs in itertools.combinations(range(G), m):
        Q = set(Qs); ids = [d for d in range(D) if not (grow[d] & Q)]
        if not ids:
            continue                                  # a subset with no ordering grown without it (possible with B < all pairs)
        S_Q = [labels[b] for b in mg.consensus(paths[ids], k_g, "freq")[:k_g]]
        ex, _ = prepare_export(tensor.assemble(list(Qs)), drop, blocks0, names0)
        idx = [ex.env_names.index(e) for e in S_Q]; ev = Evaluator(ex); r = ev.cumulative(idx)
        sel = np.asarray(select_indices(ev.export, idx), dtype=int); f = ev.export.full
        comp = _component_h2_stack(f.x_l[np.ix_(sel, sel)], f.y_r[sel])[1:1 + len(idx)]
        cj = _component_h2_stack(ev._jx[:, sel][:, :, sel], ev._jy[:, sel])[:, 1:1 + len(idx)]
        li = [labels.index(e) for e in S_Q]; comps_sum[li] += comp
        if cjack_sum is None:
            cjack_sum = np.zeros((cj.shape[0], K))
        cjack_sum[:, li] += cj; used += 1
        taus.append(r["cum_h2"]); jacks.append(np.asarray(r["jack_cum"], float))
        rows.append(dict(subset=list(Qs), n_orderings=len(ids), tau=r["cum_h2"], S_Q=S_Q))
        if log and used % 50 == 1:
            log(f"[heldout] subset {used}  ({len(ids)} orderings)  tau {r['cum_h2']:.4f}")
    jack = np.mean(jacks, axis=0)
    return dict(tau=float(np.mean(taus)), se=float(jk_se(jack)), jack=jack, comps=comps_sum / used,
                comps_jack=cjack_sum / used, S_hat=S_hat, k_g=k_g, m=m, n_subsets=used, rows=rows)


def n_eff(comps) -> float:
    """Effective number of environments of a set of components: 1 / sum s^2 over the shares of the
    positive components (negative components, which reflect noise, set to zero)."""
    v = np.clip(np.asarray(comps, float), 0, None)
    return float(1.0 / np.sum((v / v.sum()) ** 2)) if v.sum() > 0 else float("nan")
