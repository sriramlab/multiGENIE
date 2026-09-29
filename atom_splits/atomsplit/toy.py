"""Simulated data for V1 / V2: genotypes with missingness, continuous and binary
environments, covariates, a G + GxE + NxE phenotype, and every file the engine needs."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .bedio import MISSING, write_bed
from .model import Design


def make_toy(N: int = 3000, M: int = 5000, L: int = 5, G: int = 6, n_cov: int = 2,
             seed: int = 1, miss_rate: float = 0.005, n_binary: int = 1,
             h2_g: float = 0.3, h2_gxe=(0.05, 0.03), h2_nxe: float = 0.02) -> Dict:
    rng = np.random.default_rng(seed)
    maf = rng.uniform(0.05, 0.5, M)
    codes = (rng.random((N, M)) < maf).astype(np.int8) + (rng.random((N, M)) < maf).astype(np.int8)
    miss = rng.random((N, M)) < miss_rate
    codes[miss] = MISSING
    C = rng.standard_normal((N, n_cov))
    E = rng.standard_normal((N, L))
    for i in range(n_binary):
        E[:, L - 1 - i] = (rng.random(N) < 0.3).astype(float)
    E = (E - E.mean(0)) / E.std(0)
    # phenotype from the OBSERVED-imputed-as-mean standardised genotypes
    obs = codes != MISSING
    mean = np.where(obs, codes, 0).sum(0) / obs.sum(0)
    sd = np.sqrt(mean * (1 - mean / 2))
    Xs = np.where(obs, (codes - mean) / sd, 0.0)
    g = Xs @ rng.standard_normal(M) / np.sqrt(M)
    y = np.sqrt(h2_g) * g / g.std()
    for i, h in enumerate(h2_gxe):
        gi = (E[:, i] * (Xs @ rng.standard_normal(M) / np.sqrt(M)))
        y = y + np.sqrt(h) * gi / gi.std()
    y = y + C @ np.array([0.3, -0.2][:n_cov] + [0.1] * max(0, n_cov - 2)) + 0.5 * E[:, 0]
    noise = rng.standard_normal(N) * np.sqrt(1 - h2_g - sum(h2_gxe) - h2_nxe + 1e-9)
    noise = noise + np.sqrt(h2_nxe) * E[:, 1] * rng.standard_normal(N)
    y = y + noise
    # atoms: random, roughly balanced, "families" of size 1-3 kept together
    fam = np.repeat(np.arange(N), 1)
    atom = rng.permutation(np.arange(N) % G)
    ids = np.array([f"id{i:06d}" for i in range(N)])
    return dict(codes=codes, y=y, E=E, C=C, atom=atom, ids=ids, G=G,
                env_names=[f"e{i}" for i in range(L)])


def design_from_toy(t: Dict, trait: str = "toy") -> Design:
    return Design(y=t["y"], E=t["E"], C=t["C"], atom=t["atom"], G=t["G"], ids=t["ids"],
                  env_names=t["env_names"], trait=trait)


def write_toy(t: Dict, outdir: str | Path, prefix: str = "toy",
              half_atoms: Optional[np.ndarray] = None) -> Dict[str, str]:
    """Write .bed/.bim/.fam, .pheno, .covar, .env, .atoms.tsv and (optionally) the two
    -9-masked half .env files the engine needs for a half-sample run."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / prefix
    write_bed(p, t["codes"], t["ids"])
    ids = t["ids"]
    with open(f"{p}.pheno", "w") as fh:
        fh.write("FID IID pheno\n")
        for s, v in zip(ids, t["y"]):
            fh.write(f"{s} {s} {v:.10g}\n")
    with open(f"{p}.covar", "w") as fh:
        fh.write("FID IID " + " ".join(f"c{j}" for j in range(t["C"].shape[1])) + "\n")
        for s, row in zip(ids, t["C"]):
            fh.write(f"{s} {s} " + " ".join(f"{v:.10g}" for v in row) + "\n")
    names = t["env_names"]

    def write_env(path, keep):
        with open(path, "w") as fh:
            fh.write("FID IID " + " ".join(names) + "\n")
            for k, (s, row) in enumerate(zip(ids, t["E"])):
                if keep[k]:
                    fh.write(f"{s} {s} " + " ".join(f"{v:.10g}" for v in row) + "\n")
                else:
                    fh.write(f"{s} {s} " + " ".join("-9" for _ in row) + "\n")
    write_env(f"{p}.env", np.ones(len(ids), dtype=bool))
    files = dict(geno=str(p), pheno=f"{p}.pheno", covar=f"{p}.covar", env=f"{p}.env",
                 atoms=f"{p}.atoms.tsv")
    with open(files["atoms"], "w") as fh:
        fh.write("FID\tIID\tatom\tfold\tfamily\n")
        G = t["G"]
        for k, s in enumerate(ids):
            a = int(t["atom"][k])
            fh.write(f"{s}\t{s}\t{a}\t{'A' if a < G // 2 else 'B'}\t{k}\n")
    if half_atoms is not None:
        inA = np.isin(t["atom"], half_atoms)
        write_env(f"{p}.halfA.env", inA)
        write_env(f"{p}.halfB.env", ~inA)
        files["envA"], files["envB"] = f"{p}.halfA.env", f"{p}.halfB.env"
    return files
