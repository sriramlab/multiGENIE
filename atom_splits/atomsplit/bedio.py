"""Genotype blocks exactly as the engine's BED reader produces them.

Mirrors the engine's BED reader:

    orientation      00 -> 0, 01 -> missing, 10 -> 1, 11 -> 2   (A2 dosage; bed_reader count_A1=False)
    statistics       mean = sum / n_observed over the analysed rows, sd = sqrt(mean (1 - mean/2)),
                     monomorphic (mean 0 or 2) -> sd = 1                    (compute_snp_stats, 959-1003)
    imputation       a missing genotype is drawn from HWE with the deterministic seed
                     42 ^ hash_index(global SNP index) ^ hash_index(LOCAL sample index),
                     SplitMix64 -> U(0,1)                                   (genotype_reader.h 28-50, impute_missing)
    standardisation  (g - mean) / sd for every entry, imputed ones included

The local sample index is the rank of the individual within the analysed subset, in
.fam order, which is how the C++ loops over `decoded_subset`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

MISSING = -127          # bed_reader's int8 missing code


# ------------------------------------------------------------------ the engine's hash
def hash_index(idx) -> np.ndarray:
    """uint32 hash of an integer index, `genotype_reader.h:28-34`."""
    x = np.asarray(idx, dtype=np.int64).astype(np.uint32)      # static_cast<uint32_t>
    x = x ^ (x >> np.uint32(16))
    x = x * np.uint32(0x45d9f3b)                               # wraps mod 2^32
    x = x ^ (x >> np.uint32(16))
    return x


def splitmix64_double(seed) -> np.ndarray:
    """SplitMix64 output mapped to [0, 1), `genotype_reader.h:44-50`."""
    s = np.asarray(seed, dtype=np.uint64)
    s = s + np.uint64(0x9e3779b97f4a7c15)
    s = (s ^ (s >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    s = (s ^ (s >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    s = s ^ (s >> np.uint64(31))
    return (s >> np.uint64(11)).astype(np.float64) * (2.0 ** -53)


def snp_stats(codes: np.ndarray):
    """(allele_freq, mean, sd) per SNP from an (n, m) int8 block with MISSING entries."""
    obs = codes != MISSING
    n_obs = obs.sum(axis=0)
    if np.any(n_obs == 0):
        raise ValueError("All genotypes missing for a SNP")
    s = np.where(obs, codes, 0).astype(np.int64).sum(axis=0)
    af = s / (2.0 * n_obs)
    mean = 2.0 * af
    sd = np.sqrt(mean * (1.0 - 0.5 * mean))
    mono = (mean == 0.0) | (mean == 2.0)
    sd[mono] = 1.0
    return af, mean, sd


def hwe_impute(codes: np.ndarray, snp_global: np.ndarray, sample_local: np.ndarray,
                af: Optional[np.ndarray] = None) -> np.ndarray:
    """Replace MISSING entries in place with the engine's deterministic HWE draw."""
    miss = codes == MISSING
    if not miss.any():
        return codes
    if af is None:
        af = snp_stats(codes)[0]
    ii, jj = np.nonzero(miss)
    seed = (np.uint64(42)
            ^ hash_index(np.asarray(snp_global)[jj]).astype(np.uint64)
            ^ hash_index(np.asarray(sample_local)[ii]).astype(np.uint64))
    r = splitmix64_double(seed)
    p = af[jj]
    p0 = (1.0 - p) ** 2
    p1 = 2.0 * p * (1.0 - p)
    g = np.where(r < p0, 0, np.where(r < p0 + p1, 1, 2)).astype(np.int8)
    codes[ii, jj] = g
    return codes


def standardize_block(codes: np.ndarray, snp_global: np.ndarray,
                      sample_local: Optional[np.ndarray] = None,
                      dtype=np.float32) -> np.ndarray:
    """Standardised (n, m) block, the engine's conventions.  `codes` is consumed (imputed in place)."""
    n = codes.shape[0]
    if sample_local is None:
        sample_local = np.arange(n)
    af, mean, sd = snp_stats(codes)
    hwe_impute(codes, snp_global, sample_local, af)
    X = codes.astype(dtype, order="K")
    X -= mean.astype(dtype)[None, :]
    X *= (1.0 / sd).astype(dtype)[None, :]
    return X


# ------------------------------------------------------------------- readers
class BedBlockReader:
    """Blocks of SNPs for a fixed row subset, straight from the .bed via bed_reader."""

    def __init__(self, prefix: str | Path, fam_index: Sequence[int], num_threads: int = 1):
        from bed_reader import open_bed
        prefix = str(prefix)
        self.path = prefix if prefix.endswith(".bed") else prefix + ".bed"
        # count_A1=False: value = A2 dosage, i.e. 11 -> 2, which is the engine's orientation
        self.bed = open_bed(self.path, count_A1=False)
        self.rows = np.asarray(fam_index, dtype=np.int64)
        if np.any(np.diff(self.rows) <= 0):
            raise ValueError("fam_index must be strictly increasing (.fam order)")
        self.n_snps = int(self.bed.sid_count)
        self.n_rows = int(len(self.rows))
        self.num_threads = int(num_threads)

    def read(self, a: int, b: int) -> np.ndarray:
        return self.bed.read(np.s_[self.rows, a:b], dtype="int8", num_threads=self.num_threads)


class ArrayBlockReader:
    """The same interface over an in-memory (n, M) int8 matrix (tests, toy data)."""

    def __init__(self, codes: np.ndarray):
        self.codes = np.ascontiguousarray(codes, dtype=np.int8)
        self.n_rows, self.n_snps = self.codes.shape

    def read(self, a: int, b: int) -> np.ndarray:
        return self.codes[:, a:b].copy()


# ------------------------------------------------------------------- writing
def write_bed(prefix: str | Path, codes: np.ndarray, iids: Sequence[str],
              chrom: Optional[Sequence[int]] = None, pos: Optional[Sequence[int]] = None) -> None:
    """PLINK .bed/.bim/.fam from an (N, M) int8 matrix (0/1/2 = A2 dosage, MISSING)."""
    prefix = str(prefix)
    codes = np.asarray(codes, dtype=np.int8)
    N, M = codes.shape
    two_bit = np.zeros(codes.shape, dtype=np.uint8)
    two_bit[codes == 1] = 0b10
    two_bit[codes == 2] = 0b11
    two_bit[codes == MISSING] = 0b01
    pad = (-N) % 4
    if pad:
        two_bit = np.vstack([two_bit, np.zeros((pad, M), dtype=np.uint8)])
    tb = two_bit.T.reshape(M, -1, 4)                       # (M, ceil(N/4), 4)
    byte = (tb[..., 0] | (tb[..., 1] << 2) | (tb[..., 2] << 4) | (tb[..., 3] << 6)).astype(np.uint8)
    with open(prefix + ".bed", "wb") as fh:
        fh.write(bytes([0x6c, 0x1b, 0x01]))
        fh.write(byte.tobytes())
    chrom = [1] * M if chrom is None else list(chrom)
    pos = list(range(1, M + 1)) if pos is None else list(pos)
    with open(prefix + ".bim", "w") as fh:
        for j in range(M):
            fh.write(f"{chrom[j]}\tsnp{j}\t0\t{pos[j]}\tA\tC\n")
    with open(prefix + ".fam", "w") as fh:
        for s in iids:
            fh.write(f"{s} {s} 0 0 0 -9\n")


def read_fam_iids(prefix: str | Path) -> np.ndarray:
    ids = []
    with open(f"{prefix}.fam") as fh:
        for line in fh:
            ids.append(line.split()[1])
    return np.asarray(ids, dtype=str)
