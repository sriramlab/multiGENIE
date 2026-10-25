"""Dependency-free PLINK .bed block reader (seek + read per block; no whole-file mmap, which the 4 GB address-space
cap on the login nodes refuses for the 33 GB array .bed), same interface and codes as bedio.BedBlockReader.

bedio.BedBlockReader needs the `bed_reader` package, which none of the cluster environments carry; this reader
decodes the SNP-major .bed directly with the engine's orientation (bedio module docstring):
    2-bit 00 -> 0, 01 -> MISSING, 10 -> 1, 11 -> 2   (A2 dosage),  samples packed low bits first.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from .bedio import MISSING

_LUT = np.array([0, MISSING, 1, 2], dtype=np.int8)


def _count_lines(path: str) -> int:
    with open(path, "rb") as fh:
        return sum(1 for _ in fh)


class MmapBedReader:                                                     # name kept for the tests/CLI
    """Blocks of SNPs for a fixed row subset (.fam order) -> int8 (n_rows, b - a)."""

    def __init__(self, prefix: str | Path, fam_index: Sequence[int], num_threads: int = 1):
        prefix = str(prefix)
        stem = prefix[:-4] if prefix.endswith(".bed") else prefix
        self.path = stem + ".bed"
        self.n_fam = _count_lines(stem + ".fam")
        self.n_snps = _count_lines(stem + ".bim")
        self.bpr = (self.n_fam + 3) // 4                                  # bytes per SNP record
        with open(self.path, "rb") as fh:
            magic = fh.read(3)
        if magic != bytes([0x6C, 0x1B, 0x01]):
            raise ValueError(f"{self.path}: not a SNP-major PLINK .bed (magic {magic.hex()})")
        expect = 3 + self.n_snps * self.bpr
        size = Path(self.path).stat().st_size
        if size != expect:
            raise ValueError(f"{self.path}: {size} bytes, expected {expect} for {self.n_snps} SNPs x {self.n_fam} samples")
        self.fh = open(self.path, "rb")
        self.rows = np.asarray(fam_index, dtype=np.int64)
        if np.any(np.diff(self.rows) <= 0) or self.rows.min() < 0 or self.rows.max() >= self.n_fam:
            raise ValueError("fam_index must be strictly increasing (.fam order) and inside the .fam")
        self.n_rows = int(len(self.rows))
        self._byte = self.rows // 4
        self._shift = ((self.rows % 4) * 2).astype(np.uint8)

    def read(self, a: int, b: int) -> np.ndarray:
        if not (0 <= a < b <= self.n_snps):
            raise IndexError(f"SNP range {a}:{b} outside 0:{self.n_snps}")
        self.fh.seek(3 + a * self.bpr)
        buf = np.fromfile(self.fh, dtype=np.uint8, count=(b - a) * self.bpr).reshape(b - a, self.bpr)
        raw = buf[:, self._byte]                                              # (m, n) uint8
        two = (raw >> self._shift[None, :]) & np.uint8(3)
        return np.ascontiguousarray(_LUT[two].T)                              # (n, m) int8

    def read_snps(self, idx) -> np.ndarray:
        """int8 (n_rows, len(idx)) for arbitrary SNP indices (one seek per SNP; returned in the given order)."""
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size and (idx.min() < 0 or idx.max() >= self.n_snps):
            raise IndexError("SNP index outside the .bim")
        out = np.empty((self.n_rows, idx.size), dtype=np.int8)
        for c, j in enumerate(idx):
            self.fh.seek(3 + int(j) * self.bpr)
            raw = np.fromfile(self.fh, dtype=np.uint8, count=self.bpr)[self._byte]
            out[:, c] = _LUT[(raw >> self._shift) & np.uint8(3)]
        return out
