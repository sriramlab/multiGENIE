"""atomsplit -- many complementary sample splits from one genotype pass.

Partition the analysed individuals once into G family-intact atoms; every half is a
union of G/2 atoms.  The randomized-trace normal equations of ANY half, and of all its
delete-one-SNP-block jackknife replicates, are then assembled from small Gram tensors
computed in two passes over the genotypes (`stream.atom_pass`), and handed to the
existing `multi_gxe` selection code as ordinary `Export` objects (`tensor.AtomTensor.
assemble`).  See `thoughts/sep_28/atom_splits_plan.md` for the derivation and the
Step 0 engine facts every convention here mirrors.

    model.Design        the fixed (full-sample) preprocessing: W, V, y~, atoms, kernels
    bedio               the engine's BED decoding, standardisation and HWE imputation hash
    stream.atom_pass    Stage B: the two genotype passes -> AtomTensor
    tensor.AtomTensor   Stage C: assemble(atoms) -> multi_gxe Export
    dense               the N x N reference implementation (validation V1)
    atoms               Stage A: family-intact, PC-balanced atoms refining a fold file
    pairs               Stage D: all C(G, G/2)/2 pairs through the paper's rule
    toy                 simulated data for V1/V2
"""
from .model import Design, kernel_names
from .tensor import AtomTensor
from .stream import atom_pass, jk_bounds
from . import bedio, dense, atoms, toy

__all__ = ["Design", "kernel_names", "AtomTensor", "atom_pass", "jk_bounds",
           "bedio", "dense", "atoms", "toy"]
