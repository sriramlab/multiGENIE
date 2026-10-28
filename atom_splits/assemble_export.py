#!/usr/bin/env python3
"""Stage C on the command line: write the normal_eq.json of one union of atoms, in
GENIE's schema, so any existing multi_gxe script can read it.

    python assemble_export.py --tensor bmi.atoms.G12.npz --atoms 0,1,2,3,4,5 -o bmi.atomsA.normal_eq.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atomsplit.tensor import AtomTensor                                   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tensor", required=True)
ap.add_argument("--atoms", required=True, help="comma-separated atom indices")
ap.add_argument("-o", "--out", required=True)
a = ap.parse_args()
t = AtomTensor.load(a.tensor)
atoms = [int(x) for x in a.atoms.split(",")]
print(t.export_json(atoms, a.out), f"(N_h = {int(t.atom_sizes[atoms].sum()):,})")
