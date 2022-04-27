#! /usr/bin/env python
from collections import namedtuple
from pathlib import Path

import h5py as h5
import numpy as np
import rich
from ase.io import read as read_ase

_cls_data = namedtuple(
    "FC_block_data",
    ["index_neighbor", "index_neighbor_unit", "lattice_vector", "force_constant"],
)
_cls_block = namedtuple("FC_block", ["atom_index", "n_neighbors", "data"])
_cls_fc = namedtuple("FC", ["n_atoms", "cutoff", "blocks"])


def get_one_block(f):
    """Read one block of force constants for a given atom in the primitive cell

    Args:
        f: file object

    Returns:
        (index of atom, element number of atom, no. of neighbors, data)
        data = (index of neighbor, index of neighbor in unit cell,
                element no. of neighbor, reciprocal lattice vector,
                force constant matrix for pair)
                for each neighbor
    """
    data = []

    line = f.readline()
    n_neighbors, iatom = [int(line.split()[zz]) for zz in (0, 6)]

    for _ in range(n_neighbors):
        line = f.readline()
        iunit, ineighbor, _ = [int(line.split()[zz]) for zz in (0, 11, 14)]
        LL = np.fromstring(f.readline(), dtype=float, sep=" ")
        MM = []
        for _ in range(3):
            FC = np.fromstring(f.readline(), dtype=float, sep=" ")
            MM.append(FC)
        MM = np.array(MM)

        data.append(_cls_data(ineighbor, iunit, LL, MM))

    return _cls_block(iatom, n_neighbors, data)


def parse_forceconstant(file: str = "outfile.forceconstant"):
    """Parse TDEP forceconstants

    Args:
        file: filename

    Returns:
        (no. atoms in primitive cell, realspace cutoff, blocks)
        blocks = data from `get_one_block` for each atom in primitive cell
    """

    blocks = []

    with open(file) as f:

        n_atoms = int(f.readline().split()[0])
        cutoff = float(f.readline().split()[0])

        for _ in range(n_atoms):

            block = get_one_block(f)
            blocks.append(block)

    return _cls_fc(n_atoms, cutoff, blocks)
