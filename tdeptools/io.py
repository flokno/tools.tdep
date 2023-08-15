#! /usr/bin/env python
from collections import namedtuple
from pathlib import Path

import h5py as h5
import numpy as np
import rich
from ase.io import read as read_ase
from rich import print as echo

from tdeptools.keys import keys

_cls_data = namedtuple(
    "FC_block_data",
    ["index_neighbor", "index_neighbor_unit", "lattice_vector", "force_constant"],
)
_cls_block = namedtuple("FC_block", ["atom_index", "n_neighbors", "data"])
_cls_fc = namedtuple("FC", ["n_atoms", "cutoff", "blocks"])

outfile_meta = "infile.meta"
outfile_stat = "infile.stat"
outfile_forces = "infile.forces"
outfile_positions = "infile.positions"
outfile_born_charges = "infile.born_charges"
outfile_dielectric_tensor = "infile.dielectric_tensor"


def _parse_forceconstant_one_block(f):
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

            block = _parse_forceconstant_one_block(f)
            blocks.append(block)

    return _cls_fc(n_atoms, cutoff, blocks)


def write_infiles(
    rows: list,
    timestep: float = 1.0,
    outfile_forces: str = outfile_forces,
    outfile_positions: str = outfile_positions,
    outfile_stat: str = outfile_stat,
    outfile_born_charges: str = outfile_born_charges,
    outfile_dielectric_tensor: str = outfile_dielectric_tensor,
):
    """write the normal input files (positions, forces, statistics)

    Args:
        rows: list with Atoms objects holding the calculation results
        timestep: simulation timestep in fs
    """
    echo("... write forces, positions, and statistics")
    with open(outfile_forces, "w") as ff, open(outfile_positions, "w") as fp, open(
        outfile_stat, "w"
    ) as fs:
        for ii, row in enumerate(rows):

            for (pos, force) in zip(row[keys.positions], row[keys.forces]):
                (px, py, pz) = pos
                (fx, fy, fz) = force
                fp.write(f"{px:23.15e} {py:23.15e} {pz:23.15e}\n")
                ff.write(f"{fx:23.15e} {fy:23.15e} {fz:23.15e}\n")

            # shorthands
            dt = timestep
            et = row[keys.energy_total]
            ep = row[keys.energy_potential]
            ek = row[keys.energy_kinetic]
            t, p, s = row[keys.temperature], row[keys.pressure], row[keys.stress]
            assert len(s) == 6, len(s)
            fs.write(f"{ii+1:7d} {ii*dt:9.3f} {et:23.15e} {ep:23.15e} {ek:23.15e} ")
            fmt = "15.9f"
            fs.write(f"{t:{fmt}} {p:{fmt}} ")
            fs.write(" ".join(f"{x:{fmt}}" for x in s))
            fs.write("\n")

    echo(f"... forces written to {outfile_forces}")
    echo(f"... positions written to {outfile_positions}")
    echo(f"... statistics written to {outfile_stat}")

    # dielectric data?
    if row.get(keys.dielectric_tensor) is not None:
        echo("... dielectric tensor found")
        with open(outfile_dielectric_tensor, "w") as f:
            for row in rows:
                eps = row[keys.dielectric_tensor]
                for vec in eps:
                    f.write(" ".join(f"{x:23.15e}" for x in vec) + "\n")
        echo(f"... dielectric tensors written to {outfile_dielectric_tensor}")

        # then we should also write born charges:
        mock_bec = -np.ones([len(row[keys.positions]), 3, 3])
        with open(outfile_born_charges, "w") as f:
            for row in rows:
                if row.get(keys.born_charges) is None:
                    mock = True
                    bec = mock_bec
                else:
                    mock = False
                    bec = row.get(keys.born_charges)
                for vec in bec.reshape(-1, 3):
                    f.write(" ".join(f"{x:23.15e}" for x in vec) + "\n")
        if mock:
            echo(f"*** mock born charges written to {outfile_born_charges}")
        else:
            echo(f"... born charges written to {outfile_born_charges}")


def write_meta(
    n_atoms: int,
    n_samples: int,
    dt: float = 1.0,
    temperature: float = -314.15,
    outfile: str = outfile_meta,
):
    """write TDEP simulation metadata

    Args:
        n_atoms: no. of atoms
        n_samples: no. of samples
        dt: time step in fs
        temperature: simulation temperature
        outfile: the output file to write to, default = infile.meta
    """
    with open(outfile, "w") as f:
        f.write(f"{n_atoms:10}     # N atoms\n")
        f.write(f"{n_samples:10}     # N timesteps\n")
        f.write(f"{dt:10}     # timestep in fs (currently not used )\n")
        f.write(f"{temperature:10}     # temperature in K (only for free energy)\n")
    echo(f"... meta info written to {outfile_meta}")
