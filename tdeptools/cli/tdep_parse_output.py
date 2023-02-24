#! /usr/bin/env python3

from pathlib import Path
from typing import List
from collections import namedtuple

import numpy as np
from rich import print as echo
import typer
from ase import units, Atoms
from ase.calculators.singlepoint import PropertyNotImplementedError
from ase.io import read


outfile_meta = "infile.meta"
outfile_stat = "infile.stat"
outfile_forces = "infile.forces"
outfile_positions = "infile.positions"
outfile_born_charges = "infile.born_charges"
outfile_dielectric_tensor = "infile.dielectric_tensor"

# keys, maybe move to a module
_keys = [
    "positions",
    "forces",
    "energy_total",
    "energy_kinetic",
    "energy_potential",
    "temperature",
    "stress",
    "pressure",
    "dielectric_tensor",
    "born_charges",
]
_dct = {key: key for key in _keys}
keys = namedtuple("keys", _dct.keys())(**_dct)


app = typer.Typer()


def cleanup():
    """Remove outfiles"""
    ...


def extract_results(atoms: Atoms, ignore_forces: bool = False) -> dict:
    """get the results and return a dictionary with normalized keys"""
    try:
        forces = atoms.get_forces()
    except PropertyNotImplementedError:
        if ignore_forces:
            echo("*** forces not found, set to -1")
            forces = -np.ones_like(atoms.positions)
        else:
            raise RuntimeError("*** FORCES NOT FOUND. Check or use `--ignore-forces`")

    row = {
        keys.positions: atoms.get_scaled_positions(),
        keys.forces: forces,
        keys.energy_total: atoms.get_kinetic_energy() + atoms.get_potential_energy(),
        keys.energy_kinetic: atoms.get_kinetic_energy(),
        keys.energy_potential: atoms.get_potential_energy(),
        keys.temperature: atoms.get_temperature(),
    }

    results = atoms.calc.results
    if "stress" in results:
        stress = atoms.get_stress() / units.GPa
    else:
        stress = np.zeros(6)
    pressure = np.mean(stress[:3])

    row.update({keys.stress: stress, keys.pressure: pressure})

    # dielectric data
    update = {
        keys.born_charges: results.get("born_effective_charges"),
        keys.dielectric_tensor: results.get("dielectric_tensor"),
    }
    row.update(update)

    return row


def write_infiles(rows: list, timestep: float = 1.0):
    """write the normal input files (positions, forces, statistics)"""
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


def write_meta(n_atoms: int, n_samples: int, dt: float = 1.0, file: str = outfile_meta):
    """write simulation metadata"""
    with open(file, "w") as f:
        f.write("{:10}     # N atoms\n".format(n_atoms))
        f.write("{:10}     # N timesteps\n".format(n_samples))
        f.write("{:10}     # timestep in fs (currently not used )\n".format(dt))
        f.write("{:10}     # temperature in K (currently not used)\n".format(-314))
    echo(f"... meta info written to {outfile_meta}")


@app.command()
def main(
    files: List[Path],
    timestep: float = 1.0,
    ignore_forces: bool = False,
    format: str = "aims-output",
):
    """Parse DFT force/stress calculations via ase.io.read"""
    echo(f"Parse {len(files)} file(s)")

    echo(f"... empty forces will be ignored: {ignore_forces}")

    # read data from files
    rows = []
    for ii, file in enumerate(files):
        echo(f"... parse file {ii+1:3d}: {str(file)}")

        try:
            atoms_list = read(file, ":", format=format)
        except (ValueError, IndexError):
            echo(f"*** problem in file {file}, SKIP.")
            continue

        for atoms in atoms_list:
            n_atoms = len(atoms)
            rows.append(extract_results(atoms, ignore_forces=ignore_forces))

    n_samples = len(rows)
    echo(f"... found {n_samples} samples")

    if n_samples < 1:
        echo("... no data found, abort.")
        return

    # write stuff
    write_infiles(rows, timestep=timestep)
    write_meta(n_atoms=n_atoms, n_samples=n_samples, dt=timestep)


if __name__ == "__main__":
    app()
