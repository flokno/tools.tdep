#! /usr/bin/env python3

from pathlib import Path
from typing import List

import numpy as np
import typer
from ase import Atoms, units
from ase.calculators.singlepoint import PropertyNotImplementedError
from ase.io import read
from rich import print as echo

from tdeptools.keys import keys
from tdeptools.io import write_infiles, write_meta


app = typer.Typer()


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
