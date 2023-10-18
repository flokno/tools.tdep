#! /usr/bin/env python3

from pathlib import Path
from typing import List

import numpy as np
import typer
from ase import Atoms, units
from ase.calculators.singlepoint import PropertyNotImplementedError
from ase.io import read
from rich import print as echo

from tdeptools.io import write_infiles, write_meta
from tdeptools.keys import keys

app = typer.Typer(pretty_exceptions_show_locals=False)


def extract_results(atoms: Atoms, ignore_forces: bool = False) -> dict:
    """get the results and return a dictionary with normalized keys"""
    try:
        forces = atoms.get_forces()
    except PropertyNotImplementedError:
        if ignore_forces:
            echo("*** forces not found, will not write")
            forces = None
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
        echo("*** stress not found, set to 0")
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
    temperature: float = None,
    discard_start: int = 0,
    ignore_forces: bool = False,
    format: str = None,
):
    """Parse DFT force/stress calculations via ase.io.read"""
    echo(f"Parse {len(files)} file(s)")

    echo(f"... empty forces will be ignored: {ignore_forces}")

    if temperature is None:
        echo("*** SIMULATION TEMPERATURE IS NOT GIVEN")
        echo("--> set to -314.15K to remind you")
        temperature = -314.15
    else:
        echo(f"... temperature:                  {temperature} K")

    # read data from files
    rows = []
    for ii, file in enumerate(files):
        echo(f"... parse file {ii+1:3d}: '{str(file)}'")

        try:
            atoms_list = read(file, ":", format=format)
        except (ValueError, IndexError):
            echo(f"*** problem in file {file}, SKIP.")
            continue

        for atoms in atoms_list:
            n_atoms = len(atoms)
            rows.append(extract_results(atoms, ignore_forces=ignore_forces))

    echo(f"... found {len(rows)} samples")

    echo(f"... discard {discard_start} steps at beginning")
    rows = rows[discard_start:]

    if len(rows) < 1:
        echo("... no data found, abort.")
        return

    # write stuff
    write_infiles(rows, timestep=timestep)
    write_meta(
        n_atoms=n_atoms, n_samples=len(rows), dt=timestep, temperature=temperature
    )


if __name__ == "__main__":
    app()
