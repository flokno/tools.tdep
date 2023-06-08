#! /usr/bin/env python3

from pathlib import Path
from typing import List

import typer
from ase.io import read
from ase.stress import voigt_6_to_full_3x3_stress
from rich import print as echo
import numpy as np


def complete_strain(strain: list):
    if len(strain) == 1:
        return strain[0] * np.eye(3)
    elif len(strain) == 3:
        return np.diag(strain)
    elif len(strain) == 6:
        return voigt_6_to_full_3x3_stress(strain)
    elif len(strain) == 9:
        return np.array(strain).reshape(3, 3)
    else:
        raise ValueError(f"Can only understand 1, 3, 6, or 9 values, not {len(strain)}")


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    file: Path,
    strain: List[float],
    scale_atoms: bool = True,
    format: str = None,
    outfile: str = None,
):
    """Apply strain to sample.

    IMPORTANT! For negative strains, use e.g. `ase_strain_sample OPTIONS -- FILE SRAIN`

    Note the `--` to separate cli options from arguments
    """
    echo(f"Read `{file}` and apply strain {strain}:")
    echo(f"... scale atoms: {scale_atoms}")

    if "geometry.in" in file.name:
        format = "aims"
        echo(f"... autodetect format `{format}` for {file}")
    elif "poscar" in file.name.lower():
        format = "vasp"
        echo(f"... autodetect format `{format}` for {file}")

    atoms = read(file, format=format)

    _strain = complete_strain(strain)
    echo("... completed strain:")
    echo(_strain)

    if np.linalg.norm(_strain - _strain.T) > 1e-12:
        raise ValueError("STRAIN IS NOT SYMMETRIC. SICK JOB!")

    deformation = _strain + np.eye(3)
    echo("--> deformation:")
    echo(deformation)

    echo("... cell before deformation:")
    echo(np.asarray(atoms.cell))

    new_cell = deformation @ atoms.cell
    echo("... cell after deformation:")
    echo(new_cell)

    atoms.set_cell(new_cell, scale_atoms=scale_atoms)

    if outfile is None:
        outfile = file.name + ".strained"

    echo(f"... write sample to {outfile}")
    atoms.write(outfile, format=format)


if __name__ == "__main__":
    app()
