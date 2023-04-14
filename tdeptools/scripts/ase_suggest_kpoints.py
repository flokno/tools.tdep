#! /usr/bin/env python3

from pathlib import Path

import numpy as np
import typer
from ase.calculators.calculator import kptdensity2monkhorstpack as density2k
from ase.io import read
from rich import print as echo


def k2densities(atoms, k_grid):
    """Generate the kpoint density per direction from given k_grid."""
    recipcell = atoms.cell.reciprocal()
    densities = np.asarray(k_grid) / (2 * np.pi * np.sqrt((recipcell ** 2).sum(axis=1)))
    return densities


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    file: Path,
    density: float = typer.Option(3.5, "-d", help="kpoint density per Angstrom"),
    even: bool = typer.Option(False, help="round up to even grids"),
    format: str = None,
):
    """Suggest k-grid based on a density"""
    echo(f"Parse {file}")

    # help guessing the file format
    if format is None and "geometry.in" in file.name:
        format = "aims"
        echo(f"... this could be an {format} file")
    elif format is None and "infile." in file.name:
        format = "vasp"
        echo(f"... this could be an {format} file")

    atoms = read(file, format=format)

    grid = density2k(atoms=atoms, kptdensity=density, even=even)
    densities = k2densities(atoms=atoms, k_grid=grid)

    echo(f"... structure:          {atoms}")
    echo(f"... number of atoms:    {len(atoms)}")
    echo("... cell:")
    echo(atoms.cell[:])
    echo(f"... target density:     {density} per 1/Ang")
    echo(f"... only even grids:    {even}")
    echo(f"--> grid:               {grid}")
    echo(f"--> resulting densites: {densities}")
    echo(f"--> resulting density:  {densities.mean():.3f}")


if __name__ == "__main__":
    app()
