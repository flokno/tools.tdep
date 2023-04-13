#! /usr/bin/env python3

from pathlib import Path

import typer
from ase.calculators.calculator import kptdensity2monkhorstpack
from ase.io import read
from rich import print as echo

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

    grid = kptdensity2monkhorstpack(atoms=atoms, kptdensity=density, even=even)

    echo(f"... structure:       {atoms}")
    echo(f"... number of atoms: {len(atoms)}")
    echo(f"... cell:")
    echo(f"{atoms.cell[:]}")
    echo(f"... density:         {density} per 1/Ang")
    echo(f"... only even grids: {even}")
    echo(f"--> grid: {grid}")


if __name__ == "__main__":
    app()
