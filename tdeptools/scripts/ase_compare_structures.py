#! /usr/bin/env python3

from pathlib import Path

import typer
from ase.io import read
from rich import print as echo
import numpy as np


def guess_format(file: Path):
    if "geometry.in" in file.name:
        return "aims"
    elif "poscar" in file.name.lower():
        return "vasp"
    else:
        return None


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    file1: Path,
    file2: Path,
    format: str = None,
):
    """Compare two structures, find supercell matrices etc."""
    echo(f"Match structures in {file1} and {file2}")
    echo(f"... read '{file1}'")
    atoms1 = read(file1, format=guess_format(file1))
    echo(atoms1)
    echo(f"... read '{file2}'")
    atoms2 = read(file2, format=guess_format(file2))
    echo(atoms2)

    echo("MATCH LATTICES")
    echo()

    cell1 = np.array(atoms1.cell)
    cell2 = np.array(atoms2.cell)
    smatrix = np.linalg.inv(cell1).T @ cell2
    echo("... supercell matrix:")
    echo(smatrix)

    echo("... round to integers:")
    echo(smatrix.round().astype(int))

    echo("... for TDEP `generate_structure`")
    echo(" ".join([f"{m:d}" for m in smatrix.round().astype(int).T.flatten()]))


if __name__ == "__main__":
    app()
