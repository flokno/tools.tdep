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
    scale_atoms: bool = True,
    tolerance_volume: float = 0.1,
    format: str = None,
    outfile: str = None,
):
    """Match lattice vectors of structure in FILE2 to that in FILE1."""
    echo(f"Match lattices of structures in {file1} and {file2}")
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
    echo()

    volume_change = np.linalg.det(smatrix - np.eye(3))

    echo(f"... volume change: {volume_change}")

    if abs(volume_change) > tolerance_volume:
        raise ValueError("DETERMINANT OF cell changes by more than tolerance!")

    atoms2.set_cell(atoms1.cell, scale_atoms=scale_atoms)

    if outfile is None:
        outfile = file2.name + ".matched"

    echo(f'... write to "{outfile}"')
    atoms2.write(outfile, format=format)


if __name__ == "__main__":
    app()
