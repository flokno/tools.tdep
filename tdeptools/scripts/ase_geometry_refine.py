#! /usr/bin/env python3
from pathlib import Path

import spglib as spg
import typer
from ase import Atoms
from ase.io import read
from rich import print as echo

from .ase_geometry_info import default_symprec, inform, to_spglib_cell

_url_std = "https://spglib.readthedocs.io/en/latest/definition.html#spglib-conventions-of-standardized-unit-cell"


def cell_to_Atoms(lattice, scaled_positions, numbers):
    """convert from spglib cell to Atoms"""
    atoms_dict = {
        "cell": lattice,
        "scaled_positions": scaled_positions,
        "numbers": numbers,
        "pbc": True,
    }

    return Atoms(**atoms_dict)


def refine_cell(atoms, symprec=default_symprec):
    """refine the structure"""
    lattice, scaled_positions, numbers = spg.refine_cell(to_spglib_cell(atoms), symprec)

    return cell_to_Atoms(lattice, scaled_positions, numbers)


def standardize_cell(
    atoms, to_primitive=False, no_idealize=False, symprec=default_symprec
):
    """wrap spglib.standardize_cell"""

    cell = to_spglib_cell(atoms)
    args = spg.standardize_cell(cell, to_primitive, no_idealize, symprec)

    return cell_to_Atoms(*args)


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    file: Path,
    format: str = None,
    symprec: float = typer.Option(default_symprec, "-t", "--symprec"),
    primitive: bool = typer.Option(False, "-prim", "--primitive"),
    conventional: bool = typer.Option(False, "-conv", "--conventional"),
):
    """Refine structure in FILE using spglib

    More information on spglib: https://spglib.readthedocs.io/en/latest/
    """
    echo(f"Read `{file}`")

    if "geometry.in" in file.name:
        format = "aims"
        echo(f"... autodetect format `{format}` for {file}")
    elif "poscar" in file.name.lower():
        format = "vasp"
        echo(f"... autodetect format `{format}` for {file}")

    atoms = read(file, format=format)

    echo()
    echo("REFINE SYMMETRY w/ SPGLIB")
    echo(f"... symprec: {symprec}")

    outfile = str(file)
    if primitive:
        echo("... find primitive unitcell")
        atoms = standardize_cell(atoms, to_primitive=True, symprec=symprec)
        outfile += ".primitive"
    elif conventional:
        echo("... find conventional unitcell")
        atoms = standardize_cell(atoms, to_primitive=False, symprec=symprec)
        outfile += ".conventional"
    else:
        echo(f"... find standardized unitcell {_url_std}")
        atoms = refine_cell(atoms, symprec=symprec)
        outfile += ".standardized"

    echo()
    echo("GEOMETRY INFORMATION FOR RESULTING STRUCTURE")
    inform(atoms, symprec=symprec)

    echo()
    echo(f"... save new structure to '{outfile}'")
    atoms.write(outfile, format=format)


if __name__ == "__main__":
    app()
