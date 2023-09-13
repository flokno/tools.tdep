#! /usr/bin/env python3
from collections import namedtuple
from pathlib import Path

import numpy as np
import spglib as spg
import typer
from ase.io import read
from rich import print as echo

from tdeptools.geometry import (
    bounding_sphere_of_box,
    get_cubicness,
    inscribed_sphere_in_box,
)

default_symprec = 1e-10


def to_spglib_cell(atoms):
    """Convert to spglib representation"""
    lattice = atoms.cell
    positions = atoms.get_scaled_positions()
    number = atoms.get_atomic_numbers()
    return (lattice, positions, number)


def get_symmetry_dataset(atoms, symprec=default_symprec):
    """Get spglib symmetry dataset"""

    dataset = spg.get_symmetry_dataset(to_spglib_cell(atoms), symprec=symprec)

    uwcks, count = np.unique(dataset["wyckoffs"], return_counts=True)
    dataset["wyckoffs_unique"] = [(w, c) for (w, c) in zip(uwcks, count)]

    ats, count = np.unique(dataset["equivalent_atoms"], return_counts=True)
    dataset["equivalent_atoms_unique"] = [(a, c) for (a, c) in zip(ats, count)]

    return namedtuple("symmetry_dataset", dataset.keys())(**dataset)


def inform(atoms, symprec=default_symprec, max_atoms=20, verbose=False):
    """print geometry information to screen"""
    unique_symbols, multiplicity = np.unique(atoms.symbols, return_counts=True)
    # Structure info:
    echo("Geometry info")
    echo(f"  input geometry:    {atoms}")
    echo(f"  Symmetry prec.:    {symprec}")
    echo(f"  Number of atoms:   {len(atoms)}")

    msg = ", ".join([f"{s} ({m})" for (s, m) in zip(unique_symbols, multiplicity)])
    echo(f"  Species:           {msg}")
    echo(f"  Periodicity:       {atoms.pbc}")
    echo("  Positions:            x (Å)            y (Å)           z (Å) ")
    for ii, (sym, pos) in enumerate(zip(atoms.get_chemical_symbols(), atoms.positions)):
        x, y, z = pos.round(decimals=6)
        rep = f"{x:15.6f}  {y:15.6f} {z:15.6f}"
        sym = f"'{sym}'"
        echo(f"    {ii:5d}, {sym:4s}: {rep}")
        if ii > max_atoms - 2 and not verbose:
            echo(f"  ... {len(atoms) - 20} more positions (show with --verbose)")
            break

    if any(atoms.pbc):
        echo("  Lattice (Å):  ")
        echo("      " + np.array2string(atoms.cell, prefix="......"))
        cub = get_cubicness(atoms.cell)
        echo(f"  Cubicness:           {cub**3:.3f}")
        r = inscribed_sphere_in_box(atoms.cell)
        echo(f"  Largest Cutoff:      {r:.3f} Å")
        r = bounding_sphere_of_box(atoms.cell)
        echo(f"  Bounding box rad.:   {r/2:.3f} Å")
        d = atoms.get_all_distances()
        echo(f"  Nearest neighbor:    {d[d > 0].min():.3f} Å")

        la = atoms.cell.cellpar()
        echo("\nCell lengths and angles:")
        echo("  a, b, c (Å): {}".format(" ".join([f"{_l:11.4f}" for _l in la[:3]])))
        angles = "  \u03B1, \u03B2, \u03B3 (°): "
        values = "{}".format(" ".join([f"{_l:11.4f}" for _l in la[3:]]))
        echo(angles + values)
        echo(f"  Volume:           {atoms.get_volume():12.3f} Å**3")
        echo(f"  Volume per atom:  {atoms.get_volume() / len(atoms):12.3f} Å**3")

        if verbose:
            echo()
            echo("Special k points:")
            for key, val in atoms.cell.bandpath().special_points.items():
                echo(f"    {key}: {val}")

    if symprec is not None:
        echo()
        echo("Report symmetry information from spglib:")
        sds = get_symmetry_dataset(atoms, symprec=symprec)

        echo(f"  Spacegroup:          {sds.international} ({sds.number})")
        if sds.number > 1:
            msg = "  Wyckoff positions:   "
            echo(msg + ", ".join(f"{c}*{w}" for (w, c) in sds.wyckoffs_unique))
            msg = "  Equivalent atoms:    "
            echo(msg + ", ".join(f"{c}*{a}" for (a, c) in sds.equivalent_atoms_unique))

        if verbose:
            echo("  Standard lattice:  ")
            for vec in sds.std_lattice:
                echo(f"    {vec}")

    # Info
    for (key, val) in atoms.info.items():
        echo(f"  {key:10s}: {val}")


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    file: Path,
    format: str = None,
    symprec: float = typer.Option(default_symprec, "-t", "--symprec"),
    max_atoms: int = typer.Option(20, help="print positions for max. this many atoms"),
    verbose: bool = False,
):
    """Report information about geometry in FILE"""
    echo(f"Read '{file}'")

    if "geometry.in" in file.name:
        format = "aims"
        echo(f"... autodetect format '{format}' for '{file}'")
    elif "poscar" in file.name.lower():
        format = "vasp"
        echo(f"... autodetect format '{format}' for '{file}'")

    atoms = read(file, format=format)

    inform(atoms, symprec=symprec, max_atoms=max_atoms, verbose=verbose)


if __name__ == "__main__":
    app()
