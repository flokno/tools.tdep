#! /usr/bin/env python3
from pathlib import Path

import typer
from ase.io import read
from rich import print as echo


default_displacement: float = 0.01


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    file: Path,
    file_unitcell: Path = None,
    nmax: int = None,
    plusminus: bool = True,
    displacement: float = typer.Option(default_displacement, help="displacement in Å"),
    format: str = "vasp",
):
    """Displace the atoms

    If FILE_SUPERCELL is given, displace only the atoms at that position
    """
    import numpy as np

    echo(f"Read structure from {file}")
    atoms = read(file, format=format)

    if file_unitcell is not None:
        echo("... match to atoms in unit cell")
        import json

        atoms_uc = read(file_unitcell, format=format)

        # which atom in the unit cell is which atom in the supercell?
        watoms = atoms.copy()
        # replace cell with that of unit cell
        watoms.cell = atoms_uc.cell
        list_match = []
        for ii, spos in enumerate(watoms.get_scaled_positions(wrap=False)):
            for _, upos in enumerate(atoms_uc.get_scaled_positions(wrap=False)):
                if np.linalg.norm(spos - upos) < 1e-9:
                    list_match.append(ii)
        assert len(list_match) == len(atoms_uc), len(list_match)

        echo(f"... matches: {list_match}")

        _outfile = "uc_indices.json"
        echo(f"... dump to {_outfile}")
        json.dump(list_match, open(_outfile, "w"))

    else:
        list_match = np.arange(len(atoms))

    if plusminus:
        signs = (1, -1)
    else:
        signs = (1,)

    natoms = len(atoms)

    if nmax is None:
        nmax = natoms

    echo(f"... number of atoms:         {natoms}")
    echo(f"... number of displacements: {nmax}")

    counter = 0
    for nn in list_match[:nmax]:
        for ii in range(3):
            for ss in signs:
                counter += 1
                rep_cart = {0: "x", 1: "y", 2: "z"}[ii]
                rep_sign = {1: "plus", -1: "minus"}[ss]

                watoms = atoms.copy()
                watoms.positions[nn, ii] += ss * displacement

                _file = str(file).lstrip("infile.")
                outfile = (
                    f"outfile.{_file}.displacement.{counter:05d}.{rep_cart}.{rep_sign}"
                )
                echo(f"... write geometry {counter:5d} to {outfile}")
                watoms.write(outfile, format=format)


if __name__ == "__main__":
    app()
