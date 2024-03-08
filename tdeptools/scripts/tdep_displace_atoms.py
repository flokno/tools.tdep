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
    plusminus: bool = True,
    displacement: float = typer.Option(default_displacement, help="displacement in Å"),
    format: str = "vasp",
):
    """Displace the atoms"""

    echo(f"Read structure from {file}")
    atoms = read(file, format=format)

    if plusminus:
        signs = (1, -1)
    else:
        signs = (1,)

    counter = 0
    for nn in range(len(atoms)):
        for ii in range(3):
            for ss in signs:
                counter += 1
                rep_cart = {0: "x", 1: "y", 2: "z"}[ii]
                rep_sign = {1: "plus", -1: "minus"}[ss]

                watoms = atoms.copy()
                watoms.positions[nn, ii] += ss * displacement

                _file = str(file).lstrip("infile.")
                outfile = (
                    f"outfile.{_file}.displacement.{nn+1:03d}.{rep_cart}.{rep_sign}"
                )
                echo(f"... write geometry {counter:5d} to {outfile}")
                watoms.write(outfile, format=format)


if __name__ == "__main__":
    app()
