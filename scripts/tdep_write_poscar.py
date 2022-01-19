#! /usr/bin/env python3

import typer
from ase.io import read

_infile_ucposcar = "infile.ucposcar"
_infile_ssposcar = "infile.ssposcar"

write_settings = {"format": "vasp", "direct": True, "vasp5": True}


def _convert_file(file: str, outfile: str, format: str = "aims"):
    """Convert geometry input file"""
    typer.echo(f".. read {file}")

    atoms = read(file, format=format)
    atoms.write(outfile, **write_settings)

    typer.echo(f"--> written to {outfile}")


app = typer.Typer()


@app.command()
def main(
    primitive: str = typer.Option(..., "--primitive", "-pc"),
    supercell: str = typer.Option(..., "--supercell", "-sc"),
    format: str = "aims",
):
    """Convert primitive and supercell to tdep/vasp format"""
    typer.echo("Convert structure infiles")

    if primitive:
        _convert_file(primitive, outfile=_infile_ucposcar, format=format)
    if supercell:
        _convert_file(supercell, outfile=_infile_ssposcar, format=format)


if __name__ == "__main__":
    app()
