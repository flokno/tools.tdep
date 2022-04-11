#! /usr/bin/env python3

import shutil
from pathlib import Path
from typing import List

import typer

app = typer.Typer()


@app.command()
def main(
    files: List[Path],
    base_folder: str = "samples",
    folder: str = "sample",
    outfile: str = "geometry.in",
    control: Path = None,
):
    typer.echo(files)

    typer.echo(f"Store {len(files)} file(s) to folders")

    for ii, file in enumerate(files):
        fol = Path(base_folder) / (folder + f".{ii+1:05d}")
        fol.mkdir(exist_ok=True, parents=True)

        typer.echo(f".. move file {ii+1:3d}: {str(file)} to {fol / outfile}")

        shutil.move(file, fol / outfile)

        if control is not None:
            typer.echo(f".. move {control} to {fol}")
            shutil.copy(control, fol)

    typer.echo("done.")


if __name__ == "__main__":
    app()
