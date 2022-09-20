#! /usr/bin/env python3

import shutil
from pathlib import Path
from typing import List

import typer
from rich import print as echo


app = typer.Typer()


@app.command()
def main(
    files: List[Path],
    base_folder: str = "samples",
    folder: str = "sample",
    outfile: str = "geometry.in",
    control: Path = None,
    force: bool = False,
):
    echo(files)

    echo(f"Store {len(files)} file(s) to folders")

    for ii, file in enumerate(files):
        fol = Path(base_folder) / (folder + f".{ii+1:05d}")
        fol.mkdir(exist_ok=force, parents=True)

        echo(f".. move file {ii+1:3d}: {str(file)} to {fol / outfile}")

        shutil.move(file, fol / outfile)

        if control is not None:
            echo(f".. move {control} to {fol}")
            shutil.copy(control, fol)

    echo("done.")


if __name__ == "__main__":
    app()
