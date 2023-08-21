#! /usr/bin/env python3

import shutil
from pathlib import Path
from typing import List

import typer
from ase.io import read
from rich import print as echo

app = typer.Typer()


@app.command()
def main(
    files: List[Path],
    base_folder: str = "samples",
    folder: str = "sample",
    outfile: str = "geometry.in",
    files_control: List[Path] = None,
    force: bool = False,
    format: str = "aims",
    format_in: str = "vasp",
):
    echo(files)

    echo(f"Store {len(files)} file(s) to folders")

    for ii, file in enumerate(files):
        fol = Path(base_folder) / (folder + f".{ii+1:05d}")
        fol.mkdir(exist_ok=force, parents=True)

        # echo(f".. move file {ii+1:3d}: {str(file)} to {fol / outfile}")
        _a, _b, _c, _d = str(file), format_in, format, str(fol / outfile)
        echo(f".. convert {_a} from `{_b}` to `{_c}`, move to {_d}")

        atoms = read(file, format=format_in)
        atoms.write(fol / outfile, format=format)

        # move original file as well for reference
        shutil.move(str(file), str(fol))

        if files_control is not None:
            for file in files_control:
                echo(f".. copy {file} to {fol}")
                shutil.copy(file, fol)

    echo("done.")


if __name__ == "__main__":
    app()
