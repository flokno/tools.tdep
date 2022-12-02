#! /usr/bin/env python3
from pathlib import Path

import typer
from rich import print as echo


app = typer.Typer()


@app.command()
def main(
    prefix: str = "iter.",
    folder_name: str = "samples",
    folder_name_new: str = "samples_prev",
):
    """ln the sample folder from the previous iteration"""
    cwd = Path().absolute()
    echo(f"We are in {cwd.name}")
    iter = int(cwd.parts[-1].lstrip(prefix))
    echo(f"..      current iteration: {iter}")

    if iter < 2:
        echo("** current iteration < 2")
        echo("** do nothing.")
        return

    folder_prev = next(Path("..").glob(f"{prefix}*{iter-1}")) / folder_name
    echo(f".. previous sample folder: {folder_prev}")

    if not folder_prev.exists():
        echo(f"** {folder_prev} does NOT exist, do NOT link")
        return 1

    folder_new = Path(folder_name_new)
    echo(f"..                link to: {folder_new}")
    if folder_new.exists():
        echo(f"** ./{folder_new} already exists, do nothing.")
    else:
        folder_new.symlink_to(folder_prev, target_is_directory=True)


if __name__ == "__main__":
    app()
