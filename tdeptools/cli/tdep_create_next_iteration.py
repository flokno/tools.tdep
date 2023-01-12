#! /usr/bin/env python3

import shutil
import subprocess as sp
from pathlib import Path

import typer
from rich import print as echo

from .tdep_create_sample_folders import main as create_sample_folders


_cmd = "canonical_configuration"
_infiles = (
    "infile.ucposcar",
    "infile.ssposcar",
    "infile.forceconstant",
    "infile.lotosplitting",
)


def _create_samples(
    temperature: float,
    nsamples: int,
    mf: float,
    quantum: bool,
    folder_new: Path,
    of: int = 4,
):
    cmd = f"{_cmd} -t {temperature} -of 4 -n {nsamples}"

    if mf is not None:
        cmd += " -mf {mf}"

    if quantum:
        cmd += " --quantum"

    ps = sp.run(cmd.split(), cwd=folder_new, capture_output=True)
    stdout = ps.stdout + ps.stderr
    with open(folder_new / (_cmd + ".log"), "wb") as f:
        f.write(cmd.encode())
        f.write("\n".encode())
        f.write(stdout)


app = typer.Typer()


@app.command()
def main(
    prefix: str = "iter.",
    temperature: float = typer.Option(None, "--temperature", "-T"),
    mf: float = None,
    of: int = 4,
    quantum: bool = True,
    force: bool = False,
    makefile: str = "Makefile",
    control: str = "control.in",
    create_sample_folder: bool = True,
):
    """..."""
    cwd = Path().absolute()
    echo(cwd)
    iter = int(cwd.parts[-1].lstrip(prefix))
    echo(f"..  current iteration: {iter}")

    folder_new = Path(f"{prefix}{iter+1:03d}")
    echo(f"..         new folder: {folder_new}")

    nsamples = 2 ** (iter + 2)
    echo(f".. new no. of samples: {nsamples}")

    echo(f"..        temperature: {temperature}")
    if temperature is None:
        raise typer.Exit("STOP: Temperature must be given.")

    folder_new.mkdir(exist_ok=force)

    for file in _infiles:
        if Path(file).exists():
            shutil.copy(file, folder_new)

    _create_samples(
        temperature=temperature,
        nsamples=nsamples,
        mf=mf,
        quantum=quantum,
        folder_new=folder_new,
        of=of,
    )

    # copy files
    echo(".. copy Makefile and control")
    shutil.copy(makefile, folder_new)
    shutil.copy(control, folder_new)

    if create_sample_folder:
        files = sorted(folder_new.glob("aims_conf*"))
        create_sample_folders(
            files=files,
            base_folder=folder_new / "samples",
            control=control,
            force=force,
        )


if __name__ == "__main__":
    app()
