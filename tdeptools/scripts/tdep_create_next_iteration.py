#! /usr/bin/env python3

import shutil
import subprocess as sp
from pathlib import Path
from typing import List

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

_refpos_dict = {
    "infile.ucposcar": "outfile.ucposcar.new_refpos",
    "infile.ssposcar": "outfile.ssposcar.new_refpos",
}


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


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    prefix: str = "iter.",
    temperature: float = typer.Option(None, "--temperature", "-T"),
    mf: float = None,
    of: int = 4,
    quantum: bool = True,
    max_samples: int = 512,
    force: bool = False,
    makefile: str = "Makefile",
    files_control: List[Path]= None,
    refpos: bool = typer.Option(False, help="Use updated reference positions."),
    create_sample_folder: bool = True,
):
    """..."""
    cwd = Path().absolute()
    echo(cwd)
    iter = int(cwd.parts[-1].lstrip(prefix))
    echo(f"..  current iteration: {iter}")

    folder_new = Path(f"{prefix}{iter+1:03d}")
    echo(f"..         new folder: {folder_new}")

    nsamples = min(2 ** (iter + 2), max_samples)
    echo(f".. new no. of samples: {nsamples} (max.: {max_samples})")

    echo(f"..        temperature: {temperature}")
    if temperature is None:
        raise typer.Exit("STOP: Temperature must be given.")

    folder_new.mkdir(exist_ok=force)

    for file in _infiles:
        if Path(file).exists():
            if refpos:
                shutil.copy(_refpos_dict.get(file, file), folder_new / file)
            else:
                shutil.copy(file, folder_new / file)

    _create_samples(
        temperature=temperature,
        nsamples=nsamples,
        mf=mf,
        quantum=quantum,
        folder_new=folder_new,
        of=of,
    )

    # copy files
    echo(".. copy Makefile")
    shutil.copy(makefile, folder_new)
    if files_control is not None:
        echo(f'.. copy control files: {files_control}')
        for file in files_control:
            shutil.copy(file, folder_new)

    if create_sample_folder:
        files = sorted(folder_new.glob("aims_conf*"))
        create_sample_folders(
            files=files,
            base_folder=folder_new / "samples",
            files_control=files_control,
            force=force,
        )


if __name__ == "__main__":
    app()
