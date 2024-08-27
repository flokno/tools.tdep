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
    "infile.ucposcar.init",
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
    imaginary: bool,
    folder_new: Path,
):
    cmd = f"{_cmd} -t {temperature} -n {nsamples}"

    if mf is not None:
        cmd += " -mf {mf}"

    if quantum:
        cmd += " --quantum"

    if imaginary:
        cmd += " --imaginary"

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
    quantum: bool = True,
    imaginary: bool = False,
    max_samples: int = 512,
    force: bool = False,
    makefile: str = "Makefile",
    refpos: bool = typer.Option(False, help="Use updated reference positions."),
    create_sample_folder: bool = True,
    file_geometry: str = "geometry.in",
    files_control: List[Path] = None,
    files_aux: List[Path] = None,
    predictions: bool = False,
    file_predictions: Path = "predictions.nc",
    format: str = "aims",
):
    """..."""
    cwd = Path().absolute()
    echo(cwd)
    iter = int(cwd.parts[-1].lstrip(prefix))
    echo(f"..  current iteration: {iter}")

    folder_new = Path(f"{prefix}{iter+1:03d}")
    echo(f"..         new folder: {folder_new}")

    nsamples = min(2 ** (iter + 1), max_samples)
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
        imaginary=imaginary,
        folder_new=folder_new,
    )

    # copy files
    echo(".. copy Makefile")
    shutil.copy(makefile, folder_new)
    if files_control is not None:
        echo(f".. copy control files: {files_control}")
        for file in files_control:
            shutil.copy(file, folder_new)

    if files_aux is not None:
        echo(f".. copy aux files: {files_aux}")
        for file in files_aux:
            shutil.copy(file, folder_new)

    if create_sample_folder:
        files = sorted(folder_new.glob("contcar_conf*"))
        create_sample_folders(
            files=files,
            outfile=file_geometry,
            base_folder=folder_new / "samples",
            files_control=files_control,
            force=force,
            format_in="vasp",
            format=format,
        )

    if predictions and file_predictions.exists():
        outfile = folder_new / (file_predictions.stem + "_prev.nc")
        echo(f"... copy '{file_predictions}' to '{outfile}'")
        shutil.copy(file_predictions, outfile)


if __name__ == "__main__":
    app()
