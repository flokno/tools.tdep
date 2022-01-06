#! /usr/bin/env python3

import shutil
import subprocess as sp
from pathlib import Path

import numpy as np
import typer

_rc2 = 20.0
_logfile = "fc12.log"
_command = "extract_forceconstants"
_infiles = ["infile.forces", "infile.positions", "infile.meta", "infile.stat"]
_infile_polar = "infile.lotosplitting"
_infile_ucposcar = "infile.ucposcar"
_infile_ssposcar = "infile.ssposcar"
_infiles_structure = [_infile_ucposcar, _infile_ssposcar]
_outfile_ucposcar = "outfile.new_ucposcar"
_outfile_ssposcar = "outfile.new_ssposcar"
_outfiles_structure = [_outfile_ucposcar, _outfile_ssposcar]
_outfile_forceconstant = "outfile.forceconstant"
_outfile_forceconstant_firstorder = "outfile.forceconstant_firstorder"


def _check_forceconstant(cwd: Path, verbose: bool = True) -> float:
    """check first-order FC and report size"""
    file = cwd / _outfile_forceconstant_firstorder
    fc = np.loadtxt(file)
    norm = np.linalg.norm(fc)
    if verbose:
        typer.echo(f".. FC norm: {norm:.4e}")

    return norm


def _extract_forceconstants(
    cwd: Path = ".",
    rc2: float = _rc2,
    polar: bool = False,
    logfile: str = _logfile,
    verbose: bool = False,
):
    """run extract_forceconstants in folder"""
    cmd = f"{_command} -rc2 {rc2} --firstorder"
    if polar:
        cmd += " --polar"

    cwd.mkdir(exist_ok=True, parents=True)
    typer.echo(f".. run `{cmd}` in {cwd}")
    ps = sp.run(cmd.split(), cwd=cwd, capture_output=True)
    stdout = ps.stdout
    if verbose:
        typer.echo(stdout)
    typer.echo(f".. write stdout to {cwd / logfile}")
    with open(cwd / logfile, "wb") as f:
        f.write(stdout)


def _copy_input_files(folder_ref: Path, folder_new: Path, mkdir=True):
    """copy input files to new folder"""
    if mkdir:
        folder_new.mkdir(exist_ok=True, parents=True)

    for file in _infiles:
        shutil.copy(folder_ref / file, folder_new)

    # structures
    shutil.copy(folder_ref / _outfile_ucposcar, folder_new / _infile_ucposcar)
    shutil.copy(folder_ref / _outfile_ssposcar, folder_new / _infile_ssposcar)


def _copy_results(folder: Path, root: Path, prefix: str = "outfile.*"):
    """copy final results: strucutres and forceconstants"""
    for file in sorted(folder.glob(prefix)):
        typer.echo(f"... copy {file} to {root}")
        shutil.copy(file, root)


def main(
    rc2: float = _rc2,
    polar: bool = False,
    logfile: str = _logfile,
    base_folder: str = "iter",
    maxiter: int = 30,
):
    """iteratively run extract_forceconstant"""
    typer.echo("Start optimization")

    if polar:
        _infiles.append(_infile_polar)

    # sanity check input files
    for file in _infiles + _infiles_structure:
        assert Path(file).exists()

    # run once to get starting point
    cwd = Path(".")
    kw = {"cwd": cwd, "rc2": rc2, "polar": polar, "logfile": cwd / logfile}
    _extract_forceconstants(**kw)

    root = cwd

    # relaxation loop
    for ii in range(maxiter):
        typer.echo(f"Iteration {ii:5d}")
        folder = root / (base_folder + f".{ii:03d}")
        folder.mkdir(exist_ok=True)

        _copy_input_files(cwd.absolute(), folder.absolute())
        _extract_forceconstants(folder)
        norm = _check_forceconstant(folder)
        if norm < 1e-21:
            typer.echo(f"--> converged with |FC| = {norm:.4e}. Break.")
            break
        cwd = folder

    typer.echo("--> Copy final results to root folder")
    _copy_results(folder, root)

    typer.echo("Done.")


if __name__ == "__main__":
    typer.run(main)
