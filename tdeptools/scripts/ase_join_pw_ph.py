#! /usr/bin/env python3

import collections
from pathlib import Path

import numpy as np
import typer
from ase.io import read
from ase.outputs import all_outputs as _all_outputs
from rich import print as echo

_dct = {key: key for key in _all_outputs}
all_outputs = collections.namedtuple("keys", _dct.keys())(**_dct)


hook_eps = "Dielectric constant in cartesian axis"
hook_bec = "Effective charges (d Force / dE) in cartesian axis without acoustic"


def _strip(xx: str) -> str:
    return xx.strip().strip("(").strip(")")


def parse_epsilon(fp) -> np.ndarray:
    """parse the dielectric tensor from ph.x output"""
    next(fp)  # skip 1 line
    lines = [_strip(next(fp)) for _ in range(3)]
    return np.array([np.fromstring(x, sep=" ") for x in lines])


def parse_bec(fp, natoms) -> np.ndarray:
    """parse the Born effective charges (BEC) from ph.x output"""
    next(fp)  # skip 1 line
    bec = []
    for _ in range(natoms):
        next(fp)
        bec.extend(next(fp).split()[2:5] for _ in range(3))

    return np.reshape(bec, (natoms, 3, 3)).astype(float)


def parse_ph_out(file: str, natoms: int) -> dict:
    """Parse BEC and dielectric tensor from QE ph.x output w/ a minor sanity check"""

    epsilon, bec = None, None
    with open(file) as f:
        for line in f:
            if "number of atoms/cell      =" in line:
                _natoms = int(line.split()[4])
                assert natoms == _natoms, (natoms, _natoms)
            if hook_eps in line:
                epsilon = parse_epsilon(f)
            if hook_bec in line:
                bec = parse_bec(f, natoms=natoms)

    return {
        all_outputs.dielectric_tensor: epsilon,
        all_outputs.born_effective_charges: bec,
    }


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    file_pw: Path,
    file_ph: Path,
    outfile: str = "qe.json",
    format_pw: str = "espresso-out",
    verbose: bool = True,
):
    """Parse and join output from QE pw.x and ph.x, dump to generic JSON output file"""
    if verbose:
        echo(f"Read QE pw.x output from {file_pw}")
        echo(f"read QE ph.x output from {file_ph}")

    atoms = read(file_pw, format=format_pw)

    results_ph = parse_ph_out(file_ph, natoms=len(atoms))

    atoms.calc.results.update(results_ph)

    if verbose:
        echo(f"... write results to {outfile}")
    atoms.write(outfile)


if __name__ == "__main__":
    app()
