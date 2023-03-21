#! /usr/bin/env python3

import numpy as np
import typer
from rich import print as echo
from typing import List

_outfile_born_charges = "infile.born_charges"
_outfile_dielectric_tensor = "infile.dielectric_tensor"


strip = lambda x: x.strip().strip("(").strip(")")

hook_eps = "Dielectric constant in cartesian axis"
hook_bec = "Effective charges (d Force / dE) in cartesian axis without acoustic"


def parse_epsilon(fp) -> np.ndarray:
    """parse the dielectric tensor from ph.x output"""
    next(fp)  # skip 1 line
    lines = [strip(next(fp)) for _ in range(3)]
    return np.array([np.fromstring(x, sep=" ") for x in lines])


def parse_bec(fp, natoms) -> np.ndarray:
    """parse the BEC from ph.x output"""
    next(fp)  # skip 1 line
    bec = []
    for _ in range(natoms):
        next(fp)
        bec.extend(next(fp).split()[2:5] for _ in range(3))

    return np.reshape(bec, (natoms, 3, 3)).astype(float)


def parse_ph_out(file: str) -> dict:
    """Parse BEC and dielectric tensor from QE ph.x output"""

    epsilon, bec = None, None
    with open(file) as f:
        for line in f:
            if "number of atoms/cell      =" in line:
                natoms = int(line.split()[4])
            if hook_eps in line:
                epsilon = parse_epsilon(f)
            if hook_bec in line:
                bec = parse_bec(f, natoms=natoms)

    return epsilon, bec


def write_results(
    results: list, outfile_dielectric_tensor: str, outfile_born_charges: str
):
    with open(outfile_dielectric_tensor, "w") as feps, open(
        outfile_born_charges, "a"
    ) as fbec:
        for result in results:

            for vec in result["dielectric_tensor"]:
                feps.write(" ".join(f"{x:23.15e}" for x in vec) + "\n")

            for vec in result["born_charges"].reshape(-1, 3):
                fbec.write(" ".join(f"{x:23.15e}" for x in vec) + "\n")


app = typer.Typer()


@app.command()
def main(
    files: List[str],
    outfile_dielectric_tensor: str = _outfile_dielectric_tensor,
    outfile_born_charges: str = _outfile_born_charges,
    verbose: bool = False,
):
    """Parse BEC and dielectric tensor from QE ph.x output files"""

    results = []
    for file in files:
        echo(f".. parse {file}")
        epsilon, bec = parse_ph_out(file)

        if epsilon is None:
            echo("** did not find dielectric tensor, SKIP")
            continue
        if bec is None:
            echo("** did not find BEC, SKIP")
            continue

        if verbose:
            echo(".. sum rule violation:")
            echo(f"{bec.sum(axis=0)}")
            bec -= bec.mean(axis=0)

        results.append({"born_charges": bec, "dielectric_tensor": epsilon})

    write_results(
        results,
        outfile_dielectric_tensor=outfile_dielectric_tensor,
        outfile_born_charges=_outfile_born_charges,
    )

    echo(f".. {len(results)} data point written to {outfile_dielectric_tensor}")
    echo(f".. {len(results)} data point written to {outfile_born_charges}")


if __name__ == "__main__":
    app()
