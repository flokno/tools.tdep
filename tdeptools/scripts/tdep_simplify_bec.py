#! /usr/bin/env python3

from pathlib import Path

import numpy as np
import typer
from rich import print as echo

app = typer.Typer()


def diagonalize(array: np.ndarray) -> np.ndarray:
    return np.diag(np.diag(array))


def scalarize(array: np.ndarray) -> np.ndarray:
    trace = np.trace(array)
    return np.eye(len(array)) * trace / len(array)


@app.command()
def main(file: Path, outfile: Path = None):
    """Diagonalize the Born charges and make everything isotropic (scalar)"""
    echo(f"Read dielectric tensor and Born charges from {file}")

    data = np.loadtxt(file).reshape(-1, 3, 3)

    eps = data[0]
    bec = data[1:]

    bec_diag = np.zeros_like(bec)
    bec_scalar = np.zeros_like(bec)
    bec_original = np.zeros_like(bec)

    echo(".. enforce acoustic sum rule")
    bec -= bec.mean(axis=0)

    echo(".. simplify BEC")
    for (ii, ZZ) in enumerate(bec):
        bec_diag[ii] = diagonalize(ZZ)
        bec_scalar[ii] = scalarize(ZZ)
        bec_original[ii] = ZZ

    violation_scalar = np.linalg.norm(bec_scalar.mean(axis=0))
    violation_diagonal = np.linalg.norm(bec_diag.mean(axis=0))
    violation_original = np.linalg.norm(bec_original.mean(axis=0))

    echo(f".. acoustic sum rule violation for original BEC: {violation_original:.3e}")
    echo(f".. acoustic sum rule violation for diagonal BEC: {violation_diagonal:.3e}")
    echo(f".. acoustic sum rule violation for isotropic BEC: {violation_scalar:.3e}")

    eps_diag = diagonalize(eps)
    eps_scalar = scalarize(eps)
    new_bec_diag = np.concatenate([eps_diag[None, :], bec_diag]).reshape(-1, 3)
    new_bec_scalar = np.concatenate([eps_scalar[None, :], bec_scalar]).reshape(-1, 3)
    new_bec_original = np.concatenate([eps[None, :], bec]).reshape(-1, 3)

    # also create non-diagonal epsilon with scalar BEC
    new_bec_scalar_bec = new_bec_scalar.copy()
    new_bec_scalar_bec[:3] = eps

    if outfile is None:
        outfile = "outfile.lotosplitting"

    kw = {"fmt": "%22.15f"}
    echo(f".. save original epsilon with   scalar BEC to {outfile}")
    np.savetxt(outfile, new_bec_scalar_bec, **kw)
    echo(f".. save   scalar epsilon with   scalar BEC to {outfile}_scalar")
    np.savetxt(outfile + "_scalar", new_bec_scalar, **kw)
    echo(f".. save   scalar epsilon with diagonal BEC to {outfile}_diagonal")
    np.savetxt(outfile + "_diagonal", new_bec_diag, **kw)
    echo(f".. save original epsilon with original BEC to {outfile}_original")
    np.savetxt(outfile + "_original", new_bec_original, **kw)

    echo("done.")


if __name__ == "__main__":
    app()
