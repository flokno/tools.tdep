#! /usr/bin/env python3

import numpy as np
import typer

outfile = "outfile.lotosplitting"

app = typer.Typer()


def diagonalize(array: np.ndarray) -> np.ndarray:
    return np.diag(np.diag(array))


def scalarize(array: np.ndarray) -> np.ndarray:
    trace = np.trace(array)
    return np.eye(len(array)) * trace / len(array)


@app.command()
def main(file):
    """Diagonalize the Born charges and make everything isotropic (scalar)"""
    typer.echo(f"Read dielectric tensor and Born charges from {file}")

    data = np.loadtxt(file).reshape(-1, 3, 3)

    eps = data[0]
    bec = data[1:]

    bec_diag = np.zeros_like(bec)
    bec_scalar = np.zeros_like(bec)

    typer.echo(".. enforce acoustic sum rule")
    bec -= bec.mean(axis=0)

    typer.echo(".. simplify BEC")
    for (ii, ZZ) in enumerate(bec):
        bec_diag[ii] = diagonalize(ZZ)
        bec_scalar[ii] = scalarize(ZZ)

    viol_diag = np.linalg.norm(bec_diag.mean(axis=0))
    viol_scalar = np.linalg.norm(bec_scalar.mean(axis=0))

    typer.echo(f".. acoustic sum rule violation for diagonal BEC: {viol_diag:.3e}")
    typer.echo(f".. acoustic sum rule violation for isotropic BEC: {viol_scalar:.3e}")

    eps_diag = diagonalize(eps)
    eps_scalar = scalarize(eps)
    new_bec_diag = np.concatenate([eps_diag[None, :], bec_diag]).reshape(-1, 3)
    new_bec_scalar = np.concatenate([eps_scalar[None, :], bec_scalar]).reshape(-1, 3)

    typer.echo(f".. save simplified BEC to {outfile}_scalar/diagonal")
    np.savetxt(outfile + "_scalar", new_bec_scalar)
    np.savetxt(outfile + "_diagonal", new_bec_diag)

    typer.echo("done.")


if __name__ == "__main__":
    app()
