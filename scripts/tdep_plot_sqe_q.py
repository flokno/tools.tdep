#! /usr/bin/env python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rich
import typer

from tdeptools.sqe import get_arrays

rich.traceback.install(show_locals=True)

echo = rich.print


def get_canvas():
    """set up canvas"""
    fig, ax = plt.subplots()
    return fig, ax


def plot(
    x: np.ndarray,
    y: np.ndarray,
    xmin: float = 0.0,
    xmax: float = 25,
    ymin: float = 1e-4,
    ymax: float = 20,
    linear: bool = False,
):
    fig, ax = get_canvas()

    ax.plot(x, y)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if not linear:
        ax.set_yscale("log")
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("Intensity")

    return fig, ax


app = typer.Typer()


@app.command()
def main(
    file: Path,
    q_index: int = 0,
    linear: bool = False,
):

    # read data
    qs, x, y, *_ = get_arrays(file)

    q = qs[q_index]

    echo(f".. plot spectral function for q = {q}")

    fig, ax = plot(x, y[:, q_index], linear=linear)

    outfile = file.stem + "_q"

    outfile += ".pdf"
    typer.echo(f".. save to {outfile}")
    fig.savefig(outfile, dpi=300)


if __name__ == "__main__":
    app()
