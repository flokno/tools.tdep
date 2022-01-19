#! /usr/bin/env python
from pathlib import Path

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import typer

sns.set_context("paper", font_scale=1.7)
to_invcm = 33.356


def get_canvas():
    fig, ax = plt.subplots()
    return fig, ax


def main(
    file: Path,
    total: bool = False,
    png: bool = False,
):
    # open the sqe file
    typer.echo(f"Read DOS function from {file}")

    # template output file name
    outfile = file.stem
    if png:
        outfile += ".png"
    else:
        outfile += ".pdf"

    # read data
    with h5.File(file) as f:
        x = np.array(f["frequencies"])
        y = np.array(f["dos"])
        ys = np.array(f["dos_per_unique_atom"])
        labels = f.attrs["unique_atom_labels"].decode().split()

    xmax = np.ceil(x.max())

    # talk
    typer.echo(f".. xlim is            {xmax:.4f} THz")

    # plot
    fig, ax = get_canvas()

    if total:
        ax.fill_between(x, y)
    else:
        ax.plot(x, y, c="k")
        labels.insert(0, "total")

        y = np.zeros_like(y)

        for _, col in enumerate(ys):
            ax.fill_between(x, y, y + col)
            y += col

    ax.set_xlabel("Freqcuency (THz)")

    ax.legend(labels, frameon=False)

    tx = ax.twiny()
    for x in (ax, tx):
        x.set_xlim((0, xmax))
        x.set_xticks(np.linspace(0, xmax + 1, 7))
    tx.set_xticklabels([f"{tick * to_invcm:.0f}" for tick in tx.get_xticks()])
    tx.set_xlabel("Omega (cm$^{-1}$)")

    typer.echo(f".. save to {outfile}")
    fig.savefig(outfile)


if __name__ == "__main__":
    typer.run(main)
