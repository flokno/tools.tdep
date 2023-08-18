#! /usr/bin/env python
from pathlib import Path

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rich import print as echo
import typer

sns.set_context("paper", font_scale=1.4)
to_invcm = 33.356


def get_canvas():
    fig, ax = plt.subplots()
    return fig, ax


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    file: Path,
    total: bool = False,
    sum_atoms: bool = False,
    xlim: float = None,
    nticks: int = 5,
    png: bool = False,
):
    # open the sqe file
    echo(f"Read DOS function from {file}")

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

    if sum_atoms:
        ys_copy = ys.copy()
        print(labels)
        symbols = [l.split("_")[0] for l in labels]
        symbols_unique, symbols_index = np.unique(symbols, return_inverse=True)
        echo("... reduce symbols:")
        echo(f"... {symbols}:")
        echo(f"--> {symbols_unique}:")
        # new ys
        ys = np.zeros([len(symbols_unique), ys.shape[1]])
        for ii, _y in enumerate(ys_copy):
            idx = symbols_index[ii]
            ys[idx] += _y
        labels = symbols_unique.tolist()

    if xlim is None:
        xlim = np.ceil(x.max())

    # talk
    echo(f".. xlim is            {xlim:.4f} THz")

    # plot
    fig, ax = get_canvas()

    if total:
        ax.fill_between(x, y)
    else:
        ax.plot(x, y, c="#313131", lw=1)
        labels.insert(0, "total")

        y = np.zeros_like(y)

        for _, col in enumerate(ys):
            ax.fill_between(x, y, y + col)
            y += col

    ax.set_xlabel("Frequency (THz)")

    ax.legend(labels, frameon=False)

    tx = ax.twiny()
    for x in (ax, tx):
        x.set_xlim((0, xlim))
        x.set_xticks(np.linspace(0, xlim, nticks))
        x.set_yticks([])
    tx.set_xticklabels([f"{tick * to_invcm:.0f}" for tick in tx.get_xticks()])
    tx.set_xlabel("Frequency (cm$^{-1}$)")
    ax.set_ylabel("Density of states (arb.)")

    echo(f".. save to {outfile}")
    fig.savefig(outfile)


if __name__ == "__main__":
    app()
