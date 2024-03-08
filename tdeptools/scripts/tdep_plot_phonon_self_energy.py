#! /usr/bin/env python
from pathlib import Path

import matplotlib.pyplot as plt
import rich
import typer
import pandas as pd
import numpy as np

from tdeptools.hdf5 import read_dataset_phonon_self_energy
from tdeptools.konstanter import lo_frequency_THz_to_icm

rich.traceback.install(show_locals=True)

echo = rich.print


def get_canvas():
    """set up canvas"""
    fig, ax = plt.subplots()
    return fig, ax


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    file: Path,
    thz: bool = False,
    xlim: float = None,
    linear: bool = False,
):
    fig, ax = get_canvas()

    echo(f"Read '{file}'")
    ds = read_dataset_phonon_self_energy(file)

    n_modes = len(ds.spectralfunction_per_mode)
    colums = ["intensity"] + [f"intensity_mode_{ii:d}" for ii in range(n_modes)]

    data = np.array(ds.spectralfunction_per_mode.T)
    data = np.insert(data, 0, ds.spectralfunction_per_mode.sum(axis=0), axis=1)

    print(data.shape)

    x = ds.frequency
    xlabel = "Frequency (THz)"
    if not thz:
        echo("... convert frequencies to inv. cm")
        xlabel = "Frequency (1/cm)"
        x = x * lo_frequency_THz_to_icm

    _index_name = "frequency"
    df = pd.DataFrame(
        data,
        columns=colums,
        index=pd.Index(x, name=_index_name),
    )

    # find highest non-zero index
    max_frequency = df[df["intensity"].gt(0)].index.max()  # / 2

    df.iloc[:, 0].plot(ax=ax, color="#313131", lw=3)
    # df.iloc[:, 1:].plot(ax=ax, lw=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Intensity (arb.)")

    ax.set_xlim(0, xlim or 1.1 * max_frequency)

    outfile = file.stem + ".pdf"
    echo(f"... save plot to       '{outfile}'")
    fig.savefig(outfile, dpi=300)

    outfile = file.stem + ".csv"
    echo(f"... write plot data to '{outfile}'")
    df[:max_frequency].to_csv(outfile, index_label=_index_name)


if __name__ == "__main__":
    app()
