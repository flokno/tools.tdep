#! /usr/bin/env python3

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import typer
from ase.io import read
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from rich import panel
from rich import print as echo

from tdeptools.konstanter import lo_frequency_THz_to_icm
from tdeptools.dos import (
    get_bose_weighted_DOS,
    get_weighted_2w_DOS,
    get_convoluted_DOS,
    get_convoluted_weighted_DOS,
)


app = typer.Typer(pretty_exceptions_show_locals=False)


def _infile(*args):
    """Input file option, must exist"""
    return typer.Option(*args, exists=True)


def plot_df(df, toicm, temperature, outfile_plot, figsize=(6, 6)):
    if toicm:
        df = df.set_index("frequency_cm")
    else:
        df = df.set_index("frequency")

    fig, axs = plt.subplots(nrows=5, sharex=True, figsize=figsize)
    axs[0].plot(df.index, df["dos"], label="DOS")
    axs[1].plot(df.index, df["dos_convoluted"], label="DOS conv.")
    axs[2].plot(df.index, df["dos_weighted"], label="weighted DOS")
    axs[3].plot(df.index, df["2w_dos_weighted"], label="weighted 2w-DOS ")
    axs[4].plot(df.index, df["dos_weighted_convoluted"], label="weighted DOS conv.")
    for ax in axs:
        ax.legend(loc=2, frameon=False)
        ax.set_xlim(df.index.min(), df.index.max())

    ax.set_xlabel("Frequency (cm$^{-1}$)" if toicm else "Frequency (THz)")

    # fig.suptitle(f"Temperature: {temperature} K", pad=0)
    axs[0].set_title(f"Temperature: {temperature} K")
    fig.savefig(outfile_plot)
    return fig


@app.command()
def main(
    file_dos: Path = _infile("outfile.phonon_dos"),
    temperature: float = typer.Option(300, help="Temperature in K"),
    plot: bool = typer.Option(True, help="Plot the DOS convolutions"),
    toicm: bool = True,
    outfile_data: Path = "outfile.phonon_dos_convolutions.csv",
    outfile_plot: Path = "outfile.phonon_dos_convolutions.pdf",
    eps: float = 1e-12,
):
    """Calculate DOS convolutions for Raman intensities"""
    echo(f"Read '{file_dos}'")

    data_dos = np.loadtxt(file_dos).T
    x_dos, y_dos, *_ = data_dos

    # mask = np.abs(x_dos) > eps
    # x_dos = x_dos[mask]
    # y_dos = y_dos[mask]

    # get weighted 2w-DOS

    kw = {"x": x_dos, "y": y_dos, "temperature": temperature}
    x_full, y_full, y_full_weighted = get_bose_weighted_DOS(**kw)
    _, _, y2_full_weighted = get_weighted_2w_DOS(**kw)
    x_c, y_c = get_convoluted_DOS(**kw)
    x_wc, y_wc = get_convoluted_weighted_DOS(**kw)

    assert np.allclose(x_full, x_wc)

    # create dataframe
    data = {
        "frequency": x_full,
        "frequency_cm": x_full * lo_frequency_THz_to_icm,
        "dos": y_full,
        "dos_weighted": y_full_weighted,
        "dos_convoluted": y_c,
        "dos_weighted_convoluted": y_wc,
        "2w_dos_weighted": y2_full_weighted,
    }
    df = pd.DataFrame(data)

    _outfile = outfile_data
    echo("DOS:")
    rep = df.head().to_string()

    echo(panel.Panel(rep, title=str(_outfile), expand=False, subtitle="..."))

    echo(f"... write data to '{_outfile}'")
    df.to_csv(_outfile, index=False)

    if plot:
        plot_df(df, toicm, temperature=temperature, outfile_plot=outfile_plot)


if __name__ == "__main__":
    app()
