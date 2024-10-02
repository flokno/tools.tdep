#! /usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from ase.io import read
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from rich import panel
from rich import print as echo
import collections
from scipy import signal as sl
import xarray as xr

from tdeptools.physics import n_BE
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


def jls_extract_def(df, toicm, temperature, outfile_plot):
    if toicm:
        _df = df.set_index("frequency_cm")
    else:
        _df = df.set_index("frequency")

    fig, axs = plt.subplots(nrows=3, sharex=True)

    ax = axs[0]
    ax.plot(_df.index, _df["2PDOS"], label="2PDOS", zorder=5)
    ax.plot(_df.index, _df["2PDOS_1"], label="2PDOS +/+")
    ax.plot(_df.index, _df["2PDOS_2"], label="2PDOS +/-")

    ax = axs[1]
    ax.plot(_df.index, _df["2wDOS"], label="2w-DOS", zorder=5)
    ax.plot(_df.index, _df["2wDOS_1"], label="2w-DOS +/+")
    ax.plot(_df.index, _df["2wDOS_2"], label="2w-DOS -/-")

    ax = axs[2]
    ax.plot(_df.index, _df["DOS_convolution"], label="DOS conv.", zorder=5)
    ax.plot(_df.index, _df["DOS_convolution_1"], label="DOS conv. +/+", zorder=5)
    ax.plot(_df.index, _df["DOS_convolution_2"], label="DOS conv. +/-", zorder=5)

    for ax in axs:
        ax.legend(loc=2, frameon=False)

    ax.set_xlabel("Frequency (cm$^{-1}$)" if toicm else "Frequency (THz)")

    axs[0].set_title(f"Temperature: {temperature} K")

    echo(f'... save plot to "{outfile_plot}"')
    fig.savefig(outfile_plot)

    return fig, axs


def delta(x, x0, eta=1e-1):
    return 1 / np.pi * eta / ((x - x0) ** 2 + eta**2)


def compute_spectral_function_convolution(
    frequencies, weights, n_frequencies=1024, temperature=300, eta=None
):
    fs = abs(np.asarray(frequencies))
    ws = np.asarray(weights)
    ns = n_BE(fs, temperature=temperature)
    Nq, Ns = fs.shape

    Nw = n_frequencies

    _w = np.linspace(-2 * fs.max(), 2 * fs.max(), Nw)
    dw = (_w[1] - _w[0]) / 2

    Jp_qs = np.zeros((Nw, Nq, Ns))
    Jm_qs = np.zeros((Nw, Nq, Ns))

    if eta is None:
        eta = 2 * dw

    for i, w in enumerate(_w):
        Jp_qs[i] = ws[:, None] * (ns + 1) * delta(w, +fs, eta=eta)
        Jm_qs[i] = ws[:, None] * ns * delta(w, -fs, eta=eta)

    # option 1: sum over q and s first
    fp = Jp_qs.sum(axis=(1, 2))
    fm = Jm_qs.sum(axis=(1, 2))

    fpp = sl.convolve(fp, fp, "same") / Nw
    fpm = sl.convolve(fp, fm, "same") / Nw
    fmp = sl.convolve(fm, fp, "same") / Nw
    fmm = sl.convolve(fm, fm, "same") / Nw
    fmm = sl.convolve(fm, fm, "same") / Nw

    J1 = fpp + fmm
    J2 = fpm + fmp

    # option 2: sum only over s first
    fp_q = Jp_qs.sum(axis=2)
    fm_q = Jm_qs.sum(axis=2)

    fpp_q = np.zeros_like(fp_q)
    fpm_q = np.zeros_like(fp_q)
    fmp_q = np.zeros_like(fp_q)
    fmm_q = np.zeros_like(fp_q)

    for iq in range(Nq):
        fpp_q[:, iq] = sl.convolve(fp_q[:, iq], fp_q[:, iq], mode="same") / Nw
        fpm_q[:, iq] = sl.convolve(fp_q[:, iq], fm_q[:, iq], mode="same") / Nw
        fmp_q[:, iq] = sl.convolve(fm_q[:, iq], fp_q[:, iq], mode="same") / Nw
        fmm_q[:, iq] = sl.convolve(fm_q[:, iq], fm_q[:, iq], mode="same") / Nw

    fpp = Nq * fpp_q.sum(axis=1)
    fpm = Nq * fpm_q.sum(axis=1)
    fmp = Nq * fmp_q.sum(axis=1)
    fmm = Nq * fmm_q.sum(axis=1)

    J1q = fpp + fmm
    J2q = fpm + fmp

    # option 3: weighted 2w-DOS
    Jp_qs = np.zeros((Nw, Nq, Ns))
    Jm_qs = np.zeros((Nw, Nq, Ns))

    for i, w in enumerate(_w):
        Jp_qs[i] = ws[:, None] * (ns + 1) ** 2 * delta(w, +2 * fs, eta=4 * eta)
        Jm_qs[i] = ws[:, None] * ns**2 * delta(w, -2 * fs, eta=4 * eta)

    # option 1: sum over q and s first
    J1s = Jp_qs.sum(axis=(1, 2))
    J2s = Jm_qs.sum(axis=(1, 2))

    return collections.namedtuple(
        "jdos", ["frequencies", "f1", "f2", "f1q", "f2q", "f1s", "f2s"]
    )(_w, J1, J2, J1q, J2q, J1s, J2s)


@app.command()
def main(
    ctx: typer.Context,
    file_dispersion: Path = _infile("outfile.grid_dispersions_irreducible.hdf5"),
    temperature: float = typer.Option(300, help="Temperature in K"),
    eta: float = None,
    plot: bool = typer.Option(True, help="Plot the DOS convolutions"),
    toicm: bool = True,
    outfile_data: Path = "outfile.2pdos.csv",
    outfile_plot: Path = "outfile.2pdos.pdf",
):
    """Calculate DOS convolutions for Raman intensities"""
    echo(f"Read '{file_dispersion}'")

    echo("Settings:")
    echo(ctx.params)

    ds = xr.load_dataset(file_dispersion)

    # frequencies from 1/s to THz
    fs = ds.frequencies.data / 1e12 / 2 / np.pi
    ws = ds.integration_weights.data

    # get different contributions to spectral function convolution
    _w, J1, J2, J1q, J2q, J1s, J2s = compute_spectral_function_convolution(
        fs, ws, n_frequencies=1024, temperature=temperature, eta=eta
    )

    # create dataframe
    data = {
        "frequency": _w,
        "frequency_cm": _w * lo_frequency_THz_to_icm,
        "2PDOS": J1q + J2q,
        "2PDOS_1": J1q,
        "2PDOS_2": J2q,
        "2wDOS": J1s + J2s,
        "2wDOS_1": J1s,
        "2wDOS_2": J2s,
        "DOS_convolution": J1 + J2,
        "DOS_convolution_1": J1,
        "DOS_convolution_2": J2,
    }
    df = pd.DataFrame(data)

    echo(f'... save data to "{outfile_data}"')
    df.to_csv(outfile_data, index=False)

    if plot:
        fig, axs = jls_extract_def(
            df, toicm=toicm, temperature=temperature, outfile_plot=outfile_plot
        )


if __name__ == "__main__":
    app()
