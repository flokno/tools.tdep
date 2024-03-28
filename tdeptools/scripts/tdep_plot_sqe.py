#! /usr/bin/env python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rich
import typer
from matplotlib.colors import LogNorm, Normalize

from tdeptools.sqe import find_ylim, get_arrays, get_arrays_dispersion

echo = rich.print


def get_canvas():
    """set up canvas"""
    fig, ax = plt.subplots()
    return fig, ax


def plot(
    x: np.ndarray,
    y: np.ndarray,
    gz: np.ndarray,
    xt: np.ndarray,
    xl: list,
    yl: list,
    ylim: float = None,
    linear: bool = False,
    ys_dispersion: np.ndarray = None,
):
    fig, ax = get_canvas()

    if linear:
        norm = Normalize(vmin=gz.min(), vmax=gz.max())
    else:
        norm = LogNorm(vmin=gz.min(), vmax=gz.max())
    kw = {"cmap": "viridis", "shading": "auto", "norm": norm}

    # for plotting, turn the axes into 2d arrays
    gx, gy = np.meshgrid(x, y)

    ax.pcolormesh(gx, gy, gz, **kw)
    # set the limits of the plot to the limits of the data
    ax.set_xlim([x.min(), x.max()])

    ax.set_ylim([y.min(), ylim])
    ax.set_xticks(xt)
    ax.set_xticklabels(xl)
    ax.set_ylabel(yl)

    if ys_dispersion is not None:
        kw = {"color": "white", "ls": "--", "lw": 0.5, "alpha": 0.5}
        for y in ys_dispersion.T:
            ax.plot(x, y, **kw)

    return fig, ax


app = typer.Typer()


@app.command()
def main(
    file: Path,
    ylim: float = None,
    max_frequency: float = 0.99,
    max_intensity: float = 1,
    min_intensity: float = 1e-4,
    linear: bool = False,
    dispersion: Path = None,
    dispersion_neutron: Path = None,
):

    # read data
    x, y, gz, xt, xl, yl = get_arrays(file)
    if dispersion is not None:
        _, ys_dispersion = get_arrays_dispersion(dispersion)
    else:
        ys_dispersion = None

    # integrate intensity in energy
    n_bands = int(np.trapz(gz, x=y, axis=0).mean())
    echo(f".. no. of bands:      {n_bands:.2f}")

    if not ylim:
        ylim = find_ylim(y, gz, max_frequency)

    # normalize intensity
    gz /= gz.max()

    # add a little bit so that the logscale does not go nuts
    gz = gz + min_intensity

    # cap intensity
    if max_intensity < 1:
        gz[gz > max_intensity] = max_intensity

    fig, ax = plot(x, y, gz, xt, xl, yl, ylim, linear, ys_dispersion)

    if dispersion_neutron is not None:
        echo(".. try to superimpose neutron data")
        df = pd.read_csv(dispersion_neutron)
        echo(df)
        echo(n_bands)
        for ii in range(n_bands):
            ax.scatter(x=df.q_tdep, y=df[str(ii+1)], s=3, color="C3")

    # talk
    echo(f".. ylim is            {ylim:.4f} THz")
    echo(f".. max intensity:     {max_intensity}")
    echo(f".. use linear scale:  {linear}")

    # template output file name
    outfile = file.stem

    if max_intensity < 1:
        outfile += "_intensity"
    if linear:
        outfile += "_linear"
    if max_frequency < 0.99:
        outfile += f"_max_{max_frequency:4.2f}"

    outfile += ".png"
    echo(f".. save to {outfile}")
    fig.savefig(outfile, dpi=300)


if __name__ == "__main__":
    app()
