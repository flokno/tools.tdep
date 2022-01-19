#! /usr/bin/env python
from pathlib import Path

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import typer
from matplotlib.colors import LogNorm, Normalize


def get_canvas():
    fig, ax = plt.subplots()
    return fig, ax


def main(
    file: Path,
    ylim: float = None,
    max_frequency: float = 0.99,
    max_intensity: float = 1,
    min_intensity: float = 1e-4,
    linear: bool = False,
):
    # open the sqe file
    typer.echo(f"Read spectral function from {file}")
    f = h5.File(file, "r")

    # typer.echo(".. keys in file:")
    # typer.echo(f.keys())

    # template output file name
    outfile = file.stem

    # get axes and intensity
    x = np.array(f.get("q_values"))
    y = np.array(f.get("energy_values"))
    try:
        gz = np.array(f["spectral_function"])
    except KeyError:
        gz = np.array(f["intensity"])  # compatibility with older sqe.hdf5 files

    # integrate intensity in energy
    n_bands = np.trapz(gz, x=y, axis=0).mean()
    typer.echo(f".. no. of bands:      {n_bands}")

    if not ylim and max_frequency < 1:
        # find ylim as fraction of full band occupation
        for nn, yy in enumerate(y):
            gz_int = np.trapz(gz[:nn], x=y[:nn], axis=0).mean()
            if gz_int > max_frequency * n_bands:
                ylim = yy
                typer.echo(f".. {max_frequency*100}% intensity at {yy:.3f} THz")
                break

    # normalize intensity
    gz /= gz.max()

    # add a little bit so that the logscale does not go nuts
    gz = gz + min_intensity
    # for plotting, turn the axes into 2d arrays
    gx, gy = np.meshgrid(x, y)
    # x-ticks
    xt = np.array(f.get("q_ticks"))
    # labels for the x-ticks
    xl = f.attrs.get("q_tick_labels").decode().split()
    # label for y-axis
    yl = f"Energy ({f.attrs.get('energy_unit').decode():s})"

    # cap intensity
    if max_intensity < 1:
        gz[gz > max_intensity] = max_intensity

    fig, ax = get_canvas()

    if linear:
        norm = Normalize(vmin=gz.min(), vmax=gz.max())
    else:
        norm = LogNorm(vmin=gz.min(), vmax=gz.max())
    kw = {"cmap": "viridis", "shading": "auto", "norm": norm}
    ax.pcolormesh(gx, gy, gz, **kw)
    # set the limits of the plot to the limits of the data
    ax.set_xlim([x.min(), x.max()])
    if ylim is None:
        ylim = y.max()

    ax.set_ylim([y.min(), ylim])
    ax.set_xticks(xt)
    ax.set_xticklabels(xl)
    ax.set_ylabel(yl)

    # talk
    typer.echo(f".. ylim is            {ylim:.4f} THz")
    typer.echo(f".. max intensity:     {max_intensity}")
    typer.echo(f".. use linear scale:  {linear}")

    if max_intensity < 1:
        outfile += "_intensity"
    if linear:
        outfile += "_linear"
    if max_frequency < 0.99:
        outfile += f"_max_{max_frequency:4.2f}"

    outfile += ".png"
    typer.echo(f".. save to {outfile}")
    fig.savefig(outfile, dpi=300)


if __name__ == "__main__":
    typer.run(main)
