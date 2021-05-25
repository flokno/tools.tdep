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
    half: bool = False,
    max_intensity: float = 1,
    min_intensity: float = 1e-4,
    linear: bool = False,
    zoom: bool = False,
):
    # open the sqe file
    typer.echo(f"Read spectral function from {file}")
    f = h5.File(file, "r")

    typer.echo(".. keys in file:")
    typer.echo(f.keys())

    # get axes and intensity
    x = np.array(f.get("q_values"))
    y = np.array(f.get("energy_values"))
    try:
        gz = np.array(f["spectral_function"])
    except KeyError:
        gz = np.array(f["intensity"])  # compatibility with older sqe.hdf5 files

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

    # zoom?
    if zoom:
        max_intensity = 0.005
        linear = True
        typer.echo(".. zoom in:")
        typer.echo(f"... max intensity: {max_intensity}")
        typer.echo("... use linear scale")

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
        if half:
            ylim /= 2
    ax.set_ylim([y.min(), ylim])
    ax.set_xticks(xt)
    ax.set_xticklabels(xl)
    ax.set_ylabel(yl)

    outfile = file.stem
    if half:
        outfile += "_half"
    if zoom:
        outfile += "_zoom"
    if not zoom and max_intensity < 1:
        outfile += "_intensity"
    if not zoom and linear:
        outfile += "_linear"

    outfile += ".png"
    typer.echo(f".. save to {outfile}")
    fig.savefig(outfile, dpi=300)


if __name__ == "__main__":
    typer.run(main)
