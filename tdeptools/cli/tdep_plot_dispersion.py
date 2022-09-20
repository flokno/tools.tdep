#! /usr/bin/env python
from pathlib import Path

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import typer

sns.set_context("paper", font_scale=1.7)
to_invcm = 33.356

dict_xyz = {0: "x", 1: "y", 2: "z"}


def get_canvas():
    fig, ax = plt.subplots()
    return fig, ax


app = typer.Typer()


@app.command()
def main(
    file: Path,
    png: bool = False,
    angmom: bool = False,
    angmom_component: int = 2,
    scale: float = 1e3,
    matrix: bool = False,
):
    # open the sqe file
    typer.echo(f"Read DOS function from {file}")

    # template output file name
    outfile = file.stem
    label = dict_xyz.get(angmom_component, "all")

    # read data
    with h5.File(file) as f:
        x = np.array(f["q_values"])
        ys = np.array(f["frequencies"])
        q_ticks = np.array(f["q_ticks"])
        q_ticklabels = f.attrs["q_tick_labels"].decode().split()
        if angmom:
            outfile += f"_angmon_{label}"
            if not matrix:
                if 0 <= angmom_component <= 2:
                    ii = angmom_component
                    zs = np.array(f["angular_momentum_vector_on_path"])[..., ii]
                else:
                    zs = np.array(f["angular_momentum_vector_on_path_norm"])
            else:
                outfile += "_matrix"
                zs = np.array(f["angular_momentum_matrix_on_path_norm"])

    # plot
    fig, ax = get_canvas()

    # dispersions
    kw = {"color": "k", "linewidth": 0.5}
    for y in ys.T:
        ax.plot(x, y, **kw)

    # angular momentum
    if angmom:
        for (y, z) in zip(ys.T, zs.T):
            colors = len(zs) * ["blue"]
            for ii, l in enumerate(z):
                if l < 0:
                    colors[ii] = "red"
            ax.scatter(x, y, abs(z) * scale, c=colors)

        if matrix:
            ax.set_title("Angular momentum matrix norm")
        else:
            ax.set_title(f"Angular momentum vector component {label}")

    ax.set_xticks(q_ticks)
    ax.set_xticklabels(q_ticklabels)

    if png:
        outfile += ".png"
    else:
        outfile += ".pdf"

    typer.echo(f".. save to {outfile}")
    fig.savefig(outfile)


if __name__ == "__main__":
    app()
