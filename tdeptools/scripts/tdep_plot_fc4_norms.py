#! /usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import rich
import typer
from ase import Atoms
from ase.data import chemical_symbols
from ase.io import read as read_ase
from matplotlib import pyplot as plt

from tdeptools.geometry import (
    compute_mean_distance_to_centroid,
    compute_min_max_distance,
)
import collections

echo = rich.print

_outfile_plot = "fc4_TAGs.pdf"

markers = ["o", "d", "*", ".", "^", "v", "s", "+", "d"]
markers += 99 * ["o"]


_cls_data = collections.namedtuple(
    "FC4_block_data",
    ["index_quartet", "i1", "i2", "i3", "i4", "L1", "L2", "L3", "L4", "force_constant"],
)
_cls_block = collections.namedtuple("FC4_block", ["atom_index", "n_neighbors", "data"])
_cls_fc = collections.namedtuple("FC", ["n_atoms", "cutoff", "blocks"])


def get_one_block(f, iatom):
    """Read one block of force constants for a given atom in the primitive cell

    Args:
        f: file object

    Returns:
        (index of atom, element number of atom, no. of neighbors, data)
        data = (index of neighbor, index of neighbor in unit cell, element no. of neighbor,
                reciprocal lattice vector, force constant matrix for pair)
                for each neighbor
    """
    data = []

    line = f.readline()
    n_neighbors = int(line)

    for iquartet in range(n_neighbors):
        i1 = int(f.readline().split()[0])
        i2 = int(f.readline().split()[0])
        i3 = int(f.readline().split()[0])
        i4 = int(f.readline().split()[0])
        L1 = np.fromstring(f.readline(), dtype=float, sep=" ")
        L2 = np.fromstring(f.readline(), dtype=float, sep=" ")
        L3 = np.fromstring(f.readline(), dtype=float, sep=" ")
        L4 = np.fromstring(f.readline(), dtype=float, sep=" ")
        MM = []
        for _ in range(27):
            FC = np.fromstring(f.readline(), dtype=float, sep=" ")
            MM.append(FC)
        MM = np.array(MM)

        data.append(_cls_data(iquartet, i1, i2, i3, i4, L1, L2, L3, L4, MM))

    return _cls_block(iatom, n_neighbors, data)


def parse_forceconstant(file: str = "outfile.forceconstant_thirdorder"):
    """Parse TDEP forceconstants

    Args:
        file: filename

    Returns:
        (no. atoms in primitive cell, realspace cutoff, blocks)
        blocks = data from `get_one_block` for each atom in primitive cell
    """

    blocks = []

    with open(file) as f:
        n_atoms = int(f.readline().split()[0])
        cutoff = float(f.readline().split()[0])

        for iatom in range(n_atoms):
            block = get_one_block(f, iatom=iatom)
            blocks.append(block)

    return _cls_fc(n_atoms, cutoff, blocks)


def process_blocks(blocks: list, atoms: Atoms = None, decimals: int = 10):
    """tuple = quartet"""
    # sort data
    tuple_indices = []  # indices in unit cell
    tuple_numbers = []  # element numbers
    tuple_symbols = []  # element symbols

    tuple_mean_distances = []
    tuple_min_distances = []
    tuple_max_distances = []

    # distance_vectors = []
    forceconstant_norms = []
    forceconstant_traces = []

    for block in blocks:
        iatom, n_neighbors, data = block
        for tuple in data:
            ituple, i1, i2, i3, i4, L1, L2, L3, L4, MM = tuple
            r1 = atoms.positions[i1 - 1] + L1 @ atoms.cell
            r2 = atoms.positions[i2 - 1] + L2 @ atoms.cell
            r3 = atoms.positions[i3 - 1] + L3 @ atoms.cell
            r4 = atoms.positions[i4 - 1] + L4 @ atoms.cell

            mean_distance = compute_mean_distance_to_centroid([r1, r2, r3, r4])
            tuple_mean_distances.append(mean_distance)

            min_distance, max_distance = compute_min_max_distance([r1, r2, r3, r4])
            tuple_min_distances.append(min_distance)
            tuple_max_distances.append(max_distance)

            tuple_indices.append([iatom, i1, i2, i3])
            n1, n2, n3 = (
                atoms.numbers[i1 - 1],
                atoms.numbers[i2 - 1],
                atoms.numbers[i3 - 1],
            )
            numbers = sorted([n1, n2, n3])
            tuple_numbers.append(numbers)
            symbols = "-".join([chemical_symbols[ii] for ii in numbers])
            tuple_symbols.append(symbols)

            # force constants
            M = MM.reshape(3, 3, 3, 3)
            traces = np.array([np.trace(m) for m in M])
            norm = np.linalg.norm(M)
            trace = np.sum(traces)
            forceconstant_norms.append(norm)
            forceconstant_traces.append(trace)

    tuple_indices = np.array(tuple_indices)
    tuple_numbers = np.array(tuple_numbers)
    tuple_symbols = np.array(tuple_symbols)
    # distance_norms = np.array(distance_norms)
    tuple_mean_distances = np.array(tuple_mean_distances)
    forceconstant_norms = np.array(forceconstant_norms)
    forceconstant_traces = np.array(forceconstant_traces)

    # create dataframe
    r_shell_12, i_shell = np.unique(
        tuple_mean_distances.round(decimals=10), return_inverse=True
    )

    data = {
        "shell_index": i_shell,
        "min_distance": tuple_min_distances,
        "max_distance": tuple_max_distances,
        "mean_distance": tuple_mean_distances,
        "fc_norm": forceconstant_norms,
        "fc_trace": forceconstant_traces,
        "symbols": tuple_symbols,
        "i1": tuple_indices[:, 0],
        "i2": tuple_indices[:, 1],
        "i3": tuple_indices[:, 2],
        "i4": tuple_indices[:, 3],
    }

    df = (
        pd.DataFrame(data)
        .sort_values(["symbols", "mean_distance"])
        .reset_index(drop=True)
    )

    return df


def plot_norms(log, trace, xmax, ymax, df):
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(8, 4))

    symbols_unique = df.symbols.unique()

    x1 = df.mean_distance
    x2 = df.shell_index
    if trace:
        y = df.fc_trace
        ax1.set_ylabel(r"FC3 trace (eV / ${\rm \AA}^3$)")
    else:
        y = df.fc_norm
        ax1.set_ylabel(r"FC3 Norm (eV / ${\rm \AA}^3$)")

    for ii, _ in enumerate(symbols_unique):
        mask = (df.symbols == symbols_unique[ii]) & (df.mean_distance > 0.1)
        _x1 = x1[mask]
        _x2 = x2[mask]
        _y = y[mask]

        ax1.plot(_x1, _y, marker=markers[ii], lw=0.5)
        ax2.plot(_x2, _y, marker=markers[ii], lw=0.5)

    for ax in (ax1, ax2):
        ax.legend(symbols_unique, markerfirst=False, framealpha=0)
        if log:
            ax.set_yscale("log")

        ax.axhline(0, color="#313131", zorder=-1, lw=0.5)

    ax1.set_xlabel(r"Distance (${\rm \AA}$)")
    ax2.set_xlabel("Shell no. (1)")

    # plot boundaries
    if not xmax:
        xmax = 1.1 * df.mean_distance.max()

    if not ymax:
        ymax = 1.1 * abs(y).max()

    ymin = 0
    if trace:
        ymin = -ymax

    ax1.set_xlim(0, 1.1 * df.mean_distance.max())
    ax1.set_ylim(ymin, ymax)

    outfile_plot = _outfile_plot.replace("TAG", "trace" if trace else "norm")

    echo(f"... save plot to {outfile_plot}")
    fig.savefig(outfile_plot)


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    file: Path = "outfile.forceconstant_fourthorder",
    file_primitive: Path = "infile.ucposcar",
    outfile_data: Path = "fc4_norms_and_traces.csv",
    plot: bool = True,
    log: bool = False,
    trace: bool = False,
    selfterms: bool = False,
    xmax: float = None,
    ymax: float = None,
):
    """Read forceconstants and write pair-resolved norm or trace per distance/shell

    Args:
        file (Path): Path to the forceconstants file.
        file_primitive (Path): Path to the primitive cell file.
        plot (bool, optional): Whether to generate plots. Defaults to True.
        log (bool, optional): Whether to use logarithmic scale in the plots. Defaults to False.
        trace (bool, optional): Whether to plot the trace instead of the norm. Defaults to False.
    """
    echo(f"Read primitive cell from {file_primitive}")
    atoms = read_ase(file_primitive, format="vasp")

    echo(f"Read forceconstants from {file}")
    n_atoms, cutoff, blocks = parse_forceconstant(file=file)

    assert len(atoms) == n_atoms, "Check no. of atoms in primtive cell"

    echo(f"... number of atoms:  {n_atoms}")
    echo(f"... realspace cutoff: {cutoff:.3f} AA")

    df = process_blocks(blocks=blocks, atoms=atoms)

    # remove self terms
    if not selfterms:
        echo("... remove self terms")
        df = df[df["min_distance"].gt(0)]

    echo(f"... unique pairs per element: {df.symbols.unique()}")

    echo(df.head())

    echo(f"... write data to {outfile_data}")
    df.to_csv(outfile_data)

    if plot:
        plot_norms(log, trace, xmax, ymax, df)

    echo("done.")


if __name__ == "__main__":
    app()
