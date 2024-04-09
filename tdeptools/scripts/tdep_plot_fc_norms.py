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

from tdeptools.io import parse_forceconstant

echo = rich.print

_outfile_plot = "fc_TAGs.pdf"

markers = ["o", "d", "*", ".", "^", "v", "s", "+", "d"]
markers += 99 * ["o"]


def process_blocks(blocks: list, atoms: Atoms = None, decimals: int = 10):
    """turn the FC blocks from TDEP into a Dataframe containing the norms per distance"""
    # sort data
    pair_indices = []  # indices in unit cell
    pair_numbers = []  # element numbers
    pair_symbols = []  # element symbols
    # distance_vectors = []
    distance_norms = []
    forceconstant_norms = []
    forceconstant_traces = []

    for block in blocks:
        iatom, n_neighbors, data = block
        for pair in data:
            ineighbor, iunit, LL, MM = pair
            r1 = atoms.positions[iatom - 1]
            r2 = atoms.positions[iunit - 1]
            dr = r1 - r2
            lv = LL @ atoms.cell
            norm = np.linalg.norm(lv - dr)
            # distance_vectors.append(lv - dr)
            distance_norms.append(norm)
            pair_indices.append([iatom, iunit])
            n1, n2 = atoms.numbers[iatom - 1], atoms.numbers[iunit - 1]
            numbers = sorted([n1, n2])
            pair_numbers.append(numbers)
            symbols = "-".join([chemical_symbols[ii] for ii in numbers])
            pair_symbols.append(symbols)
            norm = np.linalg.norm(MM)
            trace = np.trace(MM)
            forceconstant_norms.append(norm)
            forceconstant_traces.append(trace)

    pair_indices = np.array(pair_indices)
    pair_numbers = np.array(pair_numbers)
    pair_symbols = np.array(pair_symbols)
    distance_norms = np.array(distance_norms)
    forceconstant_norms = np.array(forceconstant_norms)
    forceconstant_traces = np.array(forceconstant_traces)

    r_shell, i_shell = np.unique(
        distance_norms.round(decimals=decimals), return_inverse=True
    )

    data = {
        "distance": r_shell[i_shell],
        "shell_index": i_shell,
        "fc_norm": forceconstant_norms,
        "fc_trace": forceconstant_traces,
        "symbols": pair_symbols,
        "index_atom_1": pair_indices[:, 0],
        "index_atom_2": pair_indices[:, 1],
        "number_atom_1": pair_numbers[:, 0],
        "number_atom_2": pair_numbers[:, 1],
    }

    df = pd.DataFrame(data).sort_values(["distance", "symbols"]).reset_index(drop=True)

    return df


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    file: Path = "outfile.forceconstant",
    file_primitive: Path = "infile.ucposcar",
    outfile_data: Path = "fc_norms_and_traces.csv",
    plot: bool = True,
    log: bool = False,
    trace: bool = False,
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

    echo(f".. number of atoms:  {n_atoms}")
    echo(f".. realspace cutoff: {cutoff:.3f} AA")

    df = process_blocks(blocks=blocks, atoms=atoms)

    echo(f".. unique pairs per element: {df.symbols.unique()}")

    echo(df.head())

    echo(f".. write data to {outfile_data}")
    df.to_csv(outfile_data)

    if plot:
        fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(8, 4))

        symbols_unique = df.symbols.unique()

        x1 = df.distance
        x2 = df.shell_index
        if trace:
            y = df.fc_trace
            ax1.set_ylabel(r"FC trace (eV / ${\rm \AA}^2$)")
        else:
            y = df.fc_norm
            ax1.set_ylabel(r"FC Norm (eV / ${\rm \AA}^2$)")

        for ii, _ in enumerate(symbols_unique):
            mask = (df.symbols == symbols_unique[ii]) & (df.distance > 0.1)
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

        outfile_plot = _outfile_plot.replace("TAG", "trace" if trace else "norm")

        echo(f".. save plot to {outfile_plot}")
        fig.savefig(outfile_plot)

    echo("done.")


if __name__ == "__main__":
    app()
