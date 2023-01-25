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

_file = Path("outfile.forceconstant")
_file_primitive = Path("infile.ucposcar")
_outfile_data = "fc_TAGs.csv"
_outfile_plot = "fc_TAGs.pdf"


def process_blocks(blocks: list, atoms: Atoms = None, trace: bool = False):
    """turn the FC blocks from TDEP into a Dataframe containing the norms per distance"""
    # sort data
    pair_indices = []  # indices in unit cell
    pair_numbers = []  # element numbers
    pair_symbols = []  # element symbols
    # distance_vectors = []
    distance_norms = []
    forceconstant_norms = []

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
            if trace:
                measure = np.trace(MM)
            else:
                measure = np.linalg.norm(MM)
            forceconstant_norms.append(measure)

    pair_indices = np.array(pair_indices)
    pair_numbers = np.array(pair_numbers)
    pair_symbols = np.array(pair_symbols)
    distance_norms = np.array(distance_norms)
    forceconstant_norms = np.array(forceconstant_norms)

    r_shell, i_shell = np.unique(distance_norms.round(decimals=10), return_inverse=True)

    data = {
        "distance": r_shell[i_shell],
        "shell_index": i_shell,
        "fc_norm": forceconstant_norms,
        "symbols": pair_symbols,
        "index_atom_1": pair_indices[:, 0],
        "index_atom_2": pair_indices[:, 1],
        "number_atom_1": pair_numbers[:, 0],
        "number_atom_2": pair_numbers[:, 1],
    }

    df = pd.DataFrame(data).sort_values(["distance", "symbols"]).reset_index(drop=True)

    return df


app = typer.Typer(pretty_exceptions_show_locals=True)


@app.command()
def main(
    file: Path = _file,
    file_primitive: Path = _file_primitive,
    plot: bool = True,
    log: bool = False,
    trace: bool = False,
    _asdf: str = _outfile_plot,
):
    """Read forceconstants and write pair-resolved norm or trace per distance/shell"""
    echo(f"Read primitive cell from {file_primitive}")
    atoms = read_ase(file_primitive, format="vasp")

    echo(f"Read forceconstants from {file}")
    n_atoms, cutoff, blocks = parse_forceconstant()

    assert len(atoms) == n_atoms, "Check no. of atoms in primtive cell"

    echo(f".. number of atoms:  {n_atoms}")
    echo(f".. realspace cutoff: {cutoff:.3f} AA")

    df = process_blocks(blocks=blocks, atoms=atoms, trace=trace)

    echo(f".. unique pairs per element: {df.symbols.unique()}")

    echo(df.head())

    if trace:
        tag = "trace"
    else:
        tag = "norm"

    outfile_data = _outfile_data.replace("TAG", tag)
    echo(f".. write data to {outfile_data}")
    df.to_csv(outfile_data)

    if plot:
        fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, figsize=(8, 4))

        symbols_unique = df.symbols.unique()

        for ii, _ in enumerate(symbols_unique):
            df_temp = df[df.symbols == symbols_unique[ii]]
            df_temp = df_temp[df_temp.distance > 0.1]
            ax1.plot(df_temp.distance, df_temp.fc_norm, marker="o", lw=0.5)
            ax2.plot(df_temp.shell_index, df_temp.fc_norm, marker="o", lw=0.5)

        for ax in (ax1, ax2):
            ax.legend(symbols_unique, markerfirst=False, framealpha=0)
            if log:
                ax.set_yscale("log")

            ax.axhline(0, color="#313131", zorder=-1, lw=0.5)

        ax1.set_xlabel(r"Distance (${\rm \AA}$)")
        ax2.set_xlabel("Shell no. (1)")
        if trace:
            ax1.set_ylabel(r"FC trace (eV / ${\rm \AA}^2$)")
        else:
            ax1.set_ylabel(r"FC Norm (eV / ${\rm \AA}^2$)")

        outfile_plot = _outfile_plot.replace("TAG", tag)
        echo(f".. save plot to {outfile_plot}")
        fig.savefig(outfile_plot)

    echo("done.")


if __name__ == "__main__":
    app()
