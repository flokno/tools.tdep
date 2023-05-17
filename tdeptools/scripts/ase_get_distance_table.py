#! /usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from ase.io import read
from rich import print as echo


def get_symbol_pairs(atoms) -> np.ndarray:
    symbols = atoms.get_chemical_symbols()
    symbols_all = np.empty([len(atoms), len(atoms)], dtype=object)
    for ij in np.ndindex(len(atoms), len(atoms)):
        _i, _j = ij
        symbols_all[ij] = "-".join(sorted([symbols[_i], symbols[_j]]))

    return symbols_all


def plot_distance_table(df: pd.DataFrame, outfile: Path = None):
    """Plot the distance table"""
    from matplotlib import pyplot as plt

    _df = df.groupby(["distances", "symbols"]).size().unstack()

    fig, ax = plt.subplots()

    for ii, col in enumerate(_df.columns):
        bottom = _df[_df.columns[:ii]].sum(axis=1)
        ax.bar(_df.index, _df[_df.columns[ii]], bottom=bottom, width=0.1)

    ax.legend(_df.columns, frameon=False)
    ax.set_xlabel("Distance (Å)")
    ax.set_ylabel("Count")

    fig.savefig(outfile)


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    file: Path,
    mic: bool = True,
    n_replicas: int = 1,
    decimals: int = 5,
    plot: bool = False,
    format: str = None,
    outfile: Path = "outfile.distances.csv",
):
    """Get all distances for structure in FILE and write them as pandas Dataframe"""
    echo(f"Read `{file}`")

    if "geometry.in" in file.name:
        format = "aims"
        echo(f"... autodetect format `{format}` for {file}")
    elif "poscar" in file.name.lower():
        format = "vasp"
        echo(f"... autodetect format `{format}` for {file}")

    atoms = read(file, format=format)
    echo(f"... material: {atoms}")
    echo(f"... MIC: {'':23} {mic}")

    # density
    density = len(atoms) / atoms.get_volume()
    echo(f"... density (atoms/AA**3): {density:10.3f}")
    echo(f"--> atom volume (AA**3):   {1/density:10.3f}")
    echo(f"--> atom radius (AA):      {(3/4/np.pi/density)**(1/3):10.3f}")
    echo(f"--> atoms per  1A sphere:  {4*np.pi/3 * density:10.3f}")
    echo(f"--> atoms per  3A sphere:  {27*4*np.pi/3 * density:10.3f}")
    echo(f"--> atoms per  4A sphere:  {64*4*np.pi/3 * density:10.3f}")
    echo(f"--> atoms per  5A sphere:  {125*4*np.pi/3 * density:10.3f}")
    echo(f"--> atoms per 10A sphere:  {1000*4*np.pi/3 * density:10.3f}")

    echo(f"... use {n_replicas} replicas to see more neighbors")
    atoms = atoms * [n_replicas, n_replicas, n_replicas]
    echo(f"... number of atoms: {len(atoms)}")

    # all distances
    kw = {"mic": mic, "vector": False}
    distances_all = atoms.get_all_distances(**kw).round(decimals=decimals)

    # Write all symbol pairs
    symbols_all = get_symbol_pairs(atoms)

    # create DataFrame
    data = {"symbols": symbols_all.flatten(), "distances": distances_all.flatten()}
    df = pd.DataFrame(data)
    df = df[df.distances > 1e-9].sort_values("distances").reset_index(drop=True)

    # unique pairs
    symbols_unique = df["symbols"].unique()
    echo(f"... unique pairs: {symbols_unique}")

    _nrows = 15
    echo(f"... first {_nrows} neighbors, first is nearest neighbor:")
    echo(df.drop_duplicates("distances").head(_nrows))

    echo(f"... save to {outfile}")
    df.to_csv(outfile)

    if plot:
        outfile = outfile.stem + ".pdf"
        echo(f"... plot and save to {outfile}")
        plot_distance_table(df, outfile=outfile)


if __name__ == "__main__":
    app()
