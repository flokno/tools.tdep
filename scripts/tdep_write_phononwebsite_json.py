#! /usr/bin/env python3

import json
from typing import Tuple

import numpy as np
import rich
import typer
import xarray as xr
from ase.io import read

echo = rich.print


infile_geometry = "infile.ucposcar"
infile_dispersion = "outfile.dispersion_relations.hdf5"
outfile_dispersion = "outfile.dispersion_relations.json"
default_repetitions = [2, 2, 2]

app = typer.Typer()


@app.command()
def main(
    file: str = infile_dispersion,
    file_geometry: str = infile_geometry,
    outfile: str = outfile_dispersion,
    repetitions: Tuple[int, int, int] = default_repetitions,
    format: str = "vasp",
):
    echo(f"Read geometry form {file_geometry}")
    atoms = read(infile_geometry, format=format)

    echo(f"Parse {file} ")

    ds = xr.load_dataset(file)

    qpoints = abs(atoms.cell.cartesian_positions(ds.q_vector.data)).tolist()
    distances = ds.q_values.data.tolist()

    nq_per_segment = int(len(distances) / (len(ds.q_tick_labels.split()) - 1))

    high_symmetry_points = [
        [ii * nq_per_segment, key] for ii, key in enumerate(ds.q_tick_labels.split())
    ]

    eigenvectors_re = ds.eigenvectors_re.data
    eigenvectors_im = ds.eigenvectors_im.data

    eigenvectors = eigenvectors_re + 1.0j * eigenvectors_im

    dim = (*eigenvectors.shape[:-1], len(atoms), 3, 2)
    dim
    eigenvectors.view(float).reshape(dim)

    # prepare data
    data = {
        "name": str(atoms.symbols),
        "natoms": len(atoms),
        "lattice": atoms.cell.tolist(),
        "atom_types": list(atoms.symbols),
        "atom_numbers": atoms.numbers.tolist(),
        "atomic_numbers": np.unique(atoms.numbers).tolist(),
        "chemical_symbols": np.unique(atoms.symbols).tolist(),
        "formula": str(atoms.symbols.formula),
        "repetitions": repetitions,
        "atom_pos_car": atoms.positions.tolist(),
        "atom_pos_red": atoms.get_scaled_positions().tolist(),
        "highsym_qpts": high_symmetry_points,
        "qpoints": qpoints,
        "distances": distances,
        "eigenvalues": (ds.frequencies.data * 33.356).tolist(),
        "vectors": eigenvectors.view(float).reshape(dim).tolist(),
    }

    # dump
    echo(f".. dump data to {outfile}")
    with open(outfile_dispersion, "w") as f:
        json.dump(data, f, indent=1)


if __name__ == "__main__":
    app()
