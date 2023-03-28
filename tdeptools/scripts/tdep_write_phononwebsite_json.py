#! /usr/bin/env python3

import json
import webbrowser
from typing import Tuple

import numpy as np
import rich
import typer
import xarray as xr
from ase.io import read

echo = rich.print


_url = "https://henriquemiranda.github.io/phononwebsite/phonon.html"
infile_geometry = "infile.ucposcar"
infile_dispersion = "outfile.dispersion_relations.hdf5"
outfile_dispersion = "outfile.dispersion_relations.json"
default_repetitions = [2, 2, 2]

# REM:
# * this solution to directly open the file is currently not working:
# * https://github.com/henriquemiranda/phononwebsite/blob/gh-pages/scripts/phononwebsite.py
# * https://github.com/henriquemiranda/phononwebsite/blob/b8fc28920d1aee999ecc345ed16b01d599430a65/phononweb/phononweb.py#L15


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    file: str = infile_dispersion,
    file_geometry: str = infile_geometry,
    outfile: str = outfile_dispersion,
    repetitions: Tuple[int, int, int] = default_repetitions,
    split_bands: float = typer.Option(0.0, help="split bands by this many cm^1"),
    web: bool = typer.Option(False, help="open the webpage"),
    format: str = "vasp",
):
    """Convert dispersion relations into json file for
    https://henriquemiranda.github.io/phononwebsite/phonon.html
    """
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

    if split_bands > 1e-5:
        echo(f"** split the bands artificially by {33.356 * split_bands} cm^-1")
        split = split_bands * np.arange(ds.frequencies.data.shape[1])[None, :]
        ds.frequencies.data += split

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

    if web:
        echo(f".. open {_url}")
        echo(f'.. drag `{outfile}` to "Custom file: [Browse]"')
        echo(".. (direct opening currently not supported)")
        webbrowser.open(_url)


if __name__ == "__main__":
    app()
