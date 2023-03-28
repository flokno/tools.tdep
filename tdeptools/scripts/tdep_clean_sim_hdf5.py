#! /usr/bin/env python3

import xarray as xr
import typer
from rich import print as echo

_file = "outfile.sim.hdf5"
time, atom, cart = "time", "atom", "cart"

app = typer.Typer()


@app.command()
def update_sim_dimensions(file: str = _file, outfile: str = _file):
    """Read FILE and replace dimension names"""
    echo(f"Read dataset from {file}")
    ds = xr.load_dataset(file)

    dims = ds.positions.dims
    new_dims = {dims[0]: time, dims[1]: atom, dims[2]: cart}

    dims = ds.unitcell_latticevectors.dims
    new_dims.update({dims[0]: cart, dims[1]: cart})

    echo(".. update dimensions:")
    echo(new_dims)

    ds = ds.rename_dims(new_dims)

    echo(f".. write to {outfile}")
    ds.to_netcdf(outfile)


if __name__ == "__main__":
    app()
