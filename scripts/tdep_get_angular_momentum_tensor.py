#! /usr/bin/env python3
from pathlib import Path

import h5py as h5
import numpy as np
import typer
import xarray as xr

app = typer.Typer()


@app.command()
def main(file: str = "outfile.cumulative_kappa.hdf5"):
    typer.echo(f"Read {file}")

    with h5.File(file) as f:
        keys = list(f.keys())
        # datasets = [k for k in keys if isinstance(f[k], h5._hl.dataset.Dataset)]
        groups = [k for k in keys if isinstance(f[k], h5._hl.group.Group)]

    temperatures = np.zeros(len(groups))
    angular_momentum_tensors = np.zeros([len(groups), 3, 3])

    for ii, group in enumerate(sorted(groups)):
        ds = xr.load_dataset(file, group=group)
        temperatures[ii] = float(ds.temperature)
        angular_momentum_tensors[ii] = np.asfarray(ds.angular_momentum_tensor)

    da = xr.DataArray(
        angular_momentum_tensors,
        coords={"temperature": temperatures},
        dims=("temperature", "a", "b"),
        name="phonon_angular_momentum",
    ).sortby("temperature")

    da_stack = da.stack(ab=("a", "b"))

    da_norm = xr.DataArray(
        np.linalg.norm(da_stack, axis=1),
        coords=da.coords,
        dims=("temperature"),
        name="phonon_angular_momentum_norm",
    )

    df = da.to_dataframe()
    s = da_norm.to_series()

    outfile = Path(file).parent / "angular_momentum_tensor.csv"
    typer.echo(f".. write angular momentum tensors to {outfile}")
    df.to_csv(outfile)

    outfile = Path(file).parent / "angular_momentum_tensor_norm.csv"
    typer.echo(f".. write norm of angular momentum tensors to {outfile}")
    s.to_csv(outfile)

    outfile = Path(file).parent / "angular_momentum_tensor_rt.dat"
    typer.echo(f".. write angular momentum tensor at room temperature to {outfile}")
    np.savetxt(outfile, da.loc[300])

    typer.echo(f".. norm at 300K: {np.linalg.norm(da.loc[300]):7.3e}")


if __name__ == "__main__":
    app()
