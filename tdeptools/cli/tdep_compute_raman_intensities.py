#! /usr/bin/env python3

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import typer
import xarray as xr
from matplotlib import pyplot as plt
from rich import print as echo

from tdeptools.geometry import get_orthonormal_directions
from tdeptools.physics import freq2amplitude
from tdeptools.raman import intensity_isotropic, po_average
from tdeptools.konstanter import lo_frequency_THz_to_icm

_default_po_direction = (None, None, None)

key_intensity_raman = "intensity_raman"


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    file_activity: Path = "outfile.mode_activity.csv",
    file_dielectric: Path = "infile.dielectric_tensor",
    outfile: Path = "outfile.mode_intensity.csv",
    outfile_po: Path = "outfile.mode_intensity_po.h5",
    po_direction: Tuple[int, int, int] = _default_po_direction,
    temperature: float = 0.0,
    quantum: bool = True,
    iq: int = 0,
):
    """Compute Raman activity per mode"""

    # activity
    echo(f".. read mode activity from {file_activity}")
    df_activity = pd.read_csv(file_activity, comment="#")
    echo(f".. active q-points: {df_activity.iq.unique()}")
    echo(f".. select:          {df_activity.iq.unique()[iq]}")
    df_activity = df_activity[df_activity.iq == df_activity.iq.unique()[iq]]
    n_modes = len(df_activity)
    echo(f".. no. of modes:    {n_modes}")

    # dielectric
    echo(f".. read dielectric tensors from {file_activity}")
    data_dielectric = np.loadtxt(file_dielectric).reshape([-1, 3, 3])
    n_tensors = len(data_dielectric)
    echo(f".. found {n_tensors} tensors")

    assert n_tensors == 2 * n_modes, f"Please provide {2*n_modes} tensors for now."

    amplitudes = freq2amplitude(
        df_activity.frequency, temperature=temperature, quantum=quantum
    )

    # convention: Matrices are stores alternating between + and - displacement
    dielectric_matrices_pos = data_dielectric[0::2]
    dielectric_matrices_neg = data_dielectric[1::2]

    dielectric_matrices_diff = dielectric_matrices_pos - dielectric_matrices_neg

    # Raman tensor:
    # I_abq = d eps_ab / d Q_q  # a,b: Cart. coords, q: mode

    # let's start w/ isotropic averaging
    I_abq = np.zeros_like(dielectric_matrices_diff)
    # mask away where amplitudes are small:
    mask = amplitudes > 1e-9
    I_abq[mask] = dielectric_matrices_diff[mask] / amplitudes[mask, None, None]

    # compute 1 intensity per mode
    I_q = np.zeros_like(amplitudes)
    for ii, I_ab in enumerate(I_abq):
        I_q[ii] = intensity_isotropic(I_ab)

    # add to dataframe
    df_activity[key_intensity_raman] = I_q

    echo(f".. write intensities to {outfile}")
    df_activity.to_csv(outfile, index=None)

    # PO?
    if not (po_direction == _default_po_direction):
        echo(f".. compute PO intensity map for E_in = {po_direction}")
        echo(".. find orthonormal directions:")
        directions = get_orthonormal_directions(po_direction)
        for ii, d in enumerate(directions):
            echo(f"... direction {ii}: {d}")

        I_qp_para, I_qp_perp, angles = po_average(
            I_abq=I_abq, direction1=directions[1], direction2=directions[2]
        )

        attrs = {f"direction{ii+1}": d for ii, d in enumerate(directions)}
        coords = {
            "frequency": df_activity.frequency * lo_frequency_THz_to_icm,
            "angle": angles,
        }
        dims = coords.keys()  # ("frequency", "angle")
        kw = {"dims": dims, "coords": coords, "attrs": attrs}

        name = key_intensity_raman + "_PO"

        da_para = xr.DataArray(I_qp_para, name=name + "_parallel", **kw)
        da_perp = xr.DataArray(I_qp_perp, name=name + "_perpendicular", **kw)
        arrays = [da_para, da_perp]

        ds = xr.merge(arrays)

        echo(f".. save PO data to {outfile_po}")
        ds.to_netcdf(outfile_po)

        fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(10, 5))
        x = np.linspace(0, ds.frequency.max(), 100)
        a, *_, b = x
        echo(f".. interpolate data to evenly spaced frequencies on [{a}, {b:.2f}]")
        kw = {"fill_value": 0}
        for ax, da in zip(axs, arrays):
            # interpolate to uniformly spaced frequency for plotting
            da = da[3:].interp({"frequency": x}, method="nearest", kwargs=kw)
            da.name = da.name.split("_")[-1]
            xr.plot.imshow(da.T, ax=ax)

        fig.suptitle(f"PO Raman intensity for {po_direction} orientation")

        a, b, c = po_direction
        outfile = "outfile." + name + f"_{a}{b}{c}" + ".pdf"
        echo(f".. save plot to {outfile}")
        fig.savefig(outfile)


if __name__ == "__main__":
    app()
