#! /usr/bin/env python3

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import typer
import xarray as xr
from ase.io import read
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from rich import panel
from rich import print as echo

from tdeptools.geometry import get_orthonormal_directions
from tdeptools.konstanter import lo_amu_to_emu, lo_frequency_THz_to_icm
from tdeptools.physics import freq2amplitude
from tdeptools.raman import intensity_isotropic, po_average

_default_po_direction = (None, None, None)

key_intensity_raman = "intensity_raman"


def read_dataset(file: str) -> xr.Dataset:
    """Read outfile.phonon_self_energy.hdf5 into one xr.Dataset"""
    ds = xr.load_dataset(file).rename({"q-point": "q_point"})
    ds_ha = xr.load_dataset(file, group="harmonic")
    ds_an = xr.load_dataset(file, group="anharmonic")
    ds_qm = xr.load_dataset(file, group="qmesh")

    return xr.merge([ds, ds_ha, ds_an, ds_qm])


def plot_intensity(x, y, yp, xlim, outfile="outfile.intensity_raman.pdf"):
    fig, ax = plt.subplots(figsize=(6, 4))
    y /= y.max()
    yp /= yp.max()
    ax.plot(x, y)
    ax.plot(x, yp, ls="--", color="k", lw=1)
    ax.set_xlim(0, xlim)
    ax.set_yticks([])
    ax.set_xlabel("Frequency (1/cm)")
    ax.set_ylabel("Intensity")
    ax.legend(["unpolarized", "isotropic"])
    echo(f"... save intensity plot to '{outfile}'")
    fig.savefig(outfile)


def plot_po_map(
    po_direction,
    outfile,
    arrays_with_frequency,
    tol=1e-7,
    linear=False,
    figsize=(10, 5),
    xlim=None,
    vmax=None,
):
    ncols = len(arrays_with_frequency)
    fig, axs = plt.subplots(ncols=ncols, sharey=True, figsize=figsize)
    for ax, da in zip(axs, arrays_with_frequency):
        # normalize
        da = da / da.max()
        if vmax is not None:
            vmin = tol
        else:
            vmin = da.data.min() + tol
            vmax = da.data.max() + tol
        kw = {"vmin": vmin, "vmax": vmax}
        if linear:
            norm = Normalize(**kw)
        else:
            norm = LogNorm(**kw)
        # xr.plot.imshow(da + 2 * tol, ax=ax, norm=norm)
        xr.plot.imshow(da, ax=ax, norm=norm)

        if xlim is not None:
            ax.set_xlim(0, xlim)

    fig.suptitle(f"PO Raman intensity for {po_direction} orientation")

    echo(f"... save PO plot to '{outfile}'")
    fig.savefig(outfile)


def get_intensity_from_atom_displacements(
    ds: xr.Dataset,
    data_dielectric: np.ndarray,
    displacement: float,
    masses: np.ndarray,
):
    """Compute mode intensity tensors from real-space displacements

    I_abq = \sum_i v_iq * d eps_ab / d u_i  # a,b: Cart. coords, q: mode

    u_i = real-space displacement of atom i
    v_iq = eigenmode (transformation) vector, u_q =  \sum_i v_iq u_i
    """

    # convention: Matrices are stored alternating between + and - displacement
    dielectric_matrices_pos = data_dielectric[0::2]
    dielectric_matrices_neg = data_dielectric[1::2]

    dielectric_matrices_diff = dielectric_matrices_pos - dielectric_matrices_neg

    # get intensities
    dXdu_iab = np.zeros_like(dielectric_matrices_diff)

    h = 2 * displacement

    dXdu_iab = dielectric_matrices_diff / h

    # dXdu_iab
    dXdu_iab = dXdu_iab.reshape(-1, 3, 3, 3)

    # mode transformation
    evs = ds.eigenvectors_re.data

    # resulting displacements in [N_mode, N_atoms, 3]
    masses_emu = np.sqrt(masses.repeat(3) * lo_amu_to_emu)
    v_qia = (evs / masses_emu[None, :]).reshape(-1, len(masses), 3)

    # intensities
    _ = None
    I_qab = (v_qia[:, :, :, _, _] * dXdu_iab[_, :]).sum(axis=(1, 2))

    return I_qab


def get_intensity_from_mode_displacements(
    temperature: float,
    quantum: bool,
    ds: xr.Dataset,
    data_dielectric: np.ndarray,
):
    """Compute mode intensity tensors from mode displacements

    I_abq = d eps_ab / d u_q  # a,b: Cart. coords, q: mode

    u_q = real-space displacement pattern for mode q

    """
    amplitudes = freq2amplitude(
        ds.harmonic_frequencies, temperature=temperature, quantum=quantum
    )
    echo("... ignore acoustic --> add them as zeros")
    data_dielectric = np.concatenate([np.zeros([6, 3, 3]), data_dielectric])

    # convention: Matrices are stored alternating between + and - displacement
    dielectric_matrices_pos = data_dielectric[0::2]
    dielectric_matrices_neg = data_dielectric[1::2]

    dielectric_matrices_diff = dielectric_matrices_pos - dielectric_matrices_neg

    # Raman tensor:
    # I_abq = d eps_ab / d Q_q  # a,b: Cart. coords, q: mode

    # let's start w/ isotropic averaging
    I_qab = np.zeros_like(dielectric_matrices_diff)
    # mask away where amplitudes are small:
    mask = amplitudes > 1e-9
    I_qab[mask] = dielectric_matrices_diff[mask] / amplitudes[mask, None, None]

    return I_qab


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    file_geometry: Path = "infile.ucposcar",
    file_dielectric: Path = "infile.dielectric_tensor",
    file_self_energy: Path = "outfile.phonon_self_energy.hdf5",
    outfile_intensity: Path = "outfile.intensity_raman.csv",
    outfile_intensity_mode: Path = "outfile.mode_intensity.csv",
    outfile_intensity_po: Path = "outfile.intensity_raman_po.h5",
    temperature: float = 0.0,
    displacement: float = typer.Option(0.01, help="real-space displacement in Å"),
    quantum: bool = True,
    plot: bool = False,
    xlim: float = None,
    vmax: float = None,
    linear: bool = False,
    thz: bool = False,
    decimals: int = 9,
    qdir: Tuple[int, int, int] = (None, None, None),
    format_geometry: str = "vasp",
):
    """Compute Raman activity from finite differences.

    2 * 3 * Natoms       samples -> assume single atom displacements

    2 * 3 * (Natoms - 1) samples -> assume mode displacements (w/o acoustic)

    Args:
        file_geometry: the reference structure
        file_dielectric: input file with dielectric tensors
        file_self_energy: file with spectral functions from lineshape
        outfile: csv file with mode intensities
        outfile_po: hdf5 file with PO maps
        plot: creates some low effort plots
    """
    echo(f"... read structure from '{file_geometry}'")
    atoms = read(file_geometry, format=format_geometry)
    n_modes = 3 * len(atoms)

    echo(f"--> number of atoms: {len(atoms)}")
    echo(f"--> number of modes: {n_modes}")

    if thz:
        unit = 1.0
        echo("... use THz")
    else:
        unit = lo_frequency_THz_to_icm
        echo("... use inv. cm")

    # activity
    echo(f"... read spectral information from '{file_self_energy}'")
    ds = read_dataset(file=file_self_energy)

    if None in qdir:
        qdir = ds.incident_wavevector.data
    else:
        echo("!!! manually specified q direction:")
        qdir = np.asarray(qdir)
    echo(f"--> data is for incident q = {qdir}")

    # dielectric
    echo(f"... read dielectric tensors from '{file_dielectric}'")
    data_dielectric = np.loadtxt(file_dielectric).reshape([-1, 3, 3])
    n_tensors = len(data_dielectric)
    echo(f"... found {n_tensors} tensors")

    # Get the Raman tensors per mode
    if len(data_dielectric) == 2 * n_modes:
        echo("!!! 2 * 3N SAMPLES FOUND, ASSUME REAL SPACE DISPLACEMENTS")
        I_qab = get_intensity_from_atom_displacements(
            ds, data_dielectric, displacement, atoms.get_masses()
        )
    elif len(data_dielectric) == 2 * n_modes - 6:
        echo("!!! 2 * (3N - 3) SAMPLES FOUND, ASSUME MODE DISPLACEMENTS")
        I_qab = get_intensity_from_mode_displacements(
            temperature, quantum, ds, data_dielectric
        )
    else:
        n1, n2 = 2 * n_modes, 2 * (n_modes - 3)
        msg = f"got {len(data_dielectric)} dielectric tensors, need {n1} or {n2}"
        raise ValueError(msg)

    assert len(data_dielectric) == 2 * n_modes, (len(data_dielectric), 2 * n_modes)

    # get PO directions orthogonal to incident q direction
    po_direction = qdir
    echo(f"... compute PO intensity map for k_in = {po_direction}")
    echo("... find orthonormal directions:")
    directions = get_orthonormal_directions(po_direction)
    for ii, d in enumerate(directions):
        echo(f"... direction {ii}: {d}")

    # project Raman tensors on PO angles
    I_qp_para, I_qp_perp, angles = po_average(
        I_abq=I_qab, direction1=directions[1], direction2=directions[2]
    )

    # average over PO
    I_q_po = I_qp_para.mean(axis=1) + I_qp_perp.mean(axis=1)

    # compute 1 intensity per mode per isotropic averaging
    I_q = np.zeros(n_modes)
    for ii, I_ab in enumerate(I_qab):
        I_q[ii] = intensity_isotropic(I_ab)

    # create a dataframe for mode intensities
    data = {
        "imode": np.arange(n_modes),
        "frequency": ds.harmonic_frequencies,
        "intensity_raman_isotropic": I_q.round(decimals=decimals),
        "intensity_raman_unpolarized": I_q_po.round(decimals=decimals),
    }
    df_intensity = pd.DataFrame(data)

    # report
    echo("RAMAN MODE INTENSITIES:")
    rep = df_intensity.to_string()
    echo(panel.Panel(rep, title=str(outfile_intensity_mode), expand=False))

    # save mode intensities to file
    echo(f"... write intensities to '{outfile_intensity_mode}'")
    df_intensity.to_csv(outfile_intensity_mode, index=None)

    # DEV: what is this actually? needed?
    # now full spectral function per mode

    # multiply in the spectral function to get spectra
    _x = ds.frequency.data * unit
    data = I_q[:, None] * ds.spectralfunction_per_mode.data
    da_mode = xr.DataArray(
        data,
        coords={"imode": np.arange(n_modes), "frequency": _x},
        name="intensity_per_mode",
    )
    da_isotropic = da_mode.sum(dim="imode")
    da_isotropic.name = "spectral_intensity_raman_isotropic"

    # ds_intensity = xr.merge([da_total, da_mode])
    # ds_intensity.to_netcdf("outfile.intensity_raman.h5")

    # prepare datasets for the PO maps
    # coords = {
    #     "frequency": df_intensity.frequency * unit,
    #     "angle": angles,
    # }
    # dims = coords.keys()  # ("frequency", "angle")
    # attrs = {f"direction{ii+1}": d for ii, d in enumerate(directions)}
    # kw = {"dims": dims, "coords": coords, "attrs": attrs}

    # # create 1 DataArray each for perpendicular/parallel
    # name = key_intensity_raman + "_PO"
    # da_para = xr.DataArray(I_qp_para, name=name + "_parallel", **kw)
    # da_perp = xr.DataArray(I_qp_perp, name=name + "_perpendicular", **kw)
    # arrays = [da_para, da_perp]

    # echo(_x.shape)
    # echo(df_intensity.frequency.shape)
    # asdf

    # now multiply in the spectral functions for each PO orientation
    coords = {
        "angle": angles,
        "frequency": _x,
    }
    attrs = {f"direction{ii+1}": d for ii, d in enumerate(directions)}
    names = ["parallel", "perpendicular"]
    arrays = [I_qp_para, I_qp_perp]
    arrays_with_frequency = []
    for array, name in zip(arrays, names):
        data = array.T @ ds.spectralfunction_per_mode.data
        da = xr.DataArray(data, coords=coords, attrs=attrs)
        da.name = name
        arrays_with_frequency.append(da)

    ds_po = xr.merge(arrays_with_frequency)
    echo(f"... save PO data to '{outfile_intensity_po}'")
    ds_po.to_netcdf(outfile_intensity_po)

    # get the unpolarized intensity = mean over angle
    _ds = ds_po.mean(dim="angle")
    _ds["unpolarized"] = _ds.parallel + _ds.perpendicular
    _ds["isotropic"] = da_isotropic
    # more distinguishable naming:
    _rename = {var: f"intensity_{var}" for var in _ds.data_vars}
    df = _ds.rename(_rename).to_dataframe()
    echo(f"... save unpolarized intensity to '{outfile_intensity}'")
    df.to_csv(outfile_intensity)

    # some plotting
    if plot:
        a, b, c = [int(np.ceil(x)) for x in qdir / qdir.max()]
        _name = str(outfile_intensity).split(".")[1]
        _outfile = "outfile." + _name
        if xlim is None:
            xlim = 1.2 * ds.harmonic_frequencies.max() * unit
        echo(f"... xlim = {float(xlim)}")
        plot_po_map(
            po_direction,
            _outfile + f"_po_{a}{b}{c}.pdf",
            arrays_with_frequency,
            linear=linear,
            xlim=xlim,
            vmax=vmax,
        )
        plot_intensity(_x, df.intensity_unpolarized, df.intensity_isotropic, xlim=xlim, outfile=_outfile + f"_{a}{b}{c}.pdf")


if __name__ == "__main__":
    app()
