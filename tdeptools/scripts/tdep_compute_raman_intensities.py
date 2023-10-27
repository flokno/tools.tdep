#! /usr/bin/env python3

from pathlib import Path

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


def plot_intensity(x, y, xmax, outfile="outfile.intensity_raman.pdf"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lo_frequency_THz_to_icm * x, y)
    ax.set_xlim(0, lo_frequency_THz_to_icm * xmax)
    ax.set_yticks([])
    ax.set_xlabel("Frequency (1/cm)")
    ax.set_ylabel("Intensity")
    echo(f"... save intensity plot to '{outfile}'")
    fig.savefig(outfile)


def plot_po_map(
    po_direction, name, arrays_with_frequency, tol=1e-10, linear=False, figsize=(10, 5)
):
    ncols = len(arrays_with_frequency)
    fig, axs = plt.subplots(ncols=ncols, sharey=True, figsize=figsize)
    for ax, da in zip(axs, arrays_with_frequency):
        vmin = da.data.min() + tol
        vmax = da.data.max() + tol
        kw = {"vmin": vmin, "vmax": vmax}
        if linear:
            norm = Normalize(**kw)
        else:
            norm = LogNorm(**kw)
        xr.plot.imshow(da + 2 * tol, ax=ax, norm=norm)

    fig.suptitle(f"PO Raman intensity for {po_direction} orientation")

    a, b, c = [int(np.ceil(x)) for x in po_direction]
    outfile = "outfile." + name + f"_{a}{b}{c}" + ".pdf"
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
    decimals: int = 9,
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

    # activity
    echo(f"... read spectral information from '{file_self_energy}'")
    ds = read_dataset(file=file_self_energy)

    qdir = ds.incident_wavevector.data
    echo(f"--> data is for incident q = {qdir}")

    # dielectric
    echo(f"... read dielectric tensors from '{file_dielectric}'")
    data_dielectric = np.loadtxt(file_dielectric).reshape([-1, 3, 3])
    n_tensors = len(data_dielectric)
    echo(f"... found {n_tensors} tensors")

    # Check if we have 2 dielectric tensors per mode, optionally w/o acoustic
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

    # compute 1 intensity per mode
    I_q = np.zeros(n_modes)
    for ii, I_ab in enumerate(I_qab):
        I_q[ii] = intensity_isotropic(I_ab)

    # add to dataframe
    data = {
        "imode": np.arange(n_modes),
        "frequency": ds.harmonic_frequencies,
        "intensity_raman": I_q.round(decimals=decimals),
    }
    df_intensity = pd.DataFrame(data)

    echo("RAMAN MODE INTENSITIES:")
    p = panel.Panel(
        df_intensity.to_string(), title=str(outfile_intensity_mode), expand=False
    )
    echo(p)

    echo(f"... write intensities to '{outfile_intensity_mode}'")
    df_intensity.to_csv(outfile_intensity_mode, index=None)

    # now full spectral
    _x = ds.frequency
    data = I_q[:, None] * ds.spectralfunction_per_mode.data
    da_mode = xr.DataArray(
        data,
        coords={"imode": np.arange(n_modes), "frequency": _x.data},
        name="intensity_per_mode",
    )
    da_total = xr.DataArray(
        data.sum(axis=0), coords={"frequency": _x.data}, name="intensity"
    )

    ds_intensity = xr.merge([da_mode, da_total])
    ds_intensity.to_netcdf("outfile.intensity_raman.h5")

    # now for the PO
    po_direction = qdir
    echo(f"... compute PO intensity map for k_in = {po_direction}")
    echo("... find orthonormal directions:")
    directions = get_orthonormal_directions(po_direction)
    for ii, d in enumerate(directions):
        echo(f"... direction {ii}: {d}")

    # get PO data and corresponding angles
    I_qp_para, I_qp_perp, angles = po_average(
        I_abq=I_qab, direction1=directions[1], direction2=directions[2]
    )

    # prepare dataset with labels etc
    coords = {
        "frequency": df_intensity.frequency * lo_frequency_THz_to_icm,
        "angle": angles,
    }
    dims = coords.keys()  # ("frequency", "angle")
    attrs = {f"direction{ii+1}": d for ii, d in enumerate(directions)}
    kw = {"dims": dims, "coords": coords, "attrs": attrs}

    name = key_intensity_raman + "_PO"
    da_para = xr.DataArray(I_qp_para, name=name + "_parallel", **kw)
    da_perp = xr.DataArray(I_qp_perp, name=name + "_perpendicular", **kw)
    arrays = [da_para, da_perp]

    # now multiply in the spectral functions for PO
    arrays_with_frequency = []
    for da in arrays:
        _name = da.name.split("_")[-1]
        data = da.data.T @ ds.spectralfunction_per_mode.data
        da = xr.DataArray(
            data, coords={"angle": angles, "frequency": _x.data}, attrs=attrs
        )
        da.name = _name
        arrays_with_frequency.append(da)

    ds_po = xr.merge(arrays_with_frequency)
    echo(f"... save PO data to '{outfile_intensity_po}'")
    ds_po.to_netcdf(outfile_intensity_po)

    # unpolarized intensity
    _s = ds_po.parallel.sum(dim="angle") + ds_po.parallel.sum(dim="angle")
    s = _s.to_series()
    echo(f"... save unpolarized intensity to '{outfile_intensity}'")
    s.to_csv(outfile_intensity)

    if plot:
        _name = str(outfile_intensity_po).split(".")[1]
        plot_po_map(po_direction, _name, arrays_with_frequency)
        plot_intensity(_x, s, 1.2 * ds.harmonic_frequencies.max())


if __name__ == "__main__":
    app()
