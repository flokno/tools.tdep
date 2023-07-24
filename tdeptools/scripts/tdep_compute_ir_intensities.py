#! /usr/bin/env python3

from pathlib import Path

import numpy as np
import typer
from rich import print as echo
from ase.io import read
import pandas as pd

from tdeptools.hdf5 import read_grid_dispersion
from tdeptools.brillouin import get_special_points_cart
from tdeptools.infrared import get_oscillator_complex
from tdeptools.konstanter import lo_frequency_THz_to_icm

_option = typer.Option
_check = "\u2713"
key_intensity_raman = "intensity_ir"


def echo_check(msg):
    return echo(f" {_check} {msg}")


def fix(array: np.ndarray, decimals: int = 3):
    """fix the array: round and eliminate -0"""
    tmp = np.around(array, decimals=decimals)
    tmp += 1e-2 ** decimals
    return tmp.round(decimals=decimals)


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    file_dispersion: Path = "outfile.grid_dispersions.hdf5",
    file_geometry: Path = "infile.ucposcar",
    file_bec: Path = "infile.lotosplitting",
    file_intensity: Path = "ir_intensity.csv",
    strength_threshold: float = 1e-5,
    broadening: float = _option(0.05, help="artifical broadening in cm^-1"),
    verbose: bool = False,
    format_geometry: str = "vasp",
    decimals: int = _option(3, help="decimals for rounding the terminal output."),
):
    """Compute IR activity per mode"""
    echo("Read input files:")
    echo_check(f"dispersion from {file_dispersion}")
    ds = read_grid_dispersion(file_dispersion)

    echo_check(f"unit cell from {file_geometry}")
    atoms = read(file_geometry, format=format_geometry)

    echo_check(f"BEC from {file_bec}")
    born_charges = np.loadtxt(file_bec).reshape([-1, 3, 3])[1:]

    # availabe points close to gamma:
    indices_gamma = np.where(np.linalg.norm(ds.qpoints, axis=1) < 1e-9)
    echo(f"... available q-points close to Gamma: {indices_gamma}")

    index_gamma = int(indices_gamma[0])
    echo_check(f"Gamma point has index {index_gamma} on current grid")

    # make sure eigenvectors are real
    assert np.linalg.norm(ds.eigenvectors_im[index_gamma]) < 1e-12
    echo_check("eigenvectors are real")

    omegas = ds.frequencies.data[index_gamma] * 1e-12 / 2 / np.pi
    ev_gamma = ds.eigenvectors_re.data[index_gamma]

    # show bandpaths
    # echo(atoms.cell.bandpath())

    # get and report special points
    special_points_cart = get_special_points_cart(atoms)
    echo(".. Special points (cart.):")
    echo(special_points_cart)

    # mode-resolved BEC
    masses = atoms.get_masses()
    Z_mode = np.zeros([ds.number_of_bands, 3])

    for nn, ev in enumerate(ev_gamma):
        for ii in range(ds.number_of_atoms):
            ev_i = ev[3 * ii : 3 * ii + 3]
            m_i = masses[ii]
            Z_i = born_charges[ii]

            Z_s = Z_i @ ev_i / m_i ** 0.5

            Z_mode[nn] += Z_s

    oscillator_strength = Z_mode.conj()[:, :, None] * Z_mode[:, None, :]

    # figure out things for the spectra
    # 1. max. frequency:
    xmax = 3 * omegas.max()
    echo(f"... max. eigenfrequency:           {omegas.max():10.3f} (THz)")
    echo(f"... max. frequency for oscillator: {xmax:10.3f} (THz)")
    gamma = broadening
    xx, _ = get_oscillator_complex(1, 1, xmax=xmax)

    intensity_isotropic = np.zeros_like(xx, dtype=complex)

    for nn, (Z, S) in enumerate(zip(Z_mode, oscillator_strength)):
        strength = np.linalg.norm(S)
        echo()
        _w = fix(omegas, decimals=decimals)[nn]
        _w_cm = fix(lo_frequency_THz_to_icm * omegas, decimals=decimals)[nn]
        echo(f"Mode {nn:3d} w/ frequency {_w} THz ({_w_cm} cm^-1)")
        echo(f"Z = {fix(Z, decimals=decimals)}")
        echo(f"S = {strength:5g}")

        if strength < strength_threshold and not verbose:
            echo(".. inactive")
            continue

        if verbose:
            echo("S:")
            echo(f"{fix(S ,decimals=decimals)}")

        # compute spectrum
        _, yy = get_oscillator_complex(omegas[nn], gamma=gamma, xmax=xmax)
        intensity_isotropic -= strength * yy

        for qkey, qvec in special_points_cart.items():
            if qkey == "G":
                continue
            proj = qvec[:, None] * qvec[None, :] / (qvec @ qvec)
            # Zq = Z @ qvec / np.linalg.norm(qvec)
            Z_longitudinal = proj @ Z
            Z_transverse = Z - proj @ Z
            rep = str(fix(qvec, decimals=decimals))
            echo(f".. {qkey}: {rep:20s}")  # , --> Z.q = {fix(Zq)}")
            # echo(f".. |Z.q|            = {fix(Zq)}")
            echo(f".... |Z^longitudinal| = {np.linalg.norm(Z_longitudinal):6g}")
            echo(f".... |Z^transverse|   = {np.linalg.norm(Z_transverse):6g}")

    df_intensity = pd.DataFrame(
        {"chi.real": intensity_isotropic.real, "chi.imag": intensity_isotropic.imag},
        index=pd.Index(xx, name="frequency"),
    )

    echo(df_intensity.head())

    df_intensity.to_csv(file_intensity)


if __name__ == "__main__":
    app()
