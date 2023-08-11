#! /usr/bin/env python3

from pathlib import Path
from typing import Tuple

import numpy as np
import typer
from rich import print as echo, panel
from ase import units
from ase.io import read
import pandas as pd
import xarray as xr

from tdeptools.hdf5 import read_grid_dispersion, read_dispersion_relations
from tdeptools.brillouin import get_special_points_cart, get_q_points_cart
from tdeptools.infrared import get_mode_resolved_BEC
from tdeptools.konstanter import lo_frequency_THz_to_icm
from tdeptools.helpers import Fix

_option = typer.Option
_check = "\u2713"
key_intensity_raman = "intensity_ir"

eps0 = 1.112650055e-10


def echo_check(msg, blank=False):
    if blank:
        echo()
    return echo(f" {_check} {msg}")


def read_dataset(file: str) -> xr.Dataset:
    """Read outfile.phonon_self_energy.hdf5 into one xr.Dataset"""
    ds = xr.load_dataset(file).rename({"q-point": "q_point"})
    ds_ha = xr.load_dataset(file, group="harmonic")
    ds_an = xr.load_dataset(file, group="anharmonic")
    ds_qm = xr.load_dataset(file, group="qmesh")

    return xr.merge([ds, ds_ha, ds_an, ds_qm])


_default_qdir = (None, None, None)

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    file: Path = "outfile.phonon_self_energy.hdf5",
    file_geometry: Path = "infile.ucposcar",
    file_bec: Path = "infile.lotosplitting",
    file_intensity: Path = "outfile.ir_intensity.csv",
    strength_threshold: float = 1e-5,
    verbose: bool = False,
    format_geometry: str = "vasp",
    decimals: int = _option(3, help="decimals for rounding the terminal output."),
):
    """Compute IR activity per mode outfile.phonon_self_energy.hdf5"""
    echo("Read input files:")

    if not file.exists():
        echo(f'You need the phonon_self_energy at Gamma! ("lineshape --qpoint 0 0 0")')

    # set decimals
    fix = Fix(decimals=decimals)

    echo_check(f"dispersion from       '{file}'")
    ds = read_dataset(file)

    echo_check(f"unit cell from        '{file_geometry}'")
    atoms = read(file_geometry, format=format_geometry)

    echo_check(f"BEC from              '{file_bec}'")
    born_charges = np.loadtxt(file_bec).reshape([-1, 3, 3])[1:]

    # incident wave vector
    qdir = ds.incident_wavevector.data
    qdir = fix(qdir / np.linalg.norm(qdir))  # just to be sure
    echo_check(f"incident q direction: {qdir}")

    # make sure eigenvectors are real
    assert np.linalg.norm(ds.eigenvectors_im) < 1e-12
    echo_check("eigenvectors are real")

    omegas = fix(ds.harmonic_frequencies.data)
    omegas_cm = fix(lo_frequency_THz_to_icm * omegas)
    ev_gamma = ds.eigenvectors_re.data

    # mode-resolved BEC
    Z_mode = get_mode_resolved_BEC(
        born_charges=born_charges, eigenvectors=ev_gamma, masses=atoms.get_masses()
    )
    Z_mode = fix(Z_mode)

    S_mode = np.zeros(3 * len(atoms))
    Zq_transverse = np.zeros(3 * len(atoms))
    Zq_longitudinal = np.zeros(3 * len(atoms))

    echo()
    echo(f"COMPUTE MODE-RESOLVED BORN CHARGES FOR {len(Z_mode)} MODES:")

    for ss, Zs in enumerate(Z_mode):
        echo()
        strength = np.linalg.norm(Zs)
        _w = omegas[ss]
        _w_cm = omegas_cm[ss]
        echo(f"Mode {ss:3d} w/ frequency {_w} THz ({_w_cm} cm^-1)")

        if strength < strength_threshold and not verbose:
            echo("... inactive")
            continue

        # project on qdir
        proj = qdir[:, None] * qdir[None, :] / (qdir @ qdir)
        Zq = fix(Zs @ qdir / np.linalg.norm(qdir))
        Z_longitudinal = fix(proj @ Zs)
        Z_transverse = fix(Zs - proj @ Zs)
        Z = np.linalg.norm(Z_transverse)

        # permittivity, compute in SI units
        Z_si = (Z / units.C) ** 2 * units.kg
        V_si = atoms.get_volume() * 1e-30

        if _w > 1e-9:
            w_si = _w * 1e12 * 2 * np.pi
            eps = 4 * np.pi * Z_si / V_si / w_si ** 2 / eps0  # -> atomic units
        else:
            eps = 0
        S_mode[ss] = eps

        echo(f"... Z                = {fix(Zs)}")
        echo(f"... Z^transverse     = {fix(Z_transverse)}")
        echo(f"... Z^longitudinal   = {fix(Z_longitudinal)}")
        echo(f"... Z.q              = {Zq}")
        echo(f"... |Z^longitudinal| = {np.linalg.norm(Z_longitudinal):6g}")
        echo(f"... |Z^transverse|   = {Z:6g}")
        echo(f"--> S                = {Z:6g}")
        echo(f"--> epsilon          = {eps:6g}")

        Zq_transverse[ss] = Z
        Zq_longitudinal[ss] = np.linalg.norm(Z_longitudinal)


    df_intensity = pd.DataFrame(
        {
            "frequency": omegas,
            "frequency_cm": omegas_cm,
            "Z_transverse": Zq_transverse,
            "Z_longitudinal": Zq_longitudinal,
            "strength": S_mode,
        },
        index=pd.Index(np.arange(len(omegas)), name="mode"),
    )

    echo()
    echo(f"RESULTS FOR LIGHT IN DIRECTION {qdir}:")
    echo(panel.Panel(df_intensity.to_string(), title=str(file_intensity), expand=False))

    echo(f'... save data to "{file_intensity}"')
    df_intensity.to_csv(file_intensity, index_label="mode")


if __name__ == "__main__":
    app()
