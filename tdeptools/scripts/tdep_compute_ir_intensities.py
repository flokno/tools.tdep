#! /usr/bin/env python3

from pathlib import Path
from typing import Tuple

import numpy as np
import typer
from rich import print as echo
from ase.io import read
import pandas as pd

from tdeptools.hdf5 import read_grid_dispersion, read_dispersion_relations
from tdeptools.brillouin import get_special_points_cart, get_q_points_cart
from tdeptools.infrared import get_mode_resolved_BEC
from tdeptools.konstanter import lo_frequency_THz_to_icm
from tdeptools.helpers import Fix

_option = typer.Option
_check = "\u2713"
key_intensity_raman = "intensity_ir"


def echo_check(msg, blank=False):
    if blank:
        echo()
    return echo(f" {_check} {msg}")


def get_qdir(ds, index_gamma, cell_reciprocal=None):
    qg = ds.qpoints.data[index_gamma]
    qp = ds.qpoints.data[index_gamma + 1]
    qm = ds.qpoints.data[index_gamma - 1]
    if np.linalg.norm(qp) > 1e-9:
        qdir = qp - qg
    elif np.linalg.norm(qm) > 1e-9:
        qdir = qm - qg
    else:
        raise ValueError(f"q direction could not be found")

    if cell_reciprocal is not None:
        qdir = get_q_points_cart(qdir, cell_reciprocal=cell_reciprocal)

    qdir /= np.linalg.norm(qdir)
    return qdir


_default_qdir = (None, None, None)

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    file_dispersion: Path,
    file_geometry: Path = "infile.ucposcar",
    file_bec: Path = "infile.lotosplitting",
    file_intensity: Path = "outfile.ir_intensity.csv",
    iq: int = 0,
    qdir: Tuple[float, float, float] = _default_qdir,
    strength_threshold: float = 1e-5,
    verbose: bool = False,
    format_geometry: str = "vasp",
    decimals: int = _option(3, help="decimals for rounding the terminal output."),
):
    """Compute IR activity per mode"""
    echo("Read input files:")
    echo_check(f"dispersion from {file_dispersion}")

    # set decimals
    fix = Fix(decimals=decimals)

    if "grid_dispersion" in file_dispersion.name:
        ds = read_grid_dispersion(file_dispersion)
        ds.frequencies.data *= 1e-12 / 2 / np.pi  # to THz
    elif "dispersion_relations" in file_dispersion.name:
        ds = read_dispersion_relations(file_dispersion)

    echo_check(f"unit cell from {file_geometry}")
    atoms = read(file_geometry, format=format_geometry)

    # get and report special points
    special_points_cart = get_special_points_cart(atoms)
    echo("... Sepcial points (frac.)")
    echo(atoms.cell.bandpath().special_points)
    echo("... Special points (cart.):")
    echo(special_points_cart)

    echo_check(f"BEC from {file_bec}")
    born_charges = np.loadtxt(file_bec).reshape([-1, 3, 3])[1:]
    echo()

    # availabe points close to gamma:
    indices_gamma = np.where(np.linalg.norm(ds.qpoints, axis=1) < 1e-9)[0]
    echo(f"... available q-points close to Gamma: {indices_gamma}")
    for ii, idx in enumerate(indices_gamma):
        _q = ds.qpoints.data[idx]
        # _qdir = get_qdir(ds, idx, cell_reciprocal=atoms.cell.reciprocal())
        _qdir = get_qdir(ds, idx)  # , cell_reciprocal=atoms.cell.reciprocal())
        echo(f"--> {ii:2d} ({idx:3d}): qdir =  {_qdir}")

    index_gamma = int(indices_gamma[iq])
    if None in qdir:
        qdir = get_qdir(ds, index_gamma)  # , atoms.cell.reciprocal())
        echo_check(f"choose {iq:2d} ({index_gamma:3d}) w/ qdir =  {qdir}")
        echo_check(f"use qdir =  {qdir} (default from qpoint)", blank=True)
    else:
        qdir = np.array(qdir)
        echo_check(f"choose {iq:2d} ({index_gamma:3d})use qdir =  {qdir}")
        echo_check(f"use qdir =  {qdir} (given by user)", blank=True)

    # make sure eigenvectors are real
    assert np.linalg.norm(ds.eigenvectors_im[index_gamma]) < 1e-12
    echo_check("eigenvectors are real", blank=True)

    omegas = fix(ds.frequencies.data[index_gamma])
    omegas_cm = fix(lo_frequency_THz_to_icm * omegas)
    ev_gamma = ds.eigenvectors_re.data[index_gamma]

    # mode-resolved BEC
    Z_mode = get_mode_resolved_BEC(
        born_charges=born_charges, eigenvectors=ev_gamma, masses=atoms.get_masses()
    )
    Z_mode = fix(Z_mode)
    oscillator_strength = Z_mode.conj()[:, :, None] * Z_mode[:, None, :]

    ir_intensites = np.zeros(3 * len(atoms))
    Zq_transverse = np.zeros(3 * len(atoms))
    Zq_longitudinal = np.zeros(3 * len(atoms))

    for ss, (Z, S) in enumerate(zip(Z_mode, oscillator_strength)):
        strength = np.linalg.norm(S)
        echo()
        _w = omegas[ss]
        _w_cm = omegas_cm[ss]
        echo(f"Mode {ss:3d} w/ frequency {_w} THz ({_w_cm} cm^-1)")
        echo(f"... Z                = {fix(Z)}")
        echo(f"... |S|              = {strength:6g}")

        if strength < strength_threshold and not verbose:
            echo("... inactive")
            continue

        # project on qdir
        proj = qdir[:, None] * qdir[None, :] / (qdir @ qdir)
        Zq = fix(Z @ qdir / np.linalg.norm(qdir))
        Z_longitudinal = fix(proj @ Z)
        Z_transverse = fix(Z - proj @ Z)
        S_transverse = fix(S - proj @ S @ proj)
        echo(f"... |Z.q|            = {Zq}")
        echo(f"... |Z^longitudinal| = {np.linalg.norm(Z_longitudinal):6g}")
        echo(f"... |Z^transverse|   = {np.linalg.norm(Z_transverse):6g}")

        if verbose:
            echo("S:")
            echo(f"{fix(S)}")
            echo(f"Z^transverse: {Z_transverse}")
            echo("S^transverse")
            echo(f"{S_transverse}")

        Zq_transverse[ss] = np.linalg.norm(Z_transverse)
        Zq_longitudinal[ss] = np.linalg.norm(Z_longitudinal)
        ir_intensites = Zq_transverse * 4 * np.pi / atoms.get_volume() * omegas ** 2

    df_intensity = pd.DataFrame(
        {
            "frequency": omegas,
            "frequency_cm": omegas_cm,
            "Z_transverse": Zq_transverse,
            "Z_longitudinal": Zq_longitudinal,
            "ir_intensity": ir_intensites,
        },
        index=pd.Index(np.arange(len(omegas)), name="mode"),
    )

    echo(df_intensity)

    echo(f'... save data to "{file_intensity}"')
    df_intensity.to_csv(file_intensity, index_label="mode")


if __name__ == "__main__":
    app()
