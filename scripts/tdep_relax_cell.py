#! /usr/bin/env python3

import shutil
import subprocess as sp
from pathlib import Path

import numpy as np
import pandas as pd
import rich
import typer
import xarray as xr
from ase.io import read
from ase.stress import (
    full_3x3_to_voigt_6_stress,
    voigt_6_to_full_3x3_strain,
    voigt_6_to_full_3x3_stress,
)
from ase.units import GPa, kB


echo = rich.print


def _array_to_str(array, decimals=6):
    rep = ", ".join(f"{el:{decimals+3}.{decimals}f}" for el in array)
    return f"[ {rep} ]"


app = typer.Typer()


@app.command()
def main(
    temperature: float,
    pressure: float = 0,
    scale: float = 1.0,
    file_unitcell: str = typer.Option("infile.ucposcar", "--unitcell", "-uc"),
    file_supercell: str = typer.Option("infile.ssposcar", "--supercell", "-sc"),
    file_unitcell_init: str = "infile.ucposcar.init",
    file_compliances: str = "outfile.compliance_tensor",
    file_stress_dft: str = "outfile.stress_dft",
    file_stress_fc2: str = "outfile.stress_fc2",
    file_stress_fc3: str = "outfile.stress_fc3",
    file_stress_res: str = "outfile.stress_res",
    file_stress_qha: str = "outfile.stress_qha.csv",
    outfile_unitcell: str = "outfile.ucposcar.new_cell",
    outfile_supercell: str = "outfile.ssposcar.new_cell",
    outfile_deformation: str = "outfile.deformation",
    format: str = "vasp",
    decimals: int = 14,
):
    """Relax the cell degress of freedom

    Args:
        temperature: target temperature
        pressure: target pressure p_ext (in GPa)
        scale: scale the step with this factor
        file_unitcell: unitcell of the structure
        file_supercell: supercell of the structure
        file_unitcell_init: initial unitcell --> deformation tensor
        file_compliances: containts compliance tensors s_ij
        file_stress_dft: DFT stress p_DFT
        file_stress_fc2: stress from 2nd order FC (--> kinetic)
        file_stress_fc3: stress from 3rd order FC
        file_stress_res: residual stress p_DFT - p_FC
        file_stress_qha: analytic QHA stress vs. temperature
        outfile_deformation: write deformation tensor here
        format: for ase.io.read
        decimals: digits for rounding zeros

    Steps:
        1) get relaxation stress at target temperature
            p(T) = p_res(T) + p_qha (T) + p_ext
        2) predict strain that minimizes internal stress
            eps_i = - s_ij p_j
        3) update cell dof a with the strain transformation
            a' = a . (1 + eps)
    """
    echo("Start cell optimization")

    # read input files
    primitive = read(file_unitcell, format=format)
    supercell = read(file_supercell, format=format)
    if Path(file_unitcell_init).exists():
        primitive_initial = read(file_unitcell_init, format=format)
    else:
        primitive_initial = primitive.copy()

    s_ij = np.loadtxt(file_compliances).round(decimals=decimals)  # 6x6 Voigt matrix
    p_dft, p_dft_std, p_dft_err = np.loadtxt(file_stress_dft)  # 6x1 Voigt vector
    p_fc2, p_fc2_std, p_fc2_err = np.loadtxt(file_stress_fc2)  # 6x1 Voigt vector
    p_fc3, p_fc3_std, p_fc3_err = np.loadtxt(file_stress_fc3)  # 6x1 Voigt vector
    p_res, p_res_std, p_res_err = np.loadtxt(file_stress_res)  # 6x1 Voigt vector
    p_ext = pressure * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    # current deformation
    a0, an = primitive_initial.cell, primitive.cell
    Fn = np.linalg.solve(a0, an).round(decimals=decimals)
    # supercel deformation
    An = supercell.cell
    A0 = np.linalg.solve(Fn.T, An.T).T

    # read temperature-resolved QHA stress and interpolate to target temp via
    # cubic splines
    df_qha = pd.read_csv(file_stress_qha, index_col="temperature", comment="#")
    ds_qha = xr.DataArray(df_qha)
    p_qha = ds_qha.interp(temperature=temperature, method="cubic").data

    p_opt = p_res + p_qha - p_ext

    # apply deformation (PK1)
    iFnt = np.linalg.inv(Fn).T
    p_opt_def = full_3x3_to_voigt_6_stress(voigt_6_to_full_3x3_stress(p_opt) @ iFnt)

    na = len(primitive)
    vol = primitive.get_volume()

    echo(f"..            cell volume:  {primitive.get_volume():.3f} AA")
    echo(f"..     target temperature:  {temperature} K")
    echo(f".. ideal gas pressure NkT:  {na * kB * temperature / vol / GPa:.6f} GPa")

    echo()
    echo("Initial cell a_0:")
    for a in primitive_initial.cell:
        echo(28 * " " + _array_to_str(a))

    echo("Current cell a_n:")
    for a in primitive.cell:
        echo(28 * " " + _array_to_str(a))

    echo("Current deformation F_n:")
    for a in Fn:
        echo(28 * " " + _array_to_str(a))

    empty = 9 * " "
    echo(35 * " " + empty.join(["xx", "yy", "zz", "yz", "xz", "xy"]))
    echo(f"..           DFT pressure:  {_array_to_str(p_dft)} GPa")
    echo(f"..           FC2 pressure:  {_array_to_str(p_fc2)} GPa")
    echo(f"..           FC3 pressure:  {_array_to_str(p_fc3)} GPa")
    echo(f"..      residual pressure:  {_array_to_str(p_res)} GPa")
    echo(f"..     std. dev. pressure:  {_array_to_str(p_res_std)} GPa")
    echo(f"..     std. err. pressure:  {_array_to_str(p_res_err)} GPa")
    echo(f"..           QHA pressure:  {_array_to_str(p_qha)} GPa")
    echo(f"..      external pressure:  {_array_to_str(p_ext)} GPa")
    echo(f"-->       target pressure:  {_array_to_str(p_opt)} GPa")
    echo(f"-->     deformed pressure:  {_array_to_str(p_opt_def)} GPa")

    echo("..            compliances:")
    for s in s_ij:
        echo(28 * " " + _array_to_str(s))

    # predict strain and deformation
    eps = (-s_ij @ p_opt_def).round(decimals=decimals)
    # statistical error
    deps = (-s_ij @ p_res_err).round(decimals=decimals)

    echo()
    echo(f"-->      resulting strain:  {_array_to_str(eps)}")
    if scale != 1.0:
        echo(f"**      scale strain with:  {scale}")
        eps *= scale
        echo(f"-->         scaled strain:  {_array_to_str(eps)}")

    # fkdev: old naive deformation
    F = voigt_6_to_full_3x3_strain(eps)
    dF = voigt_6_to_full_3x3_strain(eps) - np.eye(3)
    dF_err = voigt_6_to_full_3x3_strain(deps) - np.eye(3)

    # deformation update
    F = Fn + dF

    echo("--> resulting deformation: ")
    for f in F:
        echo(28 * " " + _array_to_str(f))

    new_primitive = primitive.copy()
    _new_cell = (an + a0 @ dF).round(decimals=decimals)
    new_cell_err = (_new_cell @ dF_err).round(decimals=decimals)
    new_primitive.set_cell(_new_cell, scale_atoms=True)

    new_supercell = supercell.copy()
    _new_cell = (An + A0 @ dF).round(decimals=decimals)
    new_supercell.set_cell(_new_cell, scale_atoms=True)

    supercell_matrix_old = primitive.cell.reciprocal() @ supercell.cell
    supercell_matrix_new = new_primitive.cell.reciprocal() @ new_supercell.cell

    assert np.allclose(supercell_matrix_old, supercell_matrix_new), supercell_matrix_new

    dV = new_primitive.get_volume() / primitive.get_volume()

    echo("Cell before a_n:")
    for a in primitive.cell:
        echo(28 * " " + _array_to_str(a))

    echo("Cell after a_n+1:")
    for a in new_primitive.cell:
        echo(28 * " " + _array_to_str(a))

    echo("Cell change a_n+1 - a_n:")
    for a in new_primitive.cell - primitive.cell:
        echo(28 * " " + _array_to_str(a))

    echo("Cell error da_n+1:")
    for a in new_cell_err:
        echo(28 * " " + _array_to_str(a))

    echo(f"..            cell volume:  {new_primitive.get_volume():.3f} AA")
    echo(f"..       cell volume err.:  {np.linalg.det(new_cell_err):.6f} AA")
    echo(f"..          volume change:  {dV:.6f}")

    echo(f"..  write new unitcell to:  {outfile_unitcell}")
    new_primitive.write(outfile_unitcell, format=format, direct=True)
    echo(f".. write new supercell to:  {outfile_supercell}")
    new_supercell.write(outfile_supercell, format=format, direct=True)

    echo(f".. write deformation to {outfile_deformation}")
    np.savetxt(outfile_deformation, F)

    echo("Done.")


if __name__ == "__main__":
    app()
