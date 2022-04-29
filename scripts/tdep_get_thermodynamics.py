#! /usr/bin/env python
from pathlib import Path

import numpy as np
import pandas as pd
import rich
import typer
import xarray as xr
from ase.io import read

from tdeptools import konstanter as k
from tdeptools.hdf5 import (
    file_grid_dispersion,
    outfile_grid_dispersion,
    read_grid_dispersion,
)
from tdeptools.helpers import from_voigt
from tdeptools.thermodynamics import (
    get_free_energy,
    get_heat_capacity,
    get_mode_heat_capacity,
    get_pressure_volume,
    get_stress_volume,
)


_dim_temperature = "temperature"
_dim_stress = "r"
_dim_temperature_stress = (_dim_temperature, _dim_stress)
_dim_tensor_voigt = ("r", "s")


echo = rich.print

app = typer.Typer()


@app.command()
def main(
    file: Path = file_grid_dispersion,
    compliance: bool = False,
    ntemp: int = 101,
    mintemp: float = 0,
    maxtemp: float = 1000,
    outfile: Path = outfile_grid_dispersion,
    file_compliance: Path = "outfile.compliance_tensor",
    digits: int = 4,
    float_format: str = "%15.8e",
):
    typer.echo(f"Read dispersion on grid from {file}")

    np.set_printoptions(precision=digits)

    ds = read_grid_dispersion(file)

    echo(ds)

    n_atoms = ds.number_of_atoms

    try:
        Nq = ds.number_of_qpoints
        volume = ds.volume
    except AttributeError:
        volume = read("infile.ucposcar", format="vasp").get_volume()

    # sanity check gruneisen
    g = ds.gruneisen_parameters
    echo(f".. average Grueneisen parameter: {np.mean(g.data).round(decimals=14):.3f}")

    if "gruneisen_tensors" in ds:
        g_ab = ds.gruneisen_tensors
        diff = np.linalg.norm(g - 1 / 3 * np.trace(g_ab, axis1=-1, axis2=-2))
        echo(".. average Grueneisen tensor:    ")
        g_ab_mean = np.mean(g_ab.data, axis=(0, 1)).round(decimals=14)
        echo(g_ab_mean)
        echo(f".. difference between Grueneisen parameters and tensors: {diff:.2e}")
        echo(g_ab.shape)

    temperatures = np.linspace(mintemp, maxtemp, ntemp)

    # energies in eV
    energies = ds.frequencies * k.lo_frequency_Hz_to_eV

    cv = 3 * k.lo_kb_eV * n_atoms
    c_T = get_heat_capacity(energies, temperatures)
    cs_T = get_mode_heat_capacity(energies, temperatures)
    F_T = get_free_energy(energies, temperatures)

    # mode gruneisen per temp
    denom = cs_T.mean(axis=(0, 1))
    denom[denom < k.lo_tiny] = k.lo_huge
    g_T = (np.array(g)[:, :, None] * cs_T).mean(axis=(0, 1)) / denom

    # quasiharmonic pressure
    pV_ha_T = get_pressure_volume(energies, ds.gruneisen_parameters, temperatures)

    # quasiharmonic stress
    pV_ab_ha_T = get_stress_volume(energies, ds.gruneisen_tensors, temperatures)

    # sanity check
    assert np.allclose(pV_ha_T, np.mean(-pV_ab_ha_T[:, :3], axis=-1))

    index = pd.Index(data=temperatures, name="T [K]")
    data = {
        "F_ha(T) [eV]": F_T,
        "c_V(T) [meV]": 1e3 * c_T,
        "pV(T) [eV]": pV_ha_T,
        "c_V/c_V_cl(T) [1]": c_T / cv,
        "g(T) [1]": g_T,
    }

    p_ha_T = pV_ha_T / volume
    data["p_qha(T) [GPa]"] = p_ha_T

    # create dataset
    echo("Dump data")
    df = pd.DataFrame(data, index=index)

    echo("Some results:")
    pd.options.display.max_rows = len(df)
    pd.options.display.max_columns = 0  # automatic detection
    echo(df.head())

    echo(f".. save to {outfile}")
    df.to_csv(outfile, float_format=float_format)

    # qha stress for lattice optimization
    echo("Dump stress")
    index = pd.Index(data=temperatures, name="temperature")
    data = {
        "xx": pV_ab_ha_T[:, 0] / volume,
        "yy": pV_ab_ha_T[:, 1] / volume,
        "zz": pV_ab_ha_T[:, 2] / volume,
        "yz": pV_ab_ha_T[:, 3] / volume,
        "xz": pV_ab_ha_T[:, 4] / volume,
        "xy": pV_ab_ha_T[:, 5] / volume,
    }
    df = pd.DataFrame(data, index=index)

    outfile = "outfile.stress_qha.csv"
    echo(f".. save to {outfile}")
    df.to_csv(outfile, float_format=float_format)

    # # compliances
    # if compliance:
    #     s_ab = np.loadtxt(file_compliance)

    #     echo("Compliance tensor S_ab:")
    #     echo(s_ab)

    #     # compressibility and bulk modulus
    #     K0 = np.sum(np.diag(s_ab)[:3])
    #     K0 += 2.0 * (s_ab[0, 1] + s_ab[1, 2] + s_ab[2, 0])
    #     B0 = 1 / K0

    #     echo(f".. Compressibility: {K0:8.3f} GPa**-1")
    #     echo(f".. Bulk modulus :   {B0:8.3f} GPa")

    #     # add volume change
    #     dV = pV_ha_T * K0
    #     data["dV(T) [AA**3]"] = dV

    #     # nonisotropic volume change:
    #     _ = None
    #     dV_r = pV_ab_ha_T @ s_ab  # [T, r, s] -> [T, r]

    #     # assert np.allclose(dV, np.sum(dV_r[:, :3], axis=1))

    #     # TODO:
    #     # - [ ] write as strain = compliance * stress
    #     # - [ ] generalize to actual non-diagonal stress

    #     volumes = volume + dV  # = (1 + e) .V
    #     data["V(T) [AA**3]"] = volumes
    #     data["dV(T) [AA**3]"] = dV

    #     # volume expansion:
    #     # alpha = 1/V dV/dT
    #     # numeric:
    #     # alpha = 1 / volume * np.gradient(volumes, temperatures)
    #     # analytic (See e.g. Eq. 2 in Biernacki)
    #     alpha = (np.array(g)[:, :, None] * cs_T).sum(axis=(0, 1)) / Nq
    #     alpha *= K0 * k.lo_pressure_eVA_to_GPa / volume
    #     data["a_V(T) [K**-1]"] = alpha
    #     data["a_L(T) [K**-1]"] = alpha / 3

    #     data = {
    #         "free_energy": (_dim_temperature, F_T),
    #         "heat_capacity_volume": (_dim_temperature, c_T),
    #         "pressure": (_dim_temperature, p_ha_T),
    #         "stress": (_dim_temperature_stress, pV_ab_ha_T / volume),
    #         "volume_isotropic": (_dim_temperature, volumes),
    #         "volume_isotropic_change": (_dim_temperature, dV),
    #         "volume_nonisotropoic_change": (_dim_temperature_stress, dV_r),
    #         "expansion_coefficient_volume": (_dim_temperature, alpha),
    #         "expansion_coefficient_linear": (_dim_temperature, alpha / 3),
    #         "compliance_tensor": (_dim_tensor_voigt, s_ab),
    #         "gruneisen_parameter": (_dim_temperature, g_T),
    #     }

    #     echo(F_T.shape)
    #     echo(temperatures.shape)

    #     attrs = {"unit_pressure": "GPa", "unit_energy": "eV", "unit_temperature": "K"}
    #     coords = {_dim_temperature: temperatures}

    #     ds = xr.Dataset(data, coords=coords, attrs=attrs)
    #     echo(ds)
    #     ds.to_netcdf("outfile.thermodynamics.nc")


if __name__ == "__main__":
    app()
