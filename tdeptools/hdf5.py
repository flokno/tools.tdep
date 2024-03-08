#! /usr/bin/env python
from collections import namedtuple
from pathlib import Path

import h5py as h5
import numpy as np
import rich
import xarray as xr

from tdeptools.dimensions import dimensions_phonon_self_energy


_arrays = ["q", "energy", "intensity", "ticks", "ticklabels", "unit"]
data_spectral_cls = namedtuple("spectral_data", _arrays)
data_dispersion_cls = namedtuple("dispersion_data", ["q", "energy"])

file_grid_dispersion = "outfile.grid_dispersions.hdf5"
outfile_grid_dispersion = "outfile.grid_dispersions.csv"

_nq, _nb, _nD = "nq", "nb", "nD"


def get_arrays(file: Path):
    """Read spectral data from h5 file and return as numpy arrays"""

    rich.print(f"Read spectral data from {file}")

    with h5.File(file, "r") as f:
        # get axes and intensity
        x = np.array(f.get("q_values"))
        y = np.array(f.get("energy_values"))
        try:
            gz = np.array(f["spectral_function"])
        except KeyError:
            gz = np.array(f["intensity"])  # compatibility with older sqe.hdf5 files

        xt = np.array(f.get("q_ticks"))
        xl = f.attrs.get("q_tick_labels").decode().split()
        yl = f"Energy ({f.attrs.get('energy_unit').decode():s})"

    return data_spectral_cls(x, y, gz, xt, xl, yl)


def get_arrays_dispersion(file: Path):
    """Read dispersion data from h5 file and return as numpy arrays"""

    rich.print(f"Read dispersion data from {file}")

    with h5.File(file, "r") as f:
        # get axes and intensity
        x = np.array(f.get("q_values"))
        y = np.array(f.get("frequencies"))

    return data_dispersion_cls(x, y)


def find_ylim(energy: np.ndarray, intensity: np.ndarray, max_frequency: float) -> float:
    """find ylim as fraction of full band occupation"""
    # integrate intensity in energy
    n_bands = np.trapz(intensity, x=energy, axis=0).mean()
    rich.print(f".. no. of bands:      {n_bands}")

    for nn, yy in enumerate(energy):
        gz_int = np.trapz(intensity[:nn], x=energy[:nn], axis=0).mean()
        if gz_int > max_frequency * n_bands:
            rich.print(f".. {max_frequency*100}% intensity at {yy:.3f} THz")
            return yy

    return energy.max()


def read_grid_dispersion(file: Path = file_grid_dispersion) -> xr.Dataset:
    """Read dispersion on grid from file and return as xr.Dataset with proper dim names

    Args:
        file: `outfile.grid_dispersion.hdf5` or similar

    Returns:
        Dataset: contains the dispersion information on a grid
    """
    dims_dict = {
        "phony_dim_0": _nq,  # number of q-points
        "phony_dim_1": _nb,  # number of bands
        "phony_dim_2": _nb,
        "phony_dim_3": _nD,  # number of Cart. dimensions
        "phony_dim_4": _nD,
    }
    ds = xr.load_dataset(file)
    ds = ds.rename_dims({key: dims_dict[key] for key in ds.dims})
    return ds


def read_dispersion_relations(file: Path = file_grid_dispersion) -> xr.Dataset:
    """Read dispersion on path from file and return as xr.Dataset with proper dim names

    Args:
        file: `outfile.dispersion_relations.hdf5` or similar

    Returns:
        Dataset: contains the dispersion information on a BZ path
    """
    ds = xr.load_dataset(file)
    # ds = ds.rename_dims({key: dims_dict[key] for key in ds.dims})
    ds = ds.rename_vars({"q_vector": "qpoints"})
    return ds


def read_dataset_phonon_self_energy(file: str) -> xr.Dataset:
    """Read outfile.phonon_self_energy.hdf5 into one xr.Dataset with proper dim names"""
    ds_in = xr.load_dataset(file).rename({"q-point": "q_point"})
    ds_ha = xr.load_dataset(file, group="harmonic")
    ds_an = xr.load_dataset(file, group="anharmonic")
    # temporarily rename frequency axis, make coordinate later
    ds_an = ds_an.rename({"frequency": "frequencies"})
    ds_qm = xr.load_dataset(file, group="qmesh")
    ds_st = xr.load_dataset(file, group="structure")

    datasets = (ds_in, ds_ha, ds_an, ds_qm, ds_st)

    # assign correct dimension names
    datasets_w_dimensions = []
    for _ds in datasets:
        # collect dimensions
        dict_dim = {}
        for var in _ds.data_vars:
            for ii, dim in enumerate(_ds[var].dims):
                dict_dim[dim] = dimensions_phonon_self_energy[var][ii]

        datasets_w_dimensions.append(_ds.rename_dims(dict_dim))

    ds = xr.merge(datasets_w_dimensions)

    # make frequency a coordinate
    ds = ds.assign_coords({"frequency": ds.frequencies})

    return ds
