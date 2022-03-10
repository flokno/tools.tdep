#! /usr/bin/env python
from collections import namedtuple
from pathlib import Path

import h5py as h5
import numpy as np
import rich
from rich import traceback

traceback.install(show_locals=True)

_arrays = ["q", "energy", "intensity", "ticks", "ticklabels", "unit"]
data_spectral_cls = namedtuple("spectral_data", _arrays)
data_dispersion_cls = namedtuple("dispersion_data", ["q", "energy"])


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
