#! /usr/bin/env python3
from pathlib import Path

import numpy as np
import typer
import xarray as xr
from ase.io import read
from rich import print as echo

from tdeptools.konstanter import lo_amu_to_emu, lo_bohr_to_A
from tdeptools.physics import freq2amplitude

default_symprec = 1e-10


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    file_uc: Path = "infile.ucposcar",
    file_self_energy: Path = "outfile.phonon_self_energy.hdf5",
    plusminus: bool = True,
    temperature: float = 0.0,
    ignore_acoustic: bool = True,
    frequency_tolerance: float = 1e-5,
    verbose: bool = False,
):
    """Create mode displacements in unitcell"""
    echo(f"Read '{file_uc}'")

    atoms = read(file_uc, format="vasp")
    masses = atoms.get_masses()

    ds_ha = xr.load_dataset(file_self_energy, group="harmonic")

    assert np.linalg.norm(ds_ha.eigenvectors_im) < 1e-9

    # real eigenvectors
    evs = ds_ha.eigenvectors_re.data

    # amplitudes
    omegas = ds_ha.harmonic_frequencies
    amplitudes = freq2amplitude(omegas, temperature=temperature)

    # resulting displacements in [N_mode, N_atoms, 3]
    masses_emu = np.sqrt(masses.repeat(3) * lo_amu_to_emu)
    dus = (evs / masses_emu[None, :]).reshape(-1, len(atoms), 3)
    # multiply in amplitudes
    dus *= amplitudes[:, None, None]
    # -> A
    dus *= lo_bohr_to_A

    if plusminus:
        signs = (1, -1)
    else:
        signs = (1,)

    for imode, du in enumerate(dus):
        if ignore_acoustic:
            if imode < 3:
                if omegas[imode] < frequency_tolerance:
                    echo(f"... mode {imode:03d}: acoustic, skip")
                    continue
                else:
                    echo(f"... mode {imode:03d}: supposed to beacoustic, but > tol")
                    raise ValueError
        for sign in signs:
            watoms = atoms.copy()
            watoms.positions += sign * du
            msd = np.linalg.norm(watoms.positions - atoms.positions) / len(atoms)
            rep = {1: "plus", -1: "minus"}[sign]
            outfile = f"outfile.ucposcar.mode.{imode:03d}.{rep}"
            echo(f"... mode {imode:03d}, MSD: {msd:.6f} (A) -> '{outfile}'")
            watoms.write(outfile, format="vasp", direct=True)


if __name__ == "__main__":
    app()
