#! /usr/bin/env python3

from pathlib import Path
from typing import List

import numpy as np
import typer
from ase import units
from ase.io import read

outfile_meta = "infile.meta"
outfile_stat = "infile.stat"
outfile_forces = "infile.forces"
outfile_positions = "infile.positions"

app = typer.Typer()


@app.command()
def main(files: List[Path], timestep: float = 1.0, format: str = "aims-output"):
    typer.echo(files)

    typer.echo(f"Parse {len(files)} file(s)")

    with open(outfile_forces, "w") as ff, open(outfile_positions, "w") as fp, open(
        outfile_stat, "w"
    ) as fs:
        for ii, file in enumerate(files):
            typer.echo(f".. parse file {ii+1:3d}: {str(file)}")
            atoms = read(file, format=format)

            positions = atoms.get_scaled_positions()
            forces = atoms.get_forces()
            energy_potential = atoms.get_potential_energy()
            energy_kinetic = atoms.get_kinetic_energy()
            energy_total = energy_potential + energy_kinetic
            temperature = atoms.get_temperature()
            if "stress" in atoms.calc.results:
                stress = atoms.get_stress() / units.GPa
            else:
                stress = np.zeros(6)
            pressure = np.mean(stress[:3])

            for (pos, force) in zip(positions, forces):
                (px, py, pz) = pos
                (fx, fy, fz) = force
                fp.write(f"{px:23.15e} {py:23.15e} {pz:23.15e}\n")
                ff.write(f"{fx:23.15e} {fy:23.15e} {fz:23.15e}\n")

            # shorthands
            dt = timestep
            et, ep, ek = energy_total, energy_potential, energy_kinetic
            t, p, sx, sy, sz, sxz, syz, sxy = temperature, pressure, *stress
            fs.write(f"{ii+1:7d} {ii*dt:9.3f} {et:23.15e} {ep:23.15e} {ek:23.15e} ")
            fs.write(f"{t:9.3f} {p:9.3f} ")
            fs.write(f"{sx:9.3f} {sy:9.3f} {sz:9.3f} {sxz:9.3f} {syz:9.3f} {sxy:9.3f}")
            fs.write("\n")

        n_steps = len(files)
        n_atoms = len(atoms)

    typer.echo(f".. forces written to {outfile_forces}")
    typer.echo(f".. positions written to {outfile_positions}")
    typer.echo(f".. statistics written to {outfile_stat}")

    # Store Meta info:
    with open(outfile_meta, "w") as f:
        f.write(f"{n_atoms}     # N atoms\n")
        f.write(f"{n_steps}     # N timesteps\n")
        f.write(f"{1.0}         # timestep in fs (currently not used )\n")
        f.write(f"{30000}       # temperature in K (currently not used)\n")
    typer.echo(f".. meta info written to {outfile_meta}")


if __name__ == "__main__":
    app()
