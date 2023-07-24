#! /usr/bin/env python
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import typer

born_match = "BORN EFFECTIVE CHARGES (including local field effects)"
natoms_match = "number of ions"
stress_match = "FORCE on cell =-STRESS in cart. coord.  units (eV):"
forces_match = "TOTAL-FORCE (eV/Angst)"
timing_match = "Total CPU time used (sec):"
epsilon_match = "MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects "
dimension_match = "Dimension of arrays:"

app = typer.Typer()


@dataclass
class Data:
    """Container for the VASP data"""

    natoms: int = None
    stress: np.ndarray = None
    forces: np.ndarray = None
    epsilon: np.ndarray = None
    finished: bool = False
    born_charges: np.ndarray = None

    def write(self):
        """Write outfile.lotosplitting and potentially other stuff"""
        self.write_loto()

    def write_loto(self, file: str = "outfile.lotosplitting", fmt: str = "% 15.8f"):
        """Write outfile.lotosplitting"""
        if self.epsilon is None:
            typer.echo("NO DIELECTRIC TENSOR.")
            raise typer.Abort()
        if self.born_charges is None:
            typer.echo("NO DIELECTRIC TENSOR.")
            raise typer.Abort()

        repr = np.concatenate([self.epsilon[None, :], self.born_charges]).reshape(-1, 3)

        typer.echo(f".. write dielectric tensor and Born charges to {file}")
        np.savetxt(file, repr, fmt=fmt)


@app.command()
def main(file: Path, verbose: bool = False):
    typer.echo(f"Read VASP output from {file}")

    natoms_matched = False
    with file.open() as f:
        for line in f:
            # dimensions
            if natoms_match in line and not natoms_matched:
                natoms = int(line[line.rfind("NIONS =") :].split("=")[-1].strip())
                data = Data(natoms=natoms)
                natoms_matched = True

            # Forces
            if forces_match in line:
                typer.echo(".. parse forces:")
                data.forces = np.zeros([natoms, 3, 3])
                next(f)
                for ii in range(natoms):
                    data.forces[ii] = [float(x) for x in next(f).split()[3:6]]

                if verbose:
                    typer.echo(data.forces)

            # stress
            if stress_match in line:
                typer.echo(".. parse stress tensor:")
                stress = np.zeros([3, 3])
                for subline in f:
                    if "Total" in subline:
                        xx, yy, zz, xy, yx, zx = [float(x) for x in subline.split()[1:]]
                        stress[0, 0], stress[1, 1], stress[2, 2] = xx, yy, zz
                        stress[0, 1], stress[1, 2], stress[0, 2] = xy, yx, zx
                        stress[1, 0], stress[2, 1], stress[2, 0] = xy, yx, zx

                        typer.echo(stress)
                        data.stress = stress
                        break

            # dielectric tensor
            if epsilon_match in line:
                typer.echo(".. parse dielectric tensor:")
                data.epsilon = np.zeros([3, 3])
                next(f)  # skip one line
                for aa in range(3):
                    data.epsilon[aa] = [float(elem) for elem in next(f).split()]
                typer.echo(data.epsilon)

            # Born charges
            if born_match in line:
                typer.echo(".. parse Born charges")
                born_charges = np.zeros([natoms, 3, 3])
                next(f)  # skip one line
                for ii in range(natoms):
                    next(f)  # skip one line
                    for aa in range(3):
                        Za = [float(elem) for elem in next(f).split()[1:]]
                        born_charges[ii, aa] = Za

                diff = np.linalg.norm(born_charges.sum(axis=0))
                typer.echo(f".. dev. from charge neutrality: {diff:.2e}")
                data.born_charges = born_charges

            # is the calculation acutally finished?
            if timing_match in line:
                typer.echo(f".. calculation was finished after {line.split()[5]}s")
                data.finished = True

    if not data.finished:
        typer.echo("TIMINGS NOT FOUND, OUTPUT FILE BROKEN?")
        raise typer.Abort()

    typer.echo("Save data to file")
    data.write()


if __name__ == "__main__":
    app()
