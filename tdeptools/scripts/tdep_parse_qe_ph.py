#! /usr/bin/env python3

import typer
from rich import print as echo
from tdeptools.scripts.ase_join_pw_ph import parse_ph_out, all_outputs
from pathlib import Path


def write_result(result: dict, outfile_lotosplitting: Path):
    """Write result to 'infile.lotosplitting'"""
    with open(outfile_lotosplitting, "w") as fd:

        for vec in result[all_outputs.dielectric_tensor]:
            fd.write(" ".join(f"{x:23.15e}" for x in vec) + "\n")

        for vec in result[all_outputs.born_effective_charges].reshape(-1, 3):
            fd.write(" ".join(f"{x:23.15e}" for x in vec) + "\n")


app = typer.Typer()


@app.command()
def main(
    file: Path,
    outfile_lotosplitting: Path = "infile.lotosplitting",
    verbose: bool = False,
    enforce_sum_rule: bool = False,
):
    """Parse BEC and dielectric tensor from QE ph.x output to 'infile.lotosplitting'"""

    echo(f"Parse '{file}'")
    result = parse_ph_out(file)
    bec = result[all_outputs.born_effective_charges]

    if verbose:
        echo("... sum rule violation:")
        echo(f"{bec.sum(axis=0)}")

    if enforce_sum_rule:
        if verbose:
            echo("... enforce sum rule")
        result[all_outputs.born_effective_charges] -= bec.mean(axis=0)

    write_result(result, outfile_lotosplitting=outfile_lotosplitting)

    echo(f".. data written to {outfile_lotosplitting}")


if __name__ == "__main__":
    app()
