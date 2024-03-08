#! /usr/bin/env python3

import json
import shlex
import subprocess as sp
from pathlib import Path
from typing import List

import typer
from rich import print as echo


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    files: List[Path],
    outfile: Path = "tdep_dispersions.pdf",
    outfile_prefix: str = "tmp_",
    outfile_suffix: str = "pdf",
):
    _outfiles = []
    for ii, file in enumerate(files):
        echo(file)
        # file_name = file.parts[-1]
        folder = file.parts[-2]

        tag = str(folder)

        echo("... Tag:")
        echo(tag)

        _outfile = f"{outfile_prefix}{ii:03d}.{outfile_suffix}"

        cmd = f'echo "{tag}"'
        ps1 = sp.Popen(shlex.split(cmd), stdout=sp.PIPE)
        cmd = "enscript -B -f Courier-Bold32 -o-"
        ps2 = sp.Popen(shlex.split(cmd), stdin=ps1.stdout, stdout=sp.PIPE)
        cmd = f'ps2pdf - "{_outfile}" '
        sp.run(shlex.split(cmd), stdin=ps2.stdout)
        ps1.wait()
        ps2.wait()

        cmd = f'qpdf {file} --overlay "{_outfile}" -- {_outfile}'
        sp.run(shlex.split(cmd))

        _outfiles.append(_outfile)

    _cmd = "pdfjam "
    for _outfile in _outfiles:
        _cmd += f"{_outfile} '-' "
    _cmd += "--landscape "
    _cmd += f"--outfile {outfile}"

    # check if files actually exist
    for _file in _outfiles:
        assert Path(_file).exists(), f"{_file} does not exist, arguments are folders!"

    echo("... run:")
    echo(_cmd)
    sp.run(shlex.split(_cmd))

    # cleanup
    for _file in _outfiles:
        Path(_file).unlink()

    echo("done.")


if __name__ == "__main__":
    app()
