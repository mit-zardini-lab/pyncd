'''Executing a notebook from the command line, with a listing in place of each diagram.

    python notebooks/execute_notebook.py notebooks/base_features/BuildingAModel.ipynb

The kernel starts at the repository root with `PYNCD_DIAGRAMS=listing` in its
environment, so `notebook_diagrams.show_diagram` prints the `agent_display` listing
in place of every diagram, whatever mode the notebook declares. By default the notebook
file is not written. The mode in the setup cell is preserved when outputs are saved.

The text output of each code cell is printed when the cell completes, and an image
is reported as omitted. A failing cell stops the run and prints its traceback.
`--output` saves the executed notebook, including rendered images, after a successful
run.

`--diagrams` names another mode, and `off` is the fastest run. `--kernel` names the
kernelspec to run under. By default the kernelspec that launches the interpreter
running this script is used, because that interpreter is one that runs the package.
'''
from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys
import time

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
DIAGRAM_MODE_VARIABLE = 'PYNCD_DIAGRAMS'
ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*m')


def kernelspec_for(interpreter: str) -> str:
    '''The kernelspec whose command launches `interpreter`, or `python3` when none
    does.'''
    import jupyter_client.kernelspec as kernelspec

    wanted = pathlib.Path(interpreter).resolve()
    for name, found in kernelspec.KernelSpecManager().get_all_specs().items():
        command = found['spec'].get('argv') or []
        if command and pathlib.Path(command[0]).resolve() == wanted:
            return name
    return 'python3'


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE.sub('', text)


def output_text(cell) -> str:
    '''The text a code cell's outputs carry, with each image reported as omitted.'''
    pieces: list[str] = []
    for output in cell.get('outputs', ()):
        match output.get('output_type'):
            case 'stream':
                pieces.append(output.get('text', ''))
            case 'execute_result' | 'display_data':
                data = output.get('data', {})
                if 'image/png' in data:
                    pieces.append('(image omitted)\n')
                elif 'text/plain' in data:
                    pieces.append(data['text/plain'].rstrip('\n') + '\n')
            case 'error':
                pieces.append('\n'.join(output.get('traceback', ())) + '\n')
    return strip_ansi(''.join(pieces))


def execute(
    notebook_path: pathlib.Path, mode: str, kernel: str, timeout: int,
    output_path: pathlib.Path | None = None,
) -> int:
    '''Run every code cell in order, printing each cell's output. Returns an exit
    code.'''
    import nbclient
    import nbformat

    notebook = nbformat.read(notebook_path, as_version=4)
    client = nbclient.NotebookClient(
        notebook, kernel_name=kernel, timeout=timeout,
        resources={'metadata': {'path': str(REPOSITORY_ROOT)}})
    environment = dict(os.environ, PYTHONUTF8='1', PYTHONPATH=str(REPOSITORY_ROOT))
    environment[DIAGRAM_MODE_VARIABLE] = mode
    started = time.perf_counter()
    client.reset_execution_trackers()
    with client.setup_kernel(env=environment):
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != 'code':
                continue
            cell_started = time.perf_counter()
            try:
                client.execute_cell(cell, index)
            except nbclient.exceptions.CellExecutionError:
                print(f'--- cell {index} failed ---')
                print(output_text(cell))
                return 1
            print(f'--- cell {index}, {time.perf_counter() - cell_started:.1f} s ---')
            print(output_text(cell), end='')
    elapsed = time.perf_counter() - started
    print(f'--- {notebook_path.name} executed in {elapsed:.0f} s ---')
    if output_path is not None:
        nbformat.write(notebook, output_path)
        print(f'Saved executed notebook to {output_path}')
    return 0


def main(argv: list[str] | None = None) -> int:
    sys.path.insert(0, str(REPOSITORY_ROOT))
    import notebooks.display.notebook_diagrams as notebook_diagrams

    parser = argparse.ArgumentParser(
        description='Execute a notebook with a listing in place of each diagram.')
    parser.add_argument('notebook', type=pathlib.Path)
    parser.add_argument(
        '--diagrams', choices=[mode.value for mode in notebook_diagrams.DiagramMode],
        default=notebook_diagrams.DiagramMode.LISTING.value,
        help='the mode every show_diagram call uses')
    parser.add_argument(
        '--kernel', default=kernelspec_for(sys.executable),
        help='the kernelspec to run under')
    parser.add_argument(
        '--timeout', type=int, default=900, help='seconds allowed per cell')
    parser.add_argument(
        '--output', type=pathlib.Path,
        help='save successful execution outputs to this notebook path')
    args = parser.parse_args(argv)
    return execute(args.notebook, args.diagrams, args.kernel, args.timeout, args.output)


if __name__ == '__main__':
    sys.exit(main())
