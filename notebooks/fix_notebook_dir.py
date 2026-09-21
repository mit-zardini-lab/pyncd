'''Make pyncd importable from a notebook under `notebooks/`.

The packages a notebook uses (`data_structure`, `display`, `graphs` and the rest)
live in the repository root and are imported by their top-level names. Nothing
puts that root on the import path on its own, so importing this module first does
two things. It inserts the repository root into `sys.path`, so
`import data_structure.Category` resolves. It then makes the repository root the
working directory, so a relative path such as `outputs/transformer.json` resolves.

Both are derived from this file's own location rather than from the current
directory, so they hold wherever the kernel or interpreter was started.

Assets referenced from a markdown cell are resolved by the notebook renderer
relative to the notebook file rather than by Python, so the change of directory
here does not affect them.

Within VS Code none of this is needed, because `.vscode/settings.json` sets
`jupyter.notebookFileRoot` to the workspace folder and every kernel therefore
starts at the repository root. This module covers the other cases: a notebook run
by `nbclient`, or one opened in a plain Jupyter server.

A notebook two levels down needs to name this file's folder before it can import
it:

    import sys; sys.path[:0] = ['.', 'notebooks']
    import fix_notebook_dir

The two relative entries cover a kernel started in `notebooks/` or in the
repository root. Both are removed at the end of this module, once the absolute
root is in place.
'''
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.chdir(REPO_ROOT)

# The notebooks add these to find this file; the absolute root above replaces them.
for _entry in ('.', 'notebooks'):
    while _entry in sys.path:
        sys.path.remove(_entry)
del _entry
