'''Make pyncd importable from the examples in this folder.

The packages these examples use -- `data_structure`, `display`, `graphs` and the
rest -- live in the repository root, one level above this folder, and are imported
by their top-level names. Nothing puts that root on the import path on its own, so
importing this module first does two things:

  * inserts the repository root into `sys.path`, so `import data_structure.Category`
    and friends resolve;
  * makes the repository root the working directory, so relative paths such as
    `outputs/transformer.json` resolve exactly as they did when these examples
    lived in the root.

Both are derived from this file's own location rather than from the current
directory, so they hold no matter where the kernel or interpreter was started.

Note that assets referenced from markdown -- the figures in `Results.ipynb`, say --
are resolved by the notebook renderer relative to the notebook file, not by Python,
so those paths are written as `../_guide/...` and are unaffected by the chdir here.

A script sitting in this folder can simply `import fix_notebook_dir`, because Python
puts the script's own directory on `sys.path`. A notebook needs a little more, since
a kernel started in the repository root would not find this file:

    import sys; sys.path[:0] = ['.', 'example_notebooks']
    import fix_notebook_dir

Those two relative entries cover the kernel starting either in this folder or in
the repository root, and are removed at the end of this module once the absolute
root is in place.
'''
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.chdir(REPO_ROOT)

# The notebooks add these to find this file; the absolute root above replaces them.
for _entry in ('.', 'example_notebooks'):
    while _entry in sys.path:
        sys.path.remove(_entry)
del _entry
