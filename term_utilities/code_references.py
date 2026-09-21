# Claude Fable 5.1, effort 80.
'''Writing the place in a codebase a block stands for as a `cat.CodeReference`.

A block that expresses a module of a reference implementation carries references to
that module, as pinned links, and to the function in this repository that builds it.
`source_of` reads the file and the line range of a function or a module through
`inspect`, relative to the root of this repository, so a block's aesthetics can carry
where it was written without the path being typed by hand. The display links the
reference under the base url a notebook gives it, per
`notebooks/display/advanced_display.py`, and shows it as text where it gives none.

`pinned_link` writes a reference to a file at one commit of a repository on a web host,
with the line fragment the host reads, so a table of citations in a notebook and the
references on the blocks it describes are written from one function.
'''
from __future__ import annotations

import inspect
import pathlib
import types
from collections.abc import Callable

import data_structure.Category as cat


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parent.parent


def source_of(
    target: Callable[..., object] | types.ModuleType, label: str = 'pyncd',
) -> cat.CodeReference:
    '''The file and lines of `target` in this repository. A module reference names
    the file alone, and a function reference names the lines its definition spans.'''
    file = inspect.getsourcefile(target)
    if file is None:
        raise ValueError(f'{target!r} has no source file')
    path = pathlib.Path(file).resolve()
    try:
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        relative = path.as_posix()
    if isinstance(target, types.ModuleType):
        return cat.CodeReference(label=label, path=relative)
    lines, first = inspect.getsourcelines(target)
    return cat.CodeReference(
        label=label, path=relative, line=first, end_line=first + len(lines) - 1)


def pinned_link(
    base_url: str, path: str, line: int | None = None, end_line: int | None = None,
    label: str | None = None,
) -> cat.CodeReference:
    '''A reference to `path` under `base_url`, which names one commit of a repository
    on a web host, with `#L<line>` or `#L<line>-L<end_line>` appended in the form
    GitHub and Hugging Face read. The label defaults to the path with its lines.'''
    base = base_url if base_url.endswith('/') else base_url + '/'
    url = base + path
    if line is not None:
        url += f'#L{line}' if end_line is None or end_line == line else f'#L{line}-L{end_line}'
    if label is None:
        label = path if line is None else (
            f'{path} L{line}' if end_line is None or end_line == line
            else f'{path} L{line}-L{end_line}')
    return cat.CodeReference(label=label, url=url, path=path, line=line, end_line=end_line)
