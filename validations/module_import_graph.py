'''Which repository files each `.py` file and each notebook imports, directly and
transitively.

Written by Claude Opus 5 (1M context), effort high.

Every import statement in the repository is read with `ast` and resolved against the
folders that are on the path when the importing file runs. A dotted name resolves to
a module file or to a package `__init__.py`, and every existing prefix of the name
resolves as well, because importing `a.b.c` executes `a/__init__.py` and
`a/b/__init__.py` too. `from a.b import c` resolves `a.b.c` as a submodule as well as
`a.b`, because the statement reads as a submodule import whenever `c` is one.

A path is held as a string relative to the repository root with forward slashes, which
is the form `git status --porcelain` and `git diff --name-only` report, so a modified
path needs no conversion before it is looked up.

`obsidian/06-practice/Validation.md` states what the graph is used for.
'''

from __future__ import annotations

import ast
import io
import json
import os
import pathlib
import re
import warnings
from dataclasses import dataclass, field

type RepositoryPath = str

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]

FOLDERS_WITH_NO_IMPORTABLE_CODE = frozenset({
    '__pycache__', '.git', '.vscode', '.claude', '.ipynb_checkpoints', 'node_modules',
    'obsidian', 'outputs', '_guide',
})

INDEXED_TEXT_SUFFIXES = frozenset({'.py', '.ipynb', '.md', '.json', '.ts'})

CELL_MAGIC = re.compile(r'(?m)\A\s*%%')
LINE_MAGIC_OR_SHELL = re.compile(r'(?m)^(\s*)([!%?])')
IMPORT_STATEMENT = re.compile(
    r'(?m)^\s*(?:import\s+([A-Za-z_][\w.]*)'
    r'|from\s+([A-Za-z_.][\w.]*)\s+import\s+([^\n#]+))')


def repository_relative(path: pathlib.Path, root: pathlib.Path) -> RepositoryPath:
    return path.relative_to(root).as_posix()


def python_of_notebook(path: pathlib.Path) -> tuple[str, ...]:
    '''The source of each code cell, with the cell magics dropped and the line magics
    and shell lines commented out, so that `ast` can read what remains.'''
    notebook = json.loads(io.open(path, encoding='utf-8').read())
    cells = []
    for cell in notebook.get('cells', ()):
        if cell.get('cell_type') != 'code':
            continue
        source = ''.join(cell.get('source', ()))
        if CELL_MAGIC.match(source):
            continue
        cells.append(LINE_MAGIC_OR_SHELL.sub(r'\1#\2', source))
    return tuple(cells)


def python_of_module(path: pathlib.Path) -> tuple[str, ...]:
    return (io.open(path, encoding='utf-8').read(),)


def imported_module_names(
    sources: tuple[str, ...], containing_package: tuple[str, ...],
) -> frozenset[str]:
    '''Every dotted module name the sources import, with a relative import resolved
    against the package the importing file sits in.

    A source `ast` cannot read falls back to a regular expression over its import
    lines, so a notebook holding a cell that does not parse is still read for the
    modules it imports.
    '''
    names: set[str] = set()
    for source in sources:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', SyntaxWarning)
                tree = ast.parse(source)
        except SyntaxError:
            names |= module_names_by_regular_expression(source)
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                package = absolute_package_of(node, containing_package)
                if package is None:
                    continue
                names.add(package)
                names |= {f'{package}.{alias.name}' for alias in node.names
                          if alias.name != '*'}
    return frozenset(names)


def module_names_by_regular_expression(source: str) -> frozenset[str]:
    names: set[str] = set()
    for plain, package, imported in IMPORT_STATEMENT.findall(source):
        if plain:
            names.add(plain)
            continue
        names.add(package)
        for name in imported.split(','):
            bare = name.split(' as ')[0].strip().strip('()')
            if bare and bare != '*':
                names.add(f'{package}.{bare}')
    return frozenset(names)


def absolute_package_of(
    node: ast.ImportFrom, containing_package: tuple[str, ...],
) -> str | None:
    '''The dotted name `from ... import` reads from, or `None` when the relative
    level climbs above the repository root.'''
    if not node.level:
        return node.module
    climbed = len(containing_package) - (node.level - 1)
    if climbed < 0:
        return None
    parts = containing_package[:climbed]
    if node.module:
        parts = (*parts, *node.module.split('.'))
    return '.'.join(parts) or None


def search_folders_for(path: RepositoryPath) -> tuple[RepositoryPath, ...]:
    '''The folders on the path when the file at `path` runs, nearest first.

    A notebook and a script under `notebooks/` run with the repository root on the
    path and with their own folder on it, and `notebooks/fix_notebook_dir.py` puts
    `notebooks/` there as well, so a module in a notebook folder shadows a top-level
    package of the same name for every notebook in that folder.
    '''
    parts = path.split('/')
    if parts[0] != 'notebooks':
        return ('',)
    folder = '/'.join(parts[:-1])
    return (folder, 'notebooks', '')


def resolve_module_name(
    dotted: str, search_folders: tuple[RepositoryPath, ...],
    indexed_modules: frozenset[RepositoryPath],
) -> frozenset[RepositoryPath]:
    '''The repository files a dotted module name names, including every prefix
    package's `__init__.py`.'''
    parts = dotted.split('.')
    for folder in search_folders:
        prefix = f'{folder}/' if folder else ''
        found = {
            candidate
            for length in range(1, len(parts) + 1)
            for candidate in (f'{prefix}{"/".join(parts[:length])}.py',
                              f'{prefix}{"/".join(parts[:length])}/__init__.py')
            if candidate in indexed_modules}
        if found:
            return frozenset(found)
    return frozenset()


@dataclass
class ImportGraph:
    '''The import relation over the repository, with a transitive closure per file.'''

    direct_imports: dict[RepositoryPath, frozenset[RepositoryPath]]
    module_files: frozenset[RepositoryPath]
    notebook_files: frozenset[RepositoryPath]
    other_text_files: frozenset[RepositoryPath]
    _reached: dict[RepositoryPath, frozenset[RepositoryPath]] = field(
        default_factory=dict)

    @property
    def every_indexed_file(self) -> frozenset[RepositoryPath]:
        return self.module_files | self.notebook_files | self.other_text_files

    def files_reached_by(
        self, path: RepositoryPath,
    ) -> frozenset[RepositoryPath]:
        '''The file itself and every repository file it imports, at any depth.

        The walk is over nodes rather than paths, so a package imported from fifty
        modules is visited once, and an import cycle terminates.
        '''
        if path in self._reached:
            return self._reached[path]
        visited = {path}
        frontier = [path]
        while frontier:
            current = frontier.pop()
            for imported in self.direct_imports.get(current, frozenset()):
                if imported not in visited:
                    visited.add(imported)
                    frontier.append(imported)
        self._reached[path] = frozenset(visited)
        return self._reached[path]

    def imports_any_of(
        self, path: RepositoryPath, wanted: frozenset[RepositoryPath],
    ) -> bool:
        return bool(self.files_reached_by(path) & wanted)


def source_paths_under(root: pathlib.Path) -> tuple[RepositoryPath, ...]:
    '''Every indexed text file outside the skipped folders, sorted.'''
    paths = []
    for folder, subfolders, files in os.walk(root):
        subfolders[:] = [name for name in subfolders
                         if name not in FOLDERS_WITH_NO_IMPORTABLE_CODE]
        for file in files:
            if pathlib.Path(file).suffix in INDEXED_TEXT_SUFFIXES:
                paths.append(repository_relative(pathlib.Path(folder, file), root))
    return tuple(sorted(paths))


def vault_note_paths(root: pathlib.Path) -> tuple[RepositoryPath, ...]:
    '''Every note in the vault, which the `vault` check of `validate_repository`
    reads and which imports nothing.'''
    return tuple(sorted(
        repository_relative(path, root) for path in (root / 'obsidian').rglob('*.md')))


def build_import_graph(root: pathlib.Path = REPOSITORY_ROOT) -> ImportGraph:
    '''Parse every `.py` file and notebook under `root` and record what each
    imports.'''
    indexed = source_paths_under(root)
    modules = frozenset(path for path in indexed if path.endswith('.py'))
    notebooks = frozenset(path for path in indexed if path.endswith('.ipynb'))
    others = frozenset(indexed) - modules - notebooks | frozenset(
        vault_note_paths(root))
    direct: dict[RepositoryPath, frozenset[RepositoryPath]] = {}
    for path in sorted(modules | notebooks):
        full = root / path
        sources = (python_of_notebook(full) if path.endswith('.ipynb')
                   else python_of_module(full))
        containing_package = tuple(path.split('/')[:-1])
        search_folders = search_folders_for(path)
        reached: set[RepositoryPath] = set()
        for dotted in imported_module_names(sources, containing_package):
            reached |= resolve_module_name(dotted, search_folders, modules)
        direct[path] = frozenset(reached - {path})
    return ImportGraph(
        direct_imports=direct, module_files=modules, notebook_files=notebooks,
        other_text_files=others)
