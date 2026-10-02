'''Which repository files each `.py` file and each notebook imports or reads, directly
and transitively.

Written by Claude Opus 5 (1M context), effort high. The stored parse, the data files
named in string literals and the absent modules were added by Claude Opus 5.5 (1M
context), effort 40.

Every import statement in the repository is read with `ast` and resolved against the
folders that are on the path when the importing file runs. A dotted name resolves to
a module file or to a package `__init__.py`, and every existing prefix of the name
resolves as well, because importing `a.b.c` executes `a/__init__.py` and
`a/b/__init__.py` too. `from a.b import c` resolves `a.b.c` as a submodule as well as
`a.b`, because the statement reads as a submodule import whenever `c` is one.

An import can name a module inside a repository package that has no file. An import
of a deleted or renamed module leaves such a name behind. The two paths such a module
would have are recorded as absent modules named by the importing file, so that the
deletion selects every file that still imports the module.

A file also depends on the files it reads. A string literal outside a docstring that
ends in `.json` or `.ipynb`, and that names an indexed file when it is read relative to
the folder holding the file or to one of that folder's ancestors, is recorded as a data
file read by the file.
`notebooks/sota/DeepSeekV41Flash/validate_omitted_mechanisms.py` joins the name of a
notebook to `pathlib.Path(__file__).resolve().parents[1]` and reads the notebook. A
data file is a leaf of the graph, because reading a notebook as JSON does not run its
imports. A string that names a file for some other reason selects a target that did not
need to run, which costs time and misses nothing.

Parsing every file takes about nine seconds, most of it in `ast` and in loading the
notebooks that store large outputs. The parse of each file is stored in
`.cache/validations/parsed_sources.json` with the modification time and the size of
the file when it was parsed, and a later build parses again only the files whose time
or size differ. The stored parses are discarded when the source of this module changes,
because a change here can change the content of a parse.

A path is held as a string relative to the repository root with forward slashes, which
is the form `git status --porcelain` and `git diff --name-only` report, so a modified
path needs no conversion before it is looked up.

`obsidian/06-practice/Validation.md` states what the graph is used for.
'''

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import pathlib
import posixpath
import re
import warnings
from dataclasses import asdict, dataclass, field

import validations.stored_json as stored_json

type RepositoryPath = str

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]

FOLDERS_WITH_NO_IMPORTABLE_CODE = frozenset({
    '__pycache__', '.cache', '.git', '.vscode', '.claude', '.ipynb_checkpoints',
    'node_modules',
    'obsidian', 'outputs', '_guide',
})

INDEXED_TEXT_SUFFIXES = frozenset({'.py', '.ipynb', '.md', '.json', '.ts'})
DATA_FILE_SUFFIXES = frozenset({'.json', '.ipynb'})

PARSED_SOURCES_STORE = 'parsed_sources.json'
PARSER_DIGEST = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()

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


def syntax_tree_or_none(source: str) -> ast.Module | None:
    '''The syntax tree of `source`, or `None` when `ast` cannot read it, as for a
    notebook holding a cell that does not parse.'''
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SyntaxWarning)
            return ast.parse(source)
    except SyntaxError:
        return None


def imported_module_names(
    tree: ast.Module, containing_package: tuple[str, ...],
) -> frozenset[str]:
    '''Every dotted module name imported in the tree, with a relative import resolved
    against the package holding the importing file.'''
    names: set[str] = set()
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


def docstring_constants(tree: ast.Module) -> frozenset[ast.Constant]:
    '''The constant that is the docstring of the module, of a class or of a function,
    for every one of them in the tree.'''
    docstrings: set[ast.Constant] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)) or not node.body:
            continue
        first = node.body[0]
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            docstrings.add(first.value)
    return frozenset(docstrings)


def data_file_names(tree: ast.Module) -> frozenset[str]:
    '''Every string literal outside a docstring that ends in a data file suffix, with
    backslashes written as slashes and a leading `./` removed.'''
    docstrings = docstring_constants(tree)
    names: set[str] = set()
    for node in ast.walk(tree):
        if (not isinstance(node, ast.Constant) or not isinstance(node.value, str)
                or node in docstrings):
            continue
        name = node.value.strip().replace('\\', '/')
        if '\n' not in name and posixpath.splitext(name)[1] in DATA_FILE_SUFFIXES:
            names.add(name.removeprefix('./'))
    return frozenset(names)


@dataclass(frozen=True)
class ParsedSource:
    '''The module names imported by one `.py` file or notebook and the data file names
    written in it, with the modification time and the size of the file when it was
    parsed.'''

    modification_ns: int
    size: int
    imported_names: tuple[str, ...]
    named_data_files: tuple[str, ...]

    def is_current_for(self, status: os.stat_result) -> bool:
        return (self.modification_ns == status.st_mtime_ns
                and self.size == status.st_size)


def parse_source(
    path: RepositoryPath, root: pathlib.Path, status: os.stat_result,
) -> ParsedSource:
    '''Parse the file at `path`, and record `status` as the state of the file when it
    was read.

    The status is taken before the file is read, so a file written during the read
    carries a time older than its content and is parsed again by the next build.
    '''
    full = root / path
    sources = (python_of_notebook(full) if path.endswith('.ipynb')
               else python_of_module(full))
    containing_package = tuple(path.split('/')[:-1])
    imported: set[str] = set()
    named: set[str] = set()
    for source in sources:
        tree = syntax_tree_or_none(source)
        if tree is None:
            imported |= module_names_by_regular_expression(source)
            continue
        imported |= imported_module_names(tree, containing_package)
        named |= data_file_names(tree)
    return ParsedSource(
        modification_ns=status.st_mtime_ns, size=status.st_size,
        imported_names=tuple(sorted(imported)), named_data_files=tuple(sorted(named)))


def stored_parsed_sources(root: pathlib.Path) -> dict[RepositoryPath, ParsedSource]:
    '''The parses stored by an earlier build, or none when this module has changed
    since they were stored.'''
    stored = stored_json.read_stored_json(
        stored_json.stored_file_path(root, PARSED_SOURCES_STORE))
    if stored is None or stored.get('parser_digest') != PARSER_DIGEST:
        return {}
    return {
        path: ParsedSource(
            modification_ns=fields['modification_ns'], size=fields['size'],
            imported_names=tuple(fields['imported_names']),
            named_data_files=tuple(fields['named_data_files']))
        for path, fields in stored.get('sources', {}).items()}


def store_parsed_sources(
    parsed: dict[RepositoryPath, ParsedSource], root: pathlib.Path,
) -> None:
    stored_json.write_stored_json(
        stored_json.stored_file_path(root, PARSED_SOURCES_STORE),
        {'parser_digest': PARSER_DIGEST,
         'sources': {path: asdict(source) for path, source in parsed.items()}})


def current_parsed_sources(
    paths: tuple[RepositoryPath, ...], root: pathlib.Path,
) -> dict[RepositoryPath, ParsedSource]:
    '''The parse of every file in `paths`, read from the store for a file whose time
    and size are unchanged since it was stored, and the store rewritten when any file
    had to be parsed again or has gone.'''
    stored = stored_parsed_sources(root)
    current: dict[RepositoryPath, ParsedSource] = {}
    for path in paths:
        status = (root / path).stat()
        previous = stored.get(path)
        current[path] = (previous if previous is not None
                         and previous.is_current_for(status)
                         else parse_source(path, root, status))
    if current != stored:
        store_parsed_sources(current, root)
    return current


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


def module_file_candidates(
    parts: list[str], prefix: str,
) -> tuple[RepositoryPath, RepositoryPath]:
    '''The module file and the package `__init__.py` that could hold a dotted name.'''
    joined = '/'.join(parts)
    return (f'{prefix}{joined}.py', f'{prefix}{joined}/__init__.py')


@dataclass(frozen=True)
class ResolvedModuleName:
    '''The repository files named by a dotted module name, and the two paths of its
    module when the name lies inside a repository package and no file holds it.'''

    files: frozenset[RepositoryPath]
    absent_module_paths: frozenset[RepositoryPath]


def folders_holding_modules(
    indexed_modules: frozenset[RepositoryPath],
) -> frozenset[RepositoryPath]:
    '''Every folder that holds an indexed module at any depth, which includes the
    folders with no `__init__.py`, such as `utilities/`, that Python imports as
    namespace packages.'''
    folders: set[RepositoryPath] = set()
    for path in indexed_modules:
        parts = path.split('/')[:-1]
        folders |= {'/'.join(parts[:depth]) for depth in range(1, len(parts) + 1)}
    return frozenset(folders)


def resolve_module_name(
    dotted: str, search_folders: tuple[RepositoryPath, ...],
    indexed_modules: frozenset[RepositoryPath],
    module_folders: frozenset[RepositoryPath],
) -> ResolvedModuleName:
    '''The repository files named by a dotted module name, including every prefix
    package's `__init__.py`, searched in the first folder where any of them exists.

    When no file exists in any folder, the name is absent from the first folder in
    which its first part is a folder holding modules. A name that starts in no such
    folder is taken to be a third-party or standard-library module, and names nothing
    absent.
    '''
    parts = dotted.split('.')
    for folder in search_folders:
        prefix = f'{folder}/' if folder else ''
        found = frozenset(
            candidate
            for length in range(1, len(parts) + 1)
            for candidate in module_file_candidates(parts[:length], prefix)
            if candidate in indexed_modules)
        if not found:
            continue
        whole_name = module_file_candidates(parts, prefix)
        absent = (frozenset() if any(path in indexed_modules for path in whole_name)
                  else frozenset(whole_name))
        return ResolvedModuleName(files=found, absent_module_paths=absent)
    for folder in search_folders:
        prefix = f'{folder}/' if folder else ''
        if f'{prefix}{parts[0]}' in module_folders:
            return ResolvedModuleName(
                files=frozenset(),
                absent_module_paths=frozenset(module_file_candidates(parts, prefix)))
    return ResolvedModuleName(files=frozenset(), absent_module_paths=frozenset())


def resolve_data_file_name(
    name: str, reading_path: RepositoryPath,
    data_files: frozenset[RepositoryPath],
) -> RepositoryPath | None:
    '''The data file named by `name` when it is read relative to the folder holding
    `reading_path` or to one of that folder's ancestors, nearest first.'''
    folders = reading_path.split('/')[:-1]
    for depth in range(len(folders), -1, -1):
        candidate = posixpath.normpath('/'.join((*folders[:depth], name)))
        if candidate in data_files and candidate != reading_path:
            return candidate
    return None


@dataclass
class ImportGraph:
    '''The import relation over the repository, with a transitive closure per file.

    `direct_imports` holds, for each file, the repository files imported by it,
    `data_files_read` the data files named in it, and `absent_modules_named` the
    paths of the modules imported by it inside a repository package where no file
    exists.
    '''

    direct_imports: dict[RepositoryPath, frozenset[RepositoryPath]]
    data_files_read: dict[RepositoryPath, frozenset[RepositoryPath]]
    absent_modules_named: dict[RepositoryPath, frozenset[RepositoryPath]]
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
        '''The file itself, every repository file it imports at any depth, and every
        data file read by one of those files.

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
        read = {data_file for file in visited
                for data_file in self.data_files_read.get(file, frozenset())}
        self._reached[path] = frozenset(visited | read)
        return self._reached[path]

    def paths_depended_on(
        self, path: RepositoryPath,
    ) -> frozenset[RepositoryPath]:
        '''`files_reached_by(path)` together with the absent modules named by any of
        those files, so that a deleted module selects the files still importing it.'''
        reached = self.files_reached_by(path)
        return reached | frozenset(
            absent for file in reached
            for absent in self.absent_modules_named.get(file, frozenset()))

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
    '''Record what every `.py` file and notebook under `root` imports and reads,
    parsing only the files that changed since the stored parse.'''
    indexed = source_paths_under(root)
    modules = frozenset(path for path in indexed if path.endswith('.py'))
    notebooks = frozenset(path for path in indexed if path.endswith('.ipynb'))
    others = frozenset(indexed) - modules - notebooks | frozenset(
        vault_note_paths(root))
    data_files = notebooks | others
    module_folders = folders_holding_modules(modules)
    parsed = current_parsed_sources(tuple(sorted(modules | notebooks)), root)
    direct: dict[RepositoryPath, frozenset[RepositoryPath]] = {}
    read: dict[RepositoryPath, frozenset[RepositoryPath]] = {}
    absent: dict[RepositoryPath, frozenset[RepositoryPath]] = {}
    for path, source in parsed.items():
        search_folders = search_folders_for(path)
        resolved = tuple(
            resolve_module_name(dotted, search_folders, modules, module_folders)
            for dotted in source.imported_names)
        direct[path] = frozenset().union(
            *(name.files for name in resolved)) - {path}
        absent[path] = frozenset().union(
            *(name.absent_module_paths for name in resolved))
        read[path] = frozenset(
            resolved_name for name in source.named_data_files
            if (resolved_name := resolve_data_file_name(name, path, data_files))
            is not None)
    return ImportGraph(
        direct_imports=direct, data_files_read=read, absent_modules_named=absent,
        module_files=modules, notebook_files=notebooks, other_text_files=others)
