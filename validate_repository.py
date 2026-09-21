'''Repository-wide checks that a refactor did not break anything.

Five independent checks, each runnable on its own:

    imports     every module in the package imports cleanly
    notebooks   every repository module a notebook imports still exists
    calls       every module-qualified call in a notebook matches its function
    validators  the `validate_*` scripts still pass
    vault       every link and file path in `obsidian/` resolves

The `validate_*` scripts check semantics. These checks cover what those cannot
catch: a module that no longer imports because a folder moved, a notebook naming a
module or a keyword argument that has been renamed, and a note pointing at a file
that no longer exists. A notebook is never imported, so nothing else notices when
a rename breaks one.

    python validate_repository.py            # all five
    python validate_repository.py imports    # one of them

`validations/` holds the registry of every validation in the repository and the
runner that executes several at once. The `validators` check here is that runner
applied to the `validate_*.py` scripts it discovered, so the two agree on the list
without either of them keeping it by hand. Each check below registers itself as a
target of that registry, so `python validations/run_validations.py` covers these
checks, the validator scripts and the notebooks in one run.
'''

from __future__ import annotations

import importlib
import io
import os
import pathlib
import re
import sys
import traceback

import validations.run_validation_targets as run_validation_targets
import validations.validation_targets as validation_targets

REPOSITORY_ROOT = pathlib.Path(__file__).parent
VAULT_ROOT = REPOSITORY_ROOT / 'obsidian'

# Folders holding no importable package code.
SKIPPED_FOLDERS = {
    '__pycache__', '.git', '.vscode', '.claude', 'node_modules',
    'obsidian', 'outputs', '_guide',
}

# Modules that cannot be imported alongside the rest of the package.
# `run_server` starts a server at import.
UNIMPORTABLE_MODULES = {
    'run_server',
}

WIKI_LINK = re.compile(r'\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]')
FRONTMATTER_CODE_PATHS = re.compile(r'(?m)^code:\s*(.+)$')
PYTHON_PATH_MENTION = re.compile(r'`([A-Za-z_][A-Za-z_0-9]*(?:/[A-Za-z_0-9.]+)+\.py)`')
INLINE_CODE_SPAN = re.compile(r'`[^`\n]*`')
FRONTMATTER_TAGS = re.compile(r'(?m)^tags:\s*\[(.+)\]$')


def module_names_under(root: pathlib.Path) -> list[str]:
    '''Every module in the repository, as a dotted import name.'''
    names = []
    for folder, subfolders, files in os.walk(root):
        subfolders[:] = [s for s in subfolders if s not in SKIPPED_FOLDERS]
        for file in sorted(files):
            if not file.endswith('.py') or file == '__init__.py':
                continue
            path = pathlib.Path(folder, file).relative_to(root)
            name = '.'.join((*path.parts[:-1], path.stem))
            if name in UNIMPORTABLE_MODULES:
                continue
            names.append(name)
    return names


@validation_targets.repository_check(
    name='imports', dependency_rule=validation_targets.DependencyRule.EVERY_MODULE,
    cost=validation_targets.CostClass.SECONDS)
def check_imports() -> list[str]:
    '''Import every module and collect the ones that raise.'''
    sys.path.insert(0, str(REPOSITORY_ROOT))
    failures = []
    for name in module_names_under(REPOSITORY_ROOT):
        try:
            importlib.import_module(name)
        except BaseException:
            first_line = traceback.format_exc().strip().splitlines()[-1]
            failures.append(f'{name}: {first_line}')
    return failures


def check_validators() -> list[str]:
    '''Run every discovered `validate_*.py` script at once and collect the failures.

    The scripts come from `validations.validation_targets.validator_targets`, which
    finds them by their name at a feature root, so a new feature's validator runs
    here as soon as the file exists. `run_targets` prints a line per script as it
    finishes, which is the only report during the tens of seconds the slowest of them
    take.
    '''
    results = run_validation_targets.run_targets(
        validation_targets.validator_targets(REPOSITORY_ROOT), REPOSITORY_ROOT)
    return list(run_validation_targets.failures_of(results))


def source_file_suffixes() -> set[str]:
    '''Every source path, plus each of its trailing fragments.

    Notes name a file the way a reader would, often relative to its package
    rather than to the repository root. Both forms resolve against this set.
    '''
    suffixes = set()
    for folder, subfolders, files in os.walk(REPOSITORY_ROOT):
        subfolders[:] = [s for s in subfolders if s not in SKIPPED_FOLDERS]
        for file in files:
            if not file.endswith(('.py', '.md', '.ipynb', '.json', '.ts')):
                continue
            parts = pathlib.Path(folder, file).relative_to(REPOSITORY_ROOT).parts
            for start in range(len(parts)):
                suffixes.add('/'.join(parts[start:]))
    return suffixes


def is_log(text: str) -> bool:
    '''Whether a note's frontmatter tags it as a log.

    Most logs sit under `logs/`. Two long-running ones sit in the layer folder
    they belong to and carry the tag instead.
    '''
    tags = FRONTMATTER_TAGS.search(text)
    return tags is not None and 'log' in [
        tag.strip() for tag in tags.group(1).split(',')]


@validation_targets.repository_check(
    name='vault',
    dependency_rule=validation_targets.DependencyRule.EVERY_INDEXED_FILE,
    cost=validation_targets.CostClass.SECONDS)
def check_vault() -> list[str]:
    '''Every wiki link, `code:` path and file mention in the vault resolves.

    Three exemptions. A link written inside backticks is being quoted rather
    than followed. A path named in a log may be one that has since moved,
    because a log records the repository as it was on its date, and a log is
    either a note under `logs/` or any note tagged `log`. A path named in
    `research/` usually belongs to another project entirely.
    '''
    notes = {path.stem for path in VAULT_ROOT.rglob('*.md')}
    known_paths = source_file_suffixes()
    failures = []
    for path in sorted(VAULT_ROOT.rglob('*.md')):
        if path.name == 'TEMPLATE.md':
            continue
        text = io.open(path, encoding='utf-8').read()
        relative = path.relative_to(REPOSITORY_ROOT)
        # A link inside backticks is a quoted example, so it is left alone.
        linkable = INLINE_CODE_SPAN.sub('``', text)
        for target in WIKI_LINK.findall(linkable):
            target = target.strip().rstrip('\\')
            if target not in notes:
                failures.append(f'{relative}: dead link [[{target}]]')
        # A log records history, including paths that have since moved. A
        # research note cites other projects' files, which are not here at all.
        if path.parent.name in ('logs', 'research') or is_log(text):
            continue
        for line in FRONTMATTER_CODE_PATHS.findall(text):
            for mentioned in line.split(','):
                mentioned = mentioned.strip().strip('`').rstrip('/')
                if not mentioned:
                    continue
                if not (REPOSITORY_ROOT / mentioned).exists():
                    failures.append(f'{relative}: code: {mentioned} does not exist')
        for mentioned in PYTHON_PATH_MENTION.findall(text):
            if mentioned not in known_paths:
                failures.append(f'{relative}: mentions missing {mentioned}')
    return failures


PARENTHESISED_IMPORT = re.compile(r'import\s*\(([^)]*)\)')


def flatten_parenthesised_imports(source: str) -> str:
    '''`source` with every parenthesised `from x import (a, b)` that spans several
    lines written on one line, so that the line-based import pattern below reads the
    names it lists.'''
    return PARENTHESISED_IMPORT.sub(
        lambda match: 'import ' + ' '.join(match.group(1).split()), source)


NOTEBOOK_IMPORT = re.compile(
    r'(?m)^\s*(?:import\s+([A-Za-z_][\w.]*)'
    r'|from\s+([A-Za-z_][\w.]*)\s+import\s+([^\n#]+))')


def local_package_names() -> set[str]:
    '''The top-level names that resolve inside this repository.'''
    names = {'fix_notebook_dir'}   # reachable because notebooks put notebooks/ on the path
    for entry in REPOSITORY_ROOT.iterdir():
        if entry.is_dir() and entry.name not in SKIPPED_FOLDERS and not entry.name.startswith('.'):
            names.add(entry.name)
        elif entry.suffix == '.py':
            names.add(entry.stem)
    return names


@validation_targets.repository_check(
    name='notebooks',
    dependency_rule=validation_targets.DependencyRule.EVERY_MODULE_AND_NOTEBOOK,
    cost=validation_targets.CostClass.SECONDS)
def check_notebook_imports() -> list[str]:
    '''Every repository module a notebook imports still exists.

    `check_imports` does not reach these, because a notebook is not imported. A
    rename that misses one leaves a notebook that fails on its first cell, which
    is how `from data_transfer import json` survived that module being renamed.
    Third-party and standard-library imports are not this check's business.
    '''
    import importlib.util
    import json

    sys.path.insert(0, str(REPOSITORY_ROOT))
    sys.path.insert(0, str(REPOSITORY_ROOT / 'notebooks'))
    local = local_package_names()
    failures = []
    for relative in validation_targets.notebook_paths():
        path = REPOSITORY_ROOT / relative
        notebook = json.loads(io.open(path, encoding='utf-8').read())
        source = flatten_parenthesised_imports('\n'.join(
            ''.join(cell['source']) for cell in notebook.get('cells', ())
            if cell['cell_type'] == 'code'))
        seen = set()
        for plain, package, imported in NOTEBOOK_IMPORT.findall(source):
            # `from x import a, b` may name submodules, so check each of those
            # too. Shadowing is how `from data_transfer import json` survived the
            # rename: the package resolved and the submodule was never looked at.
            candidates = [plain] if plain else [package] + [
                f'{package}.{name.split(" as ")[0].strip()}'
                for name in imported.split(',') if name.strip()]
            for name in candidates:
                if name in seen or name.split('.')[0] not in local:
                    continue
                seen.add(name)
                if resolves(name) or defined_in_package(name):
                    continue
                failures.append(f'{relative}: cannot import {name}')
    return failures


def resolves(name: str) -> bool:
    '''Whether `name` imports as a module.'''
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError, AttributeError):
        return False


def defined_in_package(dotted: str) -> bool:
    '''Whether the last segment is a name the parent package's `__init__` binds.

    `from x import y` is legitimate when `y` is a class or function rather than
    a submodule, and only the parent's `__init__.py` can say so.
    '''
    parent, _, name = dotted.rpartition('.')
    if not parent:
        return False
    folder = REPOSITORY_ROOT / pathlib.Path(*parent.split('.'))
    for source in (folder / '__init__.py', folder.with_suffix('.py')):
        if not source.exists():
            continue
        if re.search(rf'(?<![\w.]){re.escape(name)}(?![\w])',
                     io.open(source, encoding='utf-8').read()):
            return True
    return False



NOTEBOOK_ALIAS = re.compile(
    r'(?m)^\s*import\s+([A-Za-z_][\w.]*)\s+as\s+(\w+)\s*$')


@validation_targets.repository_check(
    name='calls',
    dependency_rule=validation_targets.DependencyRule.EVERY_MODULE_AND_NOTEBOOK,
    cost=validation_targets.CostClass.SECONDS)
def check_notebook_calls() -> list[str]:
    '''Every module-qualified call in a notebook matches the function it names.

    A notebook is never imported, so nothing else notices when a function it calls
    is renamed or loses a keyword argument. This resolves each `alias.name(...)`
    call against the module the alias was imported from, and checks the keyword
    arguments against the signature. It found `sk._rescope(..., top=True)` after
    that parameter was renamed to `is_outermost`.
    '''
    import ast
    import inspect
    import importlib
    import json

    sys.path.insert(0, str(REPOSITORY_ROOT))
    local = local_package_names()
    failures = []
    for relative in validation_targets.notebook_paths():
        path = REPOSITORY_ROOT / relative
        notebook = json.loads(io.open(path, encoding='utf-8').read())
        source = '\n'.join(
            ''.join(cell['source']) for cell in notebook.get('cells', ())
            if cell['cell_type'] == 'code')

        modules = {}
        for dotted, alias in NOTEBOOK_ALIAS.findall(source):
            if dotted.split('.')[0] not in local:
                continue
            try:
                modules[alias] = importlib.import_module(dotted)
            except BaseException:
                continue        # check_notebook_imports reports this

        for statement in source.splitlines():
            try:
                tree = ast.parse(statement.strip())
            except SyntaxError:
                continue        # a fragment of a multi-line statement
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if not isinstance(node.func, ast.Attribute):
                    continue
                if not isinstance(node.func.value, ast.Name):
                    continue
                module = modules.get(node.func.value.id)
                if module is None:
                    continue
                function = getattr(module, node.func.attr, None)
                if function is None:
                    failures.append(
                        f'{relative}: {node.func.value.id}.{node.func.attr} '
                        f'does not exist in {module.__name__}')
                    continue
                if not callable(function):
                    continue
                try:
                    signature = inspect.signature(function)
                except (TypeError, ValueError):
                    continue
                accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD
                                  for p in signature.parameters.values())
                if accepts_any:
                    continue
                for keyword in node.keywords:
                    if keyword.arg and keyword.arg not in signature.parameters:
                        failures.append(
                            f'{relative}: {node.func.value.id}.{node.func.attr}() '
                            f'has no parameter {keyword.arg!r}')
    return sorted(set(failures))


CHECKS = {
    'imports': check_imports,
    'notebooks': check_notebook_imports,
    'calls': check_notebook_calls,
    'validators': check_validators,
    'vault': check_vault,
}


def main(selected: list[str]) -> int:
    names = selected or list(CHECKS)
    total = 0
    for name in names:
        failures = CHECKS[name]()
        total += len(failures)
        print(f'{name}: {"ok" if not failures else f"{len(failures)} failures"}')
        for failure in failures:
            print(f'  {failure}')
    return 1 if total else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
