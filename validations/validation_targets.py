'''Every validation the repository has, where it is located, and what it costs to run.

Written by Claude Opus 5 (1M context), effort high.

Three kinds of validation exist. A `validate_*.py` script sits at a feature root or
under a feature and checks that feature's semantics. A notebook under `notebooks/`
or under `example_notebooks/` demonstrates a feature and asserts what it claims, so
it is a test as well, and `notebooks/execute_notebook.py` runs one with the diagrams
off. The checks of `validate_repository.py` cover what neither reaches: a module that
no longer imports, a notebook naming a renamed function, and a vault note pointing at
a file that has moved.

A validator is found by its `validate_*.py` name rather than read from a list, so a
new feature's validator is picked up as soon as the file exists. A notebook is found
by its extension, under either of the two notebook folders `NOTEBOOK_FOLDERS` names.
The repository checks register themselves through the
`repository_check` decorator, which `validate_repository.py` applies to each of them,
so the two never disagree about which checks exist. `check_validators` is deliberately
not registered, because the validator scripts are targets in their own right and
registering the check that runs them would run each of them twice.

`MEASURED_COSTS` names every target measured to take longer than ten seconds, and a
target it does not name took seconds. Each figure was taken with several targets in
flight, so it is an upper bound on what the target costs alone. The validators finish
in seconds and the notebooks take longer, so the notebooks are what a full run costs.

`obsidian/06-practice/Validation.md` describes what each validation covers.
'''

from __future__ import annotations

import os
import pathlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum

import validations.module_import_graph as module_import_graph

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]

type RepositoryPath = module_import_graph.RepositoryPath


class TargetKind(Enum):
    VALIDATOR_SCRIPT = 'validator'
    NOTEBOOK = 'notebook'
    REPOSITORY_CHECK = 'repository-check'


class CostClass(Enum):
    '''How long a target takes, to the nearest order of magnitude.'''
    SECONDS = 'seconds'
    TENS_OF_SECONDS = 'tens of seconds'
    MINUTES = 'minutes'


class DependencyRule(Enum):
    '''How the files a target depends on are worked out.'''
    IMPORTS_OF_THE_ENTRY_POINT = 'the imports of its entry point'
    EVERY_MODULE = 'every module'
    EVERY_MODULE_AND_NOTEBOOK = 'every module and notebook'
    EVERY_INDEXED_FILE = 'every indexed file'


@dataclass(frozen=True)
class ValidationTarget:
    '''One validation, and everything needed to run it and to select it.

    `command` is the argument list after the interpreter, so the runner prepends
    `sys.executable` and runs it from the repository root. `entry_point` is the file
    whose imports decide whether a modification reaches the target, which is the
    script itself for a validator and the notebook for a notebook. `exclusion` holds
    the reason a target is left out of a default run, and is `None` for a target that
    runs by default.
    '''

    name: str
    kind: TargetKind
    command: tuple[str, ...]
    entry_point: RepositoryPath | None
    dependency_rule: DependencyRule
    cost: CostClass
    exclusion: str | None = None


@dataclass(frozen=True)
class RepositoryCheck:
    '''A check of `validate_repository.py`, as its decorator registered it.'''

    name: str
    check: Callable[[], list[str]]
    dependency_rule: DependencyRule
    cost: CostClass


REPOSITORY_CHECKS: dict[str, RepositoryCheck] = {}


def repository_check(
    name: str, dependency_rule: DependencyRule, cost: CostClass,
) -> Callable[[Callable[[], list[str]]], Callable[[], list[str]]]:
    '''Register a check of `validate_repository.py` under `name`.'''
    def register(check: Callable[[], list[str]]) -> Callable[[], list[str]]:
        REPOSITORY_CHECKS[name] = RepositoryCheck(
            name=name, check=check, dependency_rule=dependency_rule, cost=cost)
        return check
    return register


MEASURED_COSTS: dict[RepositoryPath, CostClass] = {
    'notebooks/sota/DeepSeekV41Flash.ipynb': CostClass.TENS_OF_SECONDS,
    'notebooks/sota/DeepSeekV41Flash/validate_deepseek_v41_flash_integrated.py':
        CostClass.TENS_OF_SECONDS,
    'notebooks/sota/DeepSeekV41Flash/validate_quantised_text_only_model.py':
        CostClass.TENS_OF_SECONDS,
    'quantization/validate_quantization.py': CostClass.TENS_OF_SECONDS,
}

DIAGRAMS_SENT_TO_THE_OPEN_PAGE = (
    'imports `websocket_transfer` and draws through it directly, so the diagram mode '
    'the runner passes does not reach it and the diagrams would go to whatever page '
    'holds port 8765; run it by hand with a browser open')

DIAGRAM_MODE_FOR_A_VALIDATION_RUN = 'off'

NOTEBOOK_FOLDERS = ('notebooks', 'example_notebooks')


def validator_script_paths(
    root: pathlib.Path = REPOSITORY_ROOT,
) -> tuple[RepositoryPath, ...]:
    '''Every `validate_*.py` at a feature root or under a feature, sorted.

    `validate_repository.py` at the repository root is not one of them. Its checks
    are registered separately, through `repository_check`.
    '''
    found = []
    for folder, subfolders, files in os.walk(root):
        subfolders[:] = [
            name for name in subfolders
            if name not in module_import_graph.FOLDERS_WITH_NO_IMPORTABLE_CODE]
        for file in files:
            if not (file.startswith('validate_') and file.endswith('.py')):
                continue
            path = module_import_graph.repository_relative(
                pathlib.Path(folder, file), root)
            if path != 'validate_repository.py':
                found.append(path)
    return tuple(sorted(found))


def notebook_paths(
    root: pathlib.Path = REPOSITORY_ROOT,
) -> tuple[RepositoryPath, ...]:
    '''Every notebook under one of `NOTEBOOK_FOLDERS`, sorted, excluding the
    checkpoint copies.'''
    return tuple(sorted(
        module_import_graph.repository_relative(path, root)
        for folder in NOTEBOOK_FOLDERS
        for path in (root / folder).rglob('*.ipynb')
        if '.ipynb_checkpoints' not in path.parts))


def notebook_exclusion(
    path: RepositoryPath, graph: module_import_graph.ImportGraph,
) -> str | None:
    '''The reason a notebook is left out of a default run, or `None`.'''
    if graph.direct_imports.get(path, frozenset()) & websocket_transfer_files(graph):
        return DIAGRAMS_SENT_TO_THE_OPEN_PAGE
    return None


def websocket_transfer_files(
    graph: module_import_graph.ImportGraph,
) -> frozenset[RepositoryPath]:
    '''The modules of the `tsncd` bridge, which a notebook that draws through them
    reaches directly rather than through `notebooks/display/notebook_diagrams.py`.'''
    return frozenset(path for path in graph.module_files
                     if path.startswith('websocket_transfer/'))


def validator_targets(
    root: pathlib.Path = REPOSITORY_ROOT,
) -> tuple[ValidationTarget, ...]:
    return tuple(
        ValidationTarget(
            name=path, kind=TargetKind.VALIDATOR_SCRIPT, command=(path,),
            entry_point=path,
            dependency_rule=DependencyRule.IMPORTS_OF_THE_ENTRY_POINT,
            cost=MEASURED_COSTS.get(path, CostClass.SECONDS))
        for path in validator_script_paths(root))


def notebook_targets(
    graph: module_import_graph.ImportGraph,
    root: pathlib.Path = REPOSITORY_ROOT,
) -> tuple[ValidationTarget, ...]:
    return tuple(
        ValidationTarget(
            name=path, kind=TargetKind.NOTEBOOK,
            command=('notebooks/execute_notebook.py', path,
                     '--diagrams', DIAGRAM_MODE_FOR_A_VALIDATION_RUN),
            entry_point=path,
            dependency_rule=DependencyRule.IMPORTS_OF_THE_ENTRY_POINT,
            cost=MEASURED_COSTS.get(path, CostClass.SECONDS),
            exclusion=notebook_exclusion(path, graph))
        for path in notebook_paths(root))


def repository_check_targets() -> tuple[ValidationTarget, ...]:
    '''One target per check registered through `repository_check`.

    The registry is empty until `validate_repository` has been imported, because the
    decorator runs then.
    '''
    return tuple(
        ValidationTarget(
            name=f'repository:{registered.name}',
            kind=TargetKind.REPOSITORY_CHECK,
            command=('validate_repository.py', registered.name),
            entry_point=None, dependency_rule=registered.dependency_rule,
            cost=registered.cost)
        for registered in sorted(REPOSITORY_CHECKS.values(), key=lambda r: r.name))


def all_targets(
    graph: module_import_graph.ImportGraph,
    root: pathlib.Path = REPOSITORY_ROOT,
) -> tuple[ValidationTarget, ...]:
    return (*repository_check_targets(), *validator_targets(root),
            *notebook_targets(graph, root))


def targets_named(
    targets: Iterable[ValidationTarget], wanted: Iterable[str],
) -> tuple[ValidationTarget, ...]:
    '''The targets whose name equals or contains one of `wanted`.

    Containment is what makes `--name quantization` select that feature's validator
    and `--name BuildingAModel` select one notebook by its title alone.
    '''
    patterns = tuple(wanted)
    return tuple(target for target in targets
                 if any(pattern == target.name or pattern in target.name
                        for pattern in patterns))
