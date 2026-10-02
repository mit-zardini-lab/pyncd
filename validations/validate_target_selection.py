'''Check that the runner selects the validations a modification reaches, and orders them.

Written by Claude Opus 5.5 (1M context), effort 40.

    python validations/validate_target_selection.py

One `check_` function per claim, each printing one line, and a non-zero exit when any
fails. Every check builds a small repository in a temporary directory, so that what it
selects is known. The repository holds a package `pkg` with an `__init__.py`, a folder
`feature` without one, as most feature folders here are, two validators, a data file one
of them names, a module nothing imports, and a notebook.

The claims are that imports are closed transitively and a data file named in a string
is a dependency, that deleting a module selects the files still importing it, that a
stored parse is reused and a changed file is parsed again, that a modification selects
exactly the targets depending on it and reports the files no target reaches, that a
path an agent names is read in every form its tools print, that a repository check is
passed the modified files and records its time under its own key, and that the longest
targets start first.
'''

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
from collections.abc import Callable

sys.path[0] = str(pathlib.Path(__file__).resolve().parents[1])

import validations.list_modified_files as list_modified_files  # noqa: E402
import validations.module_import_graph as module_import_graph  # noqa: E402
import validations.order_targets_by_duration as order_targets_by_duration  # noqa: E402
import validations.select_affected_targets as select_affected_targets  # noqa: E402
import validations.validation_targets as validation_targets  # noqa: E402

NOTEBOOK_IMPORTING_THE_FEATURE = {
    'cells': [{'cell_type': 'code', 'metadata': {}, 'outputs': [],
               'execution_count': None,
               'source': ['%matplotlib inline\n', 'import feature.helper as helper\n']}],
    'metadata': {}, 'nbformat': 4, 'nbformat_minor': 5,
}

SMALL_REPOSITORY: dict[str, str] = {
    'pkg/__init__.py': '',
    'pkg/core.py': 'VALUE = 1\n',
    'pkg/uses_core.py': 'import pkg.core as core\n',
    'pkg/unused.py': 'UNUSED = 0\n',
    'feature/helper.py': 'import pkg.uses_core as uses_core\n',
    'feature/expected.json': '{}\n',
    'feature/validate_feature.py': (
        "'''The data file named below is read, and this docstring names "
        "other.json.'''\n"
        'import pathlib\n'
        'import feature.helper as helper\n'
        "EXPECTED = pathlib.Path(__file__).parent / 'expected.json'\n"),
    'other/other.json': '{}\n',
    'other/validate_other.py': 'import pkg.core as core\n',
    'notebooks/Demo.ipynb': json.dumps(NOTEBOOK_IMPORTING_THE_FEATURE),
}


def written_repository(root: pathlib.Path) -> pathlib.Path:
    for path, content in SMALL_REPOSITORY.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(content, encoding='utf-8')
    return root


def graph_of(root: pathlib.Path) -> module_import_graph.ImportGraph:
    return module_import_graph.build_import_graph(root)


def validator_targets_of(
    root: pathlib.Path,
) -> tuple[validation_targets.ValidationTarget, ...]:
    return validation_targets.validator_targets(root)


def check_imports_are_closed_and_named_data_files_are_dependencies() -> None:
    with tempfile.TemporaryDirectory() as directory:
        graph = graph_of(written_repository(pathlib.Path(directory)))
        reached = graph.files_reached_by('feature/validate_feature.py')
        wanted = {'feature/helper.py', 'pkg/uses_core.py', 'pkg/core.py',
                  'pkg/__init__.py', 'feature/expected.json'}
        if not wanted <= reached:
            raise AssertionError(f'{sorted(wanted - reached)} missing from {sorted(reached)}')
        if 'other/other.json' in reached:
            raise AssertionError('a data file named only in a docstring was recorded')
        notebook = graph.files_reached_by('notebooks/Demo.ipynb')
        if 'pkg/core.py' not in notebook:
            raise AssertionError(f'the notebook reaches only {sorted(notebook)}')


def check_a_deleted_module_selects_the_files_still_importing_it() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = written_repository(pathlib.Path(directory))
        (root / 'feature/helper.py').unlink()
        graph = graph_of(root)
        depended_on = graph.paths_depended_on('feature/validate_feature.py')
        if 'feature/helper.py' not in depended_on:
            raise AssertionError(f'the deleted module is missing from {sorted(depended_on)}')
        selection = select_affected_targets.targets_affected_by(
            validator_targets_of(root), ('feature/helper.py',), graph)
        names = [target.name for target in selection.selected]
        if names != ['feature/validate_feature.py']:
            raise AssertionError(f'deleting the helper selected {names}')


def check_a_stored_parse_is_reused_and_a_changed_file_is_parsed_again() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = written_repository(pathlib.Path(directory))
        first = graph_of(root)
        store = root / '.cache/validations' / module_import_graph.PARSED_SOURCES_STORE
        if not store.exists():
            raise AssertionError(f'no store at {store}')
        second = graph_of(root)
        if second.direct_imports != first.direct_imports:
            raise AssertionError('a build from the store differs from a fresh parse')
        (root / 'pkg/core.py').write_text('import pkg.unused as unused\n', encoding='utf-8')
        third = graph_of(root)
        if 'pkg/unused.py' not in third.direct_imports['pkg/core.py']:
            raise AssertionError(
                f'the changed file imports {sorted(third.direct_imports["pkg/core.py"])}')


def check_a_modification_selects_the_targets_depending_on_it() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = written_repository(pathlib.Path(directory))
        graph = graph_of(root)
        targets = validator_targets_of(root)
        expected = {
            ('pkg/core.py',): ['feature/validate_feature.py', 'other/validate_other.py'],
            ('feature/expected.json',): ['feature/validate_feature.py'],
            ('other/validate_other.py',): ['other/validate_other.py'],
            ('pkg/unused.py',): [],
        }
        for modified, wanted in expected.items():
            selection = select_affected_targets.targets_affected_by(
                targets, modified, graph)
            names = [target.name for target in selection.selected]
            if names != wanted:
                raise AssertionError(f'{modified} selected {names}')
        unreached = select_affected_targets.targets_affected_by(
            targets, ('pkg/unused.py', 'pkg/core.py'), graph)
        if unreached.modified_files_no_validator_or_notebook_reaches != ('pkg/unused.py',):
            raise AssertionError(
                f'{unreached.modified_files_no_validator_or_notebook_reaches} reported')


def check_a_named_path_is_read_in_every_form() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = written_repository(pathlib.Path(directory))
        graph = graph_of(root)
        absolute_with_backslashes = str(root / 'pkg' / 'core.py').replace('/', '\\')
        named = list_modified_files.paths_named_as_modified(
            (absolute_with_backslashes, 'feature', './other/validate_other.py'),
            graph.every_indexed_file, root)
        wanted = ('feature/expected.json', 'feature/helper.py',
                  'feature/validate_feature.py', 'other/validate_other.py', 'pkg/core.py')
        if named != wanted:
            raise AssertionError(f'{named}')


def check_a_repository_check_is_passed_the_modified_files() -> None:
    check = validation_targets.ValidationTarget(
        name='repository:imports', kind=validation_targets.TargetKind.REPOSITORY_CHECK,
        command=('validate_repository.py', 'imports'), entry_point=None,
        dependency_rule=validation_targets.DependencyRule.EVERY_MODULE,
        cost=validation_targets.CostClass.MINUTES,
        modified_files_option=validation_targets.MODIFIED_FILES_OPTION)
    narrowed = select_affected_targets.narrowed_to_modification(
        check, frozenset({'pkg/core.py'}))
    wanted = ('validate_repository.py', 'imports',
              validation_targets.MODIFIED_FILES_OPTION, 'pkg/core.py')
    if narrowed.arguments() != wanted:
        raise AssertionError(f'{narrowed.arguments()}')
    if narrowed.duration_key() == check.duration_key():
        raise AssertionError('a narrowed run records its time under the full run\'s key')
    too_many = frozenset(
        f'pkg/module_{index}.py' for index in range(
            select_affected_targets.MODIFIED_FILES_ARGUMENT_CHARACTERS // 10))
    if select_affected_targets.narrowed_to_modification(check, too_many) != check:
        raise AssertionError('a command line over the limit was narrowed')


def check_the_longest_targets_start_first() -> None:
    targets = tuple(
        validation_targets.ValidationTarget(
            name=name, kind=validation_targets.TargetKind.VALIDATOR_SCRIPT,
            command=(name,), entry_point=name,
            dependency_rule=validation_targets.DependencyRule.IMPORTS_OF_THE_ENTRY_POINT,
            cost=validation_targets.CostClass.SECONDS)
        for name in ('a.py', 'b.py', 'c.py', 'd.py'))
    recorded = {'a.py': 4.0, 'b.py': 10.0, 'c.py': 4.0}
    order = [target.name
             for target in order_targets_by_duration.longest_first(targets, recorded)]
    if order != ['b.py', 'd.py', 'a.py', 'c.py']:
        raise AssertionError(f'{order}')
    wall = order_targets_by_duration.expected_wall_seconds(targets, recorded, jobs=2)
    if wall != 13.0:
        raise AssertionError(f'expected wall time {wall}')
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        order_targets_by_duration.record_durations({'a.py': 1.234}, root)
        order_targets_by_duration.record_durations({'b.py': 2.0}, root)
        read_back = order_targets_by_duration.recorded_durations(root)
        if read_back != {'a.py': 1.23, 'b.py': 2.0}:
            raise AssertionError(f'{read_back}')


CHECKS: tuple[Callable[[], None], ...] = (
    check_imports_are_closed_and_named_data_files_are_dependencies,
    check_a_deleted_module_selects_the_files_still_importing_it,
    check_a_stored_parse_is_reused_and_a_changed_file_is_parsed_again,
    check_a_modification_selects_the_targets_depending_on_it,
    check_a_named_path_is_read_in_every_form,
    check_a_repository_check_is_passed_the_modified_files,
    check_the_longest_targets_start_first,
)


def main() -> int:
    failures = 0
    for check in CHECKS:
        try:
            check()
            print(f'{check.__name__}: ok')
        except Exception as error:
            failures += 1
            print(f'{check.__name__}: FAILED {type(error).__name__}: {error}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
