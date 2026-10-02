'''The validation targets a set of modified files reaches.

Written by Claude Opus 5 (1M context), effort high.

Keeping each feature in its own folder is what makes this selection possible. A
validator or a notebook imports the feature it demonstrates and the features that
feature is built on, and nothing else, so a modification to one folder leaves most
targets untouched. A target is selected when a modified file is among the files its
entry point imports transitively, and skipped otherwise.

Three checks of `validate_repository.py` do not have an entry point whose imports say
what they read. `imports` imports every module, `notebooks` and `calls` read every
module and every notebook, and `vault` reads every note and tests every path a note
mentions. Each of those carries the `DependencyRule` that states the set it reads.

A repository check that takes the modified files is passed them, and checks only the
files whose own dependencies include one of them. Windows refuses a command line longer
than 32,767 characters, so when the modified files would take more than
`MODIFIED_FILES_ARGUMENT_CHARACTERS` of it the check is passed nothing and checks every
file.

`obsidian/06-practice/Validation.md` states how the selection is used.
'''

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace

import validations.module_import_graph as module_import_graph
import validations.validation_targets as validation_targets

type RepositoryPath = module_import_graph.RepositoryPath

MODIFIED_FILES_ARGUMENT_CHARACTERS = 24_000


@dataclass(frozen=True)
class TargetSelection:
    '''The targets a modification reaches, and the ones it does not.

    `modified_files_reached` gives, per selected target's name, the modified files
    that target imports, so that a run can say why each target is in it.
    `modified_files_no_validator_or_notebook_reaches` holds the modified files
    covered only by a repository check or by nothing.
    '''

    selected: tuple[validation_targets.ValidationTarget, ...]
    skipped: tuple[validation_targets.ValidationTarget, ...]
    modified_files_reached: dict[str, tuple[RepositoryPath, ...]]
    modified_files_no_validator_or_notebook_reaches: tuple[RepositoryPath, ...]


class UnknownDependencyRule(Exception):
    '''A target carries a dependency rule this module does not resolve.'''


def dependencies_of_target(
    target: validation_targets.ValidationTarget,
    graph: module_import_graph.ImportGraph,
) -> frozenset[RepositoryPath]:
    '''Every repository path whose modification would select `target`.

    For a validator or a notebook this is the entry point together with everything it
    imports and reads, plus the script the runner invokes it through, because a change
    to `notebooks/execute_notebook.py` changes how every notebook runs. It includes
    the absent modules imported by those files, which are paths and not files.
    '''
    match target.dependency_rule:
        case validation_targets.DependencyRule.IMPORTS_OF_THE_ENTRY_POINT:
            reached = set(graph.paths_depended_on(target.command[0]))
            if target.entry_point is not None:
                reached |= graph.paths_depended_on(target.entry_point)
            return frozenset(reached)
        case validation_targets.DependencyRule.EVERY_MODULE:
            return graph.module_files
        case validation_targets.DependencyRule.EVERY_MODULE_AND_NOTEBOOK:
            return graph.module_files | graph.notebook_files
        case validation_targets.DependencyRule.EVERY_INDEXED_FILE:
            return graph.every_indexed_file
    raise UnknownDependencyRule(f'{target.name} carries {target.dependency_rule}')


def feature_folders_of(
    dependencies: Iterable[RepositoryPath],
) -> tuple[str, ...]:
    '''The distinct top-level folders the dependencies sit in, sorted.

    A file at the repository root is reported under its own name, because it belongs
    to no feature folder.
    '''
    return tuple(sorted({
        path.split('/')[0] if '/' in path else path for path in dependencies}))


def number_of_targets_depending_on_each_module(
    targets: Iterable[validation_targets.ValidationTarget],
    graph: module_import_graph.ImportGraph,
) -> dict[RepositoryPath, int]:
    '''For every module, how many of `targets` depend on it, which is how many of them
    a change to the module selects.'''
    counts = dict.fromkeys(graph.module_files, 0)
    for target in targets:
        for path in dependencies_of_target(target, graph) & graph.module_files:
            counts[path] += 1
    return counts


def narrowed_to_modification(
    target: validation_targets.ValidationTarget,
    modified: frozenset[RepositoryPath],
) -> validation_targets.ValidationTarget:
    '''`target` passed the modified files, when it takes them and the command line
    has room for them, so that it checks only the files reaching one of them.'''
    if target.modified_files_option is None:
        return target
    if sum(len(path) + 1 for path in modified) > MODIFIED_FILES_ARGUMENT_CHARACTERS:
        return target
    return replace(target, checked_files_reach=tuple(sorted(modified)))


def targets_affected_by(
    targets: Iterable[validation_targets.ValidationTarget],
    modified: Iterable[RepositoryPath],
    graph: module_import_graph.ImportGraph,
) -> TargetSelection:
    '''Split `targets` into the ones that depend on a modified file and the rest.'''
    wanted = frozenset(modified)
    selected = []
    skipped = []
    reached: dict[str, tuple[RepositoryPath, ...]] = {}
    reached_by_a_validator_or_notebook: set[RepositoryPath] = set()
    for target in targets:
        overlap = dependencies_of_target(target, graph) & wanted
        if not overlap:
            skipped.append(target)
            continue
        selected.append(narrowed_to_modification(target, wanted))
        reached[target.name] = tuple(sorted(overlap))
        if target.kind is not validation_targets.TargetKind.REPOSITORY_CHECK:
            reached_by_a_validator_or_notebook |= overlap
    return TargetSelection(
        selected=tuple(selected), skipped=tuple(skipped),
        modified_files_reached=reached,
        modified_files_no_validator_or_notebook_reaches=tuple(
            sorted(wanted - reached_by_a_validator_or_notebook)))
