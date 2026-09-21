'''Running the repository's validations, all of them or the ones a change reaches.

Written by Claude Opus 5 (1M context), effort high.

    python validations/run_validations.py                  # what the change reaches
    python validations/run_validations.py --all            # every target
    python validations/run_validations.py --kind validator # the validate_*.py scripts
    python validations/run_validations.py --name attention # by name, matched loosely
    python validations/run_validations.py --since main     # what changed since a ref
    python validations/run_validations.py --list           # targets and dependencies
    python validations/run_validations.py --dependencies-of Taped

With no flag the working tree decides. When git reports modified files the run covers
the targets that import one of them and says so, and when the tree is clean the run
covers every target.

A target with an exclusion is left out unless `--include-excluded` is given. The
excluded notebooks are the ones that draw through `websocket_transfer` directly, whose
diagrams would go to whatever page holds port 8765. `--list` prints the reason beside
each.

`obsidian/06-practice/Validation.md` describes the targets and the selection.
'''

from __future__ import annotations

import argparse
import pathlib
import sys
import time

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import validate_repository
import validations.list_modified_files as list_modified_files
import validations.module_import_graph as module_import_graph
import validations.run_validation_targets as run_validation_targets
import validations.select_affected_targets as select_affected_targets
import validations.validation_targets as validation_targets


def command_line_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run the repository validations concurrently.')
    parser.add_argument(
        '--all', action='store_true',
        help='run every target, whatever git reports as modified')
    parser.add_argument(
        '--modified', action='store_true',
        help='run the targets the working tree reaches, even when it is clean')
    parser.add_argument(
        '--since', metavar='REF',
        help='treat everything that differs from REF as modified as well')
    parser.add_argument(
        '--assume-modified', metavar='PATH', action='append', default=[],
        help='treat PATH as modified without touching it, repeatable')
    parser.add_argument(
        '--name', metavar='PATTERN', action='append', default=[],
        help='select the targets whose name contains PATTERN, repeatable')
    parser.add_argument(
        '--kind', choices=[kind.value for kind in validation_targets.TargetKind],
        action='append', default=[], help='select targets of this kind, repeatable')
    parser.add_argument(
        '--include-excluded', action='store_true',
        help='run the excluded targets too, which needs a browser on port 8765 for '
             'the notebooks that draw through websocket_transfer')
    parser.add_argument(
        '--list', action='store_true',
        help='print the targets and the folders their dependencies sit in')
    parser.add_argument(
        '--dependencies-of', metavar='PATTERN',
        help='print every file the matching targets depend on')
    parser.add_argument(
        '--jobs', type=int, default=run_validation_targets.DEFAULT_JOBS,
        help='how many targets to run at once')
    parser.add_argument(
        '--timeout', type=int,
        default=run_validation_targets.DEFAULT_TIMEOUT_SECONDS,
        help='seconds allowed per target')
    return parser


def counted(quantity: int, noun: str) -> str:
    '''`quantity` followed by `noun`, with an `s` on the noun where English wants
    one.'''
    return f'{quantity} {noun}' if quantity == 1 else f'{quantity} {noun}s'


def narrowed_by_name_and_kind(
    targets: tuple[validation_targets.ValidationTarget, ...],
    names: list[str], kinds: list[str],
) -> tuple[validation_targets.ValidationTarget, ...]:
    narrowed = targets
    if names:
        narrowed = validation_targets.targets_named(narrowed, names)
    if kinds:
        narrowed = tuple(target for target in narrowed
                         if target.kind.value in kinds)
    return narrowed


def modified_paths(arguments: argparse.Namespace) -> tuple[str, ...]:
    '''Every path this run treats as modified.

    `--assume-modified` on its own replaces the working tree, so that the targets a
    file nobody has touched would select can be shown. Given beside `--modified` or
    `--since` it adds to what git reports.
    '''
    assumed = set(arguments.assume_modified)
    if assumed and not (arguments.modified or arguments.since):
        return tuple(sorted(assumed))
    paths = assumed | set(
        list_modified_files.modified_files_in_working_tree(REPOSITORY_ROOT))
    if arguments.since:
        paths |= set(list_modified_files.modified_files_since(
            arguments.since, REPOSITORY_ROOT))
    return tuple(sorted(paths))


def print_target_list(
    targets: tuple[validation_targets.ValidationTarget, ...],
    graph: module_import_graph.ImportGraph,
) -> None:
    for target in targets:
        dependencies = select_affected_targets.dependencies_of_target(target, graph)
        folders = select_affected_targets.feature_folders_of(dependencies)
        print(f'{target.name}')
        print(f'    {target.kind.value}, {target.cost.value}, '
              f'{len(dependencies)} files in {len(folders)} folders')
        print(f'    depends on {target.dependency_rule.value}: '
              f'{", ".join(folders)}')
        if target.exclusion is not None:
            print(f'    excluded, {target.exclusion}')


def print_dependencies_of(
    targets: tuple[validation_targets.ValidationTarget, ...],
    graph: module_import_graph.ImportGraph, pattern: str,
) -> None:
    for target in validation_targets.targets_named(targets, (pattern,)):
        dependencies = select_affected_targets.dependencies_of_target(target, graph)
        print(f'{target.name}, {len(dependencies)} files')
        for path in sorted(dependencies):
            print(f'    {path}')


def print_selection_reasons(selection: select_affected_targets.TargetSelection) -> None:
    for target in selection.selected:
        reached = selection.modified_files_reached[target.name]
        shown = ', '.join(reached[:4])
        remainder = '' if len(reached) <= 4 else f' and {len(reached) - 4} more'
        print(f'    {target.name}  reaches {shown}{remainder}')


def print_run_summary(
    results: tuple[run_validation_targets.TargetResult, ...], seconds: float,
) -> None:
    failed = [result for result in results if not result.passed]
    print(f'{len(results) - len(failed)} passed, {len(failed)} failed, '
          f'{seconds:.1f} s of wall time over '
          f'{sum(result.seconds for result in results):.1f} s of target time')
    run_validation_targets.print_failing_tails(results)


def narrowed_by_modification(
    arguments: argparse.Namespace,
    runnable: tuple[validation_targets.ValidationTarget, ...],
    graph: module_import_graph.ImportGraph,
) -> tuple[validation_targets.ValidationTarget, ...]:
    '''The targets a modified file reaches, or every target when `--all` was given or
    the working tree is clean, with a line saying which of the three happened.'''
    if arguments.all:
        print(f'this run covers {counted(len(runnable), "target")}')
        return runnable
    modified = modified_paths(arguments)
    if not modified:
        print('git reports nothing modified, so this run covers '
              f'{counted(len(runnable), "target")}')
        return runnable
    selection = select_affected_targets.targets_affected_by(runnable, modified, graph)
    print(f'git reports {counted(len(modified), "modified file")}')
    print(f'this run covers {counted(len(selection.selected), "target")} of '
          f'{len(runnable)}')
    print_selection_reasons(selection)
    return selection.selected


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace')
    arguments = command_line_parser().parse_args(argv)
    graph = module_import_graph.build_import_graph(REPOSITORY_ROOT)
    every_target = validation_targets.all_targets(graph, REPOSITORY_ROOT)
    candidates = narrowed_by_name_and_kind(
        every_target, arguments.name, arguments.kind)

    if arguments.dependencies_of:
        print_dependencies_of(every_target, graph, arguments.dependencies_of)
        return 0
    if arguments.list:
        print_target_list(candidates, graph)
        return 0

    runnable = candidates if arguments.include_excluded else tuple(
        target for target in candidates if target.exclusion is None)
    excluded = tuple(target for target in candidates if target.exclusion is not None)

    runnable = narrowed_by_modification(arguments, runnable, graph)

    if excluded and not arguments.include_excluded:
        print(f'this run leaves out {counted(len(excluded), "target")} that are run '
              'by hand, and --list gives the reason for each')
    if not runnable:
        print('no target was selected')
        return 0

    started = time.perf_counter()
    results = run_validation_targets.run_targets(
        runnable, REPOSITORY_ROOT, arguments.jobs, arguments.timeout)
    print_run_summary(results, time.perf_counter() - started)
    return 1 if any(not result.passed for result in results) else 0


if __name__ == '__main__':
    sys.exit(main())
