'''Running the repository's validations, all of them or the ones a change reaches.

Written by Claude Opus 5 (1M context), effort high. The paths given as arguments, the
plan and the report of unreached files were added by Claude Opus 5.5 (1M context),
effort 40.

    python validations/run_validations.py                  # what the change reaches
    python validations/run_validations.py algebra/x.py caching/
                                                           # what these paths reach
    python validations/run_validations.py --plan algebra/x.py
                                                           # the selection, not run
    python validations/run_validations.py --all            # every target
    python validations/run_validations.py --kind validator # the validate_*.py scripts
    python validations/run_validations.py --name attention # by name, matched loosely
    python validations/run_validations.py --since main     # what changed since a ref
    python validations/run_validations.py --list           # targets and dependencies
    python validations/run_validations.py --dependencies-of Taped
    python validations/run_validations.py --reach-by-folder
                                                           # how far a change reaches

With no path and no flag the working tree decides. When git reports modified files the
run covers the targets that depend on one of them and says so, and when the tree is
clean the run covers every target. Paths given as arguments replace what git reports,
so that an agent can pass its modified files and nothing else. Given beside
`--modified` or `--since`, they add to what git reports.

A target with an exclusion is left out unless `--include-excluded` is given. The
excluded notebooks are the ones that draw through `websocket_transfer` directly, whose
diagrams would go to whatever page holds port 8765. `--list` prints the reason beside
each.

`obsidian/06-practice/Validation.md` describes the targets and the selection.
'''

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import validate_repository
import validations.list_modified_files as list_modified_files
import validations.module_import_graph as module_import_graph
import validations.order_targets_by_duration as order_targets_by_duration
import validations.run_validation_targets as run_validation_targets
import validations.select_affected_targets as select_affected_targets
import validations.validation_targets as validation_targets

SELECTION_REASONS_SHOWN_PER_TARGET = 4


def command_line_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run the repository validations concurrently.')
    parser.add_argument(
        'paths', metavar='PATH', nargs='*',
        help='a file or folder to treat as modified, relative to the repository root '
             'or absolute; given alone, the paths replace what git reports')
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
        '--plan', action='store_true',
        help='print the selected targets in the order in which they would start, '
             'with the seconds each is expected to take, and run nothing')
    parser.add_argument(
        '--list', action='store_true',
        help='print the targets and the folders their dependencies sit in')
    parser.add_argument(
        '--dependencies-of', metavar='PATTERN',
        help='print every file the matching targets depend on')
    parser.add_argument(
        '--reach-by-folder', action='store_true',
        help='print, for each top-level folder, how many of the validators and '
             'notebooks depend on its modules')
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


def modified_paths(
    arguments: argparse.Namespace, graph: module_import_graph.ImportGraph,
) -> tuple[str, ...]:
    '''Every path this run treats as modified.

    The paths given as arguments on their own replace the working tree, so that an
    agent can pass its modified files, and so that the selection for an untouched
    file can be shown. Given beside `--modified` or `--since` they add to what git
    reports.
    '''
    named = set(list_modified_files.paths_named_as_modified(
        arguments.paths, graph.every_indexed_file, REPOSITORY_ROOT))
    if arguments.paths and not (arguments.modified or arguments.since):
        return tuple(sorted(named))
    paths = named | set(
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
        dependencies = (select_affected_targets.dependencies_of_target(target, graph)
                        & graph.every_indexed_file)
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
        dependencies = (select_affected_targets.dependencies_of_target(target, graph)
                        & graph.every_indexed_file)
        print(f'{target.name}, {len(dependencies)} files')
        for path in sorted(dependencies):
            print(f'    {path}')


def print_reach_by_folder(
    targets: tuple[validation_targets.ValidationTarget, ...],
    graph: module_import_graph.ImportGraph,
) -> None:
    '''For each top-level folder, the median and the largest number of `targets` that a
    change to one of its modules selects, largest median first.'''
    counts = select_affected_targets.number_of_targets_depending_on_each_module(
        targets, graph)
    by_folder: dict[str, list[int]] = {}
    for path, count in counts.items():
        by_folder.setdefault(path.split('/')[0], []).append(count)
    print(f'A change to one module of a folder selects this many of the '
          f'{len(targets)} validators and notebooks.')
    print(f'{"folder":28} {"modules":>7} {"median":>7} {"largest":>7}')
    for folder, folder_counts in sorted(
            by_folder.items(),
            key=lambda item: (-statistics.median_low(item[1]), item[0])):
        print(f'{folder:28} {len(folder_counts):7} '
              f'{statistics.median_low(folder_counts):7} {max(folder_counts):7}')


def print_selection_reasons(selection: select_affected_targets.TargetSelection) -> None:
    for target in selection.selected:
        reached = selection.modified_files_reached[target.name]
        shown = ', '.join(reached[:SELECTION_REASONS_SHOWN_PER_TARGET])
        remainder = ('' if len(reached) <= SELECTION_REASONS_SHOWN_PER_TARGET
                     else f' and {len(reached) - SELECTION_REASONS_SHOWN_PER_TARGET} '
                          'more')
        print(f'    {target.name}  reaches {shown}{remainder}')
    unreached = selection.modified_files_no_validator_or_notebook_reaches
    if unreached:
        print(f'no validator or notebook in this run reaches '
              f'{counted(len(unreached), "modified file")}, so only a repository '
              'check covers each of these, or nothing does:')
        for path in unreached:
            print(f'    {path}')


def print_plan(
    targets: tuple[validation_targets.ValidationTarget, ...], jobs: int,
) -> None:
    recorded = order_targets_by_duration.recorded_durations(REPOSITORY_ROOT)
    for target in order_targets_by_duration.longest_first(targets, recorded):
        seconds = order_targets_by_duration.expected_seconds(target, recorded)
        source = 'recorded' if target.duration_key() in recorded else 'cost class'
        print(f'{seconds:7.1f} s  {target.name}  ({source})')
    wall_seconds = order_targets_by_duration.expected_wall_seconds(
        targets, recorded, jobs)
    print(f'expected wall time {wall_seconds:.0f} s on {counted(jobs, "job")}')


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
    nothing is modified, with a line saying which of the three happened.'''
    if arguments.all:
        print(f'this run covers {counted(len(runnable), "target")}')
        return runnable
    modified = modified_paths(arguments, graph)
    if not modified:
        print('nothing is modified, so this run covers '
              f'{counted(len(runnable), "target")}')
        return runnable
    selection = select_affected_targets.targets_affected_by(runnable, modified, graph)
    source = 'the arguments name' if arguments.paths else 'git reports'
    print(f'{source} {counted(len(modified), "modified file")}')
    print(f'this run covers {counted(len(selection.selected), "target")} of '
          f'{len(runnable)}')
    print_selection_reasons(selection)
    return selection.selected


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(errors='replace', line_buffering=True)
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

    if arguments.reach_by_folder:
        print_reach_by_folder(
            tuple(target for target in runnable if target.kind
                  is not validation_targets.TargetKind.REPOSITORY_CHECK), graph)
        return 0

    runnable = narrowed_by_modification(arguments, runnable, graph)

    if excluded and not arguments.include_excluded:
        print(f'this run leaves out {counted(len(excluded), "target")} that are run '
              'by hand, and --list gives the reason for each')
    if not runnable:
        print('no target was selected')
        return 0
    if arguments.plan:
        print_plan(runnable, arguments.jobs)
        return 0

    started = time.perf_counter()
    results = run_validation_targets.run_targets(
        runnable, REPOSITORY_ROOT, arguments.jobs, arguments.timeout)
    print_run_summary(results, time.perf_counter() - started)
    return 1 if any(not result.passed for result in results) else 0


if __name__ == '__main__':
    sys.exit(main())
