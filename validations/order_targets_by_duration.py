'''Starting the longest validation targets first, from the duration of the last run of
each.

Written by Claude Opus 5.5 (1M context), effort 40.

A pool of `jobs` workers finishes no sooner than its longest target, and a long target
that is started last finishes after every other target has finished. Starting the
longest targets first lets the short ones fill the workers left free by the long ones.
The rule is the longest-processing-time rule for scheduling on identical machines.

`run_validation_targets.run_targets` records the duration of each target that passed
in `.cache/validations/target_durations.json` after every run. A target with no record
is expected to take the seconds assigned to its `CostClass` in
`SECONDS_EXPECTED_OF_COST_CLASS`, and the cost classes were declared a-priori in
`validation_targets.MEASURED_COSTS` from the runs of 2026-09-15 and 2026-09-27.
A time recorded with other targets running beside it is longer than the target takes
alone, and the order depends only on which targets are longer than which.
'''

from __future__ import annotations

import heapq
import pathlib
from collections.abc import Iterable, Mapping

import validations.stored_json as stored_json
import validations.validation_targets as validation_targets

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]

TARGET_DURATIONS_STORE = 'target_durations.json'

SECONDS_EXPECTED_OF_COST_CLASS: dict[validation_targets.CostClass, float] = {
    validation_targets.CostClass.SECONDS: 5.0,
    validation_targets.CostClass.TENS_OF_SECONDS: 30.0,
    validation_targets.CostClass.MINUTES: 120.0,
}


def recorded_durations(root: pathlib.Path = REPOSITORY_ROOT) -> dict[str, float]:
    '''The duration in seconds of the last passing run of each target, by its
    `ValidationTarget.duration_key`.'''
    stored = stored_json.read_stored_json(
        stored_json.stored_file_path(root, TARGET_DURATIONS_STORE))
    if stored is None:
        return {}
    return {name: float(seconds) for name, seconds in stored.items()
            if isinstance(seconds, (int, float))}


def record_durations(
    durations: Mapping[str, float], root: pathlib.Path = REPOSITORY_ROOT,
) -> None:
    '''Store `durations` over the recorded durations of the same targets, and keep the
    records of the targets not covered by this run.'''
    if not durations:
        return
    stored_json.write_stored_json(
        stored_json.stored_file_path(root, TARGET_DURATIONS_STORE),
        recorded_durations(root) | {name: round(seconds, 2)
                                    for name, seconds in durations.items()})


def expected_seconds(
    target: validation_targets.ValidationTarget, recorded: Mapping[str, float],
) -> float:
    return recorded.get(target.duration_key(),
                        SECONDS_EXPECTED_OF_COST_CLASS[target.cost])


def longest_first(
    targets: Iterable[validation_targets.ValidationTarget],
    recorded: Mapping[str, float],
) -> tuple[validation_targets.ValidationTarget, ...]:
    '''`targets` sorted by expected seconds, longest first, and by name among targets
    expected to take the same time.'''
    return tuple(sorted(
        targets, key=lambda target: (-expected_seconds(target, recorded), target.name)))


def expected_wall_seconds(
    targets: Iterable[validation_targets.ValidationTarget],
    recorded: Mapping[str, float], jobs: int,
) -> float:
    '''The wall time of running `targets` longest first on `jobs` workers, where each
    target takes its expected seconds and starts on the first worker to come free.'''
    finishing_times = [0.0] * max(1, jobs)
    for target in longest_first(targets, recorded):
        earliest = heapq.heappop(finishing_times)
        heapq.heappush(finishing_times, earliest + expected_seconds(target, recorded))
    return max(finishing_times)
