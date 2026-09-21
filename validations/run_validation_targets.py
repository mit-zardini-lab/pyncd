'''Running validation targets concurrently, each as its own subprocess.

Written by Claude Opus 5 (1M context), effort high.

Every target is already a process: a validator is a script and a notebook runs under
a Jupyter kernel that `notebooks/execute_notebook.py` starts. A thread per target is
therefore enough, because each thread spends its time waiting on a subprocess rather
than running Python, and the interpreter lock it holds while it waits is released.
`concurrent.futures.ThreadPoolExecutor` runs `jobs` of them at once.

`DEFAULT_JOBS` is eight rather than the core count, so that a target starting worker
processes of its own does not oversubscribe the machine.

A timeout kills the subprocess. On Windows it does not kill the Jupyter kernel that
`execute_notebook.py` started under it, so a timed-out notebook can leave a kernel
behind.
'''

from __future__ import annotations

import concurrent.futures
import os
import pathlib
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import validations.validation_targets as validation_targets

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_JOBS = 8
DEFAULT_TIMEOUT_SECONDS = 1800
FAILING_TAIL_LINES = 12


@dataclass(frozen=True)
class TargetResult:
    '''What running one target produced.'''

    target: validation_targets.ValidationTarget
    exit_code: int
    seconds: float
    output: str
    timed_out: bool

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def status(self) -> str:
        if self.timed_out:
            return 'TIMEOUT'
        return 'ok' if self.passed else 'FAIL'

    def finished_line(self) -> str:
        detail = '' if self.passed else f'  exit {self.exit_code}'
        return f'{self.status():7} {self.seconds:7.1f} s  {self.target.name}{detail}'

    def failing_tail(self) -> tuple[str, ...]:
        '''The last lines of output, which is where a validator prints the case that
        failed and where `execute_notebook.py` prints the traceback of a failing
        cell.'''
        if self.passed:
            return ()
        lines = [line for line in self.output.splitlines() if line.strip()]
        return tuple(lines[-FAILING_TAIL_LINES:])


def subprocess_environment(root: pathlib.Path) -> dict[str, str]:
    '''The environment a target runs under: the repository on the import path, and
    UTF-8 output, which a listing needs on Windows.'''
    return dict(os.environ, PYTHONUTF8='1', PYTHONPATH=str(root))


def run_one_target(
    target: validation_targets.ValidationTarget,
    root: pathlib.Path = REPOSITORY_ROOT,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> TargetResult:
    '''Run one target from the repository root and collect its exit code, its wall
    time and its combined output.'''
    command = [sys.executable, *target.command]
    started = time.perf_counter()
    try:
        finished = subprocess.run(
            command, capture_output=True, text=True, encoding='utf-8',
            errors='replace', cwd=root, env=subprocess_environment(root),
            timeout=timeout)
    except subprocess.TimeoutExpired as expired:
        return TargetResult(
            target=target, exit_code=124, seconds=time.perf_counter() - started,
            output=f'no output within {timeout} s: {expired}', timed_out=True)
    return TargetResult(
        target=target, exit_code=finished.returncode,
        seconds=time.perf_counter() - started,
        output=f'{finished.stdout}\n{finished.stderr}', timed_out=False)


def run_targets(
    targets: Iterable[validation_targets.ValidationTarget],
    root: pathlib.Path = REPOSITORY_ROOT,
    jobs: int = DEFAULT_JOBS,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    report: Callable[[str], None] = print,
) -> tuple[TargetResult, ...]:
    '''Run every target with at most `jobs` at once, reporting each as it finishes.

    The results come back in the order the targets were given, whatever order they
    finished in, so that two runs print the same summary.
    '''
    ordered = tuple(targets)
    if not ordered:
        return ()
    reporting = threading.Lock()
    results: dict[str, TargetResult] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        running = {
            pool.submit(run_one_target, target, root, timeout): target
            for target in ordered}
        for future in concurrent.futures.as_completed(running):
            result = future.result()
            results[result.target.name] = result
            with reporting:
                report(result.finished_line())
    return tuple(results[target.name] for target in ordered)


def failures_of(results: Iterable[TargetResult]) -> tuple[str, ...]:
    '''One message per failing target, carrying its exit code and its failing tail,
    in the form `validate_repository.py` prints a failure in.'''
    messages = []
    for result in results:
        if result.passed:
            continue
        tail = '\n    '.join(result.failing_tail())
        messages.append(f'{result.target.name}: exit {result.exit_code}\n    {tail}')
    return tuple(messages)


def print_failing_tails(
    results: Iterable[TargetResult], report: Callable[[str], None] = print,
) -> None:
    for result in results:
        if result.passed:
            continue
        report(f'--- {result.target.name}, exit {result.exit_code} ---')
        for line in result.failing_tail():
            report(f'    {line}')
