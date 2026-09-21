'''The repository paths git reports as modified.

Written by Claude Opus 5 (1M context), effort high.

`git status --porcelain` reports the working tree, including untracked files, and
`git diff --name-only <ref>` reports what has changed since a commit or a branch.
Both print a path relative to the repository root with forward slashes, which is the
form `module_import_graph` indexes, so a reported path is looked up directly.

A rename is reported as `R  old -> new`, and both sides are returned: the old path
because a target that imported it now imports nothing, and the new path because a
target that imports it is reaching a file that did not exist before.
'''

from __future__ import annotations

import pathlib
import subprocess

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]


class GitCommandFailed(Exception):
    '''A git command exited non-zero, so the modified paths could not be read.'''


def git_output(arguments: tuple[str, ...], root: pathlib.Path) -> str:
    finished = subprocess.run(
        ('git', *arguments), capture_output=True, text=True, encoding='utf-8',
        errors='replace', cwd=root)
    if finished.returncode != 0:
        raise GitCommandFailed(
            f'git {" ".join(arguments)} exited {finished.returncode}: '
            f'{finished.stderr.strip()}')
    return finished.stdout


def unquoted(path: str) -> str:
    '''A path as git printed it, with the quoting it adds for a non-ASCII name
    removed.'''
    stripped = path.strip()
    if stripped.startswith('"') and stripped.endswith('"'):
        return stripped[1:-1]
    return stripped


def modified_files_in_working_tree(
    root: pathlib.Path = REPOSITORY_ROOT,
) -> tuple[str, ...]:
    '''Every path `git status --porcelain` reports, staged, unstaged or untracked.

    `--untracked-files=all` is passed because the default collapses a wholly untracked
    folder to the folder's own path, and a folder path matches no indexed file, so a
    new feature's files would select nothing.
    '''
    paths: set[str] = set()
    for line in git_output(
            ('status', '--porcelain', '--untracked-files=all'), root).splitlines():
        if len(line) < 4:
            continue
        rest = line[3:]
        if ' -> ' in rest:
            before, _, after = rest.partition(' -> ')
            paths |= {unquoted(before), unquoted(after)}
        else:
            paths.add(unquoted(rest))
    return tuple(sorted(paths))


def modified_files_since(
    ref: str, root: pathlib.Path = REPOSITORY_ROOT,
) -> tuple[str, ...]:
    '''Every path that differs between `ref` and the working tree.'''
    output = git_output(('diff', '--name-only', ref), root)
    return tuple(sorted(
        {unquoted(line) for line in output.splitlines() if line.strip()}))
