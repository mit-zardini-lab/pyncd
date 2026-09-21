# Claude Fable 5.1, effort 80.
'''Check the reading, the resolving and the writing of a wording file.

    python utilities/validate_wording_json.py

One `check_` function per claim, each printing one line, and a non-zero exit when any
fails. The claims are that a string entry, a list entry, a `$NAME` reference and a
part with fills each resolve as the module states, that a file resolves against a base
so that an entry it lacks is read from the base and a sentence it changes reaches the
base's descriptions naming it, that a missing reference and a cycle are refused with the
chain named, and that a file written by `write_wording_file` reads back as written.
Whether the wording file of DeepSeek-V4.1-Flash loads into its dataclass is checked by
`notebooks/sota/DeepSeekV41Flash/validate_deepseek_v41_flash.py`.

The checks write their files into a temporary directory. The repository root replaces
the script's own directory on the path, because `utilities/utilities.py` would otherwise
shadow the package.
'''
from __future__ import annotations

import pathlib
import sys
import tempfile
from collections.abc import Callable

sys.path[0] = str(pathlib.Path(__file__).resolve().parents[1])

import utilities.wording_json as wording_json  # noqa: E402

WORDINGS: dict[str, wording_json.WordingEntry] = {
    'MIXTURE_TITLE': '\\text{Mixture of Experts}',
    'CHOOSE_SENTENCE': 'The gate chooses six experts.',
    'SUM_SENTENCE': 'The outputs are summed.',
    'MIXTURE_DESCRIPTION': ['$CHOOSE_SENTENCE', '$SUM_SENTENCE'],
    'GROUPS_SENTENCE': 'The channels are cut into groups of {size}.',
    'ENTRY_DESCRIPTION': [
        'One entry rounded.', {'ref': 'GROUPS_SENTENCE', 'fills': {'size': 16}}],
    'LAYER_DESCRIPTION': 'Layer {layer} runs the mixture: $CHOOSE_SENTENCE',
}


def written_file(directory: pathlib.Path, name: str,
                 wordings: dict[str, wording_json.WordingEntry]) -> wording_json.WordingFile:
    return wording_json.read_wording_file(
        wording_json.write_wording_file(directory / f'{name}.json', name, wordings))


def resolved(wordings: dict[str, wording_json.WordingEntry],
             base: dict[str, wording_json.WordingEntry] | None = None) -> dict[str, str]:
    with tempfile.TemporaryDirectory() as directory:
        file = written_file(pathlib.Path(directory), 'checked', wordings)
    return wording_json.resolved_wordings(file, base)


def check_each_kind_of_entry_resolves() -> None:
    table = resolved(WORDINGS)
    if table['MIXTURE_TITLE'] != '\\text{Mixture of Experts}':
        raise AssertionError(table['MIXTURE_TITLE'])
    if table['MIXTURE_DESCRIPTION'] != 'The gate chooses six experts. The outputs are summed.':
        raise AssertionError(table['MIXTURE_DESCRIPTION'])
    if table['ENTRY_DESCRIPTION'] != 'One entry rounded. The channels are cut into groups of 16.':
        raise AssertionError(table['ENTRY_DESCRIPTION'])
    if table['GROUPS_SENTENCE'] != 'The channels are cut into groups of {size}.':
        raise AssertionError('a template lost its field')
    if table['LAYER_DESCRIPTION'] != 'Layer {layer} runs the mixture: The gate chooses six experts.':
        raise AssertionError(table['LAYER_DESCRIPTION'])


def check_a_file_resolves_against_its_base() -> None:
    table = resolved({'CHOOSE_SENTENCE': 'Six experts are chosen.'}, WORDINGS)
    if table['MIXTURE_DESCRIPTION'] != 'Six experts are chosen. The outputs are summed.':
        raise AssertionError(table['MIXTURE_DESCRIPTION'])
    if table['SUM_SENTENCE'] != 'The outputs are summed.':
        raise AssertionError('an entry the file lacks is not read from the base')
    if set(table) != set(WORDINGS):
        raise AssertionError('the resolved table does not hold every name of the base')


def check_a_missing_reference_is_refused() -> None:
    try:
        resolved({'A': 'Reads $B.'})
    except wording_json.WordingReferenceMissing as error:
        if 'A names B' not in str(error):
            raise AssertionError(f'the refusal does not name the chain: {error}')
        return
    raise AssertionError('a reference to no entry was resolved')


def check_a_cycle_is_refused() -> None:
    try:
        resolved({'A': 'Reads $B.', 'B': ['$A']})
    except wording_json.WordingReferenceCycle as error:
        if 'A -> B -> A' not in str(error):
            raise AssertionError(f'the refusal does not name the cycle: {error}')
        return
    raise AssertionError('a cycle of references was resolved')


def check_a_written_file_reads_back() -> None:
    with tempfile.TemporaryDirectory() as directory:
        file = written_file(pathlib.Path(directory), 'Current', WORDINGS)
        if file.name != 'Current':
            raise AssertionError(file.name)
        if dict(file.wordings) != WORDINGS:
            raise AssertionError('the wordings read back differ from the ones written')


def check_a_malformed_entry_is_refused() -> None:
    with tempfile.TemporaryDirectory() as directory:
        try:
            written_file(pathlib.Path(directory), 'bad', {'A': [{'fills': {}}]})
        except wording_json.WordingFileError as error:
            if 'A' not in str(error):
                raise AssertionError(f'the refusal does not name the entry: {error}')
            return
    raise AssertionError('a part with no ref was accepted')


CHECKS: tuple[Callable[[], None], ...] = (
    check_each_kind_of_entry_resolves,
    check_a_file_resolves_against_its_base,
    check_a_missing_reference_is_refused,
    check_a_cycle_is_refused,
    check_a_written_file_reads_back,
    check_a_malformed_entry_is_refused,
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
