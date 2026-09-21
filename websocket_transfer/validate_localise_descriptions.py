# Claude Fable 5.1, effort 80.
'''Check that a description is read back as its constants and written from another
table.

    python websocket_transfer/validate_localise_descriptions.py

One `check_` function per claim, each printing one line, and a non-zero exit when any
fails. The claims are that a description equal to a constant takes the wording of the
alternative, that a description filled from a template is filled again from the
alternative template with the same values, that a fill which is itself a constant is
localised, that a run outside the table is kept while the constants around it change,
that a table holding the sentences of a description and not the description localises
it, that a LaTeX title is read as its own text, that the differences of a nested
auxiliary are collected under the keys of the auxiliary, and that a template whose
fields differ between the tables is refused.

The checks use a table of their own, so they build no model.
'''
from __future__ import annotations

import pathlib
import sys
from collections.abc import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import websocket_transfer.localise_descriptions as localise_descriptions  # noqa: E402
import websocket_transfer.websockets_transfer as wst  # noqa: E402

FIRST_SENTENCE = 'The gate chooses six experts.'
SECOND_SENTENCE = 'The outputs are summed.'
TABLE = {
    'MIXTURE_TITLE': '\\text{Mixture of Experts}',
    'CHOOSE_SENTENCE': FIRST_SENTENCE,
    'SUM_SENTENCE': SECOND_SENTENCE,
    'MIXTURE_DESCRIPTION': f'{FIRST_SENTENCE} {SECOND_SENTENCE}',
    'LAYER_DESCRIPTION': 'Layer {layer} runs the mixture. {tail}',
    'POSITION_DESCRIPTION': 'Reads position {position} and returns position {position}.',
}
ALTERNATIVE = {
    'CHOOSE_SENTENCE': 'Six experts are chosen by the gate.',
    'SUM_SENTENCE': 'The six outputs are added.',
    'MIXTURE_DESCRIPTION': 'The mixture, in other words.',
    'LAYER_DESCRIPTION': 'The mixture runs in layer {layer}. {tail}',
    'POSITION_DESCRIPTION': 'Position {position} is read and returned.',
}
EXPORTED = localise_descriptions.ExportedWording.of(TABLE)


def localised(description: str, alternative: dict[str, str] = ALTERNATIVE) -> str:
    return EXPORTED.localised(description, alternative)


def check_a_constant_takes_the_alternative_wording() -> None:
    if localised(FIRST_SENTENCE) != ALTERNATIVE['CHOOSE_SENTENCE']:
        raise AssertionError(localised(FIRST_SENTENCE))
    if localised(TABLE['MIXTURE_DESCRIPTION']) != ALTERNATIVE['MIXTURE_DESCRIPTION']:
        raise AssertionError('the longest constant is not preferred')


def check_a_template_is_filled_again_with_the_same_values() -> None:
    written = localised(TABLE['LAYER_DESCRIPTION'].format(layer=14, tail='Nothing.'))
    if written != 'The mixture runs in layer 14. Nothing.':
        raise AssertionError(written)
    written = localised(TABLE['POSITION_DESCRIPTION'].format(position=3))
    if written != 'Position 3 is read and returned.':
        raise AssertionError(written)


def check_a_fill_that_is_a_constant_is_localised() -> None:
    written = localised(TABLE['LAYER_DESCRIPTION'].format(layer=1, tail=SECOND_SENTENCE))
    if written != 'The mixture runs in layer 1. The six outputs are added.':
        raise AssertionError(written)


def check_a_run_outside_the_table_is_kept() -> None:
    written = localised(f'The router weight. It scores experts. {FIRST_SENTENCE}')
    if written != f'The router weight. It scores experts. {ALTERNATIVE["CHOOSE_SENTENCE"]}':
        raise AssertionError(written)
    if localised('No constant here.') != 'No constant here.':
        raise AssertionError('a run outside the table changed')


def check_a_table_of_sentences_localises_a_joined_description() -> None:
    sentences_alone = {
        'CHOOSE_SENTENCE': ALTERNATIVE['CHOOSE_SENTENCE'],
        'SUM_SENTENCE': ALTERNATIVE['SUM_SENTENCE']}
    written = localised(TABLE['MIXTURE_DESCRIPTION'], sentences_alone)
    if written != 'Six experts are chosen by the gate. The six outputs are added.':
        raise AssertionError(written)


def check_a_latex_title_is_read_as_its_own_text() -> None:
    if localise_descriptions.template_fields(TABLE['MIXTURE_TITLE']) != ():
        raise AssertionError('a LaTeX title was read as a template')
    pattern = localise_descriptions.wording_pattern(TABLE['MIXTURE_TITLE'])
    if pattern.match(TABLE['MIXTURE_TITLE']) is None:
        raise AssertionError('a title does not match its own text')


def check_the_differences_of_a_nested_auxiliary_are_collected() -> None:
    auxiliary: wst.DiagramAuxiliary = {
        'blocks': {
            '1': {'title': None, 'formula': None, 'references': [],
                  'description': FIRST_SENTENCE},
            '2': {'title': None, 'formula': None, 'references': [],
                  'description': 'Not in the table.'},
            '3': {'title': None, 'formula': None, 'references': [],
                  'description': None}},
        'expansions': {
            '7': {'operator': 'x', 'latex': None, 'formula': '', 'expansion': '',
                  'references': [], 'description': SECOND_SENTENCE,
                  'auxiliary': {'blocks': {'9': {
                      'title': None, 'formula': None, 'references': [],
                      'description': TABLE['MIXTURE_DESCRIPTION']}}}},
            '8': {'operator': 'y', 'latex': None, 'formula': '', 'expansion': '',
                  'references': [], 'description': 'Not in the table.',
                  'auxiliary': {}}}}
    embedded = localise_descriptions.embedded_localisations(auxiliary, (
        localise_descriptions.Localisation('Current', TABLE),
        localise_descriptions.Localisation('Rewritten', ALTERNATIVE)))
    expected: wst.EmbeddedLocalisations = {
        'default': 'Current',
        'localisations': {
            'Current': {'blocks': {}, 'expansions': {}},
            'Rewritten': {
                'blocks': {'1': ALTERNATIVE['CHOOSE_SENTENCE']},
                'expansions': {'7': {
                    'description': ALTERNATIVE['SUM_SENTENCE'],
                    'auxiliary': {
                        'blocks': {'9': ALTERNATIVE['MIXTURE_DESCRIPTION']},
                        'expansions': {}}}}}}}
    if embedded != expected:
        raise AssertionError(embedded)
    if localise_descriptions.unattributed_descriptions(auxiliary, EXPORTED) != [
            'Not in the table.', 'Not in the table.']:
        raise AssertionError('the runs outside the table are not listed')


def check_a_template_with_other_fields_is_refused() -> None:
    try:
        localised(TABLE['LAYER_DESCRIPTION'].format(layer=2, tail='Nothing.'),
                  {'LAYER_DESCRIPTION': 'Layer {number}. {tail}'})
    except localise_descriptions.LocalisationFieldsDiffer as error:
        if 'LAYER_DESCRIPTION' not in str(error):
            raise AssertionError(f'the refusal does not name the constant: {error}')
        return
    raise AssertionError('a template with a field the exported table lacks was written')


CHECKS: tuple[Callable[[], None], ...] = (
    check_a_constant_takes_the_alternative_wording,
    check_a_template_is_filled_again_with_the_same_values,
    check_a_fill_that_is_a_constant_is_localised,
    check_a_run_outside_the_table_is_kept,
    check_a_table_of_sentences_localises_a_joined_description,
    check_a_latex_title_is_read_as_its_own_text,
    check_the_differences_of_a_nested_auxiliary_are_collected,
    check_a_template_with_other_fields_is_refused,
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
