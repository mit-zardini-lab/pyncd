# Claude Opus 5.5 (1M context), effort 40.
'''Check the page that carries several variants of one model, without a browser.

    python websocket_transfer/validate_page_variants.py

One `check_` function per claim, each printing one line, and a non-zero exit when any
fails. The claims are that several JSON values compressed into one repository decode
back to themselves, that a page of three variants of attention carries one element
holding one repository and no single message, that the repository holds fewer records
than the messages compressed one at a time, that each message read back out of the page
is the message written, that the settings repeated uncompressed are the ones the lab
website's build plugin reads, that a derived variant carries its source and its functor
and no message, that a set of variants naming one another inconsistently is refused
with the values at fault, and that the listing of the derived variant is the listing of
the model in the reals. The last three claims are that the page is written as
`index.html` in a folder named by the page, with a redirect beside the folder under the
name the page had as one file, that the redirect keeps the query and the hash and falls
back to a `meta` refresh and a link inside `noscript`, and that a name that HTML would
read as markup is escaped in the redirect.

The page is written against the stand-in for tsncd's `dist/` that
`validate_standalone_page` builds, so the checks need no tsncd checkout. Whether tsncd
draws the page and its selector is a claim about tsncd's bundle, which tsncd's
`test/variant_pages.test.ts` and `test/display_selector.test.ts` check.
'''
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import html.parser
import io
import json
import pathlib
import re
import sys
import tempfile
from collections.abc import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import construction_helpers as ch  # noqa: E402,F401
import data_structure.Category as cat  # noqa: E402
import data_structure.Operators as ops  # noqa: E402
import data_transfer.json_compression as json_compression  # noqa: E402
import quantization.data_structure.Quantization as Quantization  # noqa: E402
import quantization.processing.quantise_model as quantise_model  # noqa: E402
import websocket_transfer.auxiliary_information as auxiliary_information  # noqa: E402
import websocket_transfer.standalone_page as standalone_page  # noqa: E402
import websocket_transfer.validate_standalone_page as validate_standalone_page  # noqa: E402
import websocket_transfer.websockets_transfer as wst  # noqa: E402
import notebooks.display.notebook_diagrams as notebook_diagrams  # noqa: E402

# The two patterns of `detect_forms` in `_plugins/diagrams.rb` of the lab website,
# read on 2026-09-27 from the working copy at
# `_zardinilabwebsite/mit-zardini-lab.github.io`, lines 96 and 98.
WEBSITE_FORM_PATTERN = re.compile(r'"settings": \{[^}]*"form": "([a-z-]+)"')
WEBSITE_THEME_PATTERN = re.compile(r'"settings": \{[^}]*"darkMode": (true|false)')


def attention() -> cat.Morphism:
    return (ops.Einops.template('q d,x d->q x')
            @ ops.SoftMax.template(contracted=True)
            @ ops.Einops.template('q x,x v->q v'))


ATTENTION = attention()
POLICY = quantise_model.QuantizationPolicy(
    inputs=Quantization.BF16, activations=Quantization.BF16,
    scalars=Quantization.FP32, rounded_operands=Quantization.E4M3,
    weights_by_default=Quantization.E4M3)
QUANTISED_ATTENTION = quantise_model.quantise_model(ATTENTION, POLICY).morphism

PAGE_SETTINGS = notebook_diagrams.DiagramSettings(
    mode=notebook_diagrams.DiagramMode.LISTING,
    dark_mode=notebook_diagrams.ColorMode.LIGHT,
    form=notebook_diagrams.DiagramForm.ARROWS_AND_BOXES,
    advanced_display=notebook_diagrams.AdvancedDisplay.INTERACTIVE)
DECODE = notebook_diagrams.PageVariantGroup(identifier='decode', title='Decode')
CACHED = notebook_diagrams.PageVariantGroup(identifier='cached', title='Cached')
VARIANTS = (
    notebook_diagrams.PageVariant(
        identifier='decode-quantised', group=DECODE, title='Quantised',
        detail='BF16 activations and FP32 arithmetic, a </script> in the detail',
        term=QUANTISED_ATTENTION),
    notebook_diagrams.PageVariant(
        identifier='decode-unquantised', group=DECODE, title='Unquantised',
        detail='The same model in the reals', derived_from='decode-quantised',
        functor=notebook_diagrams.PageFunctor.DEQUANTISE,
        settings=dataclasses.replace(
            PAGE_SETTINGS, dark_mode=notebook_diagrams.ColorMode.DARK)),
    notebook_diagrams.PageVariant(
        identifier='cached-unquantised', group=CACHED, title='Unquantised',
        detail='Attention written in the reals', term=ATTENTION),
)
INITIAL = 'decode-quantised'


MESSAGES: dict[str, standalone_page.VariantMessage] = {
    variant.identifier: variant
    for variant in notebook_diagrams.variant_messages(VARIANTS, PAGE_SETTINGS)}
'''Each variant as the page writer receives it, computed once, because presenting a
term under `AdvancedDisplay.INTERACTIVE` wraps each cast in a block with a fresh tag
and two computations of one message differ in those tags.'''


def written_page() -> str:
    embedded = standalone_page.embedded_variants(tuple(MESSAGES.values()), INITIAL)
    with tempfile.TemporaryDirectory() as directory:
        return standalone_page.page_with_variants(
            embedded, validate_standalone_page.built_bundle(pathlib.Path(directory)))


PAGE = written_page()


def script_elements() -> list[tuple[dict[str, str | None], str]]:
    parser = validate_standalone_page.ScriptElements()
    parser.feed(PAGE)
    return parser.elements


def embedded_read_back() -> wst.EmbeddedVariants:
    attributes, text = script_elements()[0]
    if attributes.get('id') != standalone_page.EMBEDDED_VARIANTS_ID:
        raise AssertionError(f'the first script element has id {attributes.get("id")}')
    return json.loads(text)


def record_of(identifier: str) -> wst.PageVariantRecord:
    records = {record['id']: record for record in embedded_read_back()['variants']}
    return records[identifier]


def check_several_values_compressed_together_decode_to_themselves() -> None:
    shared_part = {'uid_repository': {'7': {'__type__': 'Axis'}}, 'data': [1, 2]}
    values = [{'first': shared_part, 'n': 1}, {'second': shared_part, 'n': 2}, 'text']
    shared = json_compression.compress_json_values(values)
    decoded = json_compression.decoded_repository(shared.value_repository)
    read = [decoded[root] for root in shared.roots]
    if read != values:
        raise AssertionError(f'{read} decoded from {values}')
    separate = sum(len(json_compression.compress_json(value)['value_repository'])
                   for value in values)
    if len(shared.value_repository) >= separate:
        raise AssertionError(
            f'{len(shared.value_repository)} records together and {separate} apart')


def check_the_page_carries_one_variants_element_and_no_message() -> None:
    elements = script_elements()
    identifiers = [attributes.get('id') for attributes, _ in elements]
    if identifiers != [standalone_page.EMBEDDED_VARIANTS_ID, None]:
        raise AssertionError(f'script elements with ids {identifiers}')
    if standalone_page.EMBEDDED_MESSAGE_ID in PAGE:
        raise AssertionError(f'the page names {standalone_page.EMBEDDED_MESSAGE_ID}')
    if elements[0][0].get('type') != 'application/json':
        raise AssertionError(f'the variants have type {elements[0][0].get("type")}')
    if '<' in standalone_page.text_of_variants_element(embedded_read_back()):
        raise AssertionError('a < survives in the variants element')


def check_every_message_reads_back_from_the_one_repository() -> None:
    embedded = embedded_read_back()
    decoded = json_compression.decoded_repository(embedded['value_repository'])
    written = MESSAGES
    for record in embedded['variants']:
        if 'message' not in record:
            continue
        expected = json.loads(json.dumps(written[record['id']].message))
        if decoded[record['message']] != expected:
            raise AssertionError(f'the message of {record["id"]} reads back differently')
        if not isinstance(decoded[record['message']]['data'], dict):
            raise AssertionError(f'the term of {record["id"]} is not a JSON object')


def check_the_repository_is_smaller_than_the_messages_compressed_apart() -> None:
    together = len(embedded_read_back()['value_repository'])
    apart = sum(
        len(json_compression.compress_json(variant.message)['value_repository'])
        for variant in MESSAGES.values() if variant.message is not None)
    if together >= apart:
        raise AssertionError(f'{together} records together and {apart} apart')


def check_the_initial_settings_are_read_by_the_website() -> None:
    embedded = embedded_read_back()
    form = WEBSITE_FORM_PATTERN.search(PAGE)
    theme = WEBSITE_THEME_PATTERN.search(PAGE)
    if form is None or form[1] != wst.DiagramForm.ARROWS_AND_BOXES.value:
        raise AssertionError(f'the website reads the form {form and form[1]!r}')
    if theme is None or theme[1] != 'false':
        raise AssertionError(f'the website reads darkMode {theme and theme[1]!r}')
    initial_message = MESSAGES[INITIAL].message
    if embedded['settings'] != standalone_page.with_form_and_theme_first(
            initial_message['settings']):
        raise AssertionError('the uncompressed settings differ from the initial ones')
    if '<meta name="tsncd-dark-mode" content="false">' not in PAGE:
        raise AssertionError('the head does not state the light theme')
    if embedded['initial'] != INITIAL:
        raise AssertionError(f'the page opens on {embedded["initial"]!r}')


def check_the_derived_variant_names_its_source_and_its_functor() -> None:
    record = record_of('decode-unquantised')
    expected = {
        'id': 'decode-unquantised', 'group': 'decode', 'title': 'Unquantised',
        'detail': 'The same model in the reals', 'derivedFrom': 'decode-quantised',
        'functor': 'dequantise'}
    if {key: record.get(key) for key in expected} != expected:
        raise AssertionError(f'the derived variant reads {record}')
    if 'message' in record:
        raise AssertionError('the derived variant carries a message')
    if record.get('settings', {}).get('darkMode') is not True:
        raise AssertionError(f'the derived variant has settings {record.get("settings")}')
    groups = embedded_read_back()['groups']
    if groups != [{'id': 'decode', 'title': 'Decode'}, {'id': 'cached', 'title': 'Cached'}]:
        raise AssertionError(f'the groups read {groups}')


def refusal(variants: tuple[notebook_diagrams.PageVariant, ...], initial: str) -> str:
    try:
        notebook_diagrams.check_page_variants(variants, initial)
    except standalone_page.InconsistentVariants as error:
        return str(error)
    raise AssertionError(f'{[variant.identifier for variant in variants]} were accepted')


def check_inconsistent_variants_are_refused_with_the_values_at_fault() -> None:
    quantised, derived, cached = VARIANTS
    derived_from_derived = notebook_diagrams.PageVariant(
        identifier='twice', group=DECODE, title='Twice', detail='',
        derived_from='decode-unquantised',
        functor=notebook_diagrams.PageFunctor.DEQUANTISE)
    drawn_with_a_functor = notebook_diagrams.PageVariant(
        identifier='drawn', group=DECODE, title='Drawn', detail='', term=ATTENTION,
        functor=notebook_diagrams.PageFunctor.DEQUANTISE)
    derived_by_nothing = notebook_diagrams.PageVariant(
        identifier='by-nothing', group=DECODE, title='By nothing', detail='',
        derived_from='decode-quantised')
    cases = (
        ((quantised, quantised), INITIAL, "['decode-quantised']"),
        ((quantised,), 'cached-unquantised', "'cached-unquantised'"),
        ((quantised, derived, derived_from_derived), INITIAL, "'decode-unquantised'"),
        ((derived, cached), 'cached-unquantised', "'decode-quantised'"),
        ((quantised, drawn_with_a_functor), INITIAL, "'drawn'"),
        ((quantised, derived_by_nothing), INITIAL, "'by-nothing'"),
    )
    for variants, initial, named in cases:
        message = refusal(variants, initial)
        if named not in message:
            raise AssertionError(f'the refusal {message!r} does not name {named}')


def check_the_listing_of_the_derived_variant_is_the_model_in_the_reals() -> None:
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        asyncio.run(notebook_diagrams.show_page_variants(
            VARIANTS, settings=PAGE_SETTINGS, slug='attention-variants'))
    _, quantised, derived, in_the_reals = re.split(
        r'^(?:Decode|Cached) / \w+\n', printed.getvalue(), flags=re.MULTILINE)
    if derived != in_the_reals:
        raise AssertionError(
            f'the derived variant lists\n{derived}\nand the model in the reals '
            f'lists\n{in_the_reals}')
    if 'BF16' not in quantised or 'BF16' in derived:
        raise AssertionError(f'the quantised variant lists\n{quantised}')


def check_the_legends_are_the_legends_the_page_carries() -> None:
    legends = notebook_diagrams.page_variant_legends(VARIANTS, PAGE_SETTINGS)
    if sorted(legends) != sorted(variant.identifier for variant in VARIANTS):
        raise AssertionError(f'legends for {sorted(legends)}')
    embedded = embedded_read_back()
    decoded = json_compression.decoded_repository(embedded['value_repository'])
    for record in embedded['variants']:
        if 'message' not in record:
            continue
        carried = decoded[record['message']]['auxiliary']['legend']
        if json.loads(json.dumps(legends[record['id']])) != carried:
            raise AssertionError(f'the legend of {record["id"]} differs from the page')
    if legends['decode-unquantised'] != legends['cached-unquantised']:
        raise AssertionError(
            f'the derived variant has the legend {legends["decode-unquantised"]} and '
            f'the model in the reals {legends["cached-unquantised"]}')
    without_legend = dataclasses.replace(
        PAGE_SETTINGS, advanced_display=notebook_diagrams.AdvancedDisplay.OFF)
    if any(notebook_diagrams.page_variant_legends(VARIANTS[2:], without_legend).values()):
        raise AssertionError('a variant drawn with no legend has legend rows')


PROJECTION = ops.Linear.template(1, 1, name='W') @ ops.SoftMax.template()
QUANTISED_ROLE = auxiliary_information.OperatorRole(
    role='The projection of the model. The weight is held in E4M3.')
UNQUANTISED_ROLE = auxiliary_information.OperatorRole(
    role='The projection of the model.')


def roles_page_variants() -> tuple[notebook_diagrams.PageVariant, ...]:
    '''A quantised projection whose role names the quantisation of its weight, and
    the unquantised variant derived from it with a role naming none.'''
    quantised_settings = dataclasses.replace(
        PAGE_SETTINGS, operator_roles={'W': QUANTISED_ROLE})
    return (
        notebook_diagrams.PageVariant(
            identifier='decode-quantised', group=DECODE, title='Quantised',
            detail='E4M3 weights', term=quantise_model.quantise_model(
                PROJECTION, POLICY).morphism,
            settings=quantised_settings),
        notebook_diagrams.PageVariant(
            identifier='decode-unquantised', group=DECODE, title='Unquantised',
            detail='The same model in the reals', derived_from='decode-quantised',
            functor=notebook_diagrams.PageFunctor.DEQUANTISE,
            settings=dataclasses.replace(
                quantised_settings, operator_roles={'W': UNQUANTISED_ROLE})),
        notebook_diagrams.PageVariant(
            identifier='decode-dark', group=DECODE, title='Dark',
            detail='The same model in the reals, dark', derived_from='decode-quantised',
            functor=notebook_diagrams.PageFunctor.DEQUANTISE,
            settings=dataclasses.replace(
                quantised_settings, dark_mode=notebook_diagrams.ColorMode.DARK)),
    )


def check_a_derived_variant_with_other_roles_carries_its_own_auxiliary() -> None:
    embedded = standalone_page.embedded_variants(
        notebook_diagrams.variant_messages(roles_page_variants(), PAGE_SETTINGS),
        'decode-quantised')
    decoded = json_compression.decoded_repository(embedded['value_repository'])
    records = {record['id']: record for record in embedded['variants']}
    if 'auxiliary' in records['decode-dark']:
        raise AssertionError('a derived variant with the roles of its source carries '
                             'an auxiliary of its own')
    if 'auxiliary' not in records['decode-unquantised']:
        raise AssertionError('a derived variant with other roles carries no auxiliary')
    quantised = decoded[records['decode-quantised']['message']]['auxiliary']
    unquantised = decoded[records['decode-unquantised']['auxiliary']]
    if sorted(unquantised['expansions']) != sorted(quantised['expansions']):
        raise AssertionError(
            f'the derived auxiliary keys the expansions {sorted(unquantised["expansions"])}'
            f' and the source {sorted(quantised["expansions"])}')
    if sorted(unquantised.get('blocks', {})) != sorted(quantised.get('blocks', {})):
        raise AssertionError('the derived auxiliary keys the blocks otherwise')
    descriptions = [expansion['description']
                    for expansion in unquantised['expansions'].values()]
    if not any(description.startswith(UNQUANTISED_ROLE.role)
               and 'E4M3' not in description for description in descriptions):
        raise AssertionError(f'no expansion says the unquantised role: {descriptions}')
    if not any(QUANTISED_ROLE.role in expansion['description']
               for expansion in quantised['expansions'].values()):
        raise AssertionError('the quantised variant does not say its role')


FOLDER_SLUG = 'Attention'


class RedirectElements(html.parser.HTMLParser):
    '''The text of every `script` element of a document, and the `content` of every
    `meta` refresh and the `href` of every link, each beside whether it stands inside
    a `noscript` element.'''

    def __init__(self) -> None:
        super().__init__()
        self.scripts: list[str] = []
        self.refreshes: list[tuple[str | None, bool]] = []
        self.links: list[tuple[str | None, bool]] = []
        self.open_noscript_elements = 0
        self.is_in_script = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        is_in_noscript = self.open_noscript_elements > 0
        if tag == 'noscript':
            self.open_noscript_elements += 1
        elif tag == 'script':
            self.is_in_script = True
            self.scripts.append('')
        elif tag == 'meta' and attributes.get('http-equiv') == 'refresh':
            self.refreshes.append((attributes.get('content'), is_in_noscript))
        elif tag == 'a':
            self.links.append((attributes.get('href'), is_in_noscript))

    def handle_endtag(self, tag: str) -> None:
        if tag == 'noscript':
            self.open_noscript_elements -= 1
        elif tag == 'script':
            self.is_in_script = False

    def handle_data(self, data: str) -> None:
        if self.is_in_script:
            self.scripts[-1] += data


def redirect_elements(slug: str) -> RedirectElements:
    parser = RedirectElements()
    parser.feed(standalone_page.redirect_to_page_folder(slug))
    return parser


def check_the_page_is_written_into_a_folder_with_a_redirect_beside_it() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        (root / 'dist').mkdir()
        dist = validate_standalone_page.built_bundle(root / 'dist')
        written = standalone_page.save_page_folder_with_variants(
            tuple(MESSAGES.values()), root / 'pages', FOLDER_SLUG, INITIAL, dist)
        expected = standalone_page.PageFolder(
            folder=root / 'pages' / FOLDER_SLUG,
            page=root / 'pages' / FOLDER_SLUG / 'index.html',
            redirect=root / 'pages' / f'{FOLDER_SLUG}.html')
        if written != expected:
            raise AssertionError(f'{written} written, expected {expected}')
        beside = sorted(path.name for path in (root / 'pages').iterdir())
        inside = sorted(path.name for path in written.folder.iterdir())
        if beside != [FOLDER_SLUG, f'{FOLDER_SLUG}.html'] or inside != ['index.html']:
            raise AssertionError(
                f'{beside} in the page directory and {inside} in the folder')
        if written.page.read_text('utf-8') != PAGE:
            raise AssertionError(f'{written.page} differs from the page with variants')
        if (written.redirect.read_text('utf-8')
                != standalone_page.redirect_to_page_folder(FOLDER_SLUG)):
            raise AssertionError(
                f'{written.redirect} is not the redirect to the folder')


def check_the_redirect_keeps_the_query_and_the_hash() -> None:
    elements = redirect_elements(FOLDER_SLUG)
    if len(elements.scripts) != 1:
        raise AssertionError(f'{len(elements.scripts)} script elements in the redirect')
    script = ' '.join(elements.scripts[0].split())
    sent_to = (f'location.replace("{FOLDER_SLUG}/" + (location.protocol === "file:" ? '
               '"index.html" : "") + location.search + location.hash);')
    if script != sent_to:
        raise AssertionError(f'the redirect runs {script!r}, expected {sent_to!r}')
    if elements.refreshes != [(f'0; url={FOLDER_SLUG}/', True)]:
        raise AssertionError(f'the refreshes of the redirect are {elements.refreshes}')
    if elements.links != [(f'{FOLDER_SLUG}/', True)]:
        raise AssertionError(f'the links of the redirect are {elements.links}')


def check_a_slug_read_as_markup_is_escaped_in_the_redirect() -> None:
    elements = redirect_elements('A&B </script>')
    escaped = 'A%26B%20%3C%2Fscript%3E/'
    if len(elements.scripts) != 1 or f'"{escaped}"' not in elements.scripts[0]:
        raise AssertionError(f'the redirect runs {elements.scripts}')
    if elements.links != [(escaped, True)]:
        raise AssertionError(f'the links of the redirect are {elements.links}')


CHECKS: tuple[Callable[[], None], ...] = (
    check_several_values_compressed_together_decode_to_themselves,
    check_the_page_carries_one_variants_element_and_no_message,
    check_every_message_reads_back_from_the_one_repository,
    check_the_repository_is_smaller_than_the_messages_compressed_apart,
    check_the_initial_settings_are_read_by_the_website,
    check_the_derived_variant_names_its_source_and_its_functor,
    check_inconsistent_variants_are_refused_with_the_values_at_fault,
    check_the_listing_of_the_derived_variant_is_the_model_in_the_reals,
    check_the_legends_are_the_legends_the_page_carries,
    check_a_derived_variant_with_other_roles_carries_its_own_auxiliary,
    check_the_page_is_written_into_a_folder_with_a_redirect_beside_it,
    check_the_redirect_keeps_the_query_and_the_hash,
    check_a_slug_read_as_markup_is_escaped_in_the_redirect,
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
