# Claude Fable 5.1, effort 80.
'''Check the HTML file `standalone_page` writes, without a browser.

    python websocket_transfer/validate_standalone_page.py

One `check_` function per claim, each printing one line, and a non-zero exit when any
fails. The claims are that the page written from a built bundle loads no script by URL
and holds the bundle and the message, that the message read back out of the page is the
message written into it, that text which would end a `script` element early is escaped
in the bundle and in the message, that a bundle which loads its fonts from files is
refused, that the localisations of a page are written as a third element between the
message and the bundle, and that the `title` and `heading` display settings reach the
message.

The checks build a stand-in for tsncd's `dist/` in a temporary directory, so they need
no tsncd checkout. Whether the written page draws when it is opened from a disk with no
network is a claim about tsncd's bundle, and a check here cannot establish it. Open a
written page in a browser with the network disabled to establish it.
'''
from __future__ import annotations

import html.parser
import json
import pathlib
import sys
import tempfile
from collections.abc import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import websocket_transfer.send_morphism as send_morphism  # noqa: E402
import websocket_transfer.standalone_page as standalone_page  # noqa: E402
import websocket_transfer.websockets_transfer as wst  # noqa: E402

BUILT_PAGE = '''<!DOCTYPE html>
<html><head><title>tsncd</title>
<script defer="defer" src="/assets/main.0123.js"></script></head>
<body><h1 id="page-heading">tsncd</h1><div id="diagram"></div></body></html>
'''
BUNDLE = (
    'const closes = "</script>"; const opens = /<!--|<SCRIPT/; render(1 < 2);\n'
    '//# sourceMappingURL=main.0123.js.map\n')
MESSAGE: wst.DataUpdate = {
    'msgType': 'dataUpdate',
    'data': json.dumps({'uid_repository': {}, 'data': {}}),
    'settings': {'title': 'A </script> in a title', 'width': 900},
    'auxiliary': {'legend': [], 'blocks': {}, 'expansions': {}},
}


class ScriptElements(html.parser.HTMLParser):
    '''The attributes and the text of every `script` element of a document, read
    as a browser reads them: the text of a `script` element runs to the first
    `</script`.'''

    def __init__(self) -> None:
        super().__init__()
        self.elements: list[tuple[dict[str, str | None], str]] = []
        self._open: dict[str, str | None] | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == 'script':
            self._open = dict(attrs)
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._open is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == 'script' and self._open is not None:
            self.elements.append((self._open, ''.join(self._text)))
            self._open = None


def built_bundle(directory: pathlib.Path) -> pathlib.Path:
    (directory / 'assets').mkdir()
    (directory / 'index.html').write_text(BUILT_PAGE, encoding='utf-8')
    (directory / 'assets' / 'main.0123.js').write_text(BUNDLE, encoding='utf-8')
    return directory


def script_elements_of_the_written_page() -> list[tuple[dict[str, str | None], str]]:
    with tempfile.TemporaryDirectory() as directory:
        page = standalone_page.standalone_page(
            MESSAGE, built_bundle(pathlib.Path(directory)))
    parser = ScriptElements()
    parser.feed(page)
    return parser.elements


def check_the_page_holds_the_bundle_and_loads_no_script() -> None:
    elements = script_elements_of_the_written_page()
    if len(elements) != 2:
        raise AssertionError(f'{len(elements)} script elements, expected 2')
    loaded = [attributes['src'] for attributes, _ in elements if 'src' in attributes]
    if loaded:
        raise AssertionError(f'the page still loads {loaded}')
    bundle = elements[1][1]
    if 'render(1 < 2);' not in bundle:
        raise AssertionError('the bundle is not written into the page')
    if 'sourceMappingURL' in bundle:
        raise AssertionError('the source map comment names a file beside the page')


def check_the_message_reads_back_as_written() -> None:
    attributes, text = script_elements_of_the_written_page()[0]
    if attributes.get('id') != standalone_page.EMBEDDED_MESSAGE_ID:
        raise AssertionError(f'the first script element has id {attributes.get("id")}')
    if attributes.get('type') != 'application/json':
        raise AssertionError(f'the message has type {attributes.get("type")}')
    if json.loads(text) != MESSAGE:
        raise AssertionError('the message read out of the page differs from the one written')


def check_no_text_ends_a_script_element_early() -> None:
    escaped = standalone_page.text_of_script_element(BUNDLE)
    for sequence in ('</script', '<!--', '<SCRIPT'):
        if sequence in escaped:
            raise AssertionError(f'{sequence} survives in the bundle')
    if 'render(1 < 2);' not in escaped:
        raise AssertionError('a comparison was escaped, which no rule asks for')
    if '<' in standalone_page.text_of_json_element(MESSAGE):
        raise AssertionError('a < survives in the message')


def check_a_bundle_that_loads_font_files_is_refused() -> None:
    with tempfile.TemporaryDirectory() as directory:
        dist = built_bundle(pathlib.Path(directory))
        (dist / '0108e89c9003e8c14ea3.woff2').write_bytes(b'')
        try:
            standalone_page.standalone_page(MESSAGE, dist)
        except standalone_page.BundleLoadsFiles as error:
            if '1 font files' not in str(error):
                raise AssertionError(f'the refusal does not count the files: {error}')
            return
    raise AssertionError('a bundle with a font file beside it was accepted')


LOCALISATIONS: wst.EmbeddedLocalisations = {
    'default': 'Current',
    'localisations': {
        'Current': {'blocks': {}, 'expansions': {}},
        'Rewritten': {'blocks': {'3': 'A </script> in a description'},
                      'expansions': {}}}}


def check_the_localisations_are_written_between_the_message_and_the_bundle() -> None:
    with tempfile.TemporaryDirectory() as directory:
        page = standalone_page.standalone_page(
            MESSAGE, built_bundle(pathlib.Path(directory)), LOCALISATIONS)
    parser = ScriptElements()
    parser.feed(page)
    if len(parser.elements) != 3:
        raise AssertionError(f'{len(parser.elements)} script elements, expected 3')
    attributes, text = parser.elements[1]
    if attributes.get('id') != standalone_page.EMBEDDED_LOCALISATIONS_ID:
        raise AssertionError(f'the second script element has id {attributes.get("id")}')
    if attributes.get('type') != 'application/json':
        raise AssertionError(f'the localisations have type {attributes.get("type")}')
    if json.loads(text) != LOCALISATIONS:
        raise AssertionError('the localisations read out of the page differ')
    if 'render(1 < 2);' not in parser.elements[2][1]:
        raise AssertionError('the bundle is not the last script element')


def check_the_title_reaches_the_message() -> None:
    if send_morphism.display_settings(title='DeepSeekV4.1') != {'title': 'DeepSeekV4.1'}:
        raise AssertionError('display_settings does not carry the title')
    if send_morphism.display_settings(heading=wst.PageHeading.TITLE) != {'heading': 'title'}:
        raise AssertionError('display_settings does not carry the heading')
    if 'heading' in send_morphism.display_settings(title='DeepSeekV4.1'):
        raise AssertionError('display_settings sends a heading nobody asked for')
    if 'title' in send_morphism.display_settings(width=900):
        raise AssertionError('a send with no title carries one')


CHECKS: tuple[Callable[[], None], ...] = (
    check_the_page_holds_the_bundle_and_loads_no_script,
    check_the_message_reads_back_as_written,
    check_no_text_ends_a_script_element_early,
    check_a_bundle_that_loads_font_files_is_refused,
    check_the_localisations_are_written_between_the_message_and_the_bundle,
    check_the_title_reaches_the_message,
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
