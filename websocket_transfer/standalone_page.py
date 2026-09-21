# Claude Fable 5.1, effort 80.
'''Writing a figure as one HTML file that opens without a server or a network.

    save_standalone_page(
        attention, 'outputs/pages/attention.html',
        settings=send_morphism.display_settings(title='Attention'))

The file is tsncd's built page with two things written into it. The bundle the page
loads from `assets/` is written into a `script` element, and the `dataUpdate` a
notebook would send to the server is written into a second `script` element of type
`application/json`. tsncd's entry point draws a message it finds in the page and then
opens no websocket, so the file shows its figure when it is opened from a disk, and a
server on port 8765 cannot replace that figure. The legend and the inspection boxes
answer the pointer as they do on the open page, because the page runs the same bundle
and the message carries the same `auxiliary` field.
`obsidian/05-backends/Diagram Wire Format.md` states the embedded form.

Nothing is fetched once the file is open. tsncd's webpack configuration writes KaTeX's
fonts into the bundle as data URIs, and the bundle holds the stylesheet. A bundle built
before that configuration loads its fonts from files beside it, which a page opened from
`file://` cannot reach, and `inline_bundle` refuses such a build and names the font
files it found. A link an inspection box lists to code on a web host still needs a
network when it is followed.

No browser is started here. The page is assembled as text, so writing one takes as long
as exporting the term does.
'''

import json
import pathlib
import re
from collections.abc import Sequence

import data_transfer.term_json as term_json
import websocket_transfer.headless as headless
import websocket_transfer.localise_descriptions as localise_descriptions
import websocket_transfer.send_morphism as send_morphism
import websocket_transfer.websockets_transfer as wst


type Sendable = send_morphism.Sendable

# The id tsncd's `src/data_transfer/embedded_message.ts` reads the message from.
EMBEDDED_MESSAGE_ID = 'tsncd-embedded-message'
# The id tsncd's `src/data_transfer/embedded_localisations.ts` reads the
# localisations from.
EMBEDDED_LOCALISATIONS_ID = 'tsncd-localisations'

LOADED_SCRIPT = re.compile(
    r'<script\b[^>]*\bsrc="(?P<source>[^"]+)"[^>]*>\s*</script>')
SOURCE_MAP_COMMENT = re.compile(r'\n//# sourceMappingURL=\S+\s*$')
OPENS_OR_ENDS_A_SCRIPT_ELEMENT = re.compile(
    r'<(?=!--|/?script)', re.IGNORECASE)
JAVASCRIPT_ESCAPE_OF_LESS_THAN = r'\x3C'
JSON_ESCAPE_OF_LESS_THAN = f'\\u{ord("<"):04x}'
END_OF_BODY = '</body>'
FONT_SUFFIXES = ('.woff2', '.woff', '.ttf', '.eot')


class BundleLoadsFiles(ValueError):
    '''The built page depends on a file that cannot be written into it.'''


def text_of_script_element(script: str) -> str:
    '''`script` with the `<` of every `<!--`, `<script` and `</script` written as
    the JavaScript escape of that character, which the HTML standard recommends
    for a script written inline. The three sequences end or escape a `script`
    element wherever they stand, and in a bundle they stand inside a string, a
    template or a regular expression literal, each of which reads the escape as
    the same character.'''
    return OPENS_OR_ENDS_A_SCRIPT_ELEMENT.sub(
        lambda _: JAVASCRIPT_ESCAPE_OF_LESS_THAN, script)


def text_of_json_element(
    message: wst.DataUpdate | wst.EmbeddedLocalisations,
) -> str:
    '''`message` as JSON with every `<` written as its escape. A `<` occurs in JSON
    only inside a string, where the escape reads as the same character, and
    without one a `</script` in a description would end the element.'''
    return json.dumps(message).replace('<', JSON_ESCAPE_OF_LESS_THAN)


def font_files(dist: pathlib.Path) -> list[pathlib.Path]:
    return sorted(path for path in dist.iterdir() if path.suffix in FONT_SUFFIXES)


def inline_bundle(page: str, dist: pathlib.Path) -> tuple[str, str]:
    '''`page` without the elements that load a script from `dist`, and the text of
    those scripts in the order the page loads them.'''
    fonts = font_files(dist)
    if fonts:
        raise BundleLoadsFiles(
            f'{dist} holds {len(fonts)} font files, the first {fonts[0].name}, so '
            'its bundle loads KaTeX\'s fonts by URL and a page opened from a file '
            'cannot reach them. tsncd\'s webpack configuration has written the '
            'fonts into the bundle since 2026-09-16, and `npm run build` in the '
            'tsncd checkout rebuilds it.')
    sources = [match['source'] for match in LOADED_SCRIPT.finditer(page)]
    if not sources:
        raise BundleLoadsFiles(f'{dist / "index.html"} loads no script.')
    scripts = [
        SOURCE_MAP_COMMENT.sub('', (dist / source.lstrip('/')).read_text('utf-8'))
        for source in sources]
    return LOADED_SCRIPT.sub('', page), '\n'.join(scripts)


def json_element(
    element_id: str, message: wst.DataUpdate | wst.EmbeddedLocalisations,
) -> str:
    return (f'<script type="application/json" id="{element_id}">'
            f'{text_of_json_element(message)}</script>\n')


def standalone_page(
    message: wst.DataUpdate,
    dist: str | pathlib.Path | None = None,
    localisations: wst.EmbeddedLocalisations | None = None,
) -> str:
    '''tsncd's built page as one HTML document holding its bundle, `message` and,
    when given, `localisations` for the page's toggle between wordings.

    The elements are written at the end of the body, the JSON ones before the
    bundle. The page's own build loads the bundle with `defer`, which runs it once
    the document is parsed, and a script written inline at the end of the body runs
    at the same point.
    '''
    found = headless.find_dist(dist)
    page, bundle = inline_bundle((found / 'index.html').read_text('utf-8'), found)
    before, end_of_body, after = page.rpartition(END_OF_BODY)
    if not end_of_body:
        raise BundleLoadsFiles(f'{found / "index.html"} has no {END_OF_BODY}.')
    return ''.join((
        before,
        json_element(EMBEDDED_MESSAGE_ID, message),
        '' if localisations is None
        else json_element(EMBEDDED_LOCALISATIONS_ID, localisations),
        '<script>',
        text_of_script_element(bundle),
        '</script>\n',
        end_of_body,
        after))


def save_standalone_page(
    target: Sendable,
    path: str | pathlib.Path,
    recycle: bool = False,
    *,
    settings: wst.RenderHandlerSettings | None = None,
    auxiliary: wst.DiagramAuxiliary | None = None,
    dist: str | pathlib.Path | None = None,
    localisations: Sequence[localise_descriptions.Localisation] = (),
) -> pathlib.Path:
    '''Write `target` to `path` as a page that opens from the file.

    `settings` is what `send_morphism.display_settings` collects, and its `title`
    names the page. `auxiliary` is assembled from the morphism this call exports,
    so a caller that passes it converts first and leaves `recycle` off.
    `localisations`, when two or more are given, are the wordings the page can
    switch its descriptions between, the first being the table `auxiliary` was
    written from.
    '''
    if localisations and auxiliary is None:
        raise ValueError(
            f'{len(localisations)} localisations and no auxiliary to localise')
    morphism = send_morphism.to_morphism(target, recycle=recycle)
    message: wst.DataUpdate = wst.with_auxiliary({
        'msgType': 'dataUpdate',
        'data': term_json.TermJSONConverter.export_to_json(morphism),
        'settings': settings if settings is not None
        else wst.RenderHandlerSettings(),
    }, auxiliary)
    embedded = (
        localise_descriptions.embedded_localisations(auxiliary, localisations)
        if localisations and auxiliary is not None else None)
    written = pathlib.Path(path)
    written.parent.mkdir(parents=True, exist_ok=True)
    written.write_text(standalone_page(message, dist, embedded), encoding='utf-8')
    return written
