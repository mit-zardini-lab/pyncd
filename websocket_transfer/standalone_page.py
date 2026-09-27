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

A page can carry several variants of one model in place of one message, such as the
model in the reals and the model at its released quantisations, and tsncd draws a
selector between them. `save_page_with_variants` writes such a page. Every variant
drawn from a term of its own carries a `dataUpdate`, all of them compressed into one
value repository so that the parts two variants share are stored once, and a variant
tsncd derives from another by a functor carries only the name of the functor. The
section *A page that carries several variants* of the wire format note states the
element. Written by Claude Opus 5.5 (1M context), effort 40, on 2026-09-27.

`save_page_folder_with_variants` writes a page of several variants as `index.html` in a
folder named by the page, and a redirect beside the folder under the name the page had
as one file. The redirect keeps the query and the hash of the address it was opened at.
Added by Claude Opus 5.5 (1M context), effort 40, on 2026-09-27.
'''

import html
import json
import pathlib
import re
import urllib.parse
from collections.abc import Sequence
from dataclasses import dataclass

import data_transfer.json_compression as json_compression
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
# The id tsncd reads the variants of a page from, in place of a message.
EMBEDDED_VARIANTS_ID = 'tsncd-variants'

LOADED_SCRIPT = re.compile(
    r'<script\b[^>]*\bsrc="(?P<source>[^"]+)"[^>]*>\s*</script>')
SOURCE_MAP_COMMENT = re.compile(r'\n//# sourceMappingURL=\S+\s*$')
OPENS_OR_ENDS_A_SCRIPT_ELEMENT = re.compile(
    r'<(?=!--|/?script)', re.IGNORECASE)
JAVASCRIPT_ESCAPE_OF_LESS_THAN = r'\x3C'
JSON_ESCAPE_OF_LESS_THAN = f'\\u{ord("<"):04x}'
END_OF_BODY = '</body>'
START_OF_HEAD = '<head>'
# The element the script in the head of tsncd's `public/index.html` reads the
# theme of the page from, before the bundle runs.
THEME_ELEMENT = '<meta name="tsncd-dark-mode" content="{}">'
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


def with_theme_stated(page: str, settings: wst.RenderHandlerSettings | None) -> str:
    '''`page` with the theme of `settings` stated at the start of its head.

    tsncd's page paints itself dark until its bundle has read the message, which in
    a page of several megabytes is after the first frame, so a light figure opened
    on a dark page for a moment. The script at the top of the page's head reads
    this element and paints the page light from its first frame. Settings with no
    `darkMode` are drawn dark, which the page is already painted in.
    '''
    dark_mode = (settings or {}).get('darkMode')
    if not isinstance(dark_mode, bool):
        return page
    before, start_of_head, after = page.partition(START_OF_HEAD)
    if not start_of_head:
        return page
    return ''.join((before, start_of_head,
                    THEME_ELEMENT.format('true' if dark_mode else 'false'), after))


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
    return page_holding(
        json_element(EMBEDDED_MESSAGE_ID, message)
        + ('' if localisations is None
           else json_element(EMBEDDED_LOCALISATIONS_ID, localisations)),
        message.get('settings'), dist)


def page_holding(
    json_elements: str,
    settings: wst.RenderHandlerSettings | None,
    dist: str | pathlib.Path | None = None,
) -> str:
    '''tsncd's built page as one HTML document holding its bundle and
    `json_elements`, with the theme of `settings` stated at the start of its head.'''
    found = headless.find_dist(dist)
    page, bundle = inline_bundle((found / 'index.html').read_text('utf-8'), found)
    page = with_theme_stated(page, settings)
    before, end_of_body, after = page.rpartition(END_OF_BODY)
    if not end_of_body:
        raise BundleLoadsFiles(f'{found / "index.html"} has no {END_OF_BODY}.')
    return ''.join((
        before,
        json_elements,
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


# ==========================================================================
# A page that carries several variants.
# ==========================================================================
class InconsistentVariants(ValueError):
    '''The variants of a page do not name one another consistently.'''


@dataclass(frozen=True)
class VariantOutline:
    '''What a check of the variants of a page reads of each one: its identifier,
    the variant it is derived from, the functor deriving it, and whether it
    carries a figure of its own.'''
    identifier: str
    derived_from: str | None
    functor: str | None
    carries_its_own_figure: bool


def check_variant_outlines(outlines: Sequence[VariantOutline], initial: str) -> None:
    '''Raise `InconsistentVariants` unless the identifiers are distinct, `initial`
    is one of them, every variant either carries its own figure or is derived and
    is not both, a variant names a functor exactly when it is derived, and every
    derived variant is derived from a variant carrying its own figure.'''
    identifiers = [outline.identifier for outline in outlines]
    repeated = sorted({identifier for identifier in identifiers
                       if identifiers.count(identifier) > 1})
    if repeated:
        raise InconsistentVariants(f'the identifiers {repeated} name two variants each')
    if initial not in identifiers:
        raise InconsistentVariants(
            f'the initial variant {initial!r} is not one of {identifiers}')
    drawn = sorted(outline.identifier for outline in outlines
                   if outline.carries_its_own_figure)
    for outline in outlines:
        is_derived = outline.derived_from is not None
        if is_derived == outline.carries_its_own_figure:
            raise InconsistentVariants(
                f'the variant {outline.identifier!r} carries '
                f'{"a" if outline.carries_its_own_figure else "no"} figure of its '
                f'own and is derived from {outline.derived_from!r}')
        if is_derived != (outline.functor is not None):
            raise InconsistentVariants(
                f'the variant {outline.identifier!r} is derived from '
                f'{outline.derived_from!r} by the functor {outline.functor!r}')
        if is_derived and outline.derived_from not in drawn:
            raise InconsistentVariants(
                f'the variant {outline.identifier!r} is derived from '
                f'{outline.derived_from!r}, and the variants carrying a figure of '
                f'their own are {drawn}')


@dataclass(frozen=True)
class VariantMessage:
    '''One variant of a page as `embedded_variants` is given it. A variant drawn
    from its own term carries the `dataUpdate` drawing it. A derived variant
    carries the identifier of the variant it is derived from, the name of the
    functor tsncd derives it by, `settings` merged over the settings of that
    variant, and, where its inspection boxes say something else, the `auxiliary`
    that replaces the one tsncd derives, keyed by the numbering of the message of
    that variant.'''
    identifier: str
    group: wst.VariantGroupRecord
    title: str
    detail: str
    message: wst.DataUpdate | None = None
    derived_from: str | None = None
    functor: str | None = None
    settings: wst.RenderHandlerSettings | None = None
    auxiliary: wst.DiagramAuxiliary | None = None

    def outline(self) -> VariantOutline:
        return VariantOutline(
            identifier=self.identifier, derived_from=self.derived_from,
            functor=self.functor, carries_its_own_figure=self.message is not None)


def groups_in_order_of_first_use(
    variants: Sequence[VariantMessage],
) -> list[wst.VariantGroupRecord]:
    '''The groups `variants` name, each once, in the order the variants first name
    them, which is the order the selector lists them in.'''
    groups: dict[str, wst.VariantGroupRecord] = {}
    for variant in variants:
        known = groups.setdefault(variant.group['id'], variant.group)
        if known != variant.group:
            raise InconsistentVariants(
                f'the group {variant.group["id"]!r} is titled {known["title"]!r} '
                f'and {variant.group["title"]!r}')
    return list(groups.values())


def settings_of_variant(
    identifier: str, variants: Sequence[VariantMessage],
) -> wst.RenderHandlerSettings:
    '''The settings the variant named `identifier` is drawn with: those of its own
    message, or those of the variant it is derived from with its own merged over
    them.'''
    by_identifier = {variant.identifier: variant for variant in variants}
    variant = by_identifier[identifier]
    if variant.message is not None:
        return variant.message.get('settings') or {}
    source = by_identifier[variant.derived_from]  # type: ignore[index]
    return {**((source.message or {}).get('settings') or {}),
            **(variant.settings or {})}  # type: ignore[typeddict-item]


def with_form_and_theme_first(
    settings: wst.RenderHandlerSettings,
) -> wst.RenderHandlerSettings:
    '''`settings` with `form` and `darkMode` stated, taking tsncd's defaults where
    they are left out, and written before every other key.

    The build plugin of the lab website, `detect_forms` in `_plugins/diagrams.rb`,
    reads the form and the theme a page starts in from its text with the patterns
    `"settings": \\{[^}]*"form": "([a-z-]+)"` and
    `"settings": \\{[^}]*"darkMode": (true|false)`. A `}` between the brace and
    either key stops the match, so the two keys come first, ahead of any value
    that could hold one. A key left out would let the pattern run on to the
    settings of a later variant.'''
    return {
        'form': settings.get('form', wst.DiagramForm.ALL_BROADCASTED.value),
        'darkMode': settings.get('darkMode', True),
        **{key: value for key, value in settings.items()
           if key not in ('form', 'darkMode')},
    }  # type: ignore[return-value]


def variant_record(
    variant: VariantMessage, message_roots: dict[str, int],
    auxiliary_roots: dict[str, int],
) -> wst.PageVariantRecord:
    record: wst.PageVariantRecord = {
        'id': variant.identifier, 'group': variant.group['id'],
        'title': variant.title, 'detail': variant.detail}
    if variant.message is not None:
        record['message'] = message_roots[variant.identifier]
    if variant.derived_from is not None:
        record['derivedFrom'] = variant.derived_from
    if variant.functor is not None:
        record['functor'] = variant.functor
    if variant.settings is not None:
        record['settings'] = variant.settings
    if variant.auxiliary is not None:
        record['auxiliary'] = auxiliary_roots[variant.identifier]
    return record


def check_only_derived_variants_carry_an_auxiliary(
    variants: Sequence[VariantMessage],
) -> None:
    carrying_both = [variant.identifier for variant in variants
                     if variant.message is not None and variant.auxiliary is not None]
    if carrying_both:
        raise InconsistentVariants(
            f'the variants {carrying_both} carry a message and an auxiliary of their '
            'own, and only a derived variant carries an auxiliary')


def embedded_variants(
    variants: Sequence[VariantMessage], initial: str,
) -> wst.EmbeddedVariants:
    '''The contents of the `tsncd-variants` element of a page carrying `variants`.
    Every message and every auxiliary of a derived variant is compressed into one
    value repository, so a value two of them hold is one record, and the settings
    of `initial` are repeated uncompressed.'''
    check_variant_outlines([variant.outline() for variant in variants], initial)
    check_only_derived_variants_carry_an_auxiliary(variants)
    drawn = [variant for variant in variants if variant.message is not None]
    explained = [variant for variant in variants if variant.auxiliary is not None]
    shared = json_compression.compress_json_values(
        [*(variant.message for variant in drawn),
         *(variant.auxiliary for variant in explained)])  # type: ignore[misc]
    message_roots = {variant.identifier: root
                     for variant, root in zip(drawn, shared.roots)}
    auxiliary_roots = {variant.identifier: root
                       for variant, root in zip(explained, shared.roots[len(drawn):])}
    return {
        'version': 1,
        'settings': with_form_and_theme_first(settings_of_variant(initial, variants)),
        'initial': initial,
        'groups': groups_in_order_of_first_use(variants),
        'variants': [variant_record(variant, message_roots, auxiliary_roots)
                     for variant in variants],
        'value_repository': shared.value_repository,
    }


def text_of_variants_element(embedded: wst.EmbeddedVariants) -> str:
    '''`embedded` as JSON with every `<` written as its escape. Every key but the
    value repository is written with a space after each colon and comma, which is
    the text `with_form_and_theme_first` expects the website to read, and the value
    repository, which holds nearly all of the text, is written with none.'''
    described = {key: value for key, value in embedded.items()
                 if key != 'value_repository'}
    repository = json.dumps(embedded['value_repository'], separators=(',', ':'))
    text = f'{json.dumps(described)[:-1]}, "value_repository": {repository}}}'
    return text.replace('<', JSON_ESCAPE_OF_LESS_THAN)


def page_with_variants(
    embedded: wst.EmbeddedVariants, dist: str | pathlib.Path | None = None,
) -> str:
    '''tsncd's built page as one HTML document holding its bundle and `embedded`,
    in the theme of the initial variant. The page holds no `tsncd-embedded-message`
    element, and tsncd draws the initial variant and a selector between them.'''
    return page_holding(
        f'<script type="application/json" id="{EMBEDDED_VARIANTS_ID}">'
        f'{text_of_variants_element(embedded)}</script>\n',
        embedded['settings'], dist)


def save_page_with_variants(
    variants: Sequence[VariantMessage],
    path: str | pathlib.Path,
    initial: str,
    dist: str | pathlib.Path | None = None,
) -> pathlib.Path:
    written = pathlib.Path(path)
    written.parent.mkdir(parents=True, exist_ok=True)
    written.write_text(
        page_with_variants(embedded_variants(variants, initial), dist),
        encoding='utf-8')
    return written


# ==========================================================================
# A page written as a folder, with a redirect from the address of one file.
# ==========================================================================
PAGE_IN_FOLDER = 'index.html'


@dataclass(frozen=True)
class PageFolder:
    '''The paths of a page written as a folder. `folder` holds the page as `page`, so
    the address a reader shares ends in the name of the folder. `redirect` stands
    beside the folder, at the address the page had when it was one file.'''
    folder: pathlib.Path
    page: pathlib.Path
    redirect: pathlib.Path


def page_folder_named(page_directory: pathlib.Path, slug: str) -> PageFolder:
    folder = page_directory / slug
    return PageFolder(folder=folder, page=folder / PAGE_IN_FOLDER,
                      redirect=page_directory / f'{slug}.html')


def redirect_to_page_folder(slug: str) -> str:
    '''A page that sends a reader of `<slug>.html?query#hash` to `<slug>/?query#hash`.

    The script keeps the query, which can name the variant the page opens on, and the
    hash. A browser shows a folder opened from a disk as a list of its files, so from a
    disk the script names `index.html` inside the folder. A reader whose browser runs
    no script is sent to the folder by the `meta` refresh, or follows the link, and the
    query and the hash are dropped. The refresh stands inside `noscript`, so that it
    cannot replace the navigation started by the script with one that drops them.'''
    address = urllib.parse.quote(slug, safe='') + '/'
    address_attribute = html.escape(address)
    title = html.escape(slug)
    return (
        '<!doctype html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '<meta charset="utf-8">\n'
        f'<title>{title}</title>\n'
        '<script>\n'
        f'location.replace({json.dumps(address)}\n'
        f'  + (location.protocol === "file:" ? {json.dumps(PAGE_IN_FOLDER)} : "")\n'
        '  + location.search + location.hash);\n'
        '</script>\n'
        '<noscript><meta http-equiv="refresh" '
        f'content="0; url={address_attribute}"></noscript>\n'
        '</head>\n'
        '<body>\n'
        f'<noscript><p><a href="{address_attribute}">{title}</a></p></noscript>\n'
        '</body>\n'
        '</html>\n')


def save_page_folder_with_variants(
    variants: Sequence[VariantMessage],
    page_directory: str | pathlib.Path,
    slug: str,
    initial: str,
    dist: str | pathlib.Path | None = None,
) -> PageFolder:
    '''Write `variants` to `<page_directory>/<slug>/index.html`, and the page of
    `redirect_to_page_folder` to `<page_directory>/<slug>.html`.'''
    written = page_folder_named(pathlib.Path(page_directory), slug)
    save_page_with_variants(variants, written.page, initial, dist)
    written.redirect.write_text(redirect_to_page_folder(slug), encoding='utf-8')
    return written
