'''
Rendering figures without a browser window open.

`capture.capture_morphism` needs a tsncd page up, because that page is what
draws. Depending on the open page is the right trade for working in a
notebook, since the diagram on screen is the diagram that comes back. It is
the wrong trade for rebuilding a directory of paper figures, where the output
should not depend on which tab happened to be focused. This module therefore
brings its own browser.

    # a named set into ./outputs, as PDFs
    await save_figures({'attention': attention, 'convolution': convolution})

    # or one at a time, with full control
    async with HeadlessRenderer() as renderer:
        await renderer.save(attention, 'figures/attention.pdf')
        await renderer.save(convolution, 'figures/convolution.png', width=1400)

It renders through the very same `termPass` the websocket path uses, reached
via the `window.tsncd` hook in `src/index.ts`, so the two agree by construction
rather than by discipline. The websocket server is not involved at all: a
figure rebuild should not require a server to be up.

Requires Playwright, which is not part of `requirements.txt` because it pulls
down a browser:

    pip install -r requirements-headless.txt
    playwright install chromium

and a built tsncd bundle - `npm run build` in the tsncd checkout, which is what
puts `dist/` there.
'''

import contextlib
import functools
import http.server
import os
import pathlib
import threading
from dataclasses import dataclass, field
from typing import Any, Literal

import data_transfer.term_json as dtj
import websocket_transfer.send_morphism as sm
import websocket_transfer.websockets_transfer as wst


type Sendable = sm.Sendable
type CaptureFormat = Literal['png', 'svg', 'pdf']

'''
Strips the page back to the diagram alone before capturing it.

A tight crop means making the diagram the whole page. The heading goes, the
body's own spacing goes, and the wanted margin is reapplied as padding. Only
the container moves, and every overlay is positioned within it, so the drawing
itself is unaffected.

The body is sized `max-content` rather than `fit-content`, because `fit-content`
is capped at the viewport width and a figure wider than the viewport then
overflows the body. `isolated` also gives the body the capture box as a minimum
size, because the container is translated to put the overlay's overhang at the
origin and the translate does not grow the body's padding box, so the page's
scroll width ended ten pixels before the clip's right edge and Playwright
trimmed the clip to it. The legend of the advanced display sits at the right
edge of the figure and was the first thing cut.
'''
ISOLATION_CSS = '''
    body > *:not(#diagram) {{ display: none !important; }}
    body {{
        margin: 0 !important;
        padding: {padding}px !important;
        width: max-content !important;
        background-color: {background} !important;
    }}
    #diagram {{
        background-color: transparent !important;
    }}
'''

DEFAULT_VIEWPORT = {'width': 1600, 'height': 1200}

# `None` already means "transparent" for a background, so "not specified" needs
# a value of its own.
UNSET: Any = object()

'''
Where to look for the built tsncd bundle. `TSNCD_DIST` wins. Failing that, the
search covers the two names a tsncd checkout beside this repository is given,
`tsncd` and `tsncd-public`.
'''
DIST_ENV_VAR = 'TSNCD_DIST'
SIBLING_DIST_CANDIDATES = ('tsncd/dist', 'tsncd-public/dist')


def find_dist(dist: str | pathlib.Path | None = None) -> pathlib.Path:
    '''Locate tsncd's `dist/`, raising an error that states how to build it.'''
    candidates: list[pathlib.Path] = []
    if dist is not None:
        candidates = [pathlib.Path(dist)]
    elif env := os.environ.get(DIST_ENV_VAR):
        candidates = [pathlib.Path(env)]
    else:
        siblings = pathlib.Path(__file__).resolve().parent.parent.parent
        candidates = [siblings / name for name in SIBLING_DIST_CANDIDATES]

    for candidate in candidates:
        if (candidate / 'index.html').is_file():
            return candidate
    raise FileNotFoundError(
        'No built tsncd bundle found (looked in '
        + ', '.join(str(c) for c in candidates)
        + f'). Run `npm run build` in the tsncd checkout, or point {DIST_ENV_VAR} '
        'at its `dist/` directory.')


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    '''
    A figure rebuild would otherwise bury its own output under one log line per
    KaTeX font file.
    '''
    def log_message(self, *args, **kwargs) -> None:
        pass


@contextlib.contextmanager
def serve_directory(directory: pathlib.Path):
    '''
    Serve `directory` on a loopback port for as long as the block runs.

    Loading the page over `file://` will not do: webpack builds with
    `publicPath: '/'`, so the bundle and the KaTeX fonts are requested at
    absolute paths that only resolve under an HTTP root.
    '''
    handler = functools.partial(_QuietHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


WEBSOCKET_THAT_NEVER_CONNECTS = '''
window.WebSocket = class {
    constructor() { this.readyState = 3; }
    addEventListener() {}
    removeEventListener() {}
    send() {}
    close() {}
};
'''
'''The page of the bundle opens a socket to the relay server on port 8765 whenever it is
loaded with no embedded message. A relay that holds a figure answers with it, and the
answer arrives after the first `render_json` of a fresh page and replaces the figure
that call drew, so the first capture of a run came out as the relay's figure. The
headless page therefore gets a socket that never opens, and `readyState` 3 is CLOSED.'''


@dataclass
class HeadlessRenderer:
    '''
    A browser held open across many renders.

    Startup costs far more than any single diagram, because it launches
    Chromium, loads the bundle and warms the KaTeX fonts. The context manager
    exists to pay that cost once for a whole batch.
    '''
    scale: float = 2.0
    padding: int = 16
    background: str | None = 'auto'
    dist: str | pathlib.Path | None = None
    viewport: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_VIEWPORT))

    _stack: contextlib.AsyncExitStack | None = field(default=None, init=False)
    _page: object | None = field(default=None, init=False)

    async def __aenter__(self) -> 'HeadlessRenderer':
        try:
            from playwright.async_api import async_playwright
        except ImportError as e:
            raise ImportError(
                'Headless capture needs Playwright: `pip install -r '
                'requirements-headless.txt` then `playwright install chromium`.'
            ) from e

        self._stack = contextlib.AsyncExitStack()
        await self._stack.__aenter__()
        try:
            url = self._stack.enter_context(serve_directory(find_dist(self.dist)))
            playwright = await self._stack.enter_async_context(async_playwright())
            browser = await playwright.chromium.launch()
            self._stack.push_async_callback(browser.close)
            context = await browser.new_context(
                viewport=self.viewport,
                # Playwright sets the pixel density per context rather than
                # per screenshot, so the resolution of every figure in a batch
                # is fixed here.
                device_scale_factor=self.scale)
            page = await context.new_page()
            page.on('pageerror', lambda error: print(f'[tsncd page error] {error}'))
            await page.add_init_script(WEBSOCKET_THAT_NEVER_CONNECTS)
            await page.goto(url, wait_until='load')
            # Installed at the end of the entry point's DOMContentLoaded
            # handler, so its presence means the renderer is fully wired up.
            await page.wait_for_function('window.tsncd !== undefined')
            self._page = page
        except BaseException:
            await self._stack.aclose()
            self._stack = None
            raise
        return self

    async def __aexit__(self, *exc_info) -> None:
        self._page = None
        stack, self._stack = self._stack, None
        if stack is not None:
            await stack.aclose()

    @property
    def page(self):
        if self._page is None:
            raise RuntimeError(
                'HeadlessRenderer is not running; use it as `async with '
                'HeadlessRenderer() as renderer:`.')
        return self._page

    async def render_json(
        self,
        data: str | dict,
        settings: wst.RenderHandlerSettings | None = None,
        auxiliary: wst.DiagramAuxiliary | None = None,
    ) -> None:
        '''
        Draw an already-exported term, leaving it on the page to be captured.

        Takes the same payload that goes over the websocket, so a `.json`
        export saved next to a notebook can be redrawn without reconstructing
        the morphism that produced it. `auxiliary` is the field of the same
        name a message carries, for the legend and the inspection boxes.
        '''
        await self.page.evaluate(
            'async ([data, settings, auxiliary]) => '
            'await window.tsncd.render(data, settings, auxiliary ?? undefined)',
            [data, settings or {}, auxiliary])

    async def render(
        self,
        target: Sendable,
        recycle: bool = False,
        *,
        darkMode: wst.ColorMode | bool | None = None,
        debugBorders: bool | None = None,
        coreDebug: bool | None = None,
        width: int | None = None,
        subBlocks: bool | None = None,
        drawnBlockTags: list[int] | None = None,
        tapeLabels: bool | None = None,
        legend: bool | None = None,
        inspectionBoxes: bool | None = None,
        axisHover: wst.AxisHover | None = None,
        axisLabelFontSize: float | None = None,
        auxiliary: wst.DiagramAuxiliary | None = None,
    ) -> None:
        '''Draw `target` in the headless page, leaving it there to be captured.
        `auxiliary` is assembled from the morphism this call draws, so a caller
        that passes it converts first.'''
        morphism = sm.to_morphism(target, recycle=recycle)
        await self.render_json(
            dtj.TermJSONConverter.export_to_json(morphism),
            sm.display_settings(
                darkMode, debugBorders, coreDebug, width, subBlocks,
                drawnBlockTags, tapeLabels, legend, inspectionBoxes,
                axisHover, axisLabelFontSize),
            auxiliary)

    async def resolve_capture_background(
        self,
        background: str | None,
    ) -> str | None:
        '''Resolve `auto` for Playwright's page-level capture.'''
        if background != 'auto':
            return background
        resolved_background = await self.page.evaluate('''() => {
            if (window.tsncd.captureBackground !== undefined) {
                return window.tsncd.captureBackground('auto');
            }
            const diagram = document.getElementById('diagram');
            const computed = getComputedStyle(diagram).backgroundColor;
            return computed === 'rgba(0, 0, 0, 0)' || computed === 'transparent'
                ? '#ffffff'
                : computed;
        }''')
        if resolved_background is not None and not isinstance(
            resolved_background, str
        ):
            raise TypeError(
                'tsncd resolved the capture background to '
                f'{resolved_background!r}, expected a CSS colour or None.')
        return resolved_background

    async def capture_rendered(
        self,
        *,
        format: CaptureFormat = 'png',
        padding: int | None = None,
        background: str | None = UNSET,
    ) -> bytes:
        '''
        Cut an image from whatever is currently on the page.

        Three formats, and the choice matters:

        `png` is Playwright's own screenshot, which is a real browser paint,
        so nothing about the fonts or the overlays is lost in serialisation.
        It is the default.

        `pdf` is Chromium's print pipeline, and is what to use for a figure
        going into a paper: true vector, real embedded text, and a file in the
        tens of KB.

        `svg` goes through the in-page serialiser, which reproduces the
        diagram by inlining the full computed style of every element. It is
        correct, and it runs to megabytes on a diagram of any size, most of it
        CSS with no bearing on the drawing. Prefer `pdf` unless something
        downstream genuinely needs SVG.
        '''
        padding = self.padding if padding is None else padding
        background = self.background if background is UNSET else background

        if format == 'svg':
            # The in-page serialiser works off the live DOM and frames itself,
            # so it needs none of the isolation the other two do.
            result = await self.page.evaluate(
                'async (options) => await window.tsncd.capture(options)',
                {'format': 'svg', 'padding': padding, 'background': background})
            return wst.result_to_bytes(result)

        resolved_background = await self.resolve_capture_background(background)
        async with self.isolated(padding, resolved_background) as box:
            if format == 'pdf':
                return await self.page.pdf(
                    width=f'{box["width"]}px',
                    height=f'{box["height"]}px',
                    margin={'top': '0', 'right': '0', 'bottom': '0', 'left': '0'},
                    print_background=resolved_background is not None,
                    prefer_css_page_size=False)
            return await self.page.screenshot(
                clip={'x': 0, 'y': 0,
                      'width': box['width'], 'height': box['height']},
                # The diagram routinely runs past the viewport. Without
                # this the clip is silently intersected with what is on
                # screen.
                full_page=True,
                omit_background=resolved_background is None,
                type='png')

    @contextlib.asynccontextmanager
    async def isolated(self, padding: int, background: str | None):
        '''
        Put the diagram alone on the page with its corner at the origin, and
        yield the box it occupies.

        Both the screenshot and the print need the region they are given to
        be a region that exists. A capture box reaches outside the page
        whenever the requested padding exceeds the page's own, because the
        overlay already overhangs the container, and Playwright clamps a clip
        to the page without reporting it, which trims the margin. Moving the
        diagram to the origin instead makes the box valid by construction.

        Everything is undone afterwards, so a batch can mix formats without one
        capture leaking into the next.
        '''
        style = await self.page.add_style_tag(content=ISOLATION_CSS.format(
            padding=padding,
            background=background if background is not None else 'transparent'))
        try:
            # Measured after the stylesheet lands, since hiding the heading and
            # dropping the body's spacing has moved everything. `bounds`
            # covers the overlay's overhang, which the page's own scroll size
            # would miss above and to the left, because overflow in the
            # negative direction does not extend a scroll box.
            box = await self.page.evaluate(
                '(padding) => window.tsncd.bounds(padding)', padding)
            await self.page.evaluate('''([x, y, width, height]) => {
                document.getElementById('diagram').style.transform =
                    `translate(${-x}px, ${-y}px)`;
                document.body.style.boxSizing = 'border-box';
                document.body.style.minWidth = `${width}px`;
                document.body.style.minHeight = `${height}px`;
            }''', [box['x'], box['y'], box['width'], box['height']])
            yield box
        finally:
            await style.evaluate('node => node.remove()')
            await self.page.evaluate('''() => {
                document.getElementById('diagram').style.transform = '';
                document.body.style.boxSizing = '';
                document.body.style.minWidth = '';
                document.body.style.minHeight = '';
            }''')

    async def capture(
        self,
        target: Sendable,
        recycle: bool = False,
        *,
        format: CaptureFormat = 'png',
        padding: int | None = None,
        background: str | None = UNSET,
        darkMode: wst.ColorMode | bool | None = None,
        debugBorders: bool | None = None,
        coreDebug: bool | None = None,
        width: int | None = None,
        subBlocks: bool | None = None,
        drawnBlockTags: list[int] | None = None,
        tapeLabels: bool | None = None,
        legend: bool | None = None,
        inspectionBoxes: bool | None = None,
        axisHover: wst.AxisHover | None = None,
        axisLabelFontSize: float | None = None,
        auxiliary: wst.DiagramAuxiliary | None = None,
    ) -> bytes:
        '''Render `target` and return the image bytes.'''
        await self.render(
            target, recycle=recycle, darkMode=darkMode,
            debugBorders=debugBorders, coreDebug=coreDebug, width=width,
            subBlocks=subBlocks, drawnBlockTags=drawnBlockTags,
            tapeLabels=tapeLabels, legend=legend,
            inspectionBoxes=inspectionBoxes, axisHover=axisHover,
            axisLabelFontSize=axisLabelFontSize,
            auxiliary=auxiliary)
        return await self.capture_rendered(
            format=format, padding=padding, background=background)

    async def save(
        self,
        target: Sendable,
        path: str | pathlib.Path,
        recycle: bool = False,
        **options,
    ) -> pathlib.Path:
        '''
        Render `target` to `path`, taking the format from the file extension.

            await renderer.save(attention, 'figures/attention.pdf')
        '''
        path = pathlib.Path(path)
        by_suffix: dict[str, CaptureFormat] = {
            '.svg': 'svg', '.pdf': 'pdf', '.png': 'png'}
        options.setdefault('format', by_suffix.get(path.suffix.lower(), 'png'))
        payload = await self.capture(target, recycle=recycle, **options)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return path


async def capture_headless(
    target: Sendable,
    recycle: bool = False,
    *,
    scale: float = 2.0,
    dist: str | pathlib.Path | None = None,
    **options,
) -> bytes:
    '''
    One-shot capture, for when there is a single figure to make.

    It launches and tears down a browser around the one render. Use
    `HeadlessRenderer` directly for more than a couple of figures.
    '''
    async with HeadlessRenderer(scale=scale, dist=dist) as renderer:
        return await renderer.capture(target, recycle=recycle, **options)


DEFAULT_OUTPUT_DIRECTORY = 'outputs'


def output_directory(directory: str | pathlib.Path | None = None) -> pathlib.Path:
    '''
    Where figures land: `directory`, or `./outputs` beside the caller.

    It is relative to the working directory, which for a notebook is the
    directory the notebook sits in, so figures land beside the work that
    produced them without anyone having to say where.

    Always absolute, including when `directory` was given as a relative path.
    The saved paths are handed back to the caller to report or open, and a
    relative one is ambiguous the moment anything changes the working
    directory.
    '''
    path = pathlib.Path(directory if directory is not None
                        else DEFAULT_OUTPUT_DIRECTORY)
    return path if path.is_absolute() else pathlib.Path.cwd() / path


async def save_figures(
    figures: dict[str, Sendable],
    directory: str | pathlib.Path | None = None,
    *,
    format: CaptureFormat = 'pdf',
    scale: float = 2.0,
    dist: str | pathlib.Path | None = None,
    **options,
) -> list[pathlib.Path]:
    '''
    Write a whole set of named figures in one browser session.

        await save_figures({
            'attention':   attention,
            'convolution': convolution_graph,
        })
        # -> ./outputs/attention.pdf, ./outputs/convolution.pdf

    Each key is a name rather than a path, and becomes a filename under
    `directory`, which defaults to `./outputs`. A name may carry its own
    extension, which overrides `format` for that one figure, and it may
    include subdirectories:

        await save_figures(
            {'fig1.pdf': attention, 'raster/fig1.png': attention},
            'paper/figures')

    `format` defaults to PDF, because a paper needs vector output with real
    embedded text, and the file is smaller than the equivalent PNG.

    The remaining options are passed to each capture, so they apply to every
    figure in the set. `width=1400` flattens them all, for instance. For a
    setting per figure, use `HeadlessRenderer` directly and call `save`.

    One browser is launched for the whole set, which is the point: startup
    costs far more than any single diagram.
    '''
    target_directory = output_directory(directory)
    written: list[pathlib.Path] = []
    async with HeadlessRenderer(scale=scale, dist=dist) as renderer:
        for name, term in figures.items():
            path = target_directory / name
            if not path.suffix:
                path = path.with_suffix(f'.{format}')
            written.append(await renderer.save(term, path, **options))
            print(f'Wrote {written[-1]}.')
    return written
