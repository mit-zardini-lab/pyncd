'''
Rendering figures without a browser window open.

`capture.capture_morphism` needs a tsncd page up, because that page is what
draws. That is the right trade for working in a notebook - the diagram you are
looking at is the one you get - but it is the wrong one for rebuilding a
directory of paper figures, where the output should not depend on which tab
happened to be focused. So this module brings its own browser.

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

import data_transfer.json as dtj
import websocket_transfer.send_morphism as sm
import websocket_transfer.websockets_transfer as wst


type Sendable = sm.Sendable
type CaptureFormat = Literal['png', 'svg', 'pdf']

'''
Strips the page back to the diagram alone before capturing it.

A tight crop means making the diagram *be* the page: the heading goes, the
body's own spacing goes, and the margin we do want is reapplied as padding.
Only the container moves - every overlay is positioned within it, so the
drawing itself is unaffected.
'''
ISOLATION_CSS = '''
    body > *:not(#diagram) {{ display: none !important; }}
    body {{
        margin: 0 !important;
        padding: {padding}px !important;
        width: fit-content !important;
        {background}
    }}
'''

DEFAULT_VIEWPORT = {'width': 1600, 'height': 1200}

# `None` already means "transparent" for a background, so "not specified" needs
# a value of its own.
UNSET: Any = object()

'''
Where to look for the built tsncd bundle. `TSNCD_DIST` wins; otherwise we try
the checkout layout these two repositories are normally cloned into, as
siblings.
'''
DIST_ENV_VAR = 'TSNCD_DIST'
SIBLING_DIST_CANDIDATES = ('tsncd/dist',)


def find_dist(dist: str | pathlib.Path | None = None) -> pathlib.Path:
    '''Locate tsncd's `dist/`, with an error that says how to produce it.'''
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


@dataclass
class HeadlessRenderer:
    '''
    A browser held open across many renders.

    Startup - launching Chromium, loading the bundle, warming the KaTeX fonts -
    costs far more than any single diagram, so the point of the context manager
    is to pay it once for a whole batch.
    '''
    scale: float = 2.0
    padding: int = 16
    background: str | None = '#ffffff'
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
                # Playwright sets pixel density per context, not per screenshot,
                # so the resolution of every figure in a batch is fixed here.
                device_scale_factor=self.scale)
            page = await context.new_page()
            page.on('pageerror', lambda error: print(f'[tsncd page error] {error}'))
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
    ) -> None:
        '''
        Draw an already-exported term, leaving it on the page to be captured.

        Takes the same payload that goes over the websocket, so a `.json`
        export saved next to a notebook can be redrawn without reconstructing
        the morphism that produced it.
        '''
        await self.page.evaluate(
            'async ([data, settings]) => await window.tsncd.render(data, settings)',
            [data, settings or {}])

    async def render(
        self,
        target: Sendable,
        recycle: bool = False,
        *,
        darkMode: bool | None = None,
        debugBorders: bool | None = False,
        coreDebug: bool | None = None,
        width: int | None = None,
    ) -> None:
        '''Draw `target` in the headless page, leaving it there to be captured.'''
        morphism = sm.to_morphism(target, recycle=recycle)
        await self.render_json(
            dtj.TermJSONConverter.export_to_json(morphism),
            sm.display_settings(darkMode, debugBorders, coreDebug, width))

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

        `png` is Playwright's own screenshot - a real browser paint, so nothing
        about fonts or overlays can be lost in translation. The safe default.

        `pdf` is Chromium's print pipeline, and is what to use for a figure
        going into a paper: true vector, real embedded text, and a file in the
        tens of KB.

        `svg` goes through the in-page serialiser, which reproduces the diagram
        by inlining the full computed style of every element - correct, but it
        runs to megabytes on a diagram of any size, most of it CSS that has
        nothing to do with the drawing. Prefer `pdf` unless something
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

        async with self.isolated(padding, background) as box:
            if format == 'pdf':
                return await self.page.pdf(
                    width=f'{box["width"]}px',
                    height=f'{box["height"]}px',
                    margin={'top': '0', 'right': '0', 'bottom': '0', 'left': '0'},
                    print_background=background is not None,
                    prefer_css_page_size=False)
            return await self.page.screenshot(
                clip={'x': 0, 'y': 0,
                      'width': box['width'], 'height': box['height']},
                # The diagram routinely runs past the viewport; without this the
                # clip is silently intersected with what is on screen.
                full_page=True,
                omit_background=background is None,
                type='png')

    @contextlib.asynccontextmanager
    async def isolated(self, padding: int, background: str | None):
        '''
        Put the diagram alone on the page with its corner at the origin, and
        yield the box it occupies.

        Both the screenshot and the print need the region they want to be a
        region that exists. A capture box reaches outside the page whenever the
        wanted padding exceeds the page's own - the overlay already overhangs
        the container - and Playwright silently clamps a clip to the page
        rather than complaining, quietly trimming the margin. Moving the
        diagram to the origin instead makes the box valid by construction.

        Everything is undone afterwards, so a batch can mix formats without one
        capture leaking into the next.
        '''
        style = await self.page.add_style_tag(content=ISOLATION_CSS.format(
            padding=padding,
            background=(f'background-color: {background} !important;'
                        if background is not None else '')))
        try:
            # Measured after the stylesheet lands, since hiding the heading and
            # dropping the body's spacing has moved everything. `bounds`
            # covers the overlay's overhang, which the page's own scroll size
            # would miss above and to the left - overflow in the negative
            # direction does not extend a scroll box.
            box = await self.page.evaluate(
                '(padding) => window.tsncd.bounds(padding)', padding)
            await self.page.evaluate('''([x, y]) => {
                document.getElementById('diagram').style.transform =
                    `translate(${-x}px, ${-y}px)`;
            }''', [box['x'], box['y']])
            yield box
        finally:
            await style.evaluate('node => node.remove()')
            await self.page.evaluate(
                "() => { document.getElementById('diagram').style.transform = ''; }")

    async def capture(
        self,
        target: Sendable,
        recycle: bool = False,
        *,
        format: CaptureFormat = 'png',
        padding: int | None = None,
        background: str | None = UNSET,
        darkMode: bool | None = None,
        debugBorders: bool | None = False,
        coreDebug: bool | None = None,
        width: int | None = None,
    ) -> bytes:
        '''Render `target` and return the image bytes.'''
        await self.render(
            target, recycle=recycle, darkMode=darkMode,
            debugBorders=debugBorders, coreDebug=coreDebug, width=width)
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

    Launches and tears down a browser around the one render; use
    `HeadlessRenderer` directly for more than a couple.
    '''
    async with HeadlessRenderer(scale=scale, dist=dist) as renderer:
        return await renderer.capture(target, recycle=recycle, **options)


DEFAULT_OUTPUT_DIRECTORY = 'outputs'


def output_directory(directory: str | pathlib.Path | None = None) -> pathlib.Path:
    '''
    Where figures land: `directory`, or `./outputs` beside the caller.

    Relative to the working directory, which for a notebook is the directory
    the notebook sits in - so figures land next to the work that produced them
    without anyone having to say where.

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

    Names, not paths: each key becomes a filename under `directory`, which
    defaults to `./outputs`. A name may carry its own extension to override
    `format` for that one figure, and may include subdirectories:

        await save_figures(
            {'fig1.pdf': attention, 'raster/fig1.png': attention},
            'paper/figures')

    `format` defaults to PDF, which is what a paper wants - vector, embedded
    text, and smaller than the equivalent PNG.

    Remaining options are passed to each capture, so they apply to every
    figure in the set - `width=1400` to flatten them all, for instance. For
    per-figure settings, use `HeadlessRenderer` directly and call `save`.

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
