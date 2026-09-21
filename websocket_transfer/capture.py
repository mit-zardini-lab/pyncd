'''
Getting the diagram back into the notebook as an image.

`send_morphism` pushes a term to whatever browser is showing the diagram and
stops there. The picture lives in that window and none of it survives into
the notebook. `capture_morphism` completes the circuit: the same render, with the
browser hands the image back and the cell displays it inline, so it is saved
with the notebook and survives export.

    img = await capture_morphism(attention)                 # inline PNG
    await capture_morphism(attention, save_to='fig.svg')    # and on disk

The rendering still happens in the browser, because it can happen nowhere else:
the diagram's geometry is CSS layout plus measured text, with the wires drawn
from rectangles read back off the laid-out DOM. So a tsncd page has to be open.
When there is no browser to hand, as when regenerating a directory of figures,
`websocket_transfer.headless` drives its own.
'''

import pathlib
from typing import Literal

from IPython.display import Image, SVG

import websocket_transfer.send_morphism as sm
import websocket_transfer.websockets_transfer as wst


type Sendable = sm.Sendable
type CaptureFormat = Literal['png', 'svg']


def capture_options(
    format: CaptureFormat = 'png',
    scale: float = 2.0,
    padding: int = 16,
    background: str | None = 'auto',
) -> wst.CaptureOptions:
    '''
    Collect the image options into the partial dict the client expects.

    `background='auto'` uses the current diagram theme. A CSS colour overrides
    the theme, while `None` leaves the image transparent.
    '''
    return wst.CaptureOptions(
        format=format,
        scale=scale,
        padding=padding,
        background=background,
    )


def as_display(payload: bytes, format: CaptureFormat) -> Image | SVG:
    '''Wrap raw image bytes so a notebook renders them inline.'''
    return SVG(payload) if format == 'svg' else Image(payload)


async def capture_morphism(
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
    format: CaptureFormat = 'png',
    scale: float = 2.0,
    padding: int = 16,
    background: str | None = 'auto',
    timeout: float = wst.DEFAULT_CAPTURE_TIMEOUT,
    disturb_display: bool = True,
    save_to: str | pathlib.Path | None = None,
) -> Image | SVG:
    '''
    Render `target` in the connected browser and return its image.

    Accepts everything `send_morphism` does, on the same terms, plus the image
    options. `scale` applies to `png` only - `svg` comes back as vector, which
    is what to reach for when the diagram is going into a paper.

    By default the capture also lands on screen, so the diagram on screen
    is the one that comes back. With `disturb_display=False` the browser draws into
    an off-screen target instead and the visible diagram is untouched, which is useful
    when capturing a variant partway through a session without losing the current
    diagram. The
    the display settings travel with the render either way, so a disturbing
    capture applies this call's `debugBorders` to what is on screen too.

        await capture_morphism(attention, disturb_display=False)

    A browser must be connected regardless: it does the rendering, and nothing
    else can. See `websocket_transfer.headless` when there is none.

    Raises `wst.CaptureError` if no diagram page is connected, if the render
    fails, or if nothing comes back within `timeout` seconds.
    '''
    payload = await wst.capture_term(
        sm.to_morphism(target, recycle=recycle),
        settings=sm.display_settings(
            darkMode, debugBorders, coreDebug, width, subBlocks,
            drawnBlockTags, tapeLabels),
        capture=capture_options(format, scale, padding, background),
        timeout=timeout,
        disturb_display=disturb_display,
    )
    if save_to is not None:
        path = pathlib.Path(save_to)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        print(f'Wrote {path} ({len(payload)} bytes).')
    return as_display(payload, format)
