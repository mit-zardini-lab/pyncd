'''
Getting the diagram back into the notebook as an image.

`send_morphism` pushes a term to whatever browser is showing the diagram and
stops there; the picture lives in that window and nothing of it survives into
the notebook. `capture_morphism` completes the circuit - same render, but the
browser hands the image back and the cell displays it inline, so it is saved
with the notebook and survives export.

    img = await capture_morphism(attention)                 # inline PNG
    await capture_morphism(attention, save_to='fig.svg')    # and on disk

The rendering still happens in the browser, because it can happen nowhere else:
the diagram's geometry is CSS layout plus measured text, with the wires drawn
from rectangles read back off the laid-out DOM. So a tsncd page has to be open.
When there is no browser to hand - regenerating a directory of figures, say -
`websocket_transfer.headless` drives its own.

When there is neither, `print_fallback` prints the term instead of raising, so a
notebook run with no page open still shows its diagrams as text.
'''

import pathlib
import socket
from typing import Literal

from IPython.display import Image, SVG

import display as dpl
import websocket_transfer.send_morphism as sm
import websocket_transfer.websockets_transfer as wst


type Sendable = sm.Sendable
type CaptureFormat = Literal['png', 'svg']


def capture_options(
    format: CaptureFormat = 'png',
    scale: float = 2.0,
    padding: int = 16,
    background: str | None = '#ffffff',
) -> wst.CaptureOptions:
    '''
    Collect the image options into the partial dict the client expects.

    `background` defaults to white rather than transparent: these figures
    mostly end up in papers and slides, and a transparent PNG dropped onto a
    dark background renders black-on-black.
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


def server_listening(timeout: float = 0.2) -> bool:
    '''
    Whether anything is accepting connections on the server's port.

    Only a probe - it says nothing about a browser being attached, which is why
    a refused capture is still handled below. It is here because a connection
    that will be refused takes seconds to fail on some platforms, and that cost
    would be paid once per display call in a notebook run with nothing up.
    '''
    with socket.socket() as probe:
        probe.settimeout(timeout)
        return probe.connect_ex((wst.SERVER_HOST, wst.SERVER_PORT)) == 0


async def capture_morphism(
    target: Sendable,
    recycle: bool = False,
    *,
    darkMode: bool | None = None,
    debugBorders: bool | None = False,
    coreDebug: bool | None = None,
    width: int | None = None,
    format: CaptureFormat = 'png',
    scale: float = 2.0,
    padding: int = 16,
    background: str | None = '#ffffff',
    timeout: float = wst.DEFAULT_CAPTURE_TIMEOUT,
    disturb_display: bool = True,
    print_fallback: bool = True,
    save_to: str | pathlib.Path | None = None,
) -> Image | SVG | None:
    '''
    Render `target` in the connected browser and return its image.

    Accepts everything `send_morphism` does, on the same terms, plus the image
    options. `scale` applies to `png` only - `svg` comes back as vector, which
    is what to reach for when the diagram is going into a paper.

    By default the capture also lands on screen, so the diagram you are looking
    at is the one you get. With `disturb_display=False` the browser draws into
    an off-screen target instead and the visible diagram is untouched - useful
    when capturing a variant mid-session without losing your place. Note that
    the display settings travel with the render either way, so a disturbing
    capture applies this call's `debugBorders` to what is on screen too.

        await capture_morphism(attention, disturb_display=False)

    A browser must be connected to get an image: it does the rendering, and
    nothing else can. See `websocket_transfer.headless` when there is none.

    Without one - no server running, no page open, a failed render, or nothing
    back within `timeout` seconds - `print_fallback` prints the term through
    `display.print_category` and returns None, which is what makes these
    notebooks readable with nothing else running. Both failures are quick: the
    server refuses a render outright when no page is attached rather than
    letting it time out. Pass `print_fallback=False` to get `wst.CaptureError`
    instead, which is what you want when the image is the point - regenerating
    a figure, say, where silently getting text would be a broken build.
    '''
    morphism = sm.to_morphism(target, recycle=recycle)
    if print_fallback and not server_listening():
        print('No server on '
              f'{wst.SERVER_URI} - printing instead. '
              'Run `python run_server.py` and open the tsncd page for images.')
        dpl.print_category(morphism) # type: ignore
        return None
    try:
        payload = await wst.capture_term(
            morphism,
            settings=sm.display_settings(darkMode, debugBorders, coreDebug, width),
            capture=capture_options(format, scale, padding, background),
            timeout=timeout,
            disturb_display=disturb_display,
        )
    except wst.CaptureError as error:
        if not print_fallback:
            raise
        print(f'No image captured ({error}) - printing instead.')
        dpl.print_category(morphism) # type: ignore
        return None
    if save_to is not None:
        path = pathlib.Path(save_to)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        print(f'Wrote {path} ({len(payload)} bytes).')
    return as_display(payload, format)
