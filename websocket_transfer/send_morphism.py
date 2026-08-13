'''
A packaged `send_term` which accepts either a morphism or a hypergraph.

Notebooks alternate between two forms of the same diagram: a morphism, which
`send_term` displays directly, and a hypergraph, which has to be converted first
(the recurring `wst.send_term(h2m.hypergraph_to_morphism(graph))` idiom). This
module folds the conversion into the send, so either form can be handed over
without the caller deciding which it holds.

Note that `graphs.UIDHypergraph` is not covered: it registers the same `Term`
class names as `graphs.Hypergraph`, so the two can never be imported together.
'''

import data_structure.Term as fd
import graphs.Hypergraph as hg
import graphs.Hypergraph2Morphism as h2m
import websocket_transfer.websockets_transfer as wst


type Sendable = fd.GeneralTerm | hg.Hypergraph


def to_morphism(target: Sendable, recycle: bool = False) -> fd.GeneralTerm:
    '''
    Coerce `target` into the morphism form `send_term` expects.

    Hypergraphs are converted; anything else is already a term and passes
    through untouched. Tuples are mapped over elementwise, matching
    `fd.GeneralTerm`'s product case.

    With `recycle=True`, the result is additionally round-tripped through a
    `StructuredHypergraph`, which is how hand-built morphisms are normalised
    before display.
    '''
    match target:
        case hg.Hypergraph():
            morphism = h2m.hypergraph_to_morphism(target)
        case tuple():
            return tuple(to_morphism(element, recycle) for element in target)
        case _:
            morphism = target
    return h2m.recycle(morphism) if recycle else morphism


def display_settings(
    darkMode: bool | None = None,
    debugBorders: bool | None = False,
    coreDebug: bool | None = None,
    width: int | None = None,
) -> wst.RenderHandlerSettings:
    '''
    Collect the display options into the partial dict the client expects.

    Each option left as `None` is omitted from the message, and the client
    falls back to its default for it - so every send fully determines the
    display rather than inheriting whatever the previous send happened to set.

    `width` is the px at which a morphism wraps onto another line, and so
    controls the figure's proportions rather than its scale.
    '''
    settings: wst.RenderHandlerSettings = {}
    if darkMode is not None:
        settings['darkMode'] = darkMode
    if debugBorders is not None:
        settings['debugBorders'] = debugBorders
    if coreDebug is not None:
        settings['coreDebug'] = coreDebug
    if width is not None:
        settings['width'] = width
    return settings


async def send_morphism(
    target: Sendable,
    recycle: bool = False,
    *,
    darkMode: bool | None = None,
    debugBorders: bool | None = False,
    coreDebug: bool | None = None,
    width: int | None = None,
):
    '''
    Display `target`, converting it to a morphism first if it is a hypergraph.

        await send_morphism(attention)               # already a morphism
        await send_morphism(attention_graph)         # converted, then sent

    The keyword-only display options mirror the client's
    `RenderHandlerSettings` and are named to match it.

        await send_morphism(attention, debugBorders=False)

    To get the drawing back as an image rather than only onto the screen, see
    `websocket_transfer.capture.capture_morphism`.
    '''
    await wst.send_term(
        to_morphism(target, recycle=recycle),
        settings=display_settings(darkMode, debugBorders, coreDebug, width))
