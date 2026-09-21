'''
A packaged `send_term` which accepts either a morphism or a hypergraph.

Notebooks alternate between two forms of the same diagram: a morphism, which
`send_term` displays directly, and a hypergraph, which has to be converted first
(the recurring `wst.send_term(h2m.hypergraph_to_morphism(graph))` idiom). This
module folds the conversion into the send, so either form can be handed over
without the caller deciding which it holds.
'''

import data_structure.Category as cat
import data_structure.Term as fd
import graphs.data_structure.Hypergraph as hg
import graphs.processing.Hypergraph2Morphism as h2m
import websocket_transfer.websockets_transfer as wst


type Sendable = fd.GeneralTerm | hg.Hypergraph


def to_morphism(target: Sendable, recycle: bool = False) -> fd.GeneralTerm:
    '''
    Coerce `target` into the morphism form `send_term` expects.

    Hypergraphs are converted, and anything else is already a term and passes
    through untouched. Tuples are mapped over elementwise, matching
    `fd.GeneralTerm`'s product case, and the two sides of a
    `cat.DefinedExpression` are each converted.

    With `recycle=True`, the result is additionally round-tripped through a
    `Multigraph`, which is how hand-built morphisms are normalised
    before display.
    '''
    match target:
        case hg.Hypergraph():
            morphism = h2m.hypergraph_to_morphism(target)
        case tuple():
            return tuple(to_morphism(element, recycle) for element in target)
        case cat.DefinedExpression(left_hand_side=left, right_hand_side=right):
            return cat.DefinedExpression(
                left_hand_side=to_morphism(left, recycle),
                right_hand_side=to_morphism(right, recycle))
        case _:
            morphism = target
    return h2m.recycle(morphism) if recycle else morphism


def convert_color_mode(
    color_mode: wst.ColorMode | bool | None,
) -> bool | None:
    if color_mode is None:
        return None
    if color_mode is wst.ColorMode.DARK:
        return True
    if color_mode is wst.ColorMode.LIGHT:
        return False
    if isinstance(color_mode, bool):
        return color_mode
    raise TypeError(f'{color_mode!r} is not a ColorMode, bool or None.')


def display_settings(
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
    title: str | None = None,
) -> wst.RenderHandlerSettings:
    '''
    Collect the display options into the partial dict the client expects.

    Each option left as `None` is omitted from the message, and the client
    falls back to its default for it, so every send fully determines the
    display rather than inheriting whatever the previous send happened to set.

    `ColorMode.DARK` and `ColorMode.LIGHT` map to the existing `darkMode`
    boolean on the wire. A boolean remains accepted by the public functions.

    `width` is the px at which a morphism wraps onto another line, and so
    controls the figure's proportions rather than its scale.

    `drawnBlockTags` names the `BlockTag`s, each by its `uid._id`, whose bodies
    the client is to leave out because an earlier send already drew them. The
    caller keeps that record, since the client wipes what it drew on every
    message. `notebooks/display/remember_drawn_blocks.py` keeps it for a
    notebook kernel.

    `legend` draws the table of axes beside the figure and `inspectionBoxes`
    opens a box over a block or an expandable operator the pointer rests on.
    Both read the `auxiliary` field sent beside the term, which
    `websocket_transfer/auxiliary_information.py` assembles.

    `axisHover` says where an axis answers the pointer, and is sent as the
    value of the `AxisHover` given.

    `axisLabelFontSize` is the size, in em, of the label an axis carries on its
    wire.

    `title` names what the page shows. The heading of the page and the name of
    its tab then read `tsncd - <title>`.
    '''
    settings: wst.RenderHandlerSettings = {}
    converted_dark_mode = convert_color_mode(darkMode)
    if converted_dark_mode is not None:
        settings['darkMode'] = converted_dark_mode
    if debugBorders is not None:
        settings['debugBorders'] = debugBorders
    if coreDebug is not None:
        settings['coreDebug'] = coreDebug
    if width is not None:
        settings['width'] = width
    if subBlocks is not None:
        settings['subBlocks'] = subBlocks
    if drawnBlockTags is not None:
        settings['drawnBlockTags'] = drawnBlockTags
    if tapeLabels is not None:
        settings['tapeLabels'] = tapeLabels
    if legend is not None:
        settings['legend'] = legend
    if inspectionBoxes is not None:
        settings['inspectionBoxes'] = inspectionBoxes
    if axisHover is not None:
        settings['axisHover'] = axisHover.value
    if axisLabelFontSize is not None:
        settings['axisLabelFontSize'] = axisLabelFontSize
    if title is not None:
        settings['title'] = title
    return settings


async def send_morphism(
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
    title: str | None = None,
    auxiliary: wst.DiagramAuxiliary | None = None,
) -> None:
    '''
    Display `target`, converting it to a morphism first if it is a hypergraph.

        await send_morphism(attention)               # already a morphism
        await send_morphism(attention_graph)         # converted, then sent

    The keyword-only display options mirror the client's
    `RenderHandlerSettings` and are named to match it.

        await send_morphism(attention, debugBorders=False)

    To get the drawing back as an image rather than only onto the screen, see
    `websocket_transfer.capture.capture_morphism`.

    `auxiliary` is what the legend and the inspection boxes show, assembled
    by `websocket_transfer.auxiliary_information.auxiliary_information` from
    the morphism this call sends, so a caller that passes it converts first.
    '''
    await wst.send_term(
        to_morphism(target, recycle=recycle),
        settings=display_settings(
            darkMode, debugBorders, coreDebug, width, subBlocks,
            drawnBlockTags, tapeLabels, legend, inspectionBoxes,
            axisHover, axisLabelFontSize, title),
        auxiliary=auxiliary)
