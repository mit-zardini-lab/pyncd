'''Diagram delivery for the notebooks under `notebooks/sota/`.

A thin adapter over `notebook_diagrams`, which holds the actual mechanism. The
only thing kept here is the default of leaving `BlockOperator` bodies in, and
the `show` name the sota notebooks already call.

Declare the settings at the top of a notebook and pass them to every call:

    import notebooks.display.sota_figures as figures
    DIAGRAMS = figures.DiagramSettings(mode=figures.DiagramMode.OFF)
    await figures.show(term, 'A caption.', settings=DIAGRAMS)

One figure that wants another route is given `mode`, which stands in for
`settings.mode` for that call alone, so a notebook committed in
`DiagramMode.INLINE` can send one figure to the open page. `PYNCD_DIAGRAMS`
still overrides both, so a background run of the notebook draws nothing it has
no browser for. `DiagramMode.HTML` writes the figure as one HTML file that opens
with no server and no network, and `slug` names that file.

`assigned_sizes` stands in for the field of the same name. It holds the integer
a configuration assigned each named symbol, and under `AxisSizes.EXPONENT` or
`AxisSizes.SUBSCRIPT` the assignments are written onto the symbolic term, so each
symbol of a label built from several of them keeps its letter and carries its own
size, raised after the letter or lowered into its subscript.
`obsidian/05-backends/Compound Axis Labels.md` states the rule.

`forget_drawn_blocks()` empties the record of the bodies already delivered, so
the next figure draws the body of every block it holds rather than the box
alone. `remember_drawn_blocks.py` keeps the record and says why. The record
lives as long as the kernel, so a notebook run a second time in one kernel
draws fewer bodies than its first run did. A figure that must come out the
same on every run asks for `SubBlocks.EVERY_BODY`, which sends no record at
all.

`BlockRecycling.RECYCLED`, passed as the `block_recycling` of the settings,
normalises the body of every box before the figure is drawn, so a body a model
wrote by hand is drawn in the same form as the figure that holds it.

`casts` stands in for the field of the same name. The default,
`CastPresentation.THIN`, draws every conversion that changes a width as no glyph on
a box of no width, so the rounding is read from the format each wire is labelled
with, and `CastPresentation.DRAWN` draws each of them as a chevron.
`cast_presentation.py` under `notebooks/display/` holds the pass.

`advanced_display` stands in for the field of the same name. `AdvancedDisplay.LEGEND`
draws the table of the term's axes beside the figure and `INTERACTIVE` opens
inspection boxes on the page as well. `advanced_display.py` under
`notebooks/display/` says what each sends.
'''

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import websocket_transfer.send_morphism as send_morphism

from notebooks.display.notebook_diagrams import (
    DiagramMode as DiagramMode,
    DisplayMode as DisplayMode,
    DiagramSettings as DiagramSettings,
    SETTINGS as SETTINGS,
    SubBlocks as SubBlocks,
    forget_drawn_blocks as forget_drawn_blocks,
    show_diagram,
)
from notebooks.display.advanced_display import AdvancedDisplay as AdvancedDisplay
from notebooks.display.axis_sizes import AxisSizes as AxisSizes
from notebooks.display.block_recycling import BlockRecycling as BlockRecycling
from notebooks.display.cast_presentation import CastPresentation as CastPresentation
from notebooks.display.expand_with_parameters import (
    ExpandedParameters as ExpandedParameters)
from notebooks.display.tape_presentation import TapePresentation as TapePresentation


async def show(term: send_morphism.Sendable, caption: str | None = None,
               width: int | None = None, recycle: bool = True,
               sub_blocks: SubBlocks | None = None,
               mode: DiagramMode | None = None,
               assigned_sizes: Mapping[str, int] | None = None,
               advanced_display: AdvancedDisplay | None = None,
               casts: CastPresentation | None = None,
               slug: str = 'term',
               settings: DiagramSettings | None = None,
               display_mode: DisplayMode | None = None) -> None:
    '''Draw a term. `width` sets the wrap width, so it sets the figure's
    proportions rather than its scale, and a wide term wants more.
    `sub_blocks=SubBlocks.NO_BODIES` leaves the `BlockOperator` bodies out,
    giving the high-level view alone, and `SubBlocks.EVERY_BODY` draws every
    body whatever an earlier figure drew. `assigned_sizes` is a configuration's
    `assigned_integers_by_name()`, and under `AxisSizes.EXPONENT` or
    `AxisSizes.SUBSCRIPT` its integers are written onto the symbolic term as the
    exponent of each axis and each named symbol, raised or lowered. `sub_blocks`, `mode`, `assigned_sizes`, `advanced_display` and
    `casts` stand in for the fields of the same name for this call. `slug` names the file
    `DiagramMode.HTML` or `DiagramMode.DUMP` writes.'''
    chosen = settings if settings is not None else SETTINGS
    if display_mode is not None:
        chosen = dataclasses.replace(chosen, display_mode=display_mode)
    if mode is not None:
        chosen = dataclasses.replace(chosen, mode=mode)
    if sub_blocks is not None:
        chosen = dataclasses.replace(chosen, sub_blocks=sub_blocks)
    if assigned_sizes is not None:
        chosen = dataclasses.replace(chosen, assigned_sizes=assigned_sizes)
    if advanced_display is not None:
        chosen = dataclasses.replace(chosen, advanced_display=advanced_display)
    if casts is not None:
        chosen = dataclasses.replace(chosen, casts=casts)
    await show_diagram(term, caption, settings=chosen, slug=slug, width=width,
                       recycle=recycle)
