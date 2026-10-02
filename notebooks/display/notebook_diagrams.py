'''Drawing a morphism from a notebook.

Every notebook draws through `show_diagram`. A notebook builds one
`DiagramSettings` at the top of its setup cell and passes it to every call:

    import notebooks.display.notebook_diagrams as notebook_diagrams

    DIAGRAMS = notebook_diagrams.DiagramSettings(
        mode=notebook_diagrams.DiagramMode.OFF)

    await notebook_diagrams.show_diagram(
        morphism, 'The fused attention core.', settings=DIAGRAMS)

There are six modes:

    INLINE   Render a PNG and embed it in the cell output. Use it for the
             run you commit. Takes 166 ms for the transformer figure once the
             headless browser is warm, and about 1.7 s for the first diagram,
             which starts it. `DisplayFormat` chooses which browser cuts the
             image and in what format.
    BROWSER  Send the diagram to the open tsncd page and embed nothing. Takes
             about 15 milliseconds. Use it while working.
    HTML     Write the figure under `page_directory` as one HTML file, named by
             the `slug` of the call, and print the path. The file holds tsncd's
             bundle and the message BROWSER would send, so it opens from the
             disk with no server and no network, and under
             `AdvancedDisplay.INTERACTIVE` its inspection boxes answer the
             pointer. `websocket_transfer/standalone_page.py` assembles it, and
             starts no browser to do so. With two or more `localisations` the
             page carries every wording of its descriptions and a control to
             switch between them, per `websocket_transfer/localise_descriptions.py`.
    DUMP     Write the term to JSON under `dump_directory` and print the path.
    LISTING  Print the SSA listing `agent_display` renders, and draw nothing.
             Takes tens of milliseconds and needs no browser.
    OFF      Skip the diagram. Captions still print, so a notebook run without
             a browser is still a complete run.

`show_page_variants` draws several variants of one model as one page, such as
the model at its released quantisations and the same model in the reals, and
the page draws a selector between them. Each `PageVariant` names its
`PageVariantGroup`, and a variant either carries a term or is derived in the
browser from another by a `PageFunctor`. Under HTML the page holds every
variant and is written as `index.html` in the folder `<page_directory>/<slug>/`,
so its address ends in the name of the folder. `<page_directory>/<slug>.html`
redirects to the folder with the query and the hash kept. A figure written by
`show_diagram` stays one file. Under LISTING each variant's listing is printed
under its group and title, a derived one through the Python statement of its
functor. Under DUMP each variant is written to a file of its own. INLINE and
BROWSER draw the initial variant alone, and OFF prints the caption alone. The
variants were added by Claude Opus 5.5 (1M context), effort 40, on 2026-09-27,
and the folder on the same day.

`block_recycling` says whether a box's body is normalised before the figure
is drawn. The transport recycles what it is handed, which reaches the level it
was handed and not the body of any box. `RECYCLED` recycles the body of every
box and of every box inside those, so a hand-written body is drawn from the
same normal form as the figure holding it. `AS_WRITTEN`, the default, hands the
term over untouched. `block_recycling.py` holds the enum and
`algebra/broadcasted_recycle.py` the rewrite.

`sub_blocks` says which `BlockOperator` bodies are drawn as sub-diagrams
beside the main figure. `SubBlocks.BODIES_NOT_YET_DRAWN`, the default, draws a
body the kernel has not delivered beside an earlier figure, so a notebook that
boxes one attention core into five layers gets the core drawn beside the first
figure and the box alone in the other four.
`remember_drawn_blocks.DRAWN_BLOCK_TAGS` holds the tag of every body already
delivered and lives as long as the kernel does, which is why a notebook run a
second time in one kernel draws fewer bodies than its first run did.
`SubBlocks.EVERY_BODY` sends no tag at all, so every body is drawn however many
figures came before, and a figure asked for that way comes out the same on
every run. `SubBlocks.NO_BODIES` draws the boxes alone, giving the high-level
view. `remember_drawn_blocks.py` holds the enum and the record, and
`notebook_diagrams.forget_drawn_blocks()` empties the record.

`tape` says how a `Para`'s grabs and drops are shown. `BOXED` draws the term
as it stands. `ABSORBED` passes it through `to_para_wrap` just before it is
drawn, in whatever mode, so each grab sits on the operand port it feeds and
each drop on the result it saves. `tape_presentation.py` holds the enum and
the conversion.

`loop_initializers` says whether the initializer of a loop variable is drawn.
`HIDDEN`, the default, removes each initializer and the drop that starts its
variable just before the term is drawn, in whatever mode, because the starting
value is the universal unit of the accumulator and the loop grab inside the
loop shows the variable. `DRAWN` draws them. `loop_initializers.py` holds the
enum and the removal.

`clean_quantisation_labels`, true by default, leaves the quantisation of a datatype
on the wires where it changes and removes it from every other wire just before the
term is drawn, in whatever mode, so tsncd labels an array only where an operation
changed its quantisation. `clean_quantisation_labels.py` holds the pass.

`casts` says how a cast is drawn. `CastPresentation.THIN`, the default since
2026-09-20, takes the name off every `Quantization.TypeConvert` that changes a width
just before the term is drawn, in whatever mode, and tsncd draws an unnamed
conversion as no glyph on a box of no width, so the rounding is read from the format
each wire is labelled with rather than from an operation. The format the conversion
wrote is drawn in blue, and the gap it stands in leaves out the names of the axes,
which the gap the operand came from carries. `DRAWN` draws each of them
as the chevron tsncd gives it, whose two halves are as tall as the widths it reads
and writes, which is what a notebook about inserting the conversions asks for.
`cast_presentation.py` holds the enum and the pass.

`form` says which of three forms a figure is drawn in.
`DiagramForm.ALL_BROADCASTED`, the default, draws every array as one wire per
axis and every operator with its glyph, its cups and the wires of the axes it is
broadcast over, as every figure was drawn before the setting existed.
`ARROWS_AND_BROADCASTED` draws each array between two operators as one arrow
labelled with its datatype and its shape, and every operator keeps its glyph on
a plate whose edges name its axes. `ARROWS_AND_BOXES` draws every operator as a
box named by what it does, a contraction of two operands as `Matmul`, and nothing
of the broadcasting. The setting is sent to tsncd as `form` in the display
settings of the message. It changes nothing in the term, so a LISTING or DUMP run
is the same under every value. `websocket_transfer/websockets_transfer.py` holds
the enum, and `obsidian/05-backends/Diagram Wire Format.md` states what each form
draws.

`controls` says whether a page draws, under its heading, the buttons that switch
its form and its theme. `PageControls.SHOWN`, the default, draws them on the open
page and in a file HTML writes, outside the diagram container, so an INLINE
capture holds none of them. `HIDDEN` leaves them out, for a page that a host
drives through the address of its iframe or a message it posts. A switch redraws
the term the page holds, so a page written once carries every form.

`multiline_sizing` says how a figure wider than `width` is divided into rows.
`MultilineSizing.DYNAMIC`, the default since 2026-10-01, plans the rows so that
whole blocks stand on one row, with `width` as their target. `FIXED` fills each
row until the width runs out and cuts the block open at that point. A page
carries both, and
its row of controls switches between them. tsncd's
`src/display/Framework/dynamicMultilineSizing.ts` states the plan.

`axis_sizes` says where the size of an axis a configuration has sized is
drawn. `WIRE_LABEL`, the default, leaves every name as the expression wrote it,
and tsncd labels the wire of an axis carrying an integer size with that
integer. `EXPONENT` writes each integer size as the exponent of the name it
belongs to just before the term is drawn, in whatever mode, so the residual
width of DeepSeek-V4.1 reads `m^{5120}` and tsncd prints the name in place of
the integer. `SUBSCRIPT` writes the same value and lowers it into the
subscript, so the same axis reads `m_{5120}`. `axis_sizes.py` holds the enum
and the rewrite.

`assigned_sizes` holds the integer a configuration assigned each named symbol,
keyed by the bodies of the symbol's name, which is what
`term_utilities.generate_config.NumericConfig.assigned_integers_by_name()`
returns. Under `EXPONENT` the assignments are written onto the symbolic term, so
a label built from several symbols keeps each letter and gives each its own
exponent, and the sparse axis of the router reads `|k|^{6} \\text{ of } e^{384}`.
A notebook that draws a term a configuration has already sized passes none, and
the sizes are read off that term.

`advanced_display` says whether the figure carries a legend of its axes and
whether its boxes and expandable operators open inspection boxes on the page.
`AdvancedDisplay.OFF`, the default, sends the message an older page reads.
`LEGEND` draws the table of axes beside the figure, in the image of an INLINE
run. `INTERACTIVE` draws the legend and opens a box, with its top left corner at
the pointer, over a block or an operator with a standard expansion, holding the
title, the description, the code references and the body or the expansion drawn
as a diagram. `code_link_base` is where a reference carrying a path and no url is
linked. `advanced_display.py` holds the enum and says what is sent, and
`websocket_transfer/auxiliary_information.py` assembles it.

`operator_explanations` is a table from operator class to the title, the formula,
the description and the references an inspection box shows over an operator that
has no expansion, such as a top-k selection. Under `INTERACTIVE` each operator the
table explains is wrapped in a block that tsncd draws as the operator alone, so
the figure is unchanged and the operator answers the pointer.
`explain_operators.py` holds the pass. `operator_references` gives the references
the box of an expanded operator lists, by operator class. A model's package holds
both tables, as `notebooks/sota/DeepSeekV41Flash/operator_explanations.py` does.

`axis_label_font_size` is the size, in em, of the label an axis carries on its
wire. tsncd draws the label at 0.8 em where the setting is `None`, and measures
the room the label needs at the size it draws it.

`title` names what the page shows. The name of the tsncd page's tab reads
`tsncd - <title>` under BROWSER and in a file HTML writes, and `tsncd` where the
setting is `None`. A captured image holds the figure alone, so the title does
not appear in an INLINE run.

`heading` says whether the page writes that text as a heading over the figure.
`PageHeading.NONE`, the default, holds the figure alone, so a file HTML writes
stands as a page of a site that writes its own heading above it.
`PageHeading.TITLE` writes the heading.

An agent executing a notebook in the background does not edit the notebook to
change its mode. It sets `PYNCD_DIAGRAMS` in the environment of the process
that starts the kernel, and every `show_diagram` call then uses the mode that
variable names in place of the one the notebook declares. The mode a person
set at the top of the setup cell survives the run. `notebooks/execute_notebook.py`
starts a kernel that way:

    python notebooks/execute_notebook.py notebooks/base_features/BuildingAModel.ipynb

`PYNCD_DIAGRAMS` is the only environment variable this module reads, and a
notebook must not read it.

An INLINE capture goes to whichever browser `DisplayFormat.renderer` names.
HEADLESS is the default because it is the faster of the two by an order of
magnitude: Chromium paints a layout it already holds, where the open page has
to rebuild one from inline styles through `html-to-image`. PAGE cuts the image
out of the tab the user is looking at, which is the only route that guarantees
the picture matches what is on screen.

Either renderer falls back to the other when it cannot run, and the failure is
recorded for the session, so a run with no page open or no built bundle to hand
waits for the failure once.

A render fails when tsncd has no box for a term, which happens whenever an
operator is registered in Python before its TypeScript counterpart exists. The
failure is reported and the run continues.

The headless renderer runs on its own daemon thread with a proactor event
loop. On Windows, ipykernel runs a selector loop, which cannot spawn the
Chromium subprocess. The renderer is opened once and held until the kernel
exits, because starting it costs more than drawing any single figure.
'''

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import enum
import io
import os
import pathlib
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import data_structure.Category as cat
import data_transfer.term_json as term_json
import quantization.algebra.strip_quantisations as strip_quantisations
import term_utilities.term_utilities as tutil
import websocket_transfer.auxiliary_information as auxiliary_information
import websocket_transfer.capture as capture
import websocket_transfer.headless as headless
import websocket_transfer.localise_descriptions as localise_descriptions
import websocket_transfer.send_morphism as sending
import websocket_transfer.standalone_page as standalone_page
import websocket_transfer.websockets_transfer as wst

import algebra.registries.expansion_wording as expansion_wording
import utilities.wording_json as wording_json

import notebooks.display.advanced_display as advanced_display
import notebooks.display.axis_sizes as axis_sizes
import notebooks.display.block_recycling as block_recycling
import notebooks.display.clean_quantisation_labels as clean_quantisation_labels
import notebooks.display.cast_presentation as cast_presentation
import notebooks.display.display_wording as display_wording
import notebooks.display.expand_with_parameters as expand_with_parameters
import notebooks.display.explain_operators as explain_operators
import notebooks.display.explain_reindexings as explain_reindexings
import notebooks.display.loop_initializers as loop_initializers
import notebooks.display.notebook_listings as notebook_listings
import notebooks.display.remember_drawn_blocks as remember_drawn_blocks
import notebooks.display.tape_naming as tape_naming
import notebooks.display.tape_presentation as tape_presentation

TSNCD_DIST_FALLBACK = '../tsncd/dist'

ColorMode = wst.ColorMode
DisplayMode = wst.DisplayMode
MultilineSizing = wst.MultilineSizing


class DiagramMode(enum.Enum):
    INLINE = 'inline'
    BROWSER = 'browser'
    HTML = 'html'
    DUMP = 'dump'
    LISTING = 'listing'
    OFF = 'off'


class DiagramRenderer(enum.Enum):
    '''Which browser turns the drawn diagram into an image.'''
    HEADLESS = 'headless'
    PAGE = 'page'


@dataclasses.dataclass
class DisplayFormat:
    '''How the drawn diagram is turned into an image.

    `renderer` picks the browser. HEADLESS drives a Chromium held open for the
    kernel's life and captures it with `page.screenshot`, which is a real
    browser paint. PAGE sends a `renderRequest` to the open tsncd tab, which
    answers through `html-to-image`. Drawing and capturing the transformer
    figure measured 166 ms under HEADLESS against 2271 ms under PAGE.

    HEADLESS is the default for that reason. Prefer PAGE when the picture has
    to be the one on screen, since a headless capture is drawn in a browser
    nobody is looking at.

    `format` is `png` for either renderer. `svg` is available on either and
    runs to megabytes, because both produce it through `html-to-image`. `pdf`
    is available on HEADLESS alone, comes from Chromium's print pipeline, and
    cannot be shown inline, so it needs `save_to`. The transformer figure took
    430 ms as a pdf and came to 139 KB, against 187 KB as a png.

    `scale` applies to `png`. Under HEADLESS it is fixed per browser, so
    changing it between calls restarts the held Chromium.
    '''
    renderer: DiagramRenderer = DiagramRenderer.HEADLESS
    format: Literal['png', 'svg', 'pdf'] = 'png'
    scale: float = 2.0
    padding: int = 16
    background: str | None = 'auto'
    save_to: pathlib.Path | None = None


PAGE_FORMATS = ('png', 'svg')


def check_display_format(display_format: DisplayFormat) -> None:
    '''Reject a combination no renderer can produce, naming the alternative.'''
    if (display_format.renderer is DiagramRenderer.PAGE
            and display_format.format not in PAGE_FORMATS):
        raise ValueError(
            f'The open page cannot produce {display_format.format}. It comes '
            'from Chromium\'s print pipeline, so it needs '
            'DiagramRenderer.HEADLESS.')
    if display_format.format == 'pdf' and display_format.save_to is None:
        raise ValueError(
            'A pdf cannot be shown in a cell, so it needs save_to.')


@dataclasses.dataclass
class DiagramSettings:
    '''How a notebook draws its diagrams.

    A notebook builds one of these in its setup cell and passes it to every
    `show_diagram` call. Any field can be overridden for a single call by
    passing it to `show_diagram` as a keyword argument.
    '''
    mode: DiagramMode = DiagramMode.INLINE
    display_mode: DisplayMode = DisplayMode.FAST
    width: int | None = None
    multiline_sizing: MultilineSizing = MultilineSizing.DYNAMIC
    block_recycling: block_recycling.BlockRecycling = (
        block_recycling.BlockRecycling.AS_WRITTEN)
    sub_blocks: remember_drawn_blocks.SubBlocks = (
        remember_drawn_blocks.SubBlocks.BODIES_NOT_YET_DRAWN)
    recycle: bool = True
    tape: tape_presentation.TapePresentation = tape_presentation.TapePresentation.ABSORBED
    loop_initializers: loop_initializers.LoopInitializers = (
        loop_initializers.LoopInitializers.HIDDEN)
    clean_quantisation_labels: bool = True
    casts: cast_presentation.CastPresentation = (
        cast_presentation.CastPresentation.THIN)
    form: wst.DiagramForm = wst.DiagramForm.ALL_BROADCASTED
    controls: wst.PageControls = wst.PageControls.SHOWN
    tape_naming: tape_naming.TapeNaming = tape_naming.TapeNaming.NAMED
    axis_sizes: axis_sizes.AxisSizes = axis_sizes.AxisSizes.WIRE_LABEL
    assigned_sizes: Mapping[str, int] | None = None
    advanced_display: advanced_display.AdvancedDisplay = (
        advanced_display.AdvancedDisplay.OFF)
    code_link_base: str | None = None
    # What stands for a weight in the expansion an inspection box draws: a weight
    # array, or a box that reads the weight from the tape, presented under `tape`.
    expanded_parameters: expand_with_parameters.ExpandedParameters = (
        expand_with_parameters.ExpandedParameters.WEIGHT_ARRAYS)
    # The explanation of each operator that has no expansion, by operator class.
    # Under `AdvancedDisplay.INTERACTIVE` each such operator is wrapped in a block
    # drawn as the operator alone, whose inspection box shows the explanation.
    operator_explanations: explain_operators.OperatorExplanations | None = None
    # The code references the inspection box of an expanded operator lists, by
    # operator class.
    operator_references: auxiliary_information.OperatorReferences | None = None
    # What one named operator does in the model that is drawn, by the text of its
    # name, which the inspection box of an expanded operator says first.
    operator_roles: auxiliary_information.OperatorRoles | None = None
    # The explanation of each named reindexing, by the text of its name. Under
    # `AdvancedDisplay.INTERACTIVE` each such reindexing is wrapped in a block drawn
    # as the reindexing alone, whose inspection box shows the explanation.
    reindexing_explanations: explain_reindexings.ReindexingExplanations | None = None
    # Where an axis answers the pointer on the open page: from its legend row
    # alone, from every wire and name of it as well, or nowhere.
    axis_hover: wst.AxisHover = wst.AxisHover.LEGEND
    # The size, in em, of the label an axis carries on its wire. `None` leaves
    # the page's default, 0.8.
    axis_label_font_size: float | None = None
    # Each box's own bounding box, for when the question is what nests inside
    # what. Sent explicitly on every call, because display settings persist
    # between sends on a live page and a toggle left on in the browser would
    # otherwise leak into every later figure.
    debug_borders: bool = False
    dark_mode: ColorMode | bool | None = None
    timeout: int = 30
    title: str | None = None
    heading: wst.PageHeading = wst.PageHeading.NONE
    # The wordings an HTML page switches its descriptions between, the first
    # being the text module the figure was built from.
    localisations: tuple[localise_descriptions.Localisation, ...] = ()
    page_directory: pathlib.Path = pathlib.Path('outputs/pages')
    dump_directory: pathlib.Path = pathlib.Path('outputs/dumps')
    display_format: DisplayFormat = dataclasses.field(default_factory=DisplayFormat)


SETTINGS = DiagramSettings()

SubBlocks = remember_drawn_blocks.SubBlocks
CastPresentation = cast_presentation.CastPresentation
AdvancedDisplay = advanced_display.AdvancedDisplay
DiagramForm = wst.DiagramForm
PageControls = wst.PageControls
AxisHover = wst.AxisHover
PageHeading = wst.PageHeading
forget_drawn_blocks = remember_drawn_blocks.forget_drawn_blocks

# The one environment variable read here. `notebooks/execute_notebook.py` sets
# it for the kernel it starts, and `show_diagram` uses the mode it names in
# place of the one the notebook declares.
MODE_OVERRIDE_VARIABLE = 'PYNCD_DIAGRAMS'


def mode_override() -> DiagramMode | None:
    '''The mode `PYNCD_DIAGRAMS` names, or None when it is unset or empty.'''
    value = os.environ.get(MODE_OVERRIDE_VARIABLE, '').strip().lower()
    if not value:
        return None
    try:
        return DiagramMode(value)
    except ValueError:
        names = ', '.join(mode.value for mode in DiagramMode)
        raise ValueError(
            f'{MODE_OVERRIDE_VARIABLE}={value!r} names no diagram mode. '
            f'The modes are {names}.') from None


_renderer_loop = None
_renderer = None
# The scale the held renderer was opened at. Playwright fixes the pixel density
# per browser context rather than per screenshot, so a call asking for another
# scale has to restart it.
_renderer_scale: float | None = None
# Whether the page the user has open still answers. Set to False by the first
# capture that times out, so the rest of the session goes straight to headless.
_page_answers = True
# Whether a headless browser can be started at all. Set to False by the first
# attempt that fails, so a machine with no built bundle and no Playwright pays
# the failure once.
_headless_runs = True


def tsncd_dist() -> str:
    '''The built tsncd bundle: the locations `headless.find_dist` searches, then a
    tsncd checkout beside the folder this repository stands in.'''
    try:
        return headless.find_dist()
    except FileNotFoundError:
        return headless.find_dist(TSNCD_DIST_FALLBACK)


def _loop_on_its_own_thread() -> asyncio.AbstractEventLoop:
    global _renderer_loop
    if _renderer_loop is None:
        loop = (asyncio.ProactorEventLoop() if sys.platform == 'win32'
                else asyncio.new_event_loop())
        threading.Thread(target=loop.run_forever, daemon=True).start()
        _renderer_loop = loop
    return _renderer_loop


async def _capture_headless(
    term, settings: DiagramSettings,
    auxiliary: wst.DiagramAuxiliary | None = None,
) -> bytes | None:
    '''Render in a browser started here, on its own thread.

    Returns None once a headless browser has been found not to start, so the
    rest of the session goes to the open page instead.
    '''
    global _headless_runs
    if not _headless_runs:
        return None
    display_format = settings.display_format

    async def render() -> bytes:
        global _renderer, _renderer_scale
        if _renderer is not None and _renderer_scale != display_format.scale:
            await _renderer.__aexit__(None, None, None)
            _renderer = None
        if _renderer is None:
            renderer = headless.HeadlessRenderer(
                dist=tsncd_dist(), scale=display_format.scale)
            await renderer.__aenter__()   # closed when the kernel exits
            _renderer = renderer
            _renderer_scale = display_format.scale
        return await _renderer.capture(
            term, recycle=settings.recycle,
            format=display_format.format,
            padding=display_format.padding,
            background=display_format.background,
            width=settings.width,
            subBlocks=remember_drawn_blocks.draws_any_body(settings.sub_blocks),
            drawnBlockTags=remember_drawn_blocks.tags_to_skip(
                settings.sub_blocks),
            debugBorders=settings.debug_borders, darkMode=settings.dark_mode,
            legend=advanced_display.draws_legend(settings.advanced_display),
            inspectionBoxes=advanced_display.opens_inspection_boxes(
                settings.advanced_display),
            axisHover=settings.axis_hover,
            axisLabelFontSize=settings.axis_label_font_size,
            form=settings.form, controls=settings.controls,
            displayMode=settings.display_mode,
            multilineSizing=settings.multiline_sizing,
            auxiliary=auxiliary)

    try:
        future = asyncio.run_coroutine_threadsafe(
            render(), _loop_on_its_own_thread())
        return await asyncio.wrap_future(future)
    except (ImportError, FileNotFoundError):
        _headless_runs = False
        return None


async def close_renderer() -> None:
    '''Shut down the headless browser, if one was started.

    A notebook that draws many diagrams and then continues for a long time can
    call this to release the Chromium process early. Otherwise the renderer is
    held until the kernel exits.
    '''
    global _renderer, _renderer_scale
    if _renderer is None:
        return
    renderer = _renderer
    _renderer = None
    _renderer_scale = None
    future = asyncio.run_coroutine_threadsafe(
        renderer.__aexit__(None, None, None), _loop_on_its_own_thread())
    await asyncio.wrap_future(future)


def dump_term(term, slug: str, settings: DiagramSettings) -> pathlib.Path:
    '''Write `term` to the JSON tsncd's loader reads, and return the path.'''
    import json

    import data_transfer.term_json as term_json

    settings.dump_directory.mkdir(parents=True, exist_ok=True)
    path = settings.dump_directory / f'{slug}.json'
    payload = term_json.TermJSONConverter.export_to_json(sending.to_morphism(term))
    io.open(path, 'w', encoding='utf-8').write(
        payload if isinstance(payload, str) else json.dumps(payload))
    return path


def print_listing(term, settings: DiagramSettings) -> None:
    '''Print `term` as the SSA listing `agent_display` renders, in place of a diagram.

    The conversion is the one the diagram transport performs. A hypergraph is
    converted to a morphism, and a hand-built morphism is recycled when
    `settings.recycle` is set. A term the listing cannot render is reported and
    stepped over, as a diagram that cannot be drawn is.
    '''
    try:
        notebook_listings.print_listing(
            sending.to_morphism(term, recycle=settings.recycle), recycle=False)
    except Exception as error:
        # Deliberately broad, for the reason given in `show_diagram`.
        first_line = (str(error).strip().splitlines() or [''])[0]
        print(f'(no listing: {type(error).__name__}: {first_line[:120]})')


async def show_diagram(term, caption: str | None = None, *,
                       settings: DiagramSettings | None = None,
                       slug: str = 'term', **overrides):
    '''Draw `term` as a neural circuit diagram, and display it.

    `term` may be a morphism or a hypergraph, and the transport converts.
    `width` sets the wrap width, so it sets the figure's proportions rather
    than its scale. `sub_blocks=SubBlocks.NO_BODIES` leaves the
    `BlockOperator` bodies out, giving the high-level view alone.
    `SubBlocks.EVERY_BODY` draws every body, whatever an earlier figure drew.
    The default, `SubBlocks.BODIES_NOT_YET_DRAWN`, leaves out a body drawn
    beside an earlier figure and still draws its box, per the record
    `remember_drawn_blocks` keeps and `forget_drawn_blocks()` empties.
    `recycle=False` skips the round trip
    through a hypergraph that normalises a hand-built morphism.
    `display_format` chooses which browser cuts the image, in what format, and
    where it is saved. The default `tape`, `TapePresentation.ABSORBED`,
    re-expresses a term holding grabs and drops through `to_para_wrap` before
    it is drawn, so each grab sits on the operand port it feeds and each drop
    on the result it saves, and a term holding none is drawn as it stands.
    `tape=TapePresentation.BOXED` draws each grab and drop as a box of its own.
    The default `loop_initializers`, `LoopInitializers.HIDDEN`, leaves out the
    initializer of every loop variable and the drop that starts it, and
    `LoopInitializers.DRAWN` draws them. The default `clean_quantisation_labels`
    leaves the quantisation of a datatype only on the wires where it changes,
    and `clean_quantisation_labels=False` labels every wire that carries one.
    The default `casts`, `CastPresentation.THIN`, draws every conversion that
    changes a width as no glyph on a box of no width, so the rounding is read from
    the format each wire is labelled with, and `casts=CastPresentation.DRAWN` draws
    each of them as a chevron.
    `form=DiagramForm.ARROWS_AND_BROADCASTED` draws each array that passes from
    one operator to another as one arrow labelled with its datatype and its
    shape, with every operator's glyph on a plate, and `ARROWS_AND_BOXES` draws
    every operator as a named box. The default, `ALL_BROADCASTED`, draws one wire
    for each axis of every array. `controls=PageControls.HIDDEN` leaves the
    buttons that switch the form and the theme off the page.
    `tape_naming=TapeNaming.INDEXED` writes the index of every loop that
    selects a member of the tape after each slot's label, so a weight grabbed
    inside a repeated layer reads `W_{G}[l]`, and `TapeNaming.CODE_FORM` writes
    the member's code form with the same indices in typewriter. The default,
    `NAMED`, draws the labels as the term names them.
    `axis_sizes=AxisSizes.EXPONENT` writes the integer size a configuration
    assigned an axis as the exponent of that axis's name, so the residual width
    of DeepSeek-V4.1 reads `m^{5120}` where the default, `WIRE_LABEL`, labels
    the wire `5120`, and `AxisSizes.SUBSCRIPT` lowers the same value into the
    subscript, `m_{5120}`. `assigned_sizes`, which is a configuration's
    `assigned_integers_by_name()`, writes those exponents onto the symbolic term
    rather than onto one the configuration has sized, so every symbol of a label
    built from several of them keeps its letter and carries its own size.
    `block_recycling=BlockRecycling.RECYCLED` normalises the
    body of every box before anything else is done to the term, so a body
    written by hand is drawn in the form the transport draws the rest in.
    `advanced_display=AdvancedDisplay.LEGEND` draws the table of the term's
    axes, sizes and code names beside the figure, and `INTERACTIVE` opens
    inspection boxes on the page as well, with the code references the blocks
    carry linked under `code_link_base`. The default, `OFF`, sends neither.

    When `PYNCD_DIAGRAMS` is set, the mode it names is used in place of
    `settings.mode`.

    A missing server is reported and stepped over rather than raised, so a run
    with no browser to hand is still a complete run.
    '''
    settings = dataclasses.replace(settings or SETTINGS, **overrides)
    mode = mode_override() or settings.mode
    if caption:
        print(caption)
    if mode is DiagramMode.OFF:
        return
    term = present_each_side(term, settings)
    term, settings, auxiliary = package_auxiliary(term, settings)

    if mode is DiagramMode.LISTING:
        print_listing(term, settings)
        return

    if mode is DiagramMode.DUMP:
        print(f'(dumped to {dump_term(term, slug, settings)})')
        return

    if mode is DiagramMode.HTML:
        print(f'(written to {save_page(term, slug, settings, auxiliary)})')
        return

    if mode is DiagramMode.BROWSER:
        await _send_to_open_page(term, settings, auxiliary)
        record_bodies_drawn(term, settings)
        return

    check_display_format(settings.display_format)
    try:
        payload = await _capture_inline(term, settings, auxiliary)
    except Exception as error:
        # Deliberately broad. A missing browser, a missing bundle and a diagram
        # too large to lay out are all reasons to carry on without a picture. A
        # renderer error arrives with a JavaScript stack attached, none of which
        # points into this repository, so only its first line is printed.
        first_line = (str(error).strip().splitlines() or [''])[0]
        print(f'(no diagram: {type(error).__name__}: {first_line[:120]})')
        return

    if payload is None:
        print('(no diagram: no browser answered)')
        return
    deliver_image(payload, settings.display_format)
    record_bodies_drawn(term, settings)


def present_each_side(term, settings: DiagramSettings):
    '''`term` as the settings present it. A `cat.DefinedExpression` is presented one
    side at a time, because every presentation reads a morphism or a hypergraph.'''
    if isinstance(term, cat.DefinedExpression):
        return cat.DefinedExpression(
            left_hand_side=present_each_side(term.left_hand_side, settings),
            right_hand_side=present_each_side(term.right_hand_side, settings))
    term = block_recycling.present(term, settings.block_recycling)
    term = loop_initializers.present(term, settings.loop_initializers)
    if settings.clean_quantisation_labels:
        term = clean_quantisation_labels.strip_unchanged_quantisations(term)
    term = cast_presentation.present(term, settings.casts)
    term = tape_presentation.present(term, settings.tape)
    term = tape_naming.present(term, settings.tape_naming)
    return axis_sizes.present(term, settings.axis_sizes, settings.assigned_sizes)


def package_auxiliary(
    term, settings: DiagramSettings,
) -> tuple[object, DiagramSettings, wst.DiagramAuxiliary | None]:
    '''`term` as the transport sends it, the settings to send it with, and the
    auxiliary information of that morphism. Under `AdvancedDisplay.OFF` the term
    and the settings are returned as they stand and there is no auxiliary
    information. Otherwise the term is converted here, because the expansions are
    keyed by the order of the nodes of exactly what is sent, and the settings no
    longer ask the transport to recycle what has been converted. Under
    `INTERACTIVE` every operator `settings.operator_explanations` explains is
    wrapped in a block drawn as the operator alone, after the conversion, so the
    recycling does not meet the wrappers. A cast takes the row
    `cast_presentation.with_cast_explained` adds, unless the table holds one of its
    own, because a cast drawn thin has no glyph and its box is opened from the
    coloured format on the wire. Every reindexing
    `settings.reindexing_explanations` explains is wrapped the same way once the
    expansions have been assembled, because a rule that writes an operator out reads
    its reindexings, and the information of the blocks is then read off the wrapped
    term.'''
    if settings.advanced_display is advanced_display.AdvancedDisplay.OFF:
        return term, settings, None
    packaged = packaged_figure(term, settings)
    return packaged.morphism, packaged.settings, packaged.auxiliary


@dataclasses.dataclass(frozen=True)
class SentFigure:
    '''A term as the transport sends it: the morphism exported, the morphism the
    expansions of its auxiliary information were written from, which lacks only the
    blocks explaining its reindexings, the settings it is sent with, and its
    auxiliary information. The explaining blocks of a morphism are given fresh tags
    each time a term is presented, so the auxiliary information of a variant derived
    from this figure is written for this morphism and no other.'''
    morphism: object
    morphism_expanded: object
    settings: DiagramSettings
    auxiliary: wst.DiagramAuxiliary | None

    def message(self) -> wst.DataUpdate:
        '''The `dataUpdate` drawing the figure. `data` is the exported term as a JSON
        object rather than as text, so the parts it shares with the term of another
        variant are compressed into the same records.'''
        return wst.with_auxiliary({
            'msgType': 'dataUpdate',
            'data': term_json.TermJSONConverter.export_document(self.morphism),
            'settings': page_settings(self.settings),
        }, self.auxiliary)  # type: ignore[typeddict-item]


def packaged_figure(term, settings: DiagramSettings) -> SentFigure:
    '''`term` packaged as `package_auxiliary` states, for a setting of
    `advanced_display` other than `OFF`.'''
    morphism = morphism_with_operators_explained(term, settings)
    auxiliary = advanced_display.auxiliary_for(
        morphism, settings.advanced_display, assigned_sizes_of(settings),
        settings.code_link_base, settings.tape, settings.expanded_parameters,
        settings.operator_references, settings.operator_roles)
    exported = morphism
    if (settings.advanced_display is advanced_display.AdvancedDisplay.INTERACTIVE
            and settings.reindexing_explanations):
        exported = explain_reindexings.present(
            morphism, settings.reindexing_explanations)
        auxiliary['blocks'] = auxiliary_information.block_information(
            exported, settings.code_link_base)
    return SentFigure(
        morphism=exported, morphism_expanded=morphism,
        settings=dataclasses.replace(settings, recycle=False), auxiliary=auxiliary)


def morphism_with_operators_explained(term, settings: DiagramSettings) -> object:
    '''`term` converted as the transport converts it and, under
    `AdvancedDisplay.INTERACTIVE`, with every operator the settings explain wrapped
    in a block drawn as the operator alone, which is the morphism the legend, the
    blocks and the expansions of `package_auxiliary` are read off.'''
    morphism = sending.to_morphism(term, recycle=settings.recycle)
    if settings.advanced_display is advanced_display.AdvancedDisplay.INTERACTIVE:
        return explain_operators.present(
            morphism,
            cast_presentation.with_cast_explained(settings.operator_explanations))
    return morphism


def assigned_sizes_of(settings: DiagramSettings) -> dict[str, int] | None:
    return None if settings.assigned_sizes is None else dict(settings.assigned_sizes)


def record_bodies_drawn(term, settings: DiagramSettings) -> None:
    '''Record the `BlockOperator` bodies this figure delivered, so that the next
    figure asked for with `SubBlocks.BODIES_NOT_YET_DRAWN` leaves them out. A
    figure asked for with `SubBlocks.NO_BODIES` delivered none of them, so it
    records none.'''
    if remember_drawn_blocks.draws_any_body(settings.sub_blocks):
        remember_drawn_blocks.remember(term)


async def _capture_inline(
    term, settings: DiagramSettings,
    auxiliary: wst.DiagramAuxiliary | None = None,
) -> bytes | None:
    '''Capture with the renderer chosen, falling back to the other one.

    A fallback that cannot produce the format asked for is skipped, so a pdf
    requested of a headless browser that will not start is reported rather than
    arriving quietly as a png.
    '''
    display_format = settings.display_format
    if display_format.renderer is DiagramRenderer.HEADLESS:
        payload = await _capture_headless(term, settings, auxiliary)
        if payload is not None or display_format.format not in PAGE_FORMATS:
            return payload
        return await _capture_from_open_page(term, settings, auxiliary)
    payload = await _capture_from_open_page(term, settings, auxiliary)
    if payload is not None:
        return payload
    return await _capture_headless(term, settings, auxiliary)


def deliver_image(payload: bytes, display_format: DisplayFormat) -> None:
    '''Write the image where it was asked for, and show it in the cell.

    A pdf is written and nothing is shown, because a notebook cell cannot
    render one.
    '''
    if display_format.save_to is not None:
        path = pathlib.Path(display_format.save_to)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        print(f'Wrote {path} ({len(payload)} bytes).')
    if display_format.format == 'pdf':
        return
    from IPython.display import display
    display(capture.as_display(payload, display_format.format))


async def _send_to_open_page(
    term, settings: DiagramSettings,
    auxiliary: wst.DiagramAuxiliary | None = None,
):
    '''Push the diagram to the connected page. Returns nothing to display.'''
    with contextlib.redirect_stdout(io.StringIO()):
        await sending.send_morphism(
            term, recycle=settings.recycle, width=settings.width,
            subBlocks=remember_drawn_blocks.draws_any_body(settings.sub_blocks),
            drawnBlockTags=remember_drawn_blocks.tags_to_skip(
                settings.sub_blocks),
            debugBorders=settings.debug_borders,
            darkMode=settings.dark_mode,
            legend=advanced_display.draws_legend(settings.advanced_display),
            inspectionBoxes=advanced_display.opens_inspection_boxes(
                settings.advanced_display),
            axisHover=settings.axis_hover,
            axisLabelFontSize=settings.axis_label_font_size,
            displayMode=settings.display_mode,
            form=settings.form, controls=settings.controls,
            title=settings.title,
            heading=settings.heading,
            multilineSizing=settings.multiline_sizing,
            auxiliary=auxiliary)
    return None


def page_settings(settings: DiagramSettings) -> wst.RenderHandlerSettings:
    '''The display settings of a message whose figure stays on a page, which is
    the open page a capture disturbs and the page `save_page` writes.'''
    return sending.display_settings(
        darkMode=settings.dark_mode,
        debugBorders=settings.debug_borders,
        width=settings.width,
        subBlocks=remember_drawn_blocks.draws_any_body(settings.sub_blocks),
        drawnBlockTags=remember_drawn_blocks.tags_to_skip(settings.sub_blocks),
        legend=advanced_display.draws_legend(settings.advanced_display),
        inspectionBoxes=advanced_display.opens_inspection_boxes(
            settings.advanced_display),
        axisHover=settings.axis_hover,
        axisLabelFontSize=settings.axis_label_font_size,
        form=settings.form, controls=settings.controls,
        title=settings.title,
        heading=settings.heading,
        displayMode=settings.display_mode,
        multilineSizing=settings.multiline_sizing)


PACKAGE_WORDING_FILES: tuple[pathlib.Path, ...] = (
    expansion_wording.WORDING_FILE, display_wording.WORDING_FILE)


def with_package_wordings(
    localisations: tuple[localise_descriptions.Localisation, ...],
) -> tuple[localise_descriptions.Localisation, ...]:
    '''`localisations` with the wordings of the packages that draw a figure, the
    expansions and the display passes, joined into the first, which is the wording
    a page was exported with, so that a description composed by a package is read
    back as its entries. An entry of the model's file overrides an entry of a
    package's file by the same name.'''
    if not localisations:
        return localisations
    first, *rest = localisations
    joined = {**wording_json.table_of(expansion_wording.TEXT),
              **wording_json.table_of(display_wording.TEXT), **first.table}
    return (dataclasses.replace(first, table=joined), *rest)


def save_page(
    term, slug: str, settings: DiagramSettings,
    auxiliary: wst.DiagramAuxiliary | None = None,
) -> pathlib.Path:
    '''Write `term` under `settings.page_directory` as one HTML file that opens
    with no server and no network, and return the path. A second wording of
    `settings.localisations` is read with `Localisation.from_wording_file` against
    the model's file and `PACKAGE_WORDING_FILES`.'''
    return standalone_page.save_standalone_page(
        term, settings.page_directory / f'{slug}.html',
        recycle=settings.recycle, settings=page_settings(settings),
        auxiliary=auxiliary, dist=tsncd_dist(),
        localisations=with_package_wordings(settings.localisations))


async def _capture_from_open_page(
    term, settings: DiagramSettings,
    auxiliary: wst.DiagramAuxiliary | None = None,
) -> bytes | None:
    '''Render on the page the user has open, or return None if none answers.

    The verdict is remembered for the session. A notebook run with no page
    attached then waits out the timeout once rather than once per diagram.
    '''
    global _page_answers
    if not _page_answers:
        return None
    display_format = settings.display_format
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return await wst.capture_term(
                sending.to_morphism(term, recycle=settings.recycle),
                settings=page_settings(settings),
                capture=capture.capture_options(
                    display_format.format, display_format.scale,
                    display_format.padding, display_format.background),
                timeout=settings.timeout,
                auxiliary=auxiliary)
    except (OSError, wst.CaptureError):
        _page_answers = False
        return None


class PageFunctor(enum.Enum):
    '''A functor tsncd applies in the browser to the term of one variant of a page
    to draw another. `DEQUANTISE` takes every quantisation off the term and deletes
    every cast between two quantisations of one value, and
    `quantization.algebra.strip_quantisations` states it in Python.'''
    DEQUANTISE = 'dequantise'


PYTHON_STATEMENT_OF_PAGE_FUNCTOR: dict[
        PageFunctor, Callable[[cat.Morphism], cat.Morphism]] = {
    PageFunctor.DEQUANTISE: strip_quantisations.strip_quantisations}


class LocalisedPageVariants(ValueError):
    '''A page with variants was asked to carry localisations, which it does not.'''


@dataclasses.dataclass(frozen=True)
class PageVariantGroup:
    '''A group of the selector drawn by a page with variants, such as `Decode`.'''
    identifier: str
    title: str


@dataclasses.dataclass(frozen=True)
class PageVariant:
    '''One variant of a page. A variant carrying a `term` is drawn from it under
    `settings`, or under the settings of the page where `settings` is `None`. A
    variant carrying no term is derived in the browser by applying `functor` to the
    term of the variant named by `derived_from`. The browser receives that term as
    it was presented, so the `settings` of a derived variant reach the page as
    display settings merged over those of its source, such as the theme or the
    width, and do not change how the term is presented. Settings differing from the
    source's in a field of `INSPECTION_TEXT_FIELDS`, such as the roles of the
    operators, also change what the inspection boxes of the derived variant say.'''
    identifier: str
    group: PageVariantGroup
    title: str
    detail: str
    term: object | None = None
    settings: DiagramSettings | None = None
    derived_from: str | None = None
    functor: PageFunctor | None = None


def check_page_variants(variants: Sequence[PageVariant], initial: str) -> None:
    '''Raise `standalone_page.InconsistentVariants` unless the identifiers of
    `variants` are distinct, `initial` is one of them, and every variant carries a
    term or is derived, by a functor, from a variant carrying one.'''
    standalone_page.check_variant_outlines([
        standalone_page.VariantOutline(
            identifier=variant.identifier, derived_from=variant.derived_from,
            functor=None if variant.functor is None else variant.functor.value,
            carries_its_own_figure=variant.term is not None)
        for variant in variants], initial)


def apply_page_functor(functor: PageFunctor, term: object) -> object:
    '''`term` taken through the Python statement of `functor`, one side at a time
    for a `cat.DefinedExpression`. A hypergraph is converted to a morphism first.'''
    morphism = sending.to_morphism(term)
    statement = PYTHON_STATEMENT_OF_PAGE_FUNCTOR[functor]
    if isinstance(morphism, cat.DefinedExpression):
        return cat.DefinedExpression(
            left_hand_side=statement(morphism.left_hand_side),
            right_hand_side=statement(morphism.right_hand_side))
    return statement(morphism)


def term_of_page_variant(
    variant: PageVariant, by_identifier: Mapping[str, PageVariant],
) -> object:
    '''The term of `variant`, or, for a derived variant, the Python statement of its
    functor applied to the term of the variant it is derived from.'''
    if variant.term is not None:
        return variant.term
    source = by_identifier[variant.derived_from]  # type: ignore[index]
    return apply_page_functor(variant.functor, source.term)  # type: ignore[arg-type]


def settings_of_page_variant(
    variant: PageVariant, by_identifier: Mapping[str, PageVariant],
    settings: DiagramSettings,
) -> DiagramSettings:
    '''The settings `variant` is drawn under in Python: its own, those of the
    variant it is derived from where it is derived and states none, and `settings`,
    the settings of the page, otherwise.'''
    if variant.settings is not None:
        return variant.settings
    if variant.derived_from is not None:
        return settings_of_page_variant(
            by_identifier[variant.derived_from], by_identifier, settings)
    return settings


def sent_figure(term: object, settings: DiagramSettings) -> SentFigure:
    '''`term` presented and packaged under `settings` as `show_diagram` presents and
    packages a figure, and converted as the transport converts it.'''
    term = present_each_side(term, settings)
    if settings.advanced_display is not advanced_display.AdvancedDisplay.OFF:
        return packaged_figure(term, settings)
    morphism = sending.to_morphism(term, recycle=settings.recycle)
    return SentFigure(morphism=morphism, morphism_expanded=morphism,
                      settings=settings, auxiliary=None)


INSPECTION_TEXT_FIELDS = (
    'operator_roles', 'operator_explanations', 'reindexing_explanations',
    'operator_references', 'code_link_base', 'expanded_parameters')
'''The fields of `DiagramSettings` that change what an inspection box says and
change nothing in the figure outside the explaining blocks.'''


def changes_the_inspection_text(
    settings: DiagramSettings, source_settings: DiagramSettings,
) -> bool:
    return any(getattr(settings, field) != getattr(source_settings, field)
               for field in INSPECTION_TEXT_FIELDS)


def explaining_block_records(
    morphism: object, settings: DiagramSettings,
) -> dict[str, wst.BlockInformation]:
    '''The record of every block of `morphism` drawn as its body alone that explains
    one operator or one named reindexing for which the tables of `settings` hold an
    explanation, written from that explanation and keyed by the tag the block
    already carries. A block of `morphism` whose operator or reindexing the tables
    do not explain is left out, and keeps the record it had.'''
    explanations = cast_presentation.with_cast_explained(settings.operator_explanations)
    records: dict[str, wst.BlockInformation] = {}
    for block in tutil.type_search(cat.Block, morphism):
        aesthetics = block.block_tag.aesthetics
        if (aesthetics is None
                or aesthetics.drawing is not cat.BlockDrawing.BODY_IN_PLACE):
            continue
        explained = block_explaining_body(block.body, explanations, settings)
        if explained is None:
            continue
        written = auxiliary_information.block_information(
            explained, settings.code_link_base)
        records[str(block.block_tag.uid._id)] = next(iter(written.values()))
    return records


def block_explaining_body(
    body: object, explanations: explain_operators.OperatorExplanations,
    settings: DiagramSettings,
) -> cat.Block | None:
    '''The block the explanation tables would wrap `body` in, where they explain it.'''
    if isinstance(body, cat.Broadcasted):
        explanation = explain_operators.explanation_for(body, explanations)
        return (None if explanation is None
                else explain_operators.drawn_in_place(body, explanation).operator.block)
    if (isinstance(body, cat.StrideMorphism) and body.name is not None
            and settings.reindexing_explanations):
        explanation = settings.reindexing_explanations.get(body.name.to_bodies())
        return (None if explanation is None
                else explain_reindexings.drawn_in_place(body, explanation))
    return None


def auxiliary_of_derived_variant(
    source: SentFigure, settings: DiagramSettings,
) -> wst.DiagramAuxiliary | None:
    '''The auxiliary information of a variant derived from `source` whose settings
    change the text of its inspection boxes, written for the morphism `source`
    exports so that it is keyed by the numbering of the source's message, as tsncd
    requires. The legend and the expansions are written under `settings` from the
    morphism the source's expansions were written from, the records of the blocks
    are read off the morphism exported, and the record of every explaining block
    takes the explanation the tables of `settings` give it. tsncd carries it across
    the functor as it carries the source's.'''
    auxiliary = advanced_display.auxiliary_for(
        source.morphism_expanded, settings.advanced_display,
        assigned_sizes_of(settings), settings.code_link_base, settings.tape,
        settings.expanded_parameters, settings.operator_references,
        settings.operator_roles)
    if auxiliary is None:
        return None
    blocks = {
        **auxiliary_information.block_information(
            source.morphism, settings.code_link_base),
        **explaining_block_records(source.morphism, settings)}
    return {**auxiliary, **({'blocks': blocks} if blocks else {})}


def variant_messages(
    variants: Sequence[PageVariant], settings: DiagramSettings,
) -> tuple[standalone_page.VariantMessage, ...]:
    '''`variants` as the page writer receives them. A variant carrying a term is
    presented, packaged and exported under its own settings or `settings`, the
    settings of the page. A derived variant whose settings change the text of its
    inspection boxes carries an auxiliary of its own, written for the morphism its
    source exports.'''
    by_identifier = {variant.identifier: variant for variant in variants}
    sent = {variant.identifier: sent_figure(
                variant.term, settings_of_page_variant(variant, by_identifier, settings))
            for variant in variants if variant.term is not None}
    return tuple(variant_message(variant, sent, by_identifier, settings)
                 for variant in variants)


def variant_message(
    variant: PageVariant, sent: Mapping[str, SentFigure],
    by_identifier: Mapping[str, PageVariant], settings: DiagramSettings,
) -> standalone_page.VariantMessage:
    group: wst.VariantGroupRecord = {
        'id': variant.group.identifier, 'title': variant.group.title}
    if variant.term is not None:
        return standalone_page.VariantMessage(
            identifier=variant.identifier, group=group, title=variant.title,
            detail=variant.detail, message=sent[variant.identifier].message())
    source = sent[variant.derived_from]  # type: ignore[index]
    source_settings = settings_of_page_variant(
        by_identifier[variant.derived_from], by_identifier, settings)  # type: ignore[index]
    auxiliary = (
        auxiliary_of_derived_variant(source, variant.settings)
        if variant.settings is not None
        and changes_the_inspection_text(variant.settings, source_settings) else None)
    return standalone_page.VariantMessage(
        identifier=variant.identifier, group=group, title=variant.title,
        detail=variant.detail, derived_from=variant.derived_from,
        functor=None if variant.functor is None else variant.functor.value,
        settings=None if variant.settings is None else page_settings(variant.settings),
        auxiliary=auxiliary)


def check_no_localisations(
    variants: Sequence[PageVariant], settings: DiagramSettings,
) -> None:
    localised = [variant.identifier for variant in variants
                 if (variant.settings or settings).localisations]
    if localised:
        raise LocalisedPageVariants(
            f'the variants {localised} carry localisations, and a page with '
            'variants carries none')


def save_page_folder_with_variants(
    variants: Sequence[PageVariant], slug: str, settings: DiagramSettings,
    initial: str,
) -> standalone_page.PageFolder:
    '''Write `variants` to `<page_directory>/<slug>/index.html` as one HTML file that
    opens with no server and no network and draws a selector between them, write a
    redirect to that folder as `<page_directory>/<slug>.html`, and return the paths
    written.'''
    check_no_localisations(variants, settings)
    return standalone_page.save_page_folder_with_variants(
        variant_messages(variants, settings), settings.page_directory, slug, initial,
        dist=tsncd_dist())


def first_variant(variants: Sequence[PageVariant]) -> str:
    if not variants:
        raise standalone_page.InconsistentVariants('a page with no variants')
    return variants[0].identifier


async def show_page_variants(
    variants: Sequence[PageVariant], caption: str | None = None, *,
    settings: DiagramSettings, slug: str, initial: str | None = None,
) -> None:
    '''Draw several variants of one model as one page that switches between them.

    `initial` names the variant the page opens on, the first where it is `None`.
    HTML writes every variant into `<page_directory>/<slug>/index.html`, writes
    `<page_directory>/<slug>.html` to redirect to that folder with the query and the
    hash of its address kept, and prints the path of the folder. LISTING prints
    the listing of every variant under its group and title, the listing of a
    derived variant being that of the Python statement of its functor applied to
    the term of its source. DUMP writes each variant as LISTING reads it to
    `<dump_directory>/<slug>-<identifier>.json`. INLINE and BROWSER draw the initial
    variant as `show_diagram` would. OFF prints the caption alone. The mode is read
    from `settings`, and `PYNCD_DIAGRAMS` replaces it as it does in `show_diagram`.
    '''
    initial = initial if initial is not None else first_variant(variants)
    check_page_variants(variants, initial)
    mode = mode_override() or settings.mode
    if caption:
        print(caption)
    if mode is DiagramMode.OFF:
        return
    by_identifier = {variant.identifier: variant for variant in variants}

    if mode is DiagramMode.HTML:
        written = save_page_folder_with_variants(variants, slug, settings, initial)
        print(f'(written to {written.folder.as_posix()}/)')
        return

    if mode in (DiagramMode.INLINE, DiagramMode.BROWSER):
        shown = by_identifier[initial]
        await show_diagram(
            term_of_page_variant(shown, by_identifier),
            settings=dataclasses.replace(
                settings_of_page_variant(shown, by_identifier, settings), mode=mode),
            slug=slug)
        return

    for variant in variants:
        variant_settings = settings_of_page_variant(variant, by_identifier, settings)
        term = present_each_side(
            term_of_page_variant(variant, by_identifier), variant_settings)
        term, variant_settings, _ = package_auxiliary(term, variant_settings)
        if mode is DiagramMode.DUMP:
            variant_settings = dataclasses.replace(
                variant_settings, dump_directory=settings.dump_directory)
            print(f'(dumped to '
                  f'{dump_term(term, f"{slug}-{variant.identifier}", variant_settings)})')
            continue
        print(f'{variant.group.title} / {variant.title}')
        print_listing(term, variant_settings)


def page_variant_legends(
    variants: Sequence[PageVariant], settings: DiagramSettings,
) -> Mapping[str, list[wst.AxisLegendRow]]:
    '''The legend rows of each variant, by identifier, as the page carries them: read
    off the term of the variant presented by `present_each_side` and converted as
    `package_auxiliary` converts it, under the settings the variant is drawn with,
    where `settings` are the settings of the page. The term of a derived variant is
    `apply_page_functor` of the term of its source. A variant drawn under
    `AdvancedDisplay.OFF` carries no legend and has no rows. The expansions of the
    inspection boxes are not computed, because the legend is read before them.'''
    check_page_variants(variants, first_variant(variants))
    by_identifier = {variant.identifier: variant for variant in variants}
    legends: dict[str, list[wst.AxisLegendRow]] = {}
    for variant in variants:
        variant_settings = settings_of_page_variant(variant, by_identifier, settings)
        if variant_settings.advanced_display is advanced_display.AdvancedDisplay.OFF:
            legends[variant.identifier] = []
            continue
        term = present_each_side(
            term_of_page_variant(variant, by_identifier), variant_settings)
        legends[variant.identifier] = auxiliary_information.legend_rows(
            morphism_with_operators_explained(term, variant_settings),
            assigned_sizes_of(variant_settings))
    return legends
