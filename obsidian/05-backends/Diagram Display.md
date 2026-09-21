---
tags: [layer/backends, tool]
code: display/, websocket_transfer/, data_transfer/
status: stable
---

# Diagram Display

## What it is

Two renderers, both for a human reader:

| | |
|---|---|
| `display/` | a 2D ASCII neural circuit diagram with ANSI colour, in the terminal |
| `websocket_transfer/` with [`tsncd`](https://github.com/mit-zardini-lab/tsncd) | the real diagrams, drawn in a browser |

Use [[Agent Display]] to inspect the algebra as text. Browser diagrams can also be
captured as images and viewed by an agent. In tsncd, `npm run capture -- --output
tmp/current.png` captures the full term held by the server through a connected page.
The capture uses an off-screen target and preserves the displayed diagram.

[[Advanced Display]] explains the legend of axes drawn beside a figure and the
inspection boxes that open over a block or an expandable operator on the page, which
`DiagramSettings.advanced_display` turns on.

[[Diagram Themes]] explains the dark-mode design and its shared rendering pipeline.
Python mode arguments default to `None`. `ColorMode.DARK` and `ColorMode.LIGHT` select
an explicit mode. The TypeScript default is dark, and capture backgrounds follow the
resolved theme through `background='auto'`.

## The ASCII renderer

```python
import display as dpl
dpl.print_category(morphism)
dpl.print_graph(hypergraph)
```

`Box.py` is a small layout algebra, holding `Horizontal`, `Vertical`, `TextBox`, `Padded`
and `Fill`, that everything else composes. `node_category.py` renders a `Broadcasted` and
`display_graph.py` renders a hypergraph. `Color.py` handles hex and HSV, which is how a
uid becomes a hue.

> [!warning] `display` imports nothing above it
> Every layer above the backends may import `display`, so an import in the other direction
> forms a cycle. It is a real constraint rather than a style preference, per
> [[Invariants]].

## The browser diagrams

`pyncd` builds the algebra and `tsncd` draws it. Everything they share crosses a WebSocket
as JSON, and [[Diagram Wire Format]] states the contract. That note is mirrored at
`tsncd/PROTOCOL.md` in the other repository, so edit both together.

A server sits in the middle because neither end has a stable lifetime: cells run and finish,
tabs open and reload. The server holds the most recent term, which is what makes a browser
refresh work. Start it as `python run_server.py` here or as `npm run server` in `tsncd`.
The two answer identically, and only one of them may hold port 8765.

```
Jupyter kernel  <-->  DataServer :8765  <-->  Browser
 (DataClient)         holds the latest      (DiagramClient)
```

`data_transfer/term_json.py` is the term encoder, holding `TermJSONConverter` and using
`TermDirectory` and `EnumDirectory` from [[Terms]].

### Getting a diagram back

| function | what it does |
|---|---|
| `send_morphism.send_term` | pushes to the open page, and nothing comes back |
| `capture.capture_morphism(m)` | the same render, with the image handed back and displayed inline, so it is saved with the notebook |
| `capture_morphism(m, disturb_display=False)` | renders off-screen and leaves the browser as it was |
| `headless.save_figures({...})` | drives its own browser, taking names rather than paths, in PDF by default |

A `tsncd` page has to be open for the first three, because the diagram's geometry is CSS
layout plus measured text, which takes a real browser engine. `jsdom` and its relatives
report every box as zero. `headless.py` brings its own browser and renders through the same
`termPass` the websocket path uses, so the two agree by construction.

All four take the display settings of the `settings` channel in [[Diagram Wire Format]]:
`darkMode`, `debugBorders` and `width`, which sets the proportions rather than the scale.
Beside those, `subBlocks` settles whether the body of an `ops.BlockOperator` is drawn as a
sub-diagram beside the main figure.

`subBlocks=False` is how a figure gets the high-level view alone, with each box's body
rendered as its own figure. A notebook asks for that with
`sub_blocks=SubBlocks.NO_BODIES`.

A notebook says which bodies its figures draw with
`remember_drawn_blocks.SubBlocks`, which `notebook_diagrams` and `sota_figures` both
re-export. The three cases are `EVERY_BODY`, `BODIES_NOT_YET_DRAWN` and `NO_BODIES`, and
`draws_any_body` turns the choice into the `subBlocks` setting while `tags_to_skip` turns
it into the `drawnBlockTags` setting.

`BODIES_NOT_YET_DRAWN` is the default and draws a body once per notebook kernel.
`notebooks/display/remember_drawn_blocks.py` holds the `BlockTag` of every body a figure
has delivered, `show_diagram` sends that set as the `drawnBlockTags` setting, and tsncd
leaves out a body whose tag is in it while still drawing the box in the main figure. A
notebook that boxes one attention core into five layers therefore gets the core drawn
beside the first figure and the box alone in the other four. tsncd does not descend into a
body it left out, so a `BlockOperator` nested inside one is not drawn either, and
`newly_drawn_tags` walks the term the same way when it decides what a figure delivered.
`notebook_diagrams.forget_drawn_blocks()` empties the record, and so does restarting the
kernel.

The record lives as long as the kernel, which outlives a notebook. A notebook run a second
time in the kernel that ran it before therefore starts with the record its first run
filled, and each of its figures draws the boxes and none of the bodies. That is what
`DeepSeekV41Flash.ipynb` hit: on the second run its figures came back with fewer
sub-diagrams than on the first, and on the third with fewer again as later cells were
re-run on their own. A notebook whose figures must come out the same on every run empties
the record in its setup cell, as `DeepSeekV41Flash.ipynb` now does, or asks for
`SubBlocks.EVERY_BODY`.

`SubBlocks.EVERY_BODY` sends `drawnBlockTags` empty, so tsncd draws the body of every
`BlockOperator` it builds a box for, and the body of every `BlockOperator` inside those,
however many figures were sent before. The figure records what it delivered, so a later
figure asked for with the default still leaves those bodies out.

The last cell of `DeepSeekV41Flash.ipynb` draws the whole model with `EVERY_BODY`, so that
one figure carries the body of every block down to the compressor, the indexer and the
attention core, where the figures above it carry a box apiece. It passes
`mode=DiagramMode.BROWSER` to `sota_figures.show`, which stands in for the notebook's
declared `INLINE` for that call alone, so the figure goes to the open page and the notebook
stores none of it.

`block_recycling` says whether a box's body is normalised before the figure is drawn.
The transport recycles what it is handed, which reaches the level it was handed and not
the body of any box, so a body written by hand is drawn as it was written where the
figure around it is drawn from the normal form. `BlockRecycling.RECYCLED` passes the term
through `algebra.broadcasted_recycle.recycle_broadcasted` before every other presentation
pass, which recycles the body of every box and of every box inside those, per
[[Hypergraph to Morphism]]. `AS_WRITTEN` is the default. Recycling keeps each block's
tag, so the record of the bodies already delivered still names the same bodies.
`DeepSeekV41Flash.ipynb` draws under `RECYCLED`, because its attention core is boxed by
`algebra.discovering_broadcasts.broadcast_block_over_axes`, which does not recycle the
body the way `construction_idioms.boxed` does.

`tapeLabels`, which is on by default, belongs to [[Para Category]]. It labels each tape with the slot it reaches. A tape belongs to a
`ParaWrap`, per [[Para Wrap]], which is a morphism with its grabs and drops written onto it,
drawn as a box with rows of anchors along its top and bottom, where a bare `Grab` or `Drop`
is the wrap over an identity. A grab and the drop that filled it are the same parameter and
are drawn in different rows with no line between them. Their labels identify the shared
slot. Resting the pointer on the strip between a taped array's first and last tape, from
the arrowheads down to the row they reach, paints a plate behind that strip in the slot's
colour and highlights its matching grabs and drops across the rendered diagram. The slot
name beside the arrowheads stands outside the plate, as the user ruled on 2026-09-16. A small padlock in the slot's colour stands beside each plate the pointer lights, open while the slot is only lit and closed while a click holds it locked. A click on a plate or on its padlock locks the slot, a second click releases it, and several slots can be locked at once, as the user asked on 2026-09-17 with the words "Slots should also be lockable". Drawing the figure again releases every lock, so a captured figure carries neither a plate nor a padlock. The label is the slot's name where the term carries one, which `para.new_slot`
numbers `s0`, `s1` and so on in creation order, and two hex digits of its UID where it does
not, hued off that UID. Send `False` for a figure with a single tape,
where there is no pairing to show. [[Backpropagation]] is what puts a tape in a diagram at
all. Since 2026-09-16 the slot name and the array's axis names are written on two lines at
the free end of the tape, an axis name wider than the room beside its tape is turned to
run down the tape, and a tape is as tall as its names need above a small base.
[[Para Wrap]] states the rules.

An axis answers the pointer as well, since 2026-09-16, where `DiagramSettings.axis_hover`
puts it. `AxisHover.LEGEND`, the default, lights every wire and every name of an axis
while the pointer rests on the axis's row of the legend, and the wires and names answer no
pointer themselves. `AxisHover.EVERYWHERE` lights the same from a wire or a name of the
axis, in a gap or on a tape, and shades the legend row with it. `AxisHover.OFF` draws no
halo. The axis is identified by its uid, so two axes that share a name do not light
together. records the two interactions and the
setting.

Headless capture needs `pip install -r requirements-headless.txt`,
`playwright install chromium`, and a built `tsncd` bundle from `npm run build`.
`headless.find_dist` reads `TSNCD_DIST` first, then the sibling checkouts
`tsncd/dist` beside the checkout. A bundle built from a source tree newer than `dist/` is
the usual cause of a `TermDirectory` miss.

**Prefer the open page for an inline capture, over a browser started here.** A capture
through the running server costs one round trip and needs nothing installed, and the page
laying the diagram out is the one already on screen. The headless renderer is the fallback
for when no page answers. `notebook_diagrams` tries them in that order and caches the
verdict, so a session with no live page waits out the capture timeout once rather than
once per diagram.

## Notebook diagram modes

A notebook declares a `DiagramSettings` at the top of its setup cell and passes it to every
`show_diagram` call:

| mode | cost | what it does |
|---|---|---|
| `INLINE` | 2 to 4 s each | captures the image back and embeds it in the cell |
| `BROWSER` | about 15 ms | pushes it to the open page and embeds nothing |
| `HTML` | under a second, no browser | writes the figure as one HTML file that opens with no server and no network |
| `DUMP` | fast | writes the term to JSON, for a term tsncd cannot draw |
| `LISTING` | tens of ms | prints the [[Agent Display]] listing and draws nothing |
| `OFF` | none | skips the diagram |

Nearly all of a diagram's cost is the browser laying it out, and building the expression
takes tens of milliseconds. Work in `BROWSER` and use `INLINE` for the run that gets
committed.

`HTML` writes `<page_directory>/<slug>.html`, which is `outputs/pages/` unless the settings
say otherwise. The file holds tsncd's bundle, KaTeX's fonts and the message `BROWSER` would
send, so the legend and the inspection boxes of [[Advanced Display]] answer the pointer in
it. `websocket_transfer/standalone_page.py` assembles the file as text, and
[[Diagram Wire Format]] states the embedded form under *A page that carries its own
message*. The mode needs a tsncd bundle built since 2026-09-16 and refuses an older one.
`DiagramSettings.localisations`, a tuple of `localise_descriptions.Localisation` whose
first entry holds the wordings used to build the figure, makes the file carry every
wording of its descriptions and a control to switch between them, per *A page that
carries its localisations* in the same note. Titles are never switched, because a title
sets the drawn size of its block. The sentences `cast_presentation.py` and
`explain_operators.py` write into a box are entries of
`notebooks/display/display_wording.json`, loaded by `display_wording.DisplayWording`, so
that a page switches them with the rest.

`DiagramSettings.title` names what the page shows. The heading of the open page and of an
`HTML` file then reads `tsncd - <title>`, and `tsncd` where the setting is `None`. A
captured image holds the figure alone, so an `INLINE` run does not show the title.

`DiagramSettings.tape` says how a `Para`'s grabs and drops are shown. `ABSORBED`, the
default since 2026-09-12, passes a term holding grabs or drops through `to_para_wrap` just
before it is drawn, so each grab sits on the operand port it feeds and each drop on the
result it saves, per [[Para Wrap]], and leaves a term holding none as it stands, so
`recycle=False` still draws a hand-built morphism unrecycled. `BOXED` draws the term as it
stands, with every `Grab` and `Drop` a box of its own.
`notebooks/display/tape_presentation.py` holds the enum and the conversion, and
`notebook_listings.print_listing` and `listing_without_legend` take the same `tape` argument,
so a notebook that sets `ABSORBED` once shows every listing and diagram in that form while
its algebra runs on the `Para` underneath.

A boxed block carries its tape on a wrap from the moment it is built, per
[[Para Block Operator]], so a box drawn under either presentation shows one tape per slot
it reads and writes. The setting does reach the body drawn beside the figure:
`tape_presentation.wrap_inside_boxes` applies the same rule to the block of every
`ops.BlockOperator`, because `to_para_wrap` rewrites siblings in one scope and never
enters an operator. Under `ABSORBED` the drop inside a boxed attention mode therefore sits
on the result it saves in the sub-diagram, while the box in the main figure carries the
same slot on its port.

`DiagramSettings.loop_initializers` says whether the initializer of a loop variable is
drawn. `HIDDEN`, the default since 2026-09-13, removes
the drop that starts each variable of a stream loop and the operations that computed only
its value just before the term is drawn, in whatever mode, and removes the block that
grouped them with the loop once the loop is all the block holds. `DRAWN` draws them.
`notebooks/display/loop_initializers.py` holds the enum and the removal.

A datatype carrying a quantisation is drawn as a label above the names of its array's
axes, in every gap that names the axes. The label of a quantisation carries the number of values a 32-bit word holds in front of the format,
as $2\mathtt{BF16}$, and a block-scaled format is named, as MXFP8, per
[[Quantization]]. An index carrying a quantisation keeps the arrow and the bound an
unquantised index draws on its wire, and its format, INT64 or INT32, stands below the
wires where the label of a real array stands, as the user ruled on 2026-09-20. tsncd draws it, and
[[Terms Mirrored in tsncd]] lists the TypeScript files that do.

`DiagramSettings.clean_quantisation_labels`, true by default since 2026-09-13, labels an
array only where its quantisation changes.
`notebooks/display/clean_quantisation_labels.py` keeps the label on a wire entering the
figure, on a wire a `TypeConvert` writes, on a wire an operation with no input writes, and
on a wire an operation writes in a quantisation differing from one of its inputs', and
removes it from every other wire and from the weaves and tape operations at its ends, just
before the term is drawn. The pass runs in pyncd because the wiring is explicit there, and
tsncd draws whatever labels remain, so the renderer carries no code for the mode. With the
setting false, every wire carrying a quantisation is labelled.

`DiagramSettings.casts` says how a conversion that changes a quantisation is drawn.
`CastPresentation.THIN`, the default since 2026-09-20, draws it as no glyph, on a box of
no width whose operand and result are wired straight to each other, and the rounding is
read from the label on the wire changing. The format the conversion wrote is drawn in
`thin_cast_label_color`, which is blue, and the gap holding it leaves out the names of
the axes, because the gap the operand came from names them and a conversion changes
none of them. `Anchor.names_itself_in_gap`, cleared on the result, is what leaves them
out, and an anchor that answers false still carries the label of its array. `DRAWN` draws
it as a chevron whose two halves are as tall as the quantisations it reads and writes.
The wire on each side of a conversion is labelled with the format that side carries under
either presentation, so a chevron states the rounding a second time, and a model given the
quantisations a released one runs in carries eighty of them. A notebook whose subject is
where each conversion is inserted sets `DRAWN`.

That blue format is also where an inspection box over a thin conversion is opened from.
A box of no width has a rectangle two pixels across, which no reader can rest a pointer
on, so `ThinTypeConvertBox.region_element` names the label instead, and
`inspectionBoxes.measure_regions` takes the rectangle of the text inside a label rather
than the rectangle of the box holding it, which is as wide as the gap. The user asked
for the hover on 2026-09-20. `cast_presentation.cast_explanation` writes what the box
shows: the quantisation the cast reads, the quantisation it writes, what the pair of
formats does to a value, and the block scale of the written format where it has one.
`package_auxiliary` adds that row to the table a figure explains its operators with, so
every interactive figure holding casts carries it, and a model holding a row of its own
for `Quantization.TypeConvert` keeps that row, as the quantised text-only
DeepSeek-V4.1-Flash does for a cast into a stored form of a cache.

`notebooks/display/cast_presentation.py` holds the enum and takes the name off every such
conversion just before the term is drawn, and tsncd draws an unnamed conversion thin, in
`src/display/Framework/quantization/quantisationLabels.ts`. The quantisations stay in the
operator's `source` and `target`, so a listing reads alike under either presentation and
the inspection box reads them from there.
`notebooks/sota/DeepSeekV41Flash.ipynb` draws the FP4 round trip both
ways, and [[Quantization]] states the pass that writes the casts.

`DiagramSettings.tape_naming` says how a tape slot is labelled, per [[Code Forms]]. A slot
inside a repeated block stands for a different member of the tape on every iteration.
`NAMED`, the default, draws each label as the term names it. `INDEXED` writes the index
of every loop that selects a member after the label, so a weight grabbed inside a
repeated layer reads `W_{G}[l]`, and `CODE_FORM` writes the member's code form with the
same indices in typewriter, `weight_G[layer]`. A loop variable is carried across its
loop and takes the indices of the loops outside it alone. `notebooks/display/tape_naming.py`
holds the enum and the rewrite, and `notebook_listings.print_listing` takes the same
`naming` argument.

A name's exponent is drawn above its body, after its subscript, in every mode, unless
the name's settings lower it into the subscript through `fd.ExponentPlacement.SUBSCRIPT`,
`DiagramSettings.axis_sizes` says where the size of an axis a configuration has sized is
drawn, per [[Code Forms]]. An expression is written with symbolic sizes and a
`term_utilities.generate_config.NumericConfig` binds them by name, so `config(model)`
carries an integer size on every axis the configuration named. tsncd labels the wire of
such an axis with that integer, through `StrideCategoryRenderer.AxisProcessor.size_text`,
which puts the number where the axis's name would otherwise go, so a sized model is drawn
in numbers. `WIRE_LABEL`, the default, leaves it so. `EXPONENT` writes each integer size
into the name it belongs to, through `write_axis_exponents.write_axis_size_exponents`, so
the residual width of DeepSeek-V4.1 reads `m^{5120}` and the name and the number are read
together. `SUBSCRIPT` writes the same value and lowers it into the subscript, so the same
axis reads `m_{5120}`, which is the form the V4.1 notebooks draw since 2026-09-16. The size is stated once: `size_text` prints the name of an axis whose name
carries an exponent rather than the integer, because the exponent already says what the
integer would. An axis whose size still holds a free symbol is left as it stands, and so
is every axis of an expression no configuration has sized.
`notebooks/display/axis_sizes.py` holds the enum and the rewrite.

`DiagramSettings.assigned_sizes` holds the integer a configuration assigned each named
symbol, keyed by the bodies of the symbol's name, which is what
`NumericConfig.assigned_integers_by_name()` returns. Under `EXPONENT` the assignments are
written onto the symbolic expression rather than onto one the configuration has already
sized, through `write_axis_exponents.write_assigned_size_exponents`, so every symbol of a
label keeps its letter and carries its own size. The query axis of DeepSeek-V4.1, sized
`|a||b|` with the encoder's entries left symbolic, is drawn `x`, and the `Natural` counting
over it reads `|a|^{2}|b|`. A notebook that passes no assignments draws a term the
configuration has sized, through `write_axis_size_exponents`.

A label holding more than one symbol is drawn by the rule in [[Compound Axis Labels]]:
each symbol keeps its letter and carries its own size as an exponent on that letter. The
guarded axis `w|x` sized at 128 reads `w^{128}|x`, through tsncd's
`display/Framework/advanced_axis_dynamics/guardedAxisLabels.ts`, and the sparse axis of the
router reads `|k|^{6} \text{ of } e^{384}`. The requester ruled the guarded form on
2026-09-15, and the general rule the
same day.

An agent executing a notebook in the background sets `PYNCD_DIAGRAMS` in the kernel's
environment, and `show_diagram` uses the mode it names in place of the declared one, so the
notebook is not edited. `notebooks/execute_notebook.py` starts the kernel that way, with
`listing`. The variable is read in `notebook_diagrams.py` alone. A notebook that draws
through `websocket_transfer` directly is not reached by the override.

## Known traps

> [!warning] "Term type not found in TermDirectory: X" means a stale `tsncd` bundle
> It is not a `pyncd` bug. A new operator needs a box on the TypeScript side, and until it
> has one, `show_diagram` degrades to a JSON dump in `dumps/`. Rebuild the bundle and point
> `TSNCD_DIST` at it.

> [!warning] A registered `Term` with no registered box draws as nothing
> `@fd.register_term` only makes an operator parse. The glyph is a separate registration,
> `bb.opsRegistry.registerClass(ops.X)`, keyed on the exact constructor with no fallback to
> the prototype, so a subclass of `Elementwise` needs its own line even though
> `ElementwiseBox` exists. Without one the operator falls through to the default
> `OperationBox`, which builds anchors and a core and then draws nothing, and the symptom is
> a gap in the wires rather than an error.
>
> The gap is invisible for an operator whose target passes straight through. It is visible
> for one that produces axes, because those wires enter no box and so begin in mid-air.
> `para`'s former `Broadcast` operator was in that state, and the stray `x` floating into
> the first operand of the `+` in the reverse of the expanded softmax was the symptom, per
>. The operator has since gone, because a repeat is a
> `View`, drawn by `ViewBox`. `Zero` is the same trap
> taken to its limit, being nullary, and it has a box. `ops.Maximum`, `ops.ReLU` and
> `ops.Dropout` were each here before it.

> [!info] A wire drawn across an operator says the operator is broadcast over that axis
> A degree axis is one the operation is broadcast over, and `BroadcastedBox` draws it as a
> single wire running from the column before the box to the column after it. An axis in a
> weave's target is read by the operator, so its wire ends at the left column and a fresh
> wire starts at the right column. A box that loosens two target anchors and links them
> with `OperationBox.pass_anchor_through` therefore draws a broadcast, whichever fact the
> link was added for. `TopKBox` and `SelectAtPositionsBox` each did that to a consumed
> axis until 2026-09-16. The
> calls that remain pass a datatype anchor through a `CovariantView`, a `ConcatenateAxes`,
> an `Einops` or a gather, where the wire carries the same datatype on both sides and no
> axis is involved.

> [!info] A contraction group of three or more operands is a chain of cups
> `EinopsBox.setup_cups` collects one list of anchors per contraction group of the
> signature, and `update` draws the group. One anchor is an axis the operator sums over
> inside a single operand, and it takes a dot. Two anchors are the Penrose evaluation
> and take the cup `draw_cup_between_anchors` draws. Three or more anchors are three or
> more operands reading one index, which `draw_shared_index` draws as a cup between each
> operand and the one after it, with a dot where two cups meet. Until 2026-09-17 the
> method drew nothing above two, so the Engram gate of
> `notebooks/sota/DeepSeekV41Flash/engram_modules.py`, whose einsum is
> `'x n m, n m, x n m -> x n'` over the stream, the learned weight and the key, drew
> three `m` wires ending in mid-air. `einops_simplification.einsum` builds such a group
> from any shapes, so the case is reachable from every model.

> [!info] A fixed shape gets a rectangle behind it when the box outgrows it
> An `OperationBox` is as tall as its columns and as wide as its rows, and a circle or a
> diamond has one dimension, so a box carrying more than a few wires is larger than the shape
> that names it. A wire then arrives at nothing, and a tape drawn onto the operation ends in
> mid-air beside the shape. `display/Framework/Operations/GlyphBox.ts` draws the shape in the
> largest square the box holds and fills the rest of the box behind it, as a rectangle with
> one corner bitten off. `NormalizeBox`, `TopKBox` and `InjectBox` extend it, and each gives a
> fill, a corner and a shape.
>
> The condition is geometric: a box the square fills gets no rectangle. The two boxes that had
> the rectangle before tested their operand and result counts instead, which is a different
> question and answered it wrongly in both directions.
>
> `LinearBox` and `TransposeBox` are not glyph boxes, because their glyph is the rectangle.
> They bite opposite corners of it, and `bitten_rectangle` is where every bite is cut.
>
> `GlyphBox.glyph_room` is the part of the box the shape and anything written with it
> occupy, and the rectangle is drawn where the box is larger than that room. It is the
> shape alone for every box that draws nothing else, and `BitwiseXorBox` returns the
> circle and the name under it, so a box that holds the pair exactly gets no rectangle.

> [!info] A symbol a reader may not know is drawn with its name under it
> `BitwiseXorBox` draws the exclusive or as the circle a `Normalize` takes, carrying an
> upright cross, with `XOR` written under it, per the user's ruling of 2026-09-19. The
> bold `\oplus` it drew before said which symbol the operation is and not what the
> operation does, and a reader who has not met the symbol could not read it.
>
> The name hangs below the circle, so the box reserves the circle, a gap and the name,
> and `vertical_alignment_shift` returns half that strip. `BroadcastedBox` sets the
> glyph against the mean of the anchors the operator reads and then applies the shift,
> so the circle rather than the middle of the box lands on the line the wires arrive at.

> [!info] An arithmetic operation on whole numbers stands in a rectangle
> An addition of two arrays of reals joins a residual stream to what a module computed
> from it, and `AdditionOpBox` writes a bold `+` where the two wires meet. An addition
> of whole numbers is one step of an integer computation, and the user ruled on
> 2026-09-19 that such a step is written in a rectangle, as the steps around it are.
> `reads_whole_numbers` is the test, and it reads the datatype of every input weave.
>
> The registry keys on the operator's class alone, so the choice is made by a
> `bb.opsRegistry.registerFunction` on `ops.AdditionOp` and another on `ops.Modulo`,
> each returning a `NamedRectangleBox` or the bold glyph. `NamedRectangleBox` is the
> white rectangle carrying a name that `GenericOperatorBox` draws, extracted so that
> both operators draw the same figure the generic operator does.
>
> `TextEstimator` gave `\bmod` the width of an unknown command, which is 0.75 em, and
> KaTeX sets it as three letters between two thick spaces, so the name ran outside the
> rectangle it was reserved room in. The estimator now carries a width for `\bmod`,
> `\oplus` and `\otimes`.

> [!info] A reindexing carried by the operator has to draw its own node
> The `BroadcastDisplayType.NODE` of `BroadcastedBox` reads `target.reindexings[0]`, so it
> is available only to a `Broadcasted` whose reindexing is where a reindexing normally goes.
> `para.data_structure.transpose.ReindexTranspose` carries one in the operator instead,
> because a `reindexings` entry always maps output indices back to input ones and a
> transpose runs the other way, so its box builds the figure itself, per
>.
>
> The flag to set is `reversed`, and it moves two things at once: which column is the
> codomain, and which side the pentagon's point is on. `DefaultStrideRendererSettings` sets
> it `true`, which puts a forward node's codomain on the left, because a reindexing reads
> against the data flow. A transpose reads with the flow, so drawing it unreversed puts the
> domain on the left and moves the point onto the axis the fibre is summed onto. The forward
> and backward convolution then come out as exact mirror images with no drawing code aware
> of it.
>
> The constructor of `StrideRenderer` ignores the settings passed to it. It builds
> `{...DefaultStrideRendererSettings, ...super.settings}`, and `settings` is an instance
> field, so `super.settings` is `undefined` on the prototype and the spread does nothing.
> That is what the `@ts-ignore` for ts(2855) on that line is for. Set the flag after
> `super()`.

> [!info] A datatype wire that turns carries its direction mark on its longer straight run
> `DatatypeAnchor.update` draws a triangle half way along every wire it paints, saying
> which way the value travels. Until 2026-09-18 it skipped the wire whose two anchors face
> different ways, which is the wire that turns and runs down the page, so such a wire
> carried no mark at all. The user asked for one while reading the Engram figures.
> `turning_wire_direction_mark` now returns the half-way point of the longer of the two
> straight runs and the angle that run travels at, and the triangle is placed there, so
> the mark sits on the straight stretch rather than on the turn. A run shorter than two
> triangles is still left unmarked, as a short straight wire is. No figure in the
> repository draws a turning datatype wire today, because a datatype anchor is horizontal
> only on a `ParaWrap` row and the grabbed arrays there are `Reals`, which draws no wire
> of its own, so the behaviour is held by the three tests in
> `test/elementwise_arrow_geometry.test.ts`.

> [!info] A contravariant expression is drawn as its body mirrored
> `ContravariantBox` in `CategoryRenderer.ts` builds the body's box and calls `mirror()`
> on it, a method on `DiagramElement`. Layout is flex, so `mirror` reverses the children
> of every horizontally laid-out element in the subtree, which reverses the composition
> order and puts each operation's codomain on the left, while each glyph and each label
> is still drawn upright inside its own rectangle. `AnchoredBox.mirror` re-swaps
> `left_anchors` and `right_anchors`, so the two names keep meaning the drawn sides and a
> composition of contravariants wires up. `Anchor.mirror` swaps `prior` and `further`, so
> wires and their direction triangles keep drawing left to right. Mirroring twice is the
> identity, so a `Contravariant` inside a mirrored region draws forwards.
>
> Two pieces of draw code state a side rather than an anchor, and each reads the
> element's `mirrored` flag: the pointed reindexing's `points_left`, which also places
> its stride and shift labels, and the einops cup, which swaps its arc endpoints to keep
> the bulge towards the glyph. A mirrored forward node is therefore the same figure a
> `ReindexTranspose` draws, so a reindexing under a `Contravariant` is displayed as its
> reverse. A negative transform scale was rejected on 2026-09-01, because it mirrors the
> glyphs and their names with the geometry.

> [!info] A `MultiCategory` is drawn as its rows stacked, a double dashed line between them
> `render_root` in `Multiline.ts` is the entry point a whole figure goes through, and it
> dispatches a `MultiCategory` to `MultiCategoryBox`: each row rendered by
> `multiline_render`, stacked in a `Vertical`, with a `DoubleDashedSeparator` between one
> row and the next. A covariant row reads left to right and a `Contravariant` row is its
> body mirrored, so a `backprop.Taped` draws as the forward pass over the backward pass,
> with each gradient leaving at the end where its value entered. The separator's height is
> the room the tapes need: a forward row's drops run down past its box and a backward
> row's grabs run up past its own, and both land in that band. No wire crosses a row
> boundary, because the rows of a `Taped` share their tape slots and nothing else.
> `para_wrap.to_para_wrap_rows` wraps each row for display, keeping its direction and the
> class of the whole.
>
> Every row wraps at `settings.width`, the contravariant ones included.
> `multiline_render` takes a `Contravariant` as well as a `cat.ProdCategory`, splits its
> body into rows, and hands the stack to `ContravariantBox` as the inner box it would
> otherwise have built with `display_category`. Mirroring the stack reverses each row
> and leaves the order of the rows alone, because `rh.Vertical` does not reverse its
> children under `mirror`, so a contravariant row starts at the top right and each row
> below it resumes at the right. records the
> change.

> [!warning] `Cannot read properties of undefined (reading 'anchors')` is a weave bug
> It is thrown from `link_weaves` in `BroadcastedCategoryRenderer`, and it means a
> reindexing's `mapping` indexed past the output weave's degree anchors, so the
> `Broadcasted`'s weaves disagree with its `degree()` about how many TILED positions there
> are. Fix the morphism rather than the renderer, per [[Weaves and Degree]]. It caught a bug
> in `expand_to_nodes`.

> [!warning] Only a subset of LaTeX survives in an operator name
> `\sqrt` and `\frac` either raise or break the captured diagram.

> [!warning] A bare `NotImplementedError` from a diagram cell is Playwright's event loop
> It is not the renderer. Playwright needs its own loop, and the Jupyter kernel's Selector
> loop is not it.

> [!warning] The first headless figure of a session can come back as the term the server holds
> The page the headless renderer serves out of `dist/` connects to the `DataServer` on port
> 8765 as any browser tab does, and the server sends the term it holds on connect, which is
> what makes a browser refresh work. A capture made while a server is running can therefore
> be replaced by the term that server holds, and the first figure of a session is where the
> two race. Render a warm-up figure and discard it, or stop the server. The first of three
> fixtures came back as the whole DeepSeek-V4.1 model, which the server had held since a
> browser session earlier that day, and the two after it were correct.

## Related notes

- [[Agent Display]] — what to use instead, as an agent
- [[Diagram Wire Format]] — the wire format, mirrored in `tsncd`
- [[Terms]] — `TermDirectory`, which the JSON encoding keys on
