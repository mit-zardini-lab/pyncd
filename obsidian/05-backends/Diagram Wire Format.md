---
tags: [layer/backends, reference]
code: websocket_transfer/websockets_transfer.py, websocket_transfer/standalone_page.py, websocket_transfer/localise_descriptions.py, utilities/wording_json.py
status: stable
---

# The Messaging Framework Between `pyncd` and `tsncd`

> **This document is mirrored.** An identical copy lives at `tsncd/PROTOCOL.md`, beside the
> TypeScript half of the implementation. The protocol belongs to neither repository, so both
> carry it. Edit the two together, as the two implementations are edited together. Only the
> links differ, each pointing at its own side.

`pyncd` builds the algebra and `tsncd` draws it. Neither performs the other's job, so
everything they share crosses a WebSocket as JSON. This document states the contract between
them.

These files implement the contract and have to change together:

| | |
|---|---|
| Python client and server | `websocket_transfer/websockets_transfer.py` |
| TypeScript message types | [`src/data_transfer/diagram_protocol.ts`](../../tsncd/src/data_transfer/diagram_protocol.ts) |
| TypeScript browser client | [`src/data_transfer/websockets_transfer.ts`](../../tsncd/src/data_transfer/websockets_transfer.ts) |
| TypeScript server | [`src/data_transfer/diagram_server.ts`](../../tsncd/src/data_transfer/diagram_server.ts) |

## Why a server sits in the middle

Having the notebook talk to the browser directly is not available. A Jupyter kernel cannot
accept a connection a browser can reach reliably, and neither end has a stable lifetime,
because cells run and finish and tabs open and reload. A third process therefore outlives
both.

```
  ┌──────────────────┐   dataUpdate    ┌────────────┐   dataUpdate    ┌─────────────┐
  │ Jupyter kernel   │ ──────────────▶ │  DataServer│ ──────────────▶ │  Browser    │
  │ (DataClient)     │                 │ :8765      │                 │(DiagramClient)
  │                  │ ◀────────────── │            │ ◀────────────── │             │
  └──────────────────┘   renderResult  └────────────┘   renderResult  └─────────────┘
                                        holds the
                                        latest term
```

The server holds the most recent term, which is what makes a browser refresh work. The page
reconnects, identifies itself, and is sent the current diagram with no involvement from the
notebook. The server is started once and left running.

Two implementations answer identically, and either may be the one that is up:

```bash
python run_server.py     # here
npm run server           # in tsncd, which is `node src/run_server.ts`
```

Only one of them may hold port 8765 at a time. The node server binds `127.0.0.1` and `::1`
separately, because `localhost` resolves to either and node's `listen` takes one address
where Python's `websockets.serve` takes every address the name resolves to. The choice is
otherwise a matter of which runtime is already installed. The node server relays large
frames far faster, as [Where the time in a capture goes](#where-the-time-in-a-capture-goes)
measures.

**The two kinds of client are asymmetric.** A `DataClient`, meaning a notebook, connects per
send and disconnects. A `DiagramClient`, meaning a browser tab, stays connected for as long
as the tab is open. Several diagram clients may be connected at once, and they all display
the same thing.

## The messages

Every message is a JSON object carrying a `msgType`. The server raises on an unrecognised
type, deliberately, so that a version skew between the two repositories fails at the first
message rather than dropping diagrams silently.

### `identify`, from a client to the server

The first message on every connection. Until it arrives the server has no way to route
information.

```jsonc
{
  "msgType": "identify",
  "clientType": "DataClient" | "DiagramClient",
  "clientVersion": "0.1.0",
  "clientID": "unique-client-id-1234"
}
```

It is answered with `{"msgType": "Connected"}`. When a `DiagramClient` identifies while the
server holds a term, that term is pushed immediately, which is the refresh path.

### `dataUpdate`, in either direction

A term to display. A notebook sends it, and the server relays it to every diagram client. The term is a morphism, or a `cat.DefinedExpression` holding two morphisms, which the client draws as its left-hand side, `:=` and its right-hand side in one row, each side wrapped at half of `width`.

```jsonc
{
  "msgType": "dataUpdate",
  "data": "{\"uid_repository\": …, \"data\": …}",   // note: a JSON *string*
  "settings": {
    "darkMode": true, "blockBackground": "subtle", "blockHoverIntensity": 0.12,
    "debugBorders": false, "coreDebug": false,
    "width": 750, "subBlocks": true, "drawnBlockTags": [],
    "tapeLabels": true, "legend": false, "inspectionBoxes": false,
    "title": "DeepSeekV4.1", "heading": "none"
  },
  "auxiliary": { "legend": [ … ], "blocks": { … }, "expansions": { … } }
}
```

`data` is doubly encoded, as a JSON string inside a JSON object, because
`TermJSONConverter.export_to_json` returns serialised text and it is passed through without
being re-parsed. The TypeScript side calls `JSON.parse` on it a second time.

`settings` is **partial by design**. The client merges whatever arrives over its own defaults
from
[`RenderHandlerSettings.ts`](../../tsncd/src/display/Render/RenderHandlerSettings.ts),
so an omitted key takes its default rather than whatever the previous send left behind. Every
send therefore fully determines the display. On the Python side,
`websocket_transfer/send_morphism.py`'s `display_settings` assembles the partial dictionary.

`darkMode` defaults to `true`. The dark theme uses a charcoal canvas with light
text, wires and glyph outlines. Enclosing regions have dotted outlines and
faint background tints. Operator surfaces are dark and drop shadows are disabled.
Colored strokes and labels retain their hue with enough lightness for the dark
canvas. `darkMode: false` restores the light theme's colors, fills and shadows.
Python theme arguments default to `None`, which omits `darkMode` and defers
to the renderer's default. `ColorMode.DARK` and `ColorMode.LIGHT` from
`websocket_transfer.websockets_transfer` select explicit modes. The Python
settings helper converts those enum members to `true` and `false` on the
wire. Existing boolean calls remain supported. Notebook
`DiagramSettings.dark_mode` accepts the same enum and defaults to `None`.

The theme adapts drawing and annotation attributes in the render backend.
Both themes use the same geometry and element update methods. Each render
target owns its theme, so a capture can use a different mode from the display.
An omitted `darkMode` takes the default on every message.

`blockBackground` controls dark-mode enclosure fills. Its levels are `none`,
`subtle`, `medium` and `strong`. The default is `subtle`, which blends 4.5% of
the block's color into the canvas color. `medium` uses 8% and `strong` uses 14%.
Light mode retains its existing enclosure fills.

`blockHoverIntensity` controls the tint applied to a highlighted block and its
associated BlockOperator. The default is `0.12`, and values from `0` to `1`
blend the surface toward the shared highlight color. Both elements use the
same resolved fill. These optional keys can be supplied in the Python
`RenderHandlerSettings` dictionary passed to `send_term` or `render_term`.

`width` is a layout option rather than a rendering one, and it is the number of pixels at
which a morphism wraps onto another line so that `F₀; F₁ = F`. It therefore controls the
figure's proportions and leaves its scale alone. A narrower setting gives more rows and a
taller figure, and a wider one gives fewer rows and a flatter figure. It rides the settings
channel because the settings channel is the one that already reaches the renderer on every
send.

Measured on the transformer, the effect is sharp:

| `width` | page | aspect |
|---|---|---|
| 750, the default | 9.4 × 6.6 in | 1.43 |
| 1000 | 12.0 × 5.8 in | 2.09 |
| 1400 | 16.2 × 4.9 in | 3.29 |
| 2000 and above | 17.3 × 4.3 in | 4.03, unwrapped, with no further effect |

The default suits a screen. A figure spanning a paper's text block usually needs 1000 to
1400.

`subBlocks`, which defaults to `true`, settles whether the body of a `BlockOperator` is drawn
as a sub-diagram beside the main figure. A figure export needing the high-level view
alone, with each box's body as its own figure, sends `false`.

`drawnBlockTags`, which defaults to `[]`, names the `BlockTag`s whose bodies an earlier
send already drew, each as the `uid._id` of its tag, which is the integer the JSON carries
on the tag's `uid`. The client leaves out a pending body whose tag is named and still draws
the box in the main figure, and it does not descend into a body it left out, so a
`BlockOperator` nested inside one is not drawn either. The sender keeps the record, because
the client wipes its render target at the start of every message and a capture draws into a
second target that never saw the first. `notebooks/display/remember_drawn_blocks.py` keeps
it for the life of a notebook kernel, and `notebook_diagrams.forget_drawn_blocks()` empties
it.

`tapeLabels`, which defaults to `true`, labels each tape with the slot it reaches, set at the
free end of the tape. A tape belongs to a `ParaWrap`, which is the display form that
`para.data_structure.ParaWrap.to_para_wrap` puts a `Para` into before sending, with each grab
or drop written onto the morphism it touches. A bare `Grab` or `Drop` is drawn as the wrap
over an identity. A grab and the drop that filled it are the same parameter, and they are
drawn in different rows with no line between them, so the label is the only thing that pairs
them. The label is the slot's name where the term carries one, which `para.new_slot` assigns
as `s0`, `s1` and so on in creation order, and two hex digits of its UID where it does not,
hued off that UID. Send `false` for a figure with one tape,
where there is no pairing to show, or where the labels crowd a narrow row. They are drawn
outside the layout, as the tapes themselves are, so they fall over the row above or below.

A taped array answers the pointer over the strip between its first and last tape, from the
arrowheads down to the row the tapes reach, and not over the slot name beside them. Resting
the pointer there paints a plate behind that strip in the slot's colour and lights the halos
of every grab and drop of the same slot. Beside each of the plates so lit an open padlock
is drawn, in the slot's own colour, outside the edge of the plate the arrowheads are on
and at the end of that edge nearest the slot name. Clicking a plate locks the slot: its
plates and halos stay lit once the pointer has left, and the open padlock beside each of
them closes. Clicking a second time releases the slot, and several slots may be locked
at once. A lock is a highlight source of its own, so the pointer's hover comes and goes
beneath it, and a slot locked in a figure is lit in the diagram inside every inspection
box that figure opens. The click stops at the plate, so it opens, locks and closes no
inspection box. Drawing the figure again releases every lock, so a capture of a figure
nothing has pointed at carries neither a plate nor a padlock. The user asked for the
lock on 2026-09-17, and no setting on the wire turns it on.

`axisHover` (default `legend`) says where an axis answers the pointer. Under `legend`,
resting the pointer on the axis's row of the legend halos every wire of that axis in the
figure and glows every name of it, and the wires and names answer no pointer. Under
`everywhere`, resting it on a wire, or on the name of the axis in a gap or on a tape, lights
the same and shades the legend row. Under `off`, no halo is drawn and nothing answers. The
axis is identified by its uid, so two axes that happen to share a name do not light
together. A halo is a glow in the theme's highlight colour, a blue in each theme,
whatever the colour of the wire or the name surrounded by it, and [[Diagram Themes]]
gives the two colours. `DiagramSettings.axis_hover` carries the choice from a notebook
as a `wst.AxisHover`.

`axisLabelFontSize` (default `0.8`) is the size, in em, of the label an axis carries on
its wire. The layout measures a label at the size it draws it, so a larger label is given
the room it needs. The label was drawn at 0.65 em until 2026-09-16, when the user asked
for it to be about a quarter larger and for the size to be a setting.
`DiagramSettings.axis_label_font_size` carries it from a notebook.

`form` (default `all-broadcasted`) says which of three forms the figure is drawn in.
Under `all-broadcasted` every array is drawn as one wire for each of its axes, and every
operator is drawn with its glyph, its contraction cups, its reindexing node and the wires
of its broadcast axes routed around the glyph, as every figure was drawn before the
setting existed. Under `arrows-and-broadcasted` each array passing from one operator to
another is drawn as one wire. The wire is an arrow stroked heavier than an axis wire,
with a triangle on it pointing from the operator that writes the array to the operator
that reads it, and it is labelled in two lines. The shape of the array stands above the
arrow, its axes in square brackets separated by commas, in the order of the axis wires
from top to bottom. The datatype stands below the arrow. It is the array's quantisation
where the array carries one and `\mathbb{R}` otherwise, so every arrow states the format
of its array. A matrix of reals over `q` and `d` reads `[q, d]` above its arrow and
`\mathbb{R}` below it, and a scalar of reals writes no shape and `\mathbb{R}` below its
arrow. An array of naturals along `x` bounded by `\bar{v}` reads `[x]` over `\bar{v}`,
because a wire of that datatype is labelled with its bound. An array held in FP32 reads
`\mathtt{FP32}` below its arrow. The user asked for the commas and for the datatype below
the arrow on 2026-09-26. The client writes the datatype carried by the array and infers
no quantisation from a neighbouring cast.

Every operator is drawn under `arrows-and-broadcasted` as it is under `all-broadcasted`,
and it stands on a plate, a rounded rectangle with a drop shadow drawn under it in the
theme's surface tint. Resting the pointer anywhere on the plate opens the operator's
inspection box, as resting it on the box of the boxed form does. A conversion drawn
thin, which is a `TypeConvert` carrying no name, stands on an empty plate. The datatype
below the arrow of its result is written in the thin cast's blue and opens the same box.
At the left edge of the plate the arrow of each operand opens into one wire for each of
its axes, and at the right edge the wires of each result close into the result's arrow.
Each axis wire is named where it enters and where it leaves the plate, as it is named in
a gap under `all-broadcasted`. The plate carries a name in small type above the glyph.
The name says what the operator does where the operator's class registers one. It is
`Linear` for a `Linear`, `Cache` for a cache, and for an `Einops` the name read off its
signature, as the boxed form below names its box. Every other plate carries its
operator's own name. A name already written by the glyph is left off. An elementwise map
with one operand and one result gets no plate. Its arrow runs straight through, and its
name stands over the arrow between two small heads. A `BlockOperator`, whose glyph is a
titled box already, gets no plate.

Under `arrows-and-boxes` the arrays are the same arrows, and every operator is a box with
one arrow entering per operand and one leaving per result. The face of the box shows
what the operator is. Most operators, and an elementwise map, show their name in the
middle of the box. An `Einops` shows a name read off its signature, which is `Matmul`
for two operands with a contracted group, `Sum` for one operand with a contracted group,
`Product` for two or more operands with none and `Contraction` otherwise. A softmax and
the normalisations show their glyph drawn small inside a labelled box, a `Linear` shows
its named rectangle, a `View` shows the name of its reindexing, and a `BlockOperator`
shows its titled box. Each box carries above it the name carried by the plate of
`arrows-and-broadcasted`, in the same small type, and leaves the name off where the face
already shows it. A `Linear` therefore reads `Linear` above the rectangle naming its
weight, and a cache reads `Cache` above the box naming the cache. `Matmul` is written
once, inside its box. Nothing of the broadcasting is drawn in this form.

In both arrow forms an operator carrying a grab or a drop keeps the taped array in its
column. The tape comes down from its free end, turns a corner and runs level into the
array's arrow, or leaves the arrow level and turns down.
A `Contravariant` is drawn as its body mirrored, and the data of the backward pass it holds travels from right to left. In the backward pass every triangle on an arrow or on a datatype wire, the two heads of an elementwise map in every form and the head a dangling natural wire ends in point left. A grab's tape in the backward pass comes down from above and turns left into its arrow, and a drop's tape leaves its arrow to the left and turns down, with the slot name right of the arrowhead. The user asked for the arrows and the tapes of a reversed category to be drawn so on 2026-09-27.
The arrow's label stands on that
level stretch as it stands on any other arrow, so no label runs down a tape. The tape of
a bare grab carries no label, because the gap after the grab labels the arrow continuing
the tape. Only the wires between operators and the marks at each operator differ between
the three forms. A reader meets the boxed form first. The all-broadcasted form is the
full form, which shows what each operator does with each axis. The client draws the
all-broadcasted form for a message with no `form` key, so a figure sent without the key
is drawn as it was before the setting existed. `DiagramSettings.form` carries the choice
from a notebook as a `wst.DiagramForm`.

`controls` (default `hidden`) says whether the page draws one row of controls under its
heading and outside the diagram container. The row holds the selector of variants on a
page carrying several, and then the buttons that switch the form and the theme of the
figure.
[A page that switches its form and its theme](#a-page-that-switches-its-form-and-its-theme)
describes the buttons. Under `hidden` the whole row is hidden, the selector with it. A
notebook sends `shown` unless `DiagramSettings.controls` says `HIDDEN`, and a captured
image holds no button either way.

`legend` (default `false`) draws a table of the term's axes beside the figure.
Each row carries the axis, the integer its size comes to and the code name the
axis was written with. The rows themselves arrive in the `auxiliary` field, and
the table is appended inside the diagram container, so an image cut from that
container holds it. tsncd labels a row with the function that labels the wire of the
axis, which it finds in the term by the row's `uids`, so the legend and the wire read the
same, `m_{5120}` and `|k|_{6} \text{ of } e_{384}`, and the `latex` the sender wrote is
drawn only where no axis of the row is found. A row is linked to the wires of its axes as
`axisHover` says. Resting the pointer on the row halos those wires, and under `everywhere`
resting it on one of them, or on the axis's name, shades the row. A click on a row locks
the halo of its axes on, and a second click on the row releases it.

`inspectionBoxes` (default `false`) lets a block, or an operator the sender
writes an expansion for, open a box when the pointer rests on it. The box shows
the title, the formula, the description and the links the sender supplied, and under
them the block's body or the operator's expansion drawn as a diagram of its own. A
block whose aesthetics say `BlockDrawing.BODY_IN_PLACE` is drawn in the figure as its
body alone, and its box shows the text and no diagram, because the body is already in
the figure. The
blocks and operators drawn inside a box open boxes of their own, whether or not
the box is locked, so a reader opens one operator inside another, and the block
a box was opened from is highlighted while the box is open. Clicking locks a
box open. Several boxes may be locked at once, one per block or operator, and
each holds the boxes opened inside it.
Clicking the page outside every box closes them all, as does the escape key.
A box is a core width of 1000 pixels with a padding of 14 either side of it, so
a box is 1030 pixels wide. A box taller than the window scrolls, and the scrollbar
a browser draws inside such a box takes room from its content, so a box showing one
is laid out that much wider again, 1045 pixels where the scrollbar takes fifteen.
The text of a box occupies the core width either way, and the scrollbar stands
beside it. The diagram inside a box is wrapped so that the drawing and the ink
that overhangs it together occupy the core width. A window with no room for 1030
pixels holds a box of the room it has, less an eight-pixel margin either side.
What each box shows arrives in the `auxiliary` field.

`title` (no default) names what the page shows. The name of the tab reads `tsncd - <title>`,
so a message sent with `"title": "DeepSeekV4.1"` names the tab `tsncd - DeepSeekV4.1`, and
a message that sends no title returns it to `tsncd`, because the settings of each message
are merged over the defaults. The user asked for the setting on 2026-09-16.

`heading` (default `none`) says whether the same text is written as a heading over the
figure. Under `none` the page holds the figure alone, so it stands as a page of a site
that writes its own heading above it. Under `title` the heading reads what the tab reads.
Only the display target writes the tab and the heading. An off-screen capture and the
diagram inside an inspection box draw with the same settings and leave both as they were.
A captured image holds the diagram alone, so neither appears in one.
`display_settings(title=..., heading=...)` sends both, and a notebook sets them as
`DiagramSettings.title` and `DiagramSettings.heading`. The user asked on 2026-09-21 for a
page with no heading, so that it stands as a subpage of a site, and made it the default.

### A page that carries its own message

A `dataUpdate` may be written into the page in place of being sent to it. The page then
holds a `script` element of type `application/json` with the id `tsncd-embedded-message`,
whose text is the message as the server would relay it, with `data`, `settings` and
`auxiliary`. tsncd's entry point looks for the element once its registries are
established. Where it finds one, it draws the message through the `termPass` the socket
uses and opens no websocket, so a server on port 8765 cannot replace the figure. Where it
finds none, it connects as before and draws the figure carried by the build.
[The figure a page boots with](#the-figure-a-page-boots-with) states which figure that is.
`src/data_transfer/embedded_message.ts` reads the element.

`websocket_transfer/standalone_page.py` writes such a page as one HTML file. It takes
tsncd's built `index.html`, removes the element that loads the bundle from `assets/`,
and writes the message and the bundle into two `script` elements at the end of the body.
The `<` of every `<!--`, `<script` and `</script` in the bundle is written as the
JavaScript escape `\x3C`, and every `<` of the message as the JSON escape `\u003c`, which is
what the HTML standard recommends for text inside a `script` element. The file opens from
`file://` with no server and no network, and its legend and inspection boxes answer the
pointer, because the page runs the same bundle on the same message. A 2.6 MiB file holds
the whole DeepSeek-V4.1-Flash model, of which the bundle is 0.95 MiB. A notebook writes
one with `DiagramMode.HTML`, under `DiagramSettings.page_directory`, named by the `slug` of
the call. `websocket_transfer/validate_standalone_page.py` checks the assembly without a
browser. The user asked for the file on 2026-09-16.

The page is painted in the canvas colour of its theme by its own stylesheet, before
the bundle runs, and shows a turning ring in the element `page-loading` until its figure
is drawn. Once the message is read, and before its term is built, tsncd's
`src/display/loadingScreen.ts` repaints the page in the theme of the message, so a light
figure arrives on a light page and no white page stands where a dark figure is about to.
The first draw removes the ring, and a page whose figure cannot be drawn writes the reason
where the ring stood. A page with no message of its own shows the ring while it fetches
its boot figure.

A script at the top of the head of the page chooses the theme painted before the bundle
runs, by the rule followed by the bundle. The query parameter `darkMode` decides first,
then the theme requested by the system through `prefers-color-scheme`, and the page is
dark where the browser supports no such media query. `standalone_page.py` still writes
the element `<meta name="tsncd-dark-mode">` at the start of the head of a page whose
message sets `darkMode`, with `true` or `false` as its content. Since the user's ruling
of 2026-09-27 neither the script nor the bundle reads that element, or
`tsncd-dark-mode` in `localStorage`. A light page is marked `data-tsncd-theme="light"`
on its root and painted light from its first frame, so no dark frame shows before the
message is read. The user reported the dark frame on 2026-09-26.

### A page that carries its localisations

A page written by `standalone_page.py` may hold a third `script` element of type
`application/json`, with the id `tsncd-localisations`, between the message and the
bundle. Its text is an `EmbeddedLocalisations`. `default` names the wording of the
export, and `localisations` holds every wording by name, in the order the page lists
them. A wording holds only the descriptions changed by it, keyed as the `auxiliary` keys
them: `blocks` by the uid of a block's tag, and `expansions` by the importer number of
the operator, each expansion with its own `description` and with the changed
descriptions of its nested `auxiliary`. The wording named by `default` changes none.

`websocket_transfer/localise_descriptions.py` assembles the element from the `auxiliary`
and from one `Localisation` per wording. A `Localisation` is a resolved table of
wordings under the names of a wording file, and `Localisation.from_wording_file` reads
one off a JSON file in the form stated by `utilities/wording_json.py`: one entry per
name, a shared sentence as one entry that the others name as `$NAME`, a part with
`ref` and `fills` where a sentence is filled with a size, and a template's own fields
in braces. Three files hold every sentence a box of the quantised text-only page
shows. `notebooks/sota/DeepSeekV41Flash/block_titles_and_descriptions.json` holds the
titles, the descriptions, the roles of the weights and of the elementwise maps, and
the rows of the selections, the views and the cache round trips of the model.
`algebra/registries/expansion_wording.json` holds the descriptions of the standard
expansions, the sentences of a linear map and the sentences of a rotary table.
`notebooks/display/display_wording.json` holds the sentences of a cast between two
quantisations and the sentence over an elementwise map. Each file is loaded by the
dataclass beside it, `BlockTitlesAndDescriptions`, `ExpansionWording` and
`DisplayWording`, one typed field per entry, and `notebook_diagrams.save_page` joins
the two package files into the first localisation through `with_package_wordings`. A
second file holds the entries it changes, from any of the three, and resolves against
the model's file and `notebook_diagrams.PACKAGE_WORDING_FILES`, so a changed sentence
reaches every description naming it. A description composed from several sentences is
joined in the code from separate entries rather than written as one template, because
a template with two adjacent fields, one of which may be empty, cannot be read back.
A description on the wire is read back as the entries that composed it, longest first,
with the values filled into a template by a module held as fills, and is written again
from the other table with the same values. A run that no entry holds, such as the role
of a weight written in `operator_explanations.py`, is kept as exported, and the entries
around it are still switched. A wording whose template has other fields than the
exported one is refused with `LocalisationFieldsDiffer`, and a reference to no entry or
a cycle of references is refused when the file is read.

Only a description is switched. A block title sets the width and the height of its block,
so a page whose titles changed under the reader would be laid out again, and the user
asked on 2026-09-20 that switching never changes the spacing. A formula is written from
the axes of its morphism. `src/data_transfer/embedded_localisations.ts` reads the
element, and `src/advanced_display/localisationSelector.ts` draws one button per wording
under the page heading, outside the diagram container so that a capture holds no button,
when the element holds two or more wordings. Choosing one writes the descriptions of that
wording onto the `DiagramAuxiliary` object rendered by the page, after writing every
exported description back, and `inspectionBoxes.refill_open_boxes` rewrites the text of
every open box in place. No figure is drawn again, so the prerendered contents of the
boxes, every lock and every open box are kept. The chosen wording is remembered under the
`localStorage` key `tsncd-localisation`, and `window.tsncd.localise(name)` switches it
from a driving browser. On the quantised text-only page the element weighs 139 KB for
380 changed block descriptions, the figure's size and markup are identical before and
after a switch, and 73 of the 153 distinct descriptions are prose held outside the text
module, which a switch leaves as exported. [[Open Gaps]] lists that prose.

A notebook writes the element by setting `DiagramSettings.localisations`, a tuple of
`Localisation` whose first entry is read from the file the figure was built with and
whose others are read from second files against it and the package files. On the
quantised text-only page every description reads back as entries of the three files,
and a second file changing four entries, two of them of the display package, changed
456 boxes. `utilities/validate_wording_json.py` checks the reading and the resolving of
a file, `websocket_transfer/validate_localise_descriptions.py` checks the reading back
and the writing, `validate_standalone_page.py` checks the element, and tsncd's
`test/advanced_display.test.ts` checks the writing onto the object graph.

### A page that carries several variants

A page may carry several variants of one model in place of one message. The variants
may be the model at the quantisations of its released checkpoint beside the same model
in the reals, or the model decoding a token with no cache beside the model decoding it
from a cache. Such a page holds one `script` element of type `application/json` with the
id `tsncd-variants`, and no `tsncd-embedded-message` element. Its text is an
`EmbeddedVariants`:

```json
{
  "version": 1,
  "settings": {"form": "all-broadcasted", "darkMode": false, "width": 900},
  "initial": "decode-quantised",
  "groups": [{"id": "decode", "title": "Decode"}, {"id": "cached", "title": "Cached"}],
  "variants": [
    {"id": "decode-quantised", "group": "decode", "title": "Quantised",
     "detail": "FP8 weights and BF16 activations, as the released checkpoint runs",
     "message": 17},
    {"id": "decode-unquantised", "group": "decode", "title": "Unquantised",
     "detail": "The same model in the reals",
     "derivedFrom": "decode-quantised", "functor": "dequantise"},
    {"id": "cached-unquantised", "group": "cached", "title": "Unquantised",
     "detail": "The model reading its keys and values from a cache", "message": 23}
  ],
  "value_repository": [[0, "msgType"], [0, "dataUpdate"]]
}
```

`value_repository` is one repository in the form stated under
[COMPRESSED exports share JSON values by content](#compressed-exports-share-json-values-by-content).
A variant drawn from a term of its own names by `message` the root of its `dataUpdate`
in that repository. The root decodes to `{msgType, data, settings, auxiliary}`. Its
`data` is the term in the `uid_references` form written as a JSON object, where a
relayed message writes it as text. Every message of the page is compressed into the one
repository, so a record held by two variants, such as an axis, a weight or a box whose
body is drawn by both, is written once. The page decodes only the records reached by the
variant on display, and keeps them, so a second variant reuses the records shared with
the first. The `expansion` of an operator's record may likewise be the term document
itself rather than its text.

A variant derived in the browser names by `derivedFrom` a variant carrying a message,
and by `functor` the functor applied by tsncd to the term of that variant. Its
`settings`, where present, are merged over the settings of that variant. Its
`auxiliary`, where present, is a root replacing the auxiliary information derived by the
functor. `groups` lists the groups in the order of the selector, each variant names its
group by `group`, and `detail` is the line written under the title of a variant in the
selector. The query parameter `displayMode` applies to the settings of every variant, as
it applies to a page's own message. An element holding no variant is read as no
element. An element whose facts do not hold, such as a derivation from a variant
carrying no message, is refused, and the reason is written where the ring stood.

`settings` repeats the settings of the initial variant uncompressed, with `form` and
`darkMode` written first and stated even where they take tsncd's defaults. The build
plugin of the lab website, `_plugins/diagrams.rb`, parses the element as JSON. It reads
the groups, the variants and the initial variant, and it reads from `settings` the form
and the theme of the initial variant. It builds a page of the site for every variant, so
a link naming a variant absent from the page has no page. Before 2026-09-27 the plugin
read the two keys from the text of the file with regular expressions.
`standalone_page.py` therefore still writes `form` and `darkMode` first, with a space
after each colon and comma, and writes the repository with none. Every `<` of the
element is written as `\u003c`. The head states the theme of the initial
variant as it states the theme of a page with one message, in an element no longer read
by tsncd. The page opens in the form and the theme named by its address, and in the
all-broadcasted form and the system's theme where the address names none, so the lab
website names both in the address of its frame.

One functor is defined, `dequantise`, in three steps set out by the user on 2026-09-27.
The first step replaces every `Quantified` datatype, on every wire and every weight, by
the datatype wrapped inside it, so the `BlockScale` carried by the wrapper goes with it.
The second step turns into the identity on its operand every `TypeConvert` that then
reads and writes one datatype, inside the body of every box and of every `ParaWrap` as
well. A conversion that still converts stays. The third step removes the identities from
the leaves upwards. A composition drops each identity and becomes the identity when every
member is one, a product of identities is the identity, and a block or a box whose body
is the identity is the identity, whatever its tag, repetition, title or colour. A figure
drawn with inspection boxes wraps every cast in a `BlockOperator` whose block is drawn
`BODY_IN_PLACE`, and that box goes with its cast by the last rule. A `ParaWrap` that
grabs or drops acts on the tape. A figure drawing the tape on the ports of an operation
holds a cast as the body of a `ParaWrap` where the operand of the cast is grabbed or its
result is dropped, and the functor leaves that wrap holding the identity. The identities
are those of the category of arrays, so a reindexing, which is a morphism of the
category of axes, is left as it stands. tsncd's
`src/quantization/algebra/strip_quantisations.ts` states the functor in tsncd, and
`quantization/algebra/strip_quantisations.py` states it in Python, with the removal of
identities in `algebra/remove_identities.py`. [[Stripping Quantisations]] states its
rules.

The quantisation pass gives a box a tag of its own for every quantised body inside it,
so the blocks made equal again by the functor carry several tags. tsncd keys a block's
highlight, its inspection box and its sub-diagram by its tag. After the functor, tsncd's
`src/data_structure_processing/share_block_tags.ts` walks the fields depth first and
gives every block the tag of the first block met before it with an equal body,
repetition, aesthetics and display order, so each body is drawn once. The quantised
DeepSeek-V4.1-Flash carries 1209 tags, and its dequantised form carries 211. A derived
variant's operations keep the importer numbers of its source's operations, so an
`auxiliary` given for a derived variant is keyed by the numbering of the source's
message. tsncd carries an `auxiliary` given for a derived variant across the functor as
it carries the source's. The records of the operations removed by the functor are
dropped, and every expansion is marked with the functor. The settings of a derived
variant may therefore change the text of its inspection boxes as well as its display
settings.

`notebook_diagrams.show_page_variants` writes an `auxiliary` for a derived variant whose
settings differ from those of its source in the roles of the operators, the
explanations of the operators or of the reindexings, the references of the operators,
the base of the code links or the parameters drawn by an expansion. The `auxiliary` is
written for the morphism exported by the source's message. Every block explaining an
operator or a reindexing keeps its tag and takes the text given by the tables of the
derived variant. A derived variant whose settings differ in none of these carries none.
The auxiliary of the unquantised variant of GLM-5.3, whose weights take roles naming no
quantisation, adds 52 records to the repository. The auxiliary information derived by
the functor keeps the legend and the records of the blocks and the operations still held
by the derived term, and drops the rest, so a removed cast takes its record with it.
Each operator record kept is marked with the functor. tsncd applies the functor to the
term of an expansion and to its auxiliary information before drawing it in an inspection
box, so an operator opened in the derived figure shows its expansion in the reals.
tsncd's `src/advanced_display/derivedFigures.ts` applies the functor to a figure.

tsncd draws the initial variant, and draws a selector listing every group with its
variants as the first group of the row of controls under the heading. The buttons of
the form and the theme follow the selector in the row. The row wraps where the window is
narrower than the row, and the row is hidden, the selector with it, where `controls` is
`hidden`. The lab website embeds a page with `controls=hidden` and draws a toolbar of its
own, whose selector of variants is filled from the page's `tsncd-state` messages. Before
2026-09-27 the selector stood above the row and was drawn whatever `controls` said. A
page carrying one variant draws it and no selector, and offers no variant for choice.

The selector follows the lab website's `diagram-viewer.html`. Its summary names the
group in small capitals and the title of the variant on display, beside a triangle that
turns while the panel is open. The panel lists each group under a small uppercase label
and every variant as its title with its `detail` beneath. The variant on display is
marked with the accent colour and a bar on its left edge. The selector is drawn in the
theme of each draw. It closes on the escape key, which returns the focus to its summary,
on a click outside it, and when the window loses the focus. A click inside it stops
there, so it closes no inspection box. Each option is a link to the page's address
naming its variant and the form and the theme on display, written again after every
draw, so a click with a modifier opens the variant in a page of its own in the form and
the theme seen by the reader.

A variant is chosen from the selector, from the query parameter `variant` of the page's
address, from a `{type: 'tsncd-display', variant}` message posted to the window, or by
`window.tsncd.variant(id)`. The call resolves once the variant is drawn and rejects an
id absent from the page's variants. The address and the message are checked strictly,
per
[A page that reads its address strictly and reports its state to a host](#a-page-that-reads-its-address-strictly-and-reports-its-state-to-a-host).
`window.tsncd.variants()` lists the variants as `{id, group, title}`,
`window.tsncd.currentVariant()` names the variant drawn, and
`window.tsncd.variantTimings()` lists the time taken by each step of building a variant,
in milliseconds, as `{variant, step, milliseconds}` with the step `decode`, `import` or
`functor`. A variant is drawn in the form and the theme of the figure on display, so a
reader's choice of either holds from one variant to the next. A request naming a form or
a theme with the variant has the variant drawn in them, in one draw. A variant asked for
while another is being prepared replaces it, and the variant asked for last is drawn. A
form or a theme named with a replaced request is kept for that draw. A variant is built
the first time it is asked for and kept, so a return to it draws the kept term.

The element `page-loading` stands over the page, in its canvas colour, while a variant
is prepared, with a line of text under the ring. The line reads
`Loading <group>, <title>…` while the records of a variant are decoded and its term
imported, `Applying Dequantization Functor...` while the functor runs, and
`Drawing <group>, <title>…` before the variant is drawn. The element is removed once the
variant is drawn. Each line is painted before the work named by it starts. In a headless
Chromium the functor took 76 to 80 milliseconds on GLM-5.3 and 335 to 374 milliseconds
on DeepSeek-V4.1-Flash, and drawing a kept variant again took 0.4 and 3.0 seconds.
tsncd's `src/data_transfer/embedded_variants.ts` reads the element,
`src/advanced_display/variantFigures.ts` builds each variant, and
`src/advanced_display/variantSelector.ts` draws the selector and holds the switch.

`websocket_transfer/standalone_page.py` writes the page with `save_page_with_variants`,
and a notebook writes it with `notebook_diagrams.show_page_variants` under
`DiagramMode.HTML`, as [[Diagram Display]] states. The notebook writes it through
`save_page_folder_with_variants`, as `index.html` in a folder named by the page, with a
redirect beside the folder that keeps the query naming the variant. A page with variants
carries no localisations, and `show_page_variants` refuses variants carrying them with
`LocalisedPageVariants`. tsncd's `test/variant_pages.test.ts` and
`test/strip_quantisations.test.ts` check the reader, the switch and the functor. The user
asked for the variants on 2026-09-27, for the model pages of the lab website, and
[[Website Notebooks]] lists the pages.

### A page that switches its form and its theme

A page holds one term and draws it in any of the three forms and either theme without
loading anything again. A switch redraws the term held in memory by the page through the
same `termPass`, with the one setting changed. It repaints the page in the theme's
canvas colour, keeps the heading and the localisation selector, and drops the pooled
inspection content of the previous draw. The legend and the inspection boxes are drawn
again from the same auxiliary information, because a block is keyed by its tag and an
operator by its number, and no form changes either key. The switch reaches the page
three ways.

- The row of controls under the heading, drawn where the message says
  `controls: shown`. The row holds the selector of variants on a page carrying several,
  one group naming the three forms and one naming the two themes, with the current
  choice marked.
- The address of the page. The query parameters `form`, `darkMode` (`true` or `false`)
  and `controls` (`shown` or `hidden`) set the form, the theme and the controls of the
  page's own figure, as `displayMode` sets its mode, so a host page holding the figure
  in an iframe sets the form and the theme in the iframe's address. The page refuses an
  address with any other value and draws nothing.
- A driving script. `window.tsncd.display({variant, form, darkMode})` switches any of
  the three and returns a promise settled once the figure is drawn. The promise rejects,
  with nothing changed, for a refused choice. A host page posts
  `{type: 'tsncd-display', variant, form, darkMode}` to the iframe's window with
  `postMessage`. The page answers by the same switch and then with a `tsncd-state`
  message, or with a `tsncd-refused` message.

The address alone decides the form and the theme of the page's own figure when the page
opens, as the user ruled on 2026-09-27. Where the address names no `form`, the figure
opens in the all-broadcasted form. Where it names no `darkMode`, the figure opens in the
theme requested by the system through `prefers-color-scheme`, whatever the form and the
theme of the message. The rule holds for a page with variants, a page with one message
and the boot figure of the relay page. While no theme is picked, the page draws its
figure again in the system's theme each time that theme changes, through a `change`
listener on the media query. An address naming `darkMode`, a click on a button of the
theme, and a host's message or a call to `window.tsncd.display` naming `darkMode` each
pick a theme. A figure drawn by the relay or by `window.tsncd.render` with settings of
its own ends the following as well. Nothing is kept in `localStorage`. Before the ruling
the page remembered the chosen form and theme under `tsncd-form` and `tsncd-dark-mode`,
and a remembered choice, then the settings of the message, applied where the address
named none.

After every switch of the variant, the form or the theme, the page writes the three on
display into its address, so an address copied from the page names all three. A page
left unswitched keeps its opening address, so it follows the system's theme again when
it is opened again. A message from the relay is drawn with its own settings, and the
switch then applies to it as to any held figure. A switch asked for before the page
holds a figure, as a host posting on the load of its iframe asks for one, is stored and
applied to the page's own message above the query parameters. The relay page switches
the last term received the same way, and a page inside an inspection box draws no
controls. `src/advanced_display/displaySelector.ts` draws the row, holds the switch and
follows the system's theme. The user asked for the switch on 2026-09-26.

### A page that reads its address strictly and reports its state to a host

A page reads five query parameters of its address and no others.

| parameter | the values it takes |
|---|---|
| `variant` | the id of a variant offered by the page, and none on a page holding one figure |
| `form` | `arrows-and-boxes`, `arrows-and-broadcasted`, `all-broadcasted` |
| `darkMode` | `true`, `false` |
| `controls` | `shown`, `hidden` |
| `displayMode` | `slow`, `fast` |

A page offers its variants for choice where it carries two or more. A page holding one
message, a page carrying one variant and the relay page offer none. The page checks its
address once its variants are read, before anything is drawn and before a socket is
opened. The page refuses an address that names another parameter, sets a parameter
twice, or sets a parameter to a value outside its row, whatever the rest of the address
says. A value is compared as it is written, so `darkMode=True` and `displayMode=FAST`
are refused. For a refused address the page draws no figure, opens no socket, builds no
controls and sets no `window.tsncd`. It writes, where the ring stood, that its address
was refused, and a sentence naming the parameter, the value and the values accepted,
such as
`The parameter "form" is set to "bad". It takes "arrows-and-boxes", "arrows-and-broadcasted" or "all-broadcasted".`
Before 2026-09-27 a value outside its setting was dropped and the page drew its figure
with the rest of the address. A link carrying a parameter added by another site for its
own counting, such as `utm_source` or `fbclid`, is refused too. Where the address names
no form or theme, the page opens in the all-broadcasted form and the system's theme,
per [A page that switches its form and its theme](#a-page-that-switches-its-form-and-its-theme).

After every switch of the variant, the form or the theme, the page writes the three on
display into its own address with `history.replaceState`, so no entry is added to the
history. The query then reads `variant`, where the page offers variants, `form` and
`darkMode`, followed by the `controls` and the `displayMode` already carried by the
address, and the hash is kept. An address copied from the page after a switch therefore
opens the same figure for any reader, whatever the theme of that reader's system. A page
left unswitched keeps its opening address. While the page follows the system's theme, an
address written by the page is written again when that theme changes. The links of the
selector's options are written after every draw, each naming its variant with the form
and the theme on display. A frame whose origin is opaque may throw on `replaceState`, and
the page then keeps its opening address. Headless Chromium rewrites the address of a
page opened from `file://` and of a page served over HTTP in a frame sandboxed without
`allow-same-origin`.

A page held in a frame of another page posts messages to the parent frame with
`postMessage` and the target origin `*`. A page in a sandboxed frame does not know the
origin of its host, and the messages carry nothing private. A page opened on its own
posts nothing. The state message reads:

```jsonc
{"type": "tsncd-state", "variant": "cached-quantised", "form": "arrows-and-boxes",
 "darkMode": false,
 "variants": [{"id": "decode-quantised", "group": "decode", "title": "Quantised",
               "detail": "The whole model with BF16 weights and activations…"}, …],
 "groups": [{"id": "decode", "title": "Decode"}, {"id": "cached", "title": "Cached"}]}
```

`variant` is `null`, and `variants` and `groups` are empty, on a page that offers no
variants. `detail` is the empty string for a variant that carries none, and a group
holding no variant is left out. The page posts the state after every draw, including a
draw that follows the system's theme. It posts one before its first draw where it
carries its own message or variants, naming the variant about to be drawn, so a host can
build its selector while the first variant is prepared. It also posts one in answer to
every accepted `tsncd-display` message, once the switch has settled, so a switch that
draws is answered twice with the same state.

A host switches the page by posting `{type: 'tsncd-display', variant, form, darkMode}`
to the frame's window, with any of the three fields. The fields are checked as the
address is, with typed values: `variant` a string naming a variant offered by the page,
`form` one of the three forms, and `darkMode` the boolean `true` or `false`. A field
holding `undefined` is left out, and a message holding no field changes nothing and is
answered with the state. A message naming a variant with a form or a theme has the
variant drawn in them, in one draw, and a message naming `darkMode` ends the following
of the system's theme. A message holding another field, or a field holding another
value, changes nothing, and the page answers:

```json
{"type": "tsncd-refused", "parameter": "form", "value": "bad",
 "accepted": ["arrows-and-boxes", "arrows-and-broadcasted", "all-broadcasted"]}
```

For a parameter the page does not read, `accepted` lists the parameters read by the
page, and for a `variant` on a page offering none it is empty. A page whose address is
refused posts the same message once, naming the parameter of the address.
`window.tsncd.display` takes the same fields and rejects with the sentence written by the
page for a refused address.

tsncd's `src/advanced_display/pageChoices.ts` checks the address and the choices and
writes the address, and `src/advanced_display/hostMessages.ts` posts and answers the
messages. tsncd's `test/page_choices.test.ts` and `test/host_messages.test.ts` hold the
tests. The user asked on 2026-09-27 for links that follow the reader and for an invalid
link to draw nothing, so that the lab website can hold the page in a frame and supply
its own toolbar.

### The figure a page boots with

A page that carries no message of its own draws the figure carried by the build, and the
first `dataUpdate` to arrive replaces that figure. The figure is
`json_files/deepseek_v41_flash_text_only_quantised.json`, one `dataUpdate` holding the
three fields of a relayed message. The `DiagramMode.BROWSER` cell of
`notebooks/sota/DeepSeekV41Flash.ipynb` builds the message, which is the
quantised text-only DeepSeek-V4.1-Flash drawn without the bodies of its blocks, with the
legend of its axes and an inspection box over every block and every operator. The file
runs to 12.5 MiB, of which the term is 6.6 MiB and the auxiliary information 5.9 MiB.

The page fetches the file rather than holding it in the bundle, because a browser parses
the whole bundle before the renderer runs. `src/data_transfer/boot_message.ts` holds the
path and fetches the message. `webpack.config.js` writes the file from tsncd's `public/`
folder into `dist/` at that path, and the dev server serves it from the compilation.

The relay keeps whatever term it holds. tsncd's entry point opens the socket first and
claims the display only after it has fetched the message and built the term, and it tests
the claim in the same synchronous run as the draw. A term held by the relay and a term
drawn by `window.tsncd.render` therefore each keep the screen, and the boot figure is
dropped. Where the relay holds nothing, and where no relay answers at all, nothing else
claims the display and the boot figure is drawn.

The message carries its own `settings` and `auxiliary`, and it is drawn through the same
`termPass` as a relayed message, so its legend, its inspection boxes and its axis haloes
answer the pointer as a sent figure's do. It carries no localisations, because the
notebook that built it declares none, so the page draws no wording buttons over it.

### The `auxiliary` field

`tsncd` does no algebra, so every fact the legend and the boxes show is computed
in `pyncd` and sent beside the term.
`websocket_transfer/auxiliary_information.py`
assembles it and
[`src/advanced_display/`](../../tsncd/src/advanced_display/AuxiliaryInformation.ts) reads
it. The field is optional on `dataUpdate` and on `renderRequest`, and a message
without it draws as it did before the field existed. Every part of it is
optional in turn.

```jsonc
{
  "legend": [
    {"latex": "m", "text": "m", "size": 64,
     "codeName": "hidden", "sizeCodeName": "hidden_size",
     "uids": [1670598927]}
  ],
  "blocks": {
    "10175062": {
      "title": "\\text{Norm block}",
      "formula": null,
      "description": "A linear map followed by an RMSNorm over the hidden axis.",
      "references": [
        {"label": "inference/model.py L281-L293", "url": "https://huggingface.co/…",
         "path": "inference/model.py", "line": 281, "endLine": 293,
         "icon": "huggingface"}
      ]
    }
  },
  "expansions": {
    "2": {
      "operator": "Normalize",
      "latex": "RMSNorm",
      "formula": "\\mathrm{RMSNorm}_{m}(x) = …",
      "description": "Each value scaled by the inverse square root of …",
      "expansion": "{\"uid_repository\": …, \"data\": …}",
      "auxiliary": { … },
      "references": [ … ]
    }
  }
}
```

`legend` is sorted by the sender, and the client draws the rows in the order
they arrive. `size` is the integer the axis comes to, and is null where the term
is symbolic and nothing sized it. `codeName` is the code form the axis name
carries, and `sizeCodeName` the code form of its size. `uids` lists the uid of
every axis of the term the row stands for, which is the integer the JSON carries
on the axis's `uid`, and is what links the row to the wires of the figure. A
sender from before the field existed leaves it out, and the row then answers no
pointer.

`blocks` is keyed by the uid of each block's tag, written as a decimal string,
which is the integer the JSON carries on the tag's `uid`. A box is opened from
the operator glyph of a `BlockOperator`, and the block it opens is the one that
operator holds. A reference with a null `url` is written as plain text, with its
path and its line beside its label. `icon` names an icon tsncd draws before the link.
The sender writes `huggingface` for a link whose host is `huggingface.co`, from the
table `auxiliary_information.REFERENCE_ICONS`, and `null` for every other link, and
tsncd holds the drawing of each icon under the same name and tests no url itself.
`formula` is LaTeX drawn under the title, and is what a block drawn in place explains
its operator with.

`expansions` is keyed by the number the client gives each `Broadcasted` as it
builds the term. `TermJSONConverter.to_term` walks the document depth first, in
the order the fields were written, and counts a `Broadcasted` when it enters the
record, before converting that record's fields. A reference into the
`uid_repository` is descended into the first time it is met and is the already
built term on every later one, so nothing inside it is counted twice.
`pyncd`'s
`data_transfer/broadcast_occurrences.py`
reproduces that walk over the Python term, and
[`test/broadcast_occurrences.test.ts`](../../tsncd/test/broadcast_occurrences.test.ts)
checks that every key lands on a node whose operator is the class the sender
named.

`expansion` is a full term payload, doubly encoded exactly as a message's `data`
is, and `auxiliary` is the auxiliary information of that expanded morphism. The
expansion numbers its own broadcasts from zero, so an operator standing inside
one is opened the way an operator in the main figure is. `references` are the places
in a codebase the operator stands for, written as a block's references are and listed
under the description. A notebook gives them by operator class, as
`DiagramSettings.operator_references`.

### `dataRequest`, from a client to the server

Asks for the currently held term. It is answered with a `dataUpdate`, or with
`{"msgType": "No Data Available"}`.

### `renderRequest`, from a notebook through the server to the browser

A `dataUpdate` whose sender requires the picture back.

```jsonc
{
  "msgType": "renderRequest",
  "requestId": "9f2c…",           // uuid4().hex
  "data": "…",                     // as dataUpdate
  "settings": { … },               // as dataUpdate
  "auxiliary": { … },              // as dataUpdate
  "capture": {
    "format": "png",              // or "svg"
    "scale": 2,                    // device pixel ratio; png only
    "padding": 16,
    "background": "auto"          // theme canvas; null for transparent
  },
  "disturbDisplay": true
}
```

Under `disturbDisplay`, which is the default and what an absent flag means, the diagram is
drawn on screen and the image is cut from it, so a capture and a plain display are the same
render with different follow-through. The settings travel with it, so a disturbing capture
applies its own `debugBorders` to the visible diagram as well.

Under `disturbDisplay: false` the browser renders into a second, off-screen target and leaves
the display alone. The server also declines to store the term in that case, because
overwriting it would leave the display intact only until the next reload, which is a
disturbance with a delay on it. [Two render targets](#two-render-targets) covers the second
target.

`capture` is partial on the same terms as `settings`, defaulting from
[`capture.ts`](../../tsncd/src/data_transfer/capture.ts) and assembled on the Python
side by `capture_options` in `websocket_transfer/capture.py`. `padding` is measured outward
from the diagram's content box, which is not the container's own box, because the overlay
overhangs it. [Framing](#framing) below covers the difference.

Only `png` and `svg` cross the wire. PDF exists on the headless path alone, since it comes
from the browser's print pipeline rather than from anything the page can serialise itself.

An omitted `capture.background` or the value `"auto"` uses the rendered
target's canvas color. A CSS color such as `"#ffffff"` overrides the canvas.
`null` exports a transparent canvas. The choice applies to the entire image,
including padding. Glyph surfaces retain their theme colors. Explicit
backgrounds keep their meaning for existing clients. Older browser builds
require an explicit color because they do not resolve `"auto"`.

### `renderResult`, from the browser through the server to the notebook

```jsonc
{
  "msgType": "renderResult",
  "requestId": "9f2c…",
  "mime": "image/png",
  "payload": "iVBORw0KGgo…",      // base64 for png, markup for svg
  "encoding": "base64",            // or "utf-8"
  "width": 812, "height": 460      // CSS px, before scale
}
```

When the render failed:

```jsonc
{ "msgType": "renderResult", "requestId": "9f2c…", "error": "…" }
```

A failure comes back as a message rather than as a dropped connection because a notebook cell
is blocked on the reply. An exception that never arrives presents as a timeout with no cause
attached. `result_to_bytes` is where an error result becomes a `CaptureError`.

## How a capture is correlated

The `npm run capture -- --output tmp/current.png` command identifies as a
`DataClient`, fetches the held term with `dataRequest`, then sends a
`renderRequest` with `disturbDisplay: false`. The command preserves the held
settings and applies any `--dark-mode` or `--width` override only to the
preview. `--watch 2000` repeats the sequence after each capture completes and
a 2000 ms delay. The command works with either relay implementation and adds
no message types.

The browser processes incoming messages in order. A capture completes before
the next message can rebuild a render target.

The reply travels over a different connection from the request, so it cannot be identified as
the next message, which is why there is a `requestId`.

1. The notebook sends `renderRequest` and keeps its connection open.
2. The server records the `requestId` against the requesting socket, then relays to every
   diagram client so that they all stay on the same term.
3. The notebook reads past the server's acknowledgement until a `renderResult` carrying its
   own `requestId` arrives, under a total timeout.
4. The server pops the `requestId` and forwards the first result. A later reply, from another
   tab rendering the same request, finds nothing pending and is dropped.
5. When the requester disconnects partway through a capture, its pending entry is discarded,
   so no image is pushed at a closed socket.

Three failures are handled explicitly, because each would otherwise present as an unexplained
hang:

| Situation | Result |
|---|---|
| No diagram client connected | An immediate `renderResult` carrying an error |
| The browser never answers | A client-side timeout, `DEFAULT_CAPTURE_TIMEOUT`, at 60 s |
| Several tabs answer | The first wins and the rest are dropped |

## Frame size

`websockets` rejects a frame over 1 MiB by default and closes the connection rather than
reporting the problem. `ws` does the same over its `maxPayload`. A captured PNG passes 1 MiB
comfortably, so every end raises the ceiling to `MAX_MESSAGE_BYTES`, at 64 MiB. Three calls
take it: `websockets.serve` and `websockets.connect` in Python, and `WebSocketServer` in
`diagram_server.ts`. Changing the ceiling on one side alone reintroduces the failure.

## Where the time in a capture goes

The timings below were measured on the transformer figure, which holds 2219 elements at
902 x 632 CSS px. A headless Chromium captured it as a PNG at `scale` 2. The whole round
trip takes about 2.3 seconds.

| stage | ms |
|---|---|
| Building the `DiagramElement` tree and laying it out | 63 |
| Waiting on `document.fonts.ready` and one settled frame | 33 |
| `html-to-image` serialising the DOM into a 14.8 MB SVG string | 2098 |
| Decoding that string to an image and drawing it onto a canvas | 16 |
| Encoding the canvas as a PNG | 127 |
| Relaying 373 KiB of base64 inside JSON through the server | 21 |
| `json.loads` and `base64.b64decode` in the notebook | 0.7 |

Serialisation is 87% of the round trip. The format table below gives the reason. Reaching
the diagram through a `foreignObject` means reproducing the diagram from inline styles.
`html-to-image` writes each element's full computed style, some 6.9 KB across 2219
elements. The PNG that comes out is 279 KiB, so 98% of the string is built and discarded.

The socket carries 1% of the capture. Its cost still differs sharply by implementation,
measured as the round trip of a 373 KiB frame across a loopback echo:

| relay | throughput |
|---|---|
| Python `websockets`, in `run_server.py` | 34 MiB/s |
| node `ws`, in `npm run server` | 447 MiB/s |
| A raw loopback TCP socket, for reference | 1500-2500 MiB/s |

Python's `websockets` is a factor of 50 off a raw socket even with its masking extension
built. On a PNG the gap is 21 ms against 2 ms out of a 2.3 s round trip. On the 14.8 MB SVG
the gap is 1.0 s against 0.07 s. The serialisation has already cost 2.1 s before the
transfer begins. The socket cost is a further reason to prefer `pdf` for vector output.

Anything that would make a capture appreciably faster has to replace `html-to-image`.
Playwright's screenshot of the same figure takes 242 ms. The nine-fold speed-up comes from
Chromium painting a layout it already holds instead of rebuilding one from inline styles.
The headless path already takes Playwright's screenshot, and `notebook_diagrams`
`DisplayFormat` makes it the default for an `INLINE` diagram. Inside a live browser the
same saving would come from writing an SVG document out of the geometry the renderer has
already computed.

## Framing

The diagram overhangs its own container.
[`HTMLDrawHandler`](../../tsncd/src/display/HTMLRender/HTMLDrawHandler.ts) places each
SVG layer at `(-BUFFER, -BUFFER)` relative to `#diagram` and sizes it past the far edge, so an
image cut to `getBoundingClientRect()` loses the overlay on every side. `contentBox` in
[`capture.ts`](../../tsncd/src/data_transfer/capture.ts) therefore measures the union
of the container and all its descendants rather than assuming a number, so the framing follows
a change to `BUFFER`.

A zero-area element is skipped in that union. Anchors, wire stubs and spacers are structural,
several sit at the origin, and including them would drag the box out to nothing.

The headless path carries a further constraint. A capture box routinely starts at negative
page coordinates, because the content already overhangs and the requested padding usually
exceeds the page's own, and Playwright clamps a clip to the page without reporting it, which
trims the margin. `HeadlessRenderer.isolated`, in `websocket_transfer/headless.py`, therefore
strips the page to the diagram alone and moves it to the origin before screenshotting or
printing, which makes the box valid by construction. Everything is reverted afterwards, so one
batch can mix formats.

## Two render targets

An undisturbing capture needs somewhere else to draw, so the entry point builds a second
container with its own render handlers. The same handlers pointed elsewhere will not serve,
because they hold per-container state, meaning measured rectangles and pending block
references, so the second target is a second set.

The off-screen container is laid out rather than hidden. `display: none` measures zero, and
since every box position comes from `getBoundingClientRect`, the diagram would come out
collapsed onto the origin. `visibility: hidden` lays out correctly and captures blank, because
the clone inherits it. The container is therefore parked outside the viewport, to the left,
since overflow in the negative direction creates no scrollbar.

It is parked with `transform` rather than with `left`, and the distinction carries the
behaviour. `html-to-image` seeds its clone from the computed style through `cssText`, and that
text carries the logical shorthand `inset-inline` after `left`. Assigning `style.left` on the
clone updates `left` in place, so the later shorthand still wins, the clone stays parked
off-frame, and the capture comes back blank at any offset. `transform` has no competing
shorthand, and the capture overwrites it outright.

One further thing had to be isolated. An SVG `url(#…)` reference resolves document-wide, so
the drop-shadow filter, which used a single hardcoded id for every shadow in the document, let
the two targets' definitions answer for each other. It had no effect while one diagram owned
the page, and with two targets it made an off-screen render perturb the visible diagram's
shadows. Each SVG layer now mints its own id.

## Fonts, and why the stylesheet is bundled

A capture serialises the DOM into an SVG `foreignObject`, so the fonts have to be inlined as
data URIs. Inlining requires reading `cssRules`, which a browser disallows on a cross-origin
stylesheet. KaTeX loaded from a CDN would therefore capture in a fallback face, and because
the wires are drawn from measured text boxes, a fallback face moves the geometry as well as
the glyphs. KaTeX's stylesheet is therefore bundled from `node_modules` by
[`HTMLAnnotationHandler.ts`](../../tsncd/src/display/HTMLRender/HTMLAnnotationHandler.ts)
and served same-origin.

Since 2026-09-16 the fonts are written into the bundle as well. tsncd's webpack
configuration turns each woff2 file into a data URI, so the bundle requests no file once it
has loaded, and `dist/` holds `index.html` and the bundle alone. KaTeX's stylesheet lists
each face as woff2, woff and ttf in that order, and a browser takes the first format it
reads. Every browser that runs the bundle reads woff2, so the other two are written as empty
data URIs. Writing the fonts in added about 340 KiB to the bundle, which `npm run build`
reports as 958 KiB. A capture of the same figure is byte for byte the same under the two
builds.
[A page that carries its own message](#a-page-that-carries-its-own-message) depends on the
fonts being in the bundle, because a page opened from `file://` cannot reach a file the
bundle names by an absolute path.

For the same reason every capture waits on `document.fonts.ready` and one full frame before
measuring anything.

## The headless path, which uses none of the above

`websocket_transfer/headless.py` drives its own browser and does not connect to the server.
Rebuilding a figure should not depend on a server being up, or on which tab happened to be
focused. It serves the built `dist/` of tsncd on a loopback port, because `file://` will not
do, since webpack builds with `publicPath: '/'`, and it reaches the renderer through a hook
the entry point installs on `window`:

```ts
window.tsncd = {
  render(payload, settings): Promise<{width, height}>,  // the same termPass the socket uses
  capture(options): Promise<CaptureResult>,             // in-page serialiser; needed for svg
  captureBackground(background): string | null,         // resolves auto from the diagram theme
  bounds(padding): {x, y, width, height},               // page coords, for a screenshot clip
  settled(): Promise<void>,                             // fonts loaded, layout stable
}
```

Both paths render through the same `termPass`, so they agree by construction rather than by
discipline. There are three output formats, and the choice matters:

| | Produced by | Typical size | Use when |
|---|---|---|---|
| `png` | a Playwright screenshot | about 190 KB | The default. It is a real browser paint, so nothing is lost in serialisation |
| `pdf` | Chromium's print pipeline | about 140 KB | Figures for a paper. True vector, with real embedded text |
| `svg` | `window.tsncd.capture` | about **15 MB** | Only when something downstream demands SVG |

The SVG figure is measured rather than mistyped. Serialising into a `foreignObject` means
reproducing the diagram from inline styles, and `html-to-image` writes the full computed
style, some 6.9 KB, onto each of about 2000 elements. Fonts account for 650 KB of the file,
and the other 95% is CSS with no bearing on the drawing. Prefer `pdf` for vector output. A
notebook storing one such SVG output grows to 15 MB against 256 KB for the PNG, a factor of
59, so SVG returns poorly even though Jupyter renders it perfectly well.

### What is inside the PDF

One page, sized exactly to the diagram, with all three KaTeX faces embedded and subsetted, and
the labels as real selectable text. A subscript arrives as a separate glyph, so `L_q` extracts
as `L q`, which affects extraction and leaves the appearance alone.

It is not entirely vector. `feDropShadow` has no PDF equivalent, so Chromium rasterises every
shadowed element, at 24 images making up 36% of the file, at roughly 192 DPI. Wires, fills and
text stay vector. That resolution cannot be raised, because the PDF output is byte-identical
at `scale` 1, 2 and 4, because the print pipeline is not given the device scale factor. Dropping the
shadows would make the file fully vector and fully editable.

### Saving a set

`save_figures`, in `websocket_transfer/headless.py`, takes names rather than paths and writes
them into one directory, which is `./outputs` by default, and which for a notebook is beside
the notebook. It uses one browser session:

```python
await save_figures({'attention': attention, 'convolution': conv}, width=1400)
# -> ./outputs/attention.pdf, ./outputs/convolution.pdf
```

The default format is PDF, since a paper needs vector output. A name may carry its own extension, which
overrides the format for that one figure, and it may include subdirectories. The options apply
to the whole set. For control per figure, drive `HeadlessRenderer.save` directly.

## Changing the protocol

Add a message type in every implementation the table at the top names, and document it in
**both copies of this file**. Both servers refuse a type they do not recognise, so a
half-applied change fails at the first message rather than rendering nothing quietly.
Python's `match` raises and closes the connection with no reason attached. `diagram_server.ts`
closes the connection with the offending message as the close reason. The failure surfaces
either way once both repositories are updated and the server has been restarted, because a
long-running server keeps executing the code it started with.

## See also

- [[Diagram Display]] — what the diagram itself shows
- [[Agent Display]] — the SSA listing to read instead, when there is no browser
- [[Notebooks]] — how a notebook reaches this path through `notebook_diagrams.py`

## FAST selects experimental rendering optimizations

`settings.displayMode` accepts `fast` (the default) or `slow`. Python exposes
`DisplayMode.SLOW` and `DisplayMode.FAST` through `display_settings(displayMode=...)`
and `DiagramSettings.display_mode`. The delivery mode, such as HTML or BROWSER,
is independent. SLOW retains the existing renderer and inspection preparation.
FAST measures diagram positions together before drawing and prepares inspection
diagrams when opened, retaining their content for reuse. The setting propagates
to inspection diagrams and capture targets.

An exported HTML page accepts `?displayMode=slow` or `?displayMode=fast` in
its address, and the value takes precedence over its embedded setting. The page
refuses an address with any other value, per
[A page that reads its address strictly and reports its state to a host](#a-page-that-reads-its-address-strictly-and-reports-its-state-to-a-host).
The override applies to the page's own message and to the boot figure of the page,
and does not change the settings of a message from the relay.

## COMPRESSED exports share JSON values by content

`TermJSONConverter.export_to_json(term, export_form=TermExportForm.COMPRESSED)`
and `export(..., export_form=...)` select the experimental compressed format.
`TermExportForm.UID_REFERENCES` remains the default and emits the existing
`uid_repository`/`data` envelope unchanged. The Python `import_from_json` method
and the TypeScript `TermJSONConverter.import` method accept both forms.

A compressed document has this envelope:

```json
{
  "export_form": "compressed",
  "version": 1,
  "value_repository": [[0, "x"], [0, 7], [2, [0, 1]]],
  "data": 2
}
```

The example decodes to `{"x": 7}`. Each repository entry is `[kind, payload]`:
kind 0 holds a scalar, kind 1 holds an array of reference indices, and kind 2
holds alternating field-name and field-value indices. Every child reference
points to an earlier record. `data` identifies the root. For a term export, that
root is the original envelope containing `uid_repository` and `data`.

The Python compressor hashes each record with SHA-256 and checks the encoded
record inside each hash bucket, so a hash collision cannot merge distinct values.
Equal objects, arrays, field names and scalar values share records. References
are compact integer indices within the document; full hashes are not repeated
on the wire. Object field order participates in the record because TypeScript
constructors consume those fields positionally. Boolean and numeric scalar
values remain distinct. Unsupported versions, malformed records, missing
references and references to the current or a later record are rejected.

Decoding shares JSON containers, which callers treat as read-only. Term
construction still visits each non-UID occurrence separately. Only UID terms
retain the existing object-identity cache, so operation occurrence numbers and
auxiliary expansion keys keep their meaning. Compression is independent of the
FAST/SLOW rendering setting.

Repositories are currently local to each term export. The main expression and
inspection expressions can each use COMPRESSED, but do not yet share a repository
with one another. Smaller raw JSON does not guarantee a smaller HTTP-compressed
download; the experimental full DeepSeek page is 4.94 MB instead of 14.07 MB,
while Brotli sizes are 492 KB and 478 KB respectively.
