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
together. `DiagramSettings.axis_hover` carries the choice from a notebook as a
`wst.AxisHover`.

`axisLabelFontSize` (default `0.8`) is the size, in em, of the label an axis carries on
its wire. The layout measures a label at the size it draws it, so a larger label is given
the room it needs. The label was drawn at 0.65 em until 2026-09-16, when the user asked
for it to be about a quarter larger and for the size to be a setting.
`DiagramSettings.axis_label_font_size` carries it from a notebook.

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

The page is painted in the canvas colour of the dark theme by its own stylesheet, before
the bundle runs, and shows a turning ring in the element `page-loading` until its figure
is drawn. Once the message is read, and before its term is built, tsncd's
`src/display/loadingScreen.ts` repaints the page in the theme of the message, so a light
figure arrives on a light page and no white page stands where a dark figure is about to.
The first draw removes the ring, and a page whose figure cannot be drawn writes the reason
where the ring stood. A page with no message of its own shows the ring while it fetches
the figure it boots with. The page background initially uses the dark theme until the message is read.

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
