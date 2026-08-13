# The `pyncd` ↔ `tsncd` messaging framework

> **Mirrored document.** An identical copy lives at
> `tsncd/PROTOCOL.md`, next to the TypeScript half of the
> implementation. The protocol belongs to neither repository, so both carry it;
> edit the two together, exactly as you would the two implementations. Only the
> links differ, each pointing at its own side.

`pyncd` builds the algebra; `tsncd` draws it. Neither can do the other's job, so
everything they share crosses a WebSocket as JSON. This document is the contract
between them.

Two files implement it and must be changed together:

| | |
|---|---|
| Python | [`websocket_transfer/websockets_transfer.py`](websockets_transfer.py) |
| TypeScript | [`src/data_transfer/websockets_transfer.ts`](../../tsncd/src/data_transfer/websockets_transfer.ts) |

## Why there is a server in the middle

The obvious design — notebook talks to browser — is not available. A Jupyter
kernel cannot accept connections a browser can reach reliably, and neither end
has a stable lifetime: cells run and finish, tabs open and reload. So a third
process outlives both.

```
  ┌──────────────────┐   dataUpdate    ┌────────────┐   dataUpdate    ┌─────────────┐
  │ Jupyter kernel   │ ──────────────▶ │  DataServer│ ──────────────▶ │  Browser    │
  │ (DataClient)     │                 │ :8765      │                 │(DiagramClient)
  │                  │ ◀────────────── │            │ ◀────────────── │             │
  └──────────────────┘   renderResult  └────────────┘   renderResult  └─────────────┘
                                        holds the
                                        latest term
```

The server holds the most recent term, which is what makes a browser refresh
work: the page reconnects, identifies itself, and is sent the current diagram
without the notebook being involved. It is `run_server.py`, started once and
left running.

**Clients are asymmetric.** A `DataClient` (a notebook) connects per send and
disconnects. A `DiagramClient` (a browser tab) stays connected for as long as
the tab is open. There may be several diagram clients at once; they all display
the same thing.

## Messages

Every message is a JSON object with a `msgType`. Unrecognised types raise on the
server — deliberately, so a version skew between the two repositories fails
loudly rather than silently dropping diagrams.

### `identify` — client → server

First message on every connection. Until it arrives the server does not know
which way information should flow.

```jsonc
{
  "msgType": "identify",
  "clientType": "DataClient" | "DiagramClient",
  "clientVersion": "0.1.0",
  "clientID": "unique-client-id-1234"
}
```

Answered with `{"msgType": "Connected"}`. If a `DiagramClient` identifies while
the server is holding a term, that term is pushed immediately — this is the
refresh path.

### `dataUpdate` — either direction

A term to display. Sent by a notebook; relayed by the server to every diagram
client.

```jsonc
{
  "msgType": "dataUpdate",
  "data": "{\"uid_repository\": …, \"data\": …}",   // note: a JSON *string*
  "settings": {
    "darkMode": false, "debugBorders": false, "coreDebug": false,
    "width": 750
  }
}
```

`data` is doubly encoded — a JSON string inside a JSON object — because
`TermJSONConverter.export_to_json` returns serialised text and it is passed
through without re-parsing. The TypeScript side calls `JSON.parse` on it a
second time.

`settings` is **partial by design**. The client merges whatever arrives over its
own defaults from
[`RenderHandlerSettings.ts`](../../tsncd/src/display/Render/RenderHandlerSettings.ts),
so an omitted key takes its default rather than whatever the previous send left
behind. Every send therefore fully determines the display. On the Python side,
[`send_morphism.display_settings`](send_morphism.py)
is what assembles the partial dict.

`width` is the odd one out: not a rendering option but a layout one, the px at
which a morphism wraps onto another line so that `F₀; F₁ = F`. It therefore
controls the figure's **proportions**, not its scale — narrower gives more rows
and a taller figure, wider gives fewer and a flatter one. It rides the settings
channel because that is the one that already reaches the renderer per send.

Measured on the transformer, which shows how sharply it bites:

| `width` | page | aspect |
|---|---|---|
| 750 (default) | 9.4 × 6.6 in | 1.43 |
| 1000 | 12.0 × 5.8 in | 2.09 |
| 1400 | 16.2 × 4.9 in | 3.29 |
| 2000+ | 17.3 × 4.3 in | 4.03 (unwrapped; no further effect) |

The default suits a screen. A figure spanning a paper's text block usually
wants 1000–1400.

### `dataRequest` — client → server

Asks for the currently held term; answered with a `dataUpdate`, or
`{"msgType": "No Data Available"}`.

### `renderRequest` — notebook → server → browser

A `dataUpdate` whose sender wants the picture back.

```jsonc
{
  "msgType": "renderRequest",
  "requestId": "9f2c…",           // uuid4().hex
  "data": "…",                     // as dataUpdate
  "settings": { … },               // as dataUpdate
  "capture": {
    "format": "png",              // or "svg"
    "scale": 2,                    // device pixel ratio; png only
    "padding": 16,
    "background": "#ffffff"       // null for transparent
  },
  "disturbDisplay": true
}
```

With `disturbDisplay` — the default, and what an absent flag means — the diagram
is drawn on screen and the image cut from it, so a capture and a plain display
are the same render with different follow-through. Note that the settings travel
with it, so a disturbing capture applies its own `debugBorders` to the visible
diagram too.

With `disturbDisplay: false` the browser renders into a second, off-screen
target and the display is left alone. The server also **declines to store** the
term in that case: overwriting it would leave the display intact only until the
next reload, which is a disturbance with a delay on it. See
[Two render targets](#two-render-targets).

`capture` is partial on the same terms as `settings`, defaulting from
[`capture.ts`](../../tsncd/src/data_transfer/capture.ts) and assembled on the Python side by
[`capture.capture_options`](capture.py).
`padding` is measured outward from the diagram's *content* box, which is not the
container's own box: the overlay overhangs it (see [Framing](#framing) below).

Only `png` and `svg` cross the wire. PDF exists, but only on the headless path,
since it comes from the browser's print pipeline rather than from anything the
page can serialise itself.

### `renderResult` — browser → server → notebook

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

or, when the render failed:

```jsonc
{ "msgType": "renderResult", "requestId": "9f2c…", "error": "…" }
```

Failures come back as messages rather than dropped connections because a
notebook cell is blocked on this reply; an exception that never arrives shows up
only as a timeout with no cause attached. `result_to_bytes` is where an error
result becomes a `CaptureError`.

## How a capture is correlated

The reply travels over a *different* connection from the request, so it cannot
simply be "the next message" — hence `requestId`.

1. The notebook sends `renderRequest` and keeps its connection open.
2. The server records `requestId → the requesting socket`, then relays to **all**
   diagram clients so they stay on the same term.
3. The notebook reads past the server's acknowledgement until a `renderResult`
   with its own `requestId` arrives, under a total timeout.
4. The server pops the `requestId` and forwards the first result. Later replies
   — from other tabs rendering the same request — find nothing pending and are
   dropped.
5. If the requester disconnects mid-capture, its pending entry is discarded, so
   no image is pushed at a closed socket.

Three failures are handled explicitly, because each would otherwise present as
an unexplained hang:

| Situation | Result |
|---|---|
| No diagram client connected | Immediate `renderResult` carrying an error |
| Browser never answers | Client-side timeout, `DEFAULT_CAPTURE_TIMEOUT` (60 s) |
| Several tabs answer | First wins, rest dropped |

## Frame size

`websockets` rejects frames over 1 MiB by default and **closes the connection**
rather than reporting the problem. A captured PNG passes that comfortably, so
both ends raise the ceiling to `MAX_MESSAGE_BYTES` (64 MiB) — `websockets.serve`
on the server and `websockets.connect` in `DataClient`. Changing it on one side
only reintroduces the failure.

## Framing

The diagram overhangs its own container.
[`HTMLDrawHandler`](../../tsncd/src/display/HTMLRender/HTMLDrawHandler.ts) places each SVG
layer at `(-BUFFER, -BUFFER)` relative to `#diagram` and sizes it past the far
edge, so an image cut to `getBoundingClientRect()` loses the overlay on every
side. `contentBox` in [`capture.ts`](../../tsncd/src/data_transfer/capture.ts) therefore
measures the union of the container and all its descendants rather than
assuming a number — if `BUFFER` changes, the framing follows.

Zero-area elements are skipped in that union: anchors, wire stubs and spacers
are structural, several sit at the origin, and including them would drag the box
out to nothing.

The headless path has a further constraint. A capture box routinely starts at
negative page coordinates — the content already overhangs, and the requested
padding usually exceeds the page's own — and **Playwright silently clamps a clip
to the page** rather than reporting it, quietly trimming the margin. So
[`HeadlessRenderer.isolated`](headless.py)
strips the page to the diagram alone and moves it to the origin before
screenshotting or printing, making the box valid by construction. Everything is
reverted afterwards, so a batch can mix formats.

## Two render targets

An undisturbing capture needs somewhere else to draw, so the entry point builds
a second container with its own render handlers. It cannot be the same handlers
pointed elsewhere: they hold per-container state — measured rectangles, pending
block references — so the second target is a second set.

The off-screen container is **laid out, not hidden**. `display: none` measures
zero, and since every box position comes from `getBoundingClientRect`, the
diagram would come out collapsed onto the origin. `visibility: hidden` lays out
correctly but captures blank, because the clone inherits it. So it is parked
outside the viewport instead — to the *left*, since overflow in the negative
direction creates no scrollbar.

It is parked with `transform`, not `left`, and that distinction is load-bearing.
`html-to-image` seeds its clone from the computed style via `cssText`, and that
text carries the logical shorthand `inset-inline` *after* `left`. Assigning
`style.left` on the clone updates `left` in place, so the later shorthand still
wins, the clone stays parked off-frame, and the capture comes back blank at any
offset. `transform` has no competing shorthand and the capture overwrites it
outright.

One further thing had to be isolated. SVG `url(#…)` references resolve
**document-wide**, so the drop-shadow filter — which used a single hardcoded id
for every shadow in the document — let the two targets' definitions answer for
each other. Harmless while one diagram owned the page; with two targets it made
an off-screen render perturb the visible diagram's shadows. Each SVG layer now
mints its own id.

## Fonts, and why the stylesheet is bundled

Capture serialises the DOM into an SVG `foreignObject`, which means fonts have to
be inlined as data URIs — and that requires reading `cssRules`, which browsers
refuse on a cross-origin stylesheet. KaTeX loaded from a CDN would therefore
capture in a fallback face; because the wires are drawn from *measured* text
boxes, that moves the geometry, not just the glyphs. So KaTeX's stylesheet is
bundled from `node_modules` by
[`HTMLAnnotationHandler.ts`](../../tsncd/src/display/HTMLRender/HTMLAnnotationHandler.ts)
and served same-origin.

For the same reason every capture waits on `document.fonts.ready` plus a full
frame before measuring anything.

## The headless path, which uses none of this

[`headless.py`](headless.py) drives its own
browser and does **not** connect to the server. A figure rebuild should not
depend on a server being up or on which tab happened to be focused. It serves
tsncd's built `dist/` on a loopback port — `file://` will not do, since webpack
builds with `publicPath: '/'` — and reaches the renderer through a hook the
entry point installs on `window`:

```ts
window.tsncd = {
  render(payload, settings): Promise<{width, height}>,  // same termPass the socket uses
  capture(options): Promise<CaptureResult>,             // in-page serialiser; needed for svg
  bounds(padding): {x, y, width, height},               // page coords, for a screenshot clip
  settled(): Promise<void>,                             // fonts loaded, layout stable
}
```

Both paths render through the same `termPass`, so they agree by construction
rather than by discipline. Three output formats, and the choice matters:

| | Produced by | Typical size | Use when |
|---|---|---|---|
| `png` | Playwright screenshot | ~190 KB | Default. A real browser paint — nothing can be lost in serialisation. |
| `pdf` | Chromium's print pipeline | ~140 KB | Figures for a paper. True vector, real embedded text. |
| `svg` | `window.tsncd.capture` | **~15 MB** | Only when something downstream demands SVG. |

That SVG figure is not a typo. Serialising into a `foreignObject` means
reproducing the diagram from inline styles, and `html-to-image` writes the *full
computed style* — some 6.9 KB — onto each of ~2000 elements. Fonts account for
650 KB of the file; the other 95% is CSS with no bearing on the drawing. Prefer
`pdf` for vector output. A notebook storing one such SVG output grows to 15 MB
against 256 KB for the PNG — 59× — so SVG is a poor return format even though
Jupyter renders it perfectly well.

### What is inside the PDF

Worth knowing before relying on it. One page, sized exactly to the diagram, with
all three KaTeX faces embedded and subsetted, and the labels as real selectable
text (subscripts arrive as separate glyphs — `L q`, not `L_q` — which affects
extraction, not appearance).

It is **not** entirely vector. `feDropShadow` has no PDF equivalent, so Chromium
rasterises every shadowed element — 24 images, 36% of the file, at roughly
192 DPI. Wires, fills and text stay vector. That resolution cannot be raised:
PDF output is byte-identical at `scale` 1, 2 and 4, because the print pipeline
does not see the device scale factor. Dropping the shadows would make it fully
vector and fully editable.

### Saving a set

[`save_figures`](headless.py) takes names
rather than paths and writes them into one directory — `./outputs` by default,
which for a notebook is beside the notebook — in one browser session:

```python
await save_figures({'attention': attention, 'convolution': conv}, width=1400)
# -> ./outputs/attention.pdf, ./outputs/convolution.pdf
```

PDF by default, since that is what a paper wants. A name may carry its own
extension to override the format for one figure, and may include
subdirectories. Options apply to the whole set; for per-figure control, drive
`HeadlessRenderer.save` directly.

## Changing the protocol

Add a message type in both implementations and document it in **both copies of
this file**. The server's `match` raises on anything it does not recognise, so a
half-applied change fails at the first message rather than quietly rendering
nothing — but only if both repositories are updated and **`run_server.py` is
restarted**, since a long-running server keeps executing the code it started
with.
