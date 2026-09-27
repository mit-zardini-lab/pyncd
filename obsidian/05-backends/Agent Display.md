---
tags: [layer/backends, tool]
code: agent_display/
status: stable
---

# Agent Display

## What it is

**The way to read an expression as text.** Read an expression through `agent_display`
rather than through `display`.

```python
import agent_display as ad
print(ad.listing(morphism_or_hypergraph))   # the full SSA listing, under its legend
print(ad.listing_without_legend(target))    # the same, for a caller that prints many
print(ad.summary(target))                   # inputs, outputs and operation counts
print(ad.trace(target, '%4'))               # what produces %4, and what consumes it
```

Each accepts a morphism **or** a hypergraph.

## Why it exists

[[Diagram Display|`display`]] draws a neural circuit diagram, drawing wires as lines and
operations as boxes, and stacking and aligning axes. A morphism read one line at a time
loses most of what that picture carries, for five reasons.

- **Vertical alignment carries the wiring.** Which output feeds which input is stated by
  two things sitting on the same row, and a line-by-line read loses the row.
- **Colour carries the tape slot a wire reaches.** Read as text, colour is a run of ANSI
  escapes, which carries no meaning and costs tokens.
- **Width changes the output.** Padding, truncation and wrapping follow the terminal, so
  the same morphism does not render the same way twice.
- **Nothing carries a name.** The `q` on one row and the `q` on another may or may not be
  the same wire, and only position distinguishes them, and only to an eye.
- **Every axis is redrawn** on every segment it passes through.

This package renders the same object as a **static single assignment listing**, in the
manner of LLVM or MLIR. Every value is named and typed, every operation names its operands,
a declaration comes before its uses, and nesting is indentation. Five properties of that
form matter here. One line carries one fact. Identity is written out, so a claim about the
wiring is checked by matching strings. Values are declared before they are used, so one
pass builds the graph. Every value carries its type, and the broadcast structure with it.
The rendering is **deterministic**, so two versions can be compared with `diff`.

Fully expanded attention is 32 lines against the diagram's 98, and about an eighth of the
characters.

## The notation

```
%3               a value (a wire). The same %3 everywhere is the same wire.
R[q, x]          an array: datatype, then its axes
{d}              an axis the operation consumes, which is part of its target
q                an axis it is broadcast over, which is part of its degree
w|x              an axis carrying an affine form, guided by x
loop N times { } a block repeated N times
```

`listing` prints the notation above as a legend, so its output describes itself.
`listing_without_legend` omits it, and `notebooks/display/notebook_listings.py` calls
it when a notebook prints many listings in a row.

The scaled dot-product attention of `notebooks/website/tutorial/express_attention.py`
reads queries `%0`, keys `%1` and values `%2`, and lists as:

```
%4 = Einops(%0[q, {d}], %1[x, {d}]) : R[q, x]
%5 = Arithmetic<|d|^{-1/2} x>(%4[q, x]) : R[q, x]
%6 = SoftMax(%5[q, {x}]) : R[q, {x}]
%3 = Einops(%6[q, {x}], %2[{x}, v]) : R[q, v]
```

The braces are [[Weaves and Degree|the weave/target split]], printed directly, and the
datatype before them is the one carried by the array. A quantised datatype prints its
format, as `BF16[q, x]`, and names the datatype wrapped by it in brackets where that
datatype is not the reals.

A `ParaWrap` ([[Para Wrap]]) prints as its body with the tape where it touches it. A
grabbed operand is `<s1>[d, {v}]` in the operand's place, and a dropped result is `<s0>`
among the outputs, so `%3, <s0> = rewire(%0)` is a copy with one copy saved. The tokens
of a pass appended to a cache print with a `+` after the name of the slot, as `<c+>`,
per [[Caching Between Passes]]. An operand or a result that stays on its wire and is
also dropped prints as its wire followed by the slot receiving the drop, as `%0<s1>`, or
`%0<c+>` for the tokens appended to a cache.

## Where it lives

`agent_display/morphism_ir.py`. The package `__init__` carries the reasons for the shape.
`ordered_subgraphs` makes the output deterministic.

## See also

- [[Diagram Display]] — the other renderer, and when to use it
- [[Weaves and Degree]] — what the braces mean
- [[Validation]]
