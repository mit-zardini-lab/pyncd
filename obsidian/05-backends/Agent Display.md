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
`listing_without_legend` omits it, which is what `notebooks/display/notebook_listings.py`
calls when a notebook prints many listings in a row.

```
%11 = Einops(%4[qTλ, {dRλ}], %9[xSλ, {dRλ}]) : R@λ[qTλ, xSλ]
%12 = Shuffle<dRλ>() : R@λ[qTλ, xSλ]
%13 = AdditionOp<+>(%11[qTλ, xSλ], %12[qTλ, xSλ]) : R@λ[qTλ, xSλ]
```

The braces are [[Weaves and Degree|the weave/target split]], printed directly, and the
datatype before them is the one the array carries.

A `ParaWrap` ([[Para Wrap]]) prints as its body with the tape where it touches it: a
grabbed operand is `<s1>[d, {v}]` in the operand's place, a dropped result `<s0>` among
the outputs, so `%3, <s0> = rewire(%0)` is a copy with one copy saved.

## Where it lives

`agent_display/morphism_ir.py`. The package `__init__` carries the reasons for the shape.
`ordered_subgraphs` is what makes the output deterministic.

## See also

- [[Diagram Display]] — the other renderer, and when to use it
- [[Weaves and Degree]] — what the braces mean
- [[Validation]]
