---
tags: [layer/para, concept]
code: para/data_structure/ParaWrap.py, agent_display/morphism_ir.py
status: working
---

# Para Wrap

## What it is

`ParaWrap(body, grabs, drops)` is a morphism with its grabs and drops written **on** it:
`grabs[i]` is the `TapeSlot` operand `i` of `body` comes off, or `None`; `drops[j]` the slot
result `j` goes back onto. An entry that is a `Para.StreamSlot` stands for a `StreamGrab` or a
`StreamDrop`, the seeds of a loop variable per [[Para Category]], and `to_base` writes those
seeds back. The listing prints a loop drop as `<Ssx2'>` and the diagram labels it `Ssx2'`.
An entry that is a `Para.LoopSlot` stands for a `LoopGrab` or a `LoopDrop`, the seeds of a
slot indexed by the iteration of a repeated block, and carries the index beside the slot,
which the listing prints as `<e[3 - i]>` and the diagram draws as `e_{3-i}`. Its own
`dom()` and `cod()` are the operands and results that
are not on the tape. `to_base()` writes it back out as `grabs ; body ; drops`, so it adds
nothing mathematically, because [[Para Category]] needs only `Grab` and `Drop`. It exists
for the picture: a tape that arrives beside an operator, and one that arrives at a box of
its own a rearrangement away, are two different things to read.

`to_para_wrap(m)` is the layering rule that puts a `Para` into this form. `show_diagram`
applies it by default since 2026-09-12, through `DiagramSettings.tape`, to every term
that holds a grab or a drop, so a loop's variables and a model's parameters draw wrapped
unless a notebook asks for `TapePresentation.BOXED`. It is not applied to what
`agent_display` lists by default, and `notebook_listings.print_listing` takes the same
`tape` argument for a listing in the wrapped form.

A wrap is also written by hand, and a boxed block is where. `ParaBlockOperator.template`
moves each grab of its body to a leading operand of the box and each drop to a trailing
result, and returns the box inside a wrap whose entries stand at those positions, per
[[Para Block Operator]]. Such a wrap is part of the expression rather than a layer over
it, so the box draws its slots under `TapePresentation.BOXED` as under `ABSORBED`, and
`to_para_wrap` meets a wrap it did not write. Two rewrites read a wrap for that reason:
`_with_grab` and `_with_drop` add an entry to the wrap already there, translating the port
through `kept_inputs` and `kept_outputs`, so a grab standing beside a box merges onto the
operand the box still exposes.

The rule never enters an operator, and tsncd draws the block of every `ops.BlockOperator`
as a sub-diagram beside the main figure, so a boxed block would be drawn in the boxed form
whatever the figure asked for. `tape_presentation.wrap_inside_boxes` applies the rule to
the block of each box for that reason, innermost box first, and a block whose seeds have
all been absorbed holds none, so one pass reaches a fixed point. The block keeps its tag,
which is what `remember_drawn_blocks` recognises a delivered body by.

A grab standing beside a plain box and read by that box alone merges onto the port of
the box, by the third row of the table below, and the block inside still reads that
operand on a wire. Since 2026-09-17 `wrap_inside_boxes` rebuilds such a box through
`ParaBlockOperator.box_with_wrapped_tapes_as_seeds`, per [[Para Block Operator]], so its
block holds the tape as a seed and the walk then wraps that seed onto the operation that
reads it inside the body. Wrapping a body can absorb a grab onto a box standing in it, and
the block of a rebuilt box holds bare seeds, so the walk enters each again, and each of
the two moves a tape one box deeper.

## The rule

It runs on the graph, for the reason [[Expression Simplification]] gives: read once, and
by that one reader, is a question about wires. It rewrites siblings alone, innermost
scopes first, to a fixed point.

| pattern | becomes | side |
|---|---|---|
| a `Grab` whose wire enters a block | loaded inside the block. A `Grab` root producing the same node joins the body, the node leaves the block's domain, and the inner scope then splits and merges it | backward: a residual that two contractions of a blocked reverse both need. The block is a forward `Block`'s `R[...]`, since a rule's own reverse is inline |
| `Grab` read by several siblings | **split**: one `Grab` of the same slot per reader, each on a fresh node, the reader redirected onto it. The slot is what says they are one parameter | backward: a residual two contractions need loads beside each |
| `Grab ; seed`, the wire read once | the seed as a `ParaWrap` with that operand grabbed — or an existing wrap gaining the grab | backward: a taped residual enters its `Einops` |
| `seed ; Drop`, the wire read by the drop alone | the seed as a `ParaWrap` with that result dropped, writing the slot where it would write a wire | backward: a weight gradient nothing reads. Forward: the index a `TopK` emits beside its values |
| `Drop` on a wire also read elsewhere | `ParaWrap` over the copy `(0,0)`, second output dropped, attached to the wire's producer: the producer is renamed to write a fresh wire, the wrap reads it and writes the wire the readers name, so a wire leaving a block is copied inside the block and the morphism reads one `copy ; (id * drop)` per wire rather than one rearrangement `[0,0,1,1]` before a row of identities and drops. A wire arriving on the outermost domain is copied by redirecting its readers instead, and so is a wire arriving on a block's domain that the block's codomain does not carry, so the copy stands inside the block, added 2026-09-13 for the partial a reduction's loop carries in, sends and folds | forward: a value passed on and saved; a loop variable that also leaves the loop; the partial each round of an explicit reduction sends |
| anything left bare | `ParaWrap` over an identity | the old `Grab`/`Drop` box |

A seed that reads an operand through a reindexing beyond a `Rearrangement` absorbs
neither a grab nor a drop, so rows three and four pass it over and it is left bare.
`_has_non_rearrangement_reindexing` is the test. `_reindexings_of` collects the
reindexings of every `Broadcasted` in the seed and `_is_rearrangement` reports whether one
of them is built from `cat.Rearrangement` alone. Both walk the construction rules and
never enter an object, because `_wrap_here` asks the question of each candidate on each
rewrite and `tutil.type_search` would descend into every array, axis and size symbol as
well. A wrap draws a grabbed operand
on the operand's own port, and an index map that is more than a permutation, a copy or a
deletion of degree axes reads that port at indices the port does not name. The
convolution of `example_notebooks/minimum_working_example.py` is the case: its
`View` reads its operand through `x' + w`, and its grab stays a box with the reindexing
drawn between the box and the operator.

The last row is what makes `GrabBox` and `DropBox` instances of one display rule rather
than boxes of their own. A drop is left bare when its wire arrives on a nested scope's
domain and the scope's codomain carries the wire, which a sibling rewrite may not move.
Since 2026-09-13 a wire the codomain does not carry is copied inside the scope, as at
the outermost domain. Until 2026-09-12 a wire leaving a block was
left bare as well, because the block's codomain names it, and the copy at the producer
is what lets the codomain keep its wire.

## How it prints and draws

- **[[Diagram Display]] settings**: `DiagramSettings(tape=TapePresentation.ABSORBED)` makes
  `show_diagram` apply `to_para_wrap` to a term just before drawing it, and
  `notebook_listings.print_listing(term, tape=...)` does the same for a listing, so a
  notebook works on the `Para` and shows the wrap without calling it. The wrap adds no
  mathematical content, so the choice is the display's.
- **[[Agent Display]]**: a wrap prints as its body with the tape in place —
  `%1 = Einops(%0[q, {v}], <s1>[d, {v}]) : R[q, d]`, `%3, <s0> = rewire(%0)`,
  `<dW1> = Einops(<s0>[{q}, m], %4[{q}, f]) : R[m, f]`.
- **[[Diagram Display]]** (`tsncd`): `ParaWrapBox` has the domain on the left and the
  codomain on the right, and is exactly as tall as the morphism it wraps, because a wrap
  reserves no height for its tapes. The tapes are drawn past the box, to an arrowhead
  `tape_escape` beyond it and a slot label beyond that, over whatever the layout put
  there, such as a neighbouring row or a separator. The overhang is deliberate, because taking data off a tape and
  putting it back should cross the picture. A `Rearrangement` body (the identity of a
  bare grab/drop, or the copy `(0,0)` with one output dropped) has no inner box: the
  wrap's two columns are the rearrangement, and a tape is an elbow off the anchor
  concerned, the copy dotted where it leaves. A `Broadcasted` body is drawn by a
  `BroadcastedBox` given a `WrapLayout`, and has two rows as it has two columns: the
  grabbed operand's whole array on a `RowMeridian` (a meridian turned on its side) along
  the **top edge of the `BroadcastedBox`**, where the tape reaches it, and the operand's
  *target* on a row along the top of the **operator's glyph**, exactly as the left column
  holds the targets of the operands from the left. **`vertical_product` builds both rows**,
  which is what makes them agree. The order is the product's own: each operand in turn, each
  operand's anchors in the order of its shape with the datatype last, and a `SeparatorAnchor`
  between one operand and the next as a column has. A view with no anchor at a position
  leaves the slot empty, so the operator's anchor for a position sits under the anchor of the
  array it reads and the wire between them is vertical. `ArrayMeridian.row_slots` is what
  reports the slots, and it is the one thing that has to know which of the two views it is
  drawing, which `ArrayMeridian.weave` already records. `apply_wrap` moves the meridians and
  leaves their anchors where they are, so a box that captured anchors in its constructor
  goes on drawing between the same ones. It has to run **before** `BroadcastedBox` links
  its columns to the operator box, because `link` pairs two meridians by position and the
  operator box is built with every operand in its left column: link first and the first
  kept operand is paired against a grabbed one, and its wire is then drawn turning up onto
  the row that operand has since moved to. Every parametric operator hits it, since
  `grab_parameters` prepends the weight. Target links to target straight down
  , because the two rows are padded to the same slots so that they sit anchor under anchor, and the
  glyph's row is displaced to the core's edge, through `raise_operator_rows`, so that there is no hop. The
  degree axes route round the glyph to the right column, and **which way they route is the
  row's own display type**. `BroadcastDisplayType` describes the degree wires, a wrapped box
  has them on two sides, and `row_display_type` answers for the rows as
  `find_broadcast_display_type` answers for the columns. The two differ in one thing: a row
  does not read the box's override. `LinearBox` and `TopKBox` force `NODE` so that the
  reindexing they carry is drawn as a figure, and that figure stands in the core between the
  columns, so a degree axis on a row would travel into the core and back out to reach the
  right column. Where every reindexing is a rearrangement the row weaves instead and the axis
  runs straight across. A node needs an operand to draw it from, so where every operand
  carrying a degree axis is on a row and the rows weave, the whole box weaves: the selecting
  `Linear` of a mixture of experts is that case, its index being
  its only broadcast operand and arriving on the tape. The raise is conditional
  through `OperationBox.raise_rows`. A box whose grabbed operand *ends* at the glyph rather
  than cupping into a column keeps its row at the glyph's own top edge, so the tape
  visibly runs down onto the operation, and `ParaWrapBox.place_axis_label` rests
  the axis names beside the tape, which crosses no gap that could label it. The parametric
  boxes are the case, meaning `LinearBox`, `NormalizeBox` and `TransposeBox`, whose grabbed
  operand is a weight, and `InjectBox`, whose grabbed operand is a selector. A glyph that
  is not a rectangle draws one behind itself wherever the box is larger than the glyph,
  since a diamond's top edge is a point and a tape has nothing to land on. `GlyphBox` is
  where that is done, per [[Diagram Display]]: `NormalizeBox`'s circle, `TopKBox`'s diamond
  and `InjectBox`'s each sit on a rectangle with one corner bitten off.
  See [[Show Grabbed Parameters]]. Every box with rows is laid out
  by one `four_sided` helper, `Horizontal(left, Vertical(top, core, bottom), right)`,
  with a row wider than the core widening it. **An anchor records which way it faces**, in `Anchor
  .horizontal`, set by `RowMeridian`), and `wire_curve` draws the wire between a row
  anchor and a column anchor as a quarter **circle** of `turn_radius` with straight legs
  to each anchor, shrunk only where the anchors are closer than the radius. It is
    one rule, and it covers the `Einops` cup
  to a grabbed axis, a grabbed degree axis reaching the right column, and a row feeding
  a `View`'s node. **A cup between two anchors on one row is ordered right to left.** The
  two anchors are the two grabbed operands of one contraction, and the arc between them
  sweeps clockwise from its first endpoint, so taking the left anchor first raises the arc
  over the row. Each of the two operands hangs off a stub descending from the tape above
  the row, and an arc that rises meets both stubs at a cusp. Taking the right anchor first
  dips the arc below the row instead, and the arc then continues the descent of each stub.
  `row_cup` in `additionalOperationBoxes.ts` states the order, and the score contraction
  that a backward pass rebuilds from
  its two grabbed operands is the figure the rule was corrected in. Two anchors that face the SAME way get an S-curve whose control points
  are displaced along the direction they face: `Curve.flatCurve` for two column anchors,
  `Curve.verticalCurve` for two on rows, one above the other. Drawing the second pair flat
  gives the wire horizontal tangents, and it then meets each anchor at a right angle to the
  tape running into it, which is the kink `verticalCurve` exists to remove. A weight
  gradient is where it shows, its operand grabbed and its result dropped, so its wire runs
  from the top row to the bottom one.

**A tape's names are written on two lines, and a tape is as tall as its names need.**
The user ruled on 2026-09-16. A grab writes its slot name at the arrowheads, left of its array's tapes,
and the array's axis names on the line below it, each beside its own tape. A drop writes
its slot name at the arrowheads and the axis names on the line above it, and only a tape
that leaves an operator's row writes one, because a copy-drop's array continues on its
wire and is named in the gap. An axis name wider than the room between its tape and the
next tape of the row is turned a quarter turn clockwise and runs down its tape, flush
against it, and one turned name turns every name of its array so the array reads one way.
The tape's height, from the arrowhead to the row anchor, is the larger of
`tape_base_height` and the measured stack of arrowhead, slot name, `tape_label_clearance`
and the deepest axis name, where a turned name is as deep as its text is wide, all measured
bare through `TextEstimator.estimate_bare_text_width`. The half-anchor clearance the wrap
used to add on every side is gone. `tape_offsets_along_a_row`, `axis_label_rooms`,
`rotated_axis_labels` and `place_rotated_axis_label` in `ParaWrapDisplay.ts` carry the
rules.

**The wrap's own columns move with its inner box.** `ParaWrapBox.post_placement` centres
the inner box between the two label insets, and since 2026-09-16 it offsets the wrap's own
`left_anchors` and `right_anchors` by the same amount, so a wire crosses a wrap straight
and bends only in the composed gap beside it. Before that, every wrap with a label inset on
one side alone put an S-bend of half the inset inside itself.

**A drop's tape begins at the glyph's bottom edge.** `BroadcastedBox.clearance_from_rows`
moves a glyph that would reach past a row back inside the core. Until 2026-09-17
`degree_wire_layer` drew a degree wire that reaches a row anchor in the main layer, which
the glyph covers, so the descent to the row was hidden. The user ruled that day, on the
`Pool` box of DeepSeek-V4.1-Flash, that "the operation is broadcast over this axis,
therefore the axis should be drawn above it". Every degree wire is now painted in the
broadcast layer, `DEGREE_WIRE_LAYER`, over the glyph, so the token axis of the pool is
seen crossing the box and turning down into its tape, and the box's label is painted
over the wire.
A tape is extended past its row anchor towards the operation only where the wire continues
straight down from that anchor, per `wire_continues_towards_the_operation`. A degree axis
whose wire turns at the row anchor ends its tape there.

**A grab's tape ends at the glyph's top edge, whatever room the degree takes.** The user
ruled on 2026-09-16, that a ParaWrapped
grab goes down to the top of the operator box even where a degree is present.
`BroadcastedBox.core_room` reserves half an anchor per degree axis above the glyph for
the degree wires to route over it, and `raise_operator_rows` lifts the operator's row to
the core's top edge, so the room between the row and the glyph is crossed by the tape
alone. `extended_tape_terminal` had drawn the tape on by at most twelve pixels or half of
that room, and the pool's tapes into the Reindex layer's `TopK` stopped above the
diamond. It now draws the tape on to the operation rectangle's edge, at both ends of
the box, so a drop's tape begins at the glyph's bottom edge by the same rule.

## Gaps

- A grabbed operand's **degree** axes run from the operator's top row to the right column
  in one quarter turn, across the operator. Routing them round it is layout work not yet
  done.
- The slot labels still float outside the layout, as the tapes did before. A label over
  a dashed separator is the visible cost. A second instance from [[Show Grabbed Parameters]]: in a
  product of wrapped operations the lower wrap's free end is inside its neighbour, so
  $W_K$'s label lands on the $Q$ box. A third instance, a box with three grabbed arrays
  on one row whose slot names landed over the previous array's axis names, was closed on
  2026-09-16 by writing the axis names on the line below the slot names.

## See also

- [[Backpropagation]] — what writes the grabs and drops this wraps
- [[Show Grabbed Parameters]] — the other writer of grabs: weights made operands, drawn from above
- [[Para Block Operator]] — the other writer of wraps: a boxed block, taped at its ports
- [[Training Mixture of Experts Gates]] — what would write the slots, for a router
- [[Para Category]] — why two seeds suffice for the algebra
