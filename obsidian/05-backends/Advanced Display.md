---
tags: [layer/backends, tool]
code: websocket_transfer/auxiliary_information.py, notebooks/display/advanced_display.py, notebooks/display/explain_operators.py, notebooks/display/explain_reindexings.py, notebooks/display/expand_with_parameters.py, algebra/registries/standard_expansions.py, algebra/write_index_notation.py, para/processing/write_linear_formula.py, data_transfer/broadcast_occurrences.py
status: stable
---

# Advanced Display

Written by Claude Fable 5.1, effort 80.

The advanced display adds two things to a tsncd figure. A legend lists the axes of the
term beside the figure. An inspection box opens over a block or over an operator with
a standard expansion, on the open page. The user asked for both on 2026-09-16, with the
box's top left corner at the pointer, a click locking it open for recursive reading, a
click outside every box closing them, and links to the code each block stands for. The
log is.

tsncd does no algebra, so everything a legend or a box shows is computed in this
repository and sent beside the term, as the `auxiliary` field of a `dataUpdate` or a
`renderRequest`. [[Diagram Wire Format]] states the field. tsncd's `src/advanced_display/`
folder draws from it, and its `index.ts` is what tsncd's own entry point references.

## The legend

The legend is a table with one row per named axis of the term. The axis is on the left,
the integer its size comes to in the middle, and the code form its name carries on the
right. The size is evaluated under the assignments a `NumericConfig` made, passed as
`DiagramSettings.assigned_sizes` as [[Compound Axis Labels]] passes them, so the legend
is read off the symbolic model and every letter keeps its name. Two axes that read the
same in every column make one row, so the window axis of eight attention modes is one
row. An axis whose uid carries no name, which is a `ConcatenatedAxis`, is left out
because its parts are listed. `auxiliary_information.legend_rows` builds it.

Each row carries the uids of the axes it stands for, and tsncd links the row to the
figure through them. Resting the pointer on a row halos every wire of those axes and
glows every name of them. Under `DiagramSettings.axis_hover = AxisHover.EVERYWHERE`,
resting it on a wire or a name of an axis shades the row as well, and under
`AxisHover.OFF` the row answers no pointer. `AxisHover.LEGEND` is the default. The uid
is the identity of an axis, per [[Invariants]], so a row that merges two axes of one
name lights both, and hovering one of the two lights the row and that axis alone.
The user asked for the halo on 2026-09-16, together with a plate behind a taped array
when the pointer rests anywhere on it, and [[Diagram Display]] states both.

The user asked on 2026-09-16 for three more things of the legend, and the log is.

A row reads exactly as the wire of its axis reads. The sender's `latex` is the axis name
without its size, so the sparse axis of the router read `k/e` in the legend and
`|k|_{6} \text{ of } e_{384}` on its wire, and the residual width read `m` beside `m_{5120}`.
A label is drawn by tsncd, by the processor its registry gives the axis, and only tsncd
knows what that processor writes. tsncd therefore finds the axes of a row in the term by
the row's uids, in one walk that enters each node once, and labels the row with the
processor's `annotation_text()`, which is the function that labels the wire. The `latex`
the sender wrote is drawn where no axis of the row is found. Where the axes of one row
are labelled differently, the row is drawn as one line per label.

A click on a row locks the halo of its axes on, and a second click on the row releases
it. Several rows can be locked at once. The lock is a highlight source of its own in the
render handler's registry, so the hover of the row comes and goes beneath it. A locked row
keeps its shading and shows a closed padlock, and a note under the table says that a row
can be clicked. The note is written where the inspection boxes are on, because a figure
with the legend alone is usually captured as an image, which answers no click. A row
locked in the legend also halos its axes in the diagrams drawn inside that figure's
inspection boxes, each of which has a render handler of its own.

A tape slot locks the same way since 2026-09-17, when the user asked for it: a click on
the plate of a taped array holds the slot lit and closes the padlock beside the plate,
and [[Diagram Display]] states the behaviour. The lock is no longer the legend's own
machinery. The rows and the slots each hold a `LockedHighlights` of tsncd's
`src/display/Render/locked_highlights.ts`, a set of tokens held on a render handler
under a source of its own, and `inspectionBoxes.ts` carries every such set into the
diagram of a box with one call, so a slot locked in a figure is lit inside the boxes
that figure opens as a locked row is. Nothing new crosses the wire.

A lock names a slot rather than the diagram it was clicked in, so it lights every grab
and drop of that slot on the page. `attach_inspection_boxes` registers the render
handler of the figure's own diagram with every `LockedHighlights` as `adopt_content`
registers the handler of a box's diagram, and a lock therefore reaches the figure from
inside a box as it reaches a box from the figure. Until 2026-09-17 the figure's handler
was registered nowhere, so a slot locked in the figure lit its boxes while a slot
locked inside a box left the figure dark. The user found the difference on the slot
`mod` of Engram.

The label an axis carries on its wire is drawn at `DiagramSettings.axis_label_font_size`,
in em, which is the `axisLabelFontSize` display setting. tsncd's default is 0.8, where the
label was drawn at 0.65 before, and the layout measures the label at the size it draws
it. The strides and the shift written inside a reindexing, and the line that labels a
quantisation, stay at 0.65, because neither is the label of an axis.

## The inspection boxes

A box over a block shows the block's title, its formula where it carries one, its
description, its code references and its body drawn as a diagram. The description is the `description` of `cat.Block.template`, a
sentence or two saying what the block computes, written beside the title in the module
that builds the block, as the registry writes one beside each standard expansion. Every
block of the V4.1 model carries one since 2026-09-16, when the user asked for the boxes to
describe their blocks as the expansions describe their operators, and
`check_the_code_references` in the model's validator requires it. A box over an operator shows the operator's name, the formula
of its expansion, a sentence saying what it computes, and the expansion drawn as a
diagram. A box opens when the pointer rests on the operator's operation box, which is
its glyph alone, without the node box above it where the operator has one, as the user
ruled on 2026-09-16, and closes when the pointer leaves that glyph and the inspection
box. While the box is open, a block's own highlight is held on, in the figure and on the
body drawn inside the box, so it stays when the pointer moves into the box. A click locks the box. A note at the top of every box reads
"click to lock" beside an open padlock, or "click to unlock" beside a closed one, and
clicking the note locks or unlocks the box. The note and the title under it are the head
of the box. A box stands on the screen whole, as the user asked on 2026-09-18 after
the boxes of the text-only page ran off the bottom of the window and scrolling to
reach one moved the page under the pointer, which closed the box and opened another.
`boxPlacement.ts` states the placement. The box hangs below and to the right of the
pointer where it fits, stands above the pointer where it fits there and does not fit
below, so the region the pointer rests on stays visible, and is otherwise held against
the bottom edge of the window. A box too wide for the room to the right is held against
the right edge. The placement runs when the box opens with its text and again when its
diagram arrives, and the box is never taller than the window less an eight-pixel margin,
so it fits wherever the pointer is. A box taller than that room scrolls, and the head
stays at the top edge of the box while the rest scrolls beneath it, as the user asked on
2026-09-17 after scrolling the Full Attention box away from its unlock note. The head is
a sticky element that paints the background of the box. A browser measures the inset of a
sticky element from the content edge of the scrolled box, so the inset is the padding of
the box taken back. A figure holds one unlocked box at a time
and as many locked boxes as it has blocks and operators, so several blocks are
inspected at once, as the user asked on 2026-09-16. The blocks and operators drawn
inside a box open boxes of their own, one level down, whether or not the box is locked,
which is how a model is read down to its leaves from one figure. An unlocked box stays
open while the pointer is in a box opened inside it, and closing a box closes the boxes
opened inside it. A click outside every box, or the Escape key, closes them
all. tsncd draws the body and the expansion with the renderer it draws the figure with,
inside the box, so a sub-diagram carries every glyph and label the main figure carries.

The body or the expansion a box draws is drawn once per figure and kept between
openings. Once the pointer enters a figure, tsncd draws the content of every block and
operator the pointer can open, one per idle period, in a holder parked outside the
viewport, and a box takes its content from that pool. A box whose content is not yet
drawn shows its text first and its diagram a moment later. The block highlight a box
holds goes through the render handler's highlight registry, under a source of the box's
own, which is the registry the block's own hover uses, so the two compose.

The blocks are keyed by the uid of the block's tag, which is the key `drawnBlockTags`
already uses. The block's own body is in the term, so the box draws it from there and
the auxiliary information carries the title, the description and the references.

### The standard expansions are collated in a registry

A `ops.SoftMax`, an `ops.L1Norm`, an `ops.L2Norm` and an `ops.Normalize` are each written
out in their primitives by a rule in `algebra/operator_expansion.py`, and a `ops.Linear`
by `expand_parametrised_linear_root` in
`algebra/linear_expansion.py`. The
rules were functions with nothing listing them, so
`algebra/registries/standard_expansions.py` now collates them: a dictionary from operator
class to the rule, the formula it writes out in LaTeX, and a sentence of description,
extended by the `register` decorator the rules carry. `expansion_for` walks the MRO as
`algebra.registries.accumulator.accumulator_for` does. The display asks the registry
which operators of a term can be opened. The registry is filled when
`algebra.operator_expansion` is imported, so `auxiliary_information` imports it.

The formula and the description of a row are each one text, or a function that writes the text from the `cat.Broadcasted` that carries the operator, and `StandardExpansion.formula_of` and `description_of` read either. The five rows of the package hold functions, so a formula names the axes of the operator it is shown over. For an operator over an axis $m$, and a linear map from $m$ onto $o$ named $Q$, the five rows write

| operator | formula |
|---|---|
| `SoftMax` | $\mathrm{softmax}_{m}(s) = \frac{e^{s}}{\sum_{i_{m} \in m} e^{s[i_{m}]}}$ |
| `L1Norm` | $L^{1}_{m}(v) = \frac{v}{\sum_{i_{m} \in m} v[i_{m}]}$ |
| `L2Norm` | $L^{2}_{m}(v) = \frac{v}{\sqrt{\sum_{i_{m} \in m} v[i_{m}]^{2}}}$ |
| `Normalize` | $\mathrm{RMSNorm}_{m}(x) = x \left(\frac{1}{|m|}\sum_{i_{m} \in m} x[i_{m}]^{2} + \epsilon\right)^{-1/2} \odot \gamma$ |
| `Linear` | $y[i_{o}] = \sum_{i_{m} \in m} x[i_{m}]\, W_{Q}[i_{m}, i_{o}]$, and $+\, b_{Q}[i_{o}]$ after it where the map has a bias |

The user ruled on the notation of a formula on 2026-09-17. A sum names the index it iterates and the axis the index ranges over, $\sum_{i_{m} \in m}$, where it was written $\sum_{m}$. An array is read at an index in brackets, $x[i_{m}]$, where it was written with a subscript, so the top-k selection of the V4.1 table reads $\{(j, s[j]) : \ldots\}$. The gain of the RMSNorm is the last factor, $\odot \gamma$, as the expansion multiplies it in last.

The shifted softmax of `expand_shifted_softmax` is a second form of the same operator
and is not the standard one, so it is not registered. The row of a `Linear` is registered
when `algebra.linear_expansion` is imported, which
`notebooks/display/expand_with_parameters.py` does. `websocket_transfer` imports `algebra`
alone, so a caller that packages the auxiliary information without the notebook layer
gets the first four rows. `display/` still imports nothing above it.

### An operator is written out with its parameters on the tape

The user asked on 2026-09-16 for the RMSNorm of a box to be drawn with its weights
present, for the other expansions that were incomplete to be completed, and for a linear
map to be explained. A model holds the gain of a `Normalize` and the weight of a `Linear`
inside the operator, where a rule cannot reach them, so the expansion of an RMSNorm was
drawn with no gain. `notebooks/display/expand_with_parameters.py` grabs the parameters
first, through `para.processing.show_grabbed_parameters.grab_parameters`, so each is an
array read from the tape and fed to the operator as an operand, per
[[Show Grabbed Parameters]]. It then writes out every operator of the result that the
registry holds a rule for, through `operator_expansion.expand_standard_operators`.

Each grab is then followed by a weight box, and the result is presented under the tape
setting of the figure. The user ruled on the form on 2026-09-17: a map `W : a -> b`
becomes `[ParaWrap(grab(ab), W) : 1 -> ab] * hold(a)` followed by the `Einops` of `ab` and
`a` onto `b`. The weight box is a `Linear` named after the slot, whose one operand and one
result are the array the grab reads, written by
`show_grabbed_parameters.weight_box_fed_by`, per [[Show Grabbed Parameters]]. Under
`TapePresentation.ABSORBED` the grab is absorbed onto that box, so the figure draws a box
labelled with the weight, a tape running down onto it, and the contraction after it as an
operation of its own. tsncd already drew a wrap over a `Linear` whose one operand is its
weight, so nothing changed there. Until that day the grab was absorbed onto the
contraction, and the box over a weight drew one operation under a formula that names two
arrays. The user first asked for the grab and the contraction to be drawn apart, which a
bare `Para.Grab` in each expansion drew, and replaced that form with the weight box the
same day. A bias and the gain of an RMSNorm are read through a weight box
the same way, and a learned array such as the sink logit is written out as its weight box
alone.

The user asked later on 2026-09-17 for the expansions to be drawn with no `ParaWrap`.
`expand_with_parameters.expanded_with_weight_arrays` replaces each grab with a weight
array, a `Linear` with no operands named after the slot, which
`show_grabbed_parameters.weight_array_in_place_of` writes, per [[Show Grabbed Parameters]].
A map `W : a -> b` becomes `[W : 1 -> ab] * hold(a)` followed by the same `Einops`, and the
expansion holds no tape. A bias and the gain of an RMSNorm are weight arrays the same way,
and a learned array such as the sink logit is written out as its weight array alone.
`DiagramSettings.expanded_parameters` chooses between the two forms through the enum
`expand_with_parameters.ExpandedParameters`. `WEIGHT_ARRAYS` is the default, and
`READ_FROM_THE_TAPE` draws the weight box of the paragraph above.
`expand_with_parameters.write_out_under` returns the function for a setting. A weight
array inside an expansion opens no box of its own, because
`linear_expansion.is_parametrised_linear` is false for a `Linear` with no operands and the
rule returns it unchanged.

`expand_normalize` already multiplied a gain operand into its result, so the RMSNorm comes out with the product by $\gamma$. A `Linear` comes out as the `Einops` that contracts its weight.

The user reviewed the boxes on 2026-09-17 and found the expansions of the weights inconsistent. The sink logit opened no box, and the biased projection $H$ of the hyper-connections was drawn as a `Linear` reading two arrays where every other map was drawn as a contraction.

Every `Linear` of the V4.1 model is now written out as its weight
read from the tape and multiplied into what the map reads. A `Linear` with a bias comes
out as the contraction followed by an `AdditionOp` with the bias, which
`expand_parametrised_linear_root` writes since that day, per [[Linear Expansion]]. A
`Linear` with an empty domain is a learned array, and `grab_parameters` turns it into the
`Grab` of that array alone, under the array's own name, per [[Show Grabbed Parameters]],
so its box draws the array read from the tape with nothing multiplied into it. A `Linear`
that selects among weights, or that reads more than one input, is still not written out
by the rule, and its box draws the parametrised form. The V4.1 model holds neither.

`expand_normalize` wrote neither the division by the number of elements nor the epsilon under the root. It now writes both, in one `Arithmetic` after the contraction, $s \mapsto (s / |m| + \epsilon)^{-1/2}$, with the epsilon a named free symbol. The softmax and the L1
norm were complete.

`auxiliary_information.auxiliary_information` takes the function that writes an operator
out as `write_out`, and `notebooks/display/advanced_display.py` passes the one above. The
function is applied to the operators of the figure alone. Inside an expansion the
registry's rule is applied as it stands, because the parameters are operands there
already. Applying the grab at every depth did not terminate: a `Linear` the rule leaves
in its parametrised form was grabbed again inside its own expansion, and again inside
that one.

### The box over a weight says the role of that weight

The user asked on 2026-09-17 for the weights to have specific explanations of their
roles, and for the equation of a weight to be generated, to include the bias only where
it is relevant, and to use the bracket notation.

`para/processing/write_linear_formula.py` writes the formula and the description of a
`Linear` from the `cat.Broadcasted` that carries it. The operand's target axes are summed
over, the result's target axes index the result, and the axes of the degree are left
out, because the map is broadcast over them. The parameters take the names
`show_grabbed_parameters.parameter_arrays` gives them, which are the names of the slots
the expansion under the formula draws on the tape. The bias appears in the formula and
in the description only where `operator.bias` is set. A learned array reads as the array
at the result's indices and is described as an array with no input. An operand that
selects among weights is an index the weight is read at, and it is not summed over.
`algebra/write_index_notation.py` writes the pieces of the notation, and the softmax, the
L1 norm and the RMSNorm write their formulas through it as well.

The role of a weight belongs to the model. `auxiliary_information.OperatorRole` holds a
sentence or two and the released lines that declare the weight, and
`DiagramSettings.operator_roles` is a table of them keyed by the text of the operator's
name. The description an expansion carries is the role followed by the description the
registry writes, and its references are the role's followed by those of the operator's
class. `OPERATOR_ROLES` in `notebooks/sota/DeepSeekV41Flash/operator_explanations.py`
gives the twenty-three weights of the V4.1 model, with the declarations read from the
released model file at the pinned commit on 2026-09-17, and
`check_the_operator_explanations` requires a role for every `Linear` of the model.

### An operator with no expansion is explained through a block drawn in place

A top-k selection, a read at selected positions, an embedding, a concatenation of two
axes, a covariant view, the merged positions of a selected block and an elementwise map
have no expansion.
The user asked on 2026-09-16 for short explanations of such operators, and chose the
mechanism the same day: the operator is put in a `BlockOperator` whose aesthetics make it
take no room and draw no box, which still answers the pointer and still packages
auxiliary information. The user preferred it to a registry because the explanation can be
tailored to the display.

`cat.BlockAesthetics` has two trailing fields for it. `formula` is LaTeX a box shows
under the title. `drawing` is a `cat.BlockDrawing`, `BOX` or `BODY_IN_PLACE`, and `None`
reads as `BOX`, so an ordinary block exports no enum and a tsncd bundle built before the
enum existed still reads it. tsncd draws a `BlockOperator` whose block says
`BODY_IN_PLACE` as the box its body would get, registers the hover region for the
wrapper, and queues no body beside the figure. The render of a wrapped operator is
identical, pixel for pixel, to the render of the bare one. The box of such a block shows
the title, the formula, the description and the references, and no diagram, because the
body is in the figure.

The region is the glyph of the operator, which an operator drawn as no glyph does not
have. `OperationBox.region_element` is the hook for such a box, and it names the element
the pointer is tested against: the holder for a concatenation drawn on the holder's own
anchor, and for a cast drawn thin the coloured label carrying the format written by it,
per [[Diagram Display]]. `inspectionBoxes.measure_regions` reads the text inside a label
rather than the box holding it, because a label is placed in a box as wide as its gap.

`notebooks/display/explain_operators.py` is the display pass. It is given the term as the
transport sends it and a table from operator class to `OperatorExplanation`, as
`DiagramSettings.operator_explanations`, and wraps every `Broadcasted` the table explains,
inside the body of every box as well, each node once by identity. It runs under
`AdvancedDisplay.INTERACTIVE` alone, after the term has been converted and recycled, so
the recycling does not meet the wrappers and a listing or an image holds none. The model
is not edited, so every pass and every validator reads the operators where the model
wrote them. The table belongs to the model that is drawn:
`notebooks/sota/DeepSeekV41Flash/operator_explanations.py` holds the eight rows of
the V4.1 model with the released lines each stands for, and beside them
`OPERATOR_REFERENCES`, the released `RMSNorm` and `linear` that the box of an expanded
operator links, passed as `DiagramSettings.operator_references`.

The user asked for three more things of these boxes on 2026-09-17, and the log is.

A row of the table is one `OperatorExplanation` for every operator of a class, or a
function that writes the explanation from the `cat.Broadcasted` that carries the operator
and returns `None` for an operator that needs no box. The row of `dst.TopK` is a
function, `explain_top_k`, because the user asked for the descriptor of a top-k selection
to depend on the chosen form. It reads the `dst.SelectionForm` of the operator and whether
a second operand carries the positions of the entries selected over, and names the
operator's own axes. For scores $s$ over an axis $n$, a count $k$ and a result axis $r$, the
forms read

| form | formula |
|---|---|
| `WEIGHTS` | $y[i_{n}] = s[i_{n}]$ where $s[i_{n}]$ is among the $k$ largest of $s$ |
| `WEIGHTS_SELECT` | $\{(p[i_{r}], y[i_{r}])\} = \{(i_{n}, s[i_{n}]) : s[i_{n}] \text{ is among the } k \text{ largest of } s\}$ |
| `ONLY_WEIGHTS` | $\{y[i_{r}]\} = \{s[i_{n}] : s[i_{n}] \text{ is among the } k \text{ largest of } s\}$ |
| `ONLY_SELECTION` | $\{p[i_{r}]\} = \{i_{n} : s[i_{n}] \text{ is among the } k \text{ largest of } s\}$ |

and where a second operand $q$ holds the position of each entry, the position handed out is $q[i_{n}]$ in place of $i_{n}$. A result is written as a set because the operator states no order among the entries it keeps. The V4.1 model holds four selections and the four boxes differ: the router in the `WEIGHTS` form, the indexer and the candidate pool in the `ONLY_SELECTION` form, and the Reindex layer in the `ONLY_SELECTION` form over positions held as data. The released lines differ as well, read from the released model file at the pinned commit on 2026-09-17: line 822 for the router, 578 to 579 for the indexer, 607 for the pool, and 574 to 575 for the mask the Reindex layer applies first.

An elementwise map is an `ops.Arithmetic` over a formula, and its name in the figure is
the LaTeX of the formula unless the model gives it a shorter one. A map named after its
formula, such as $e^{x}$, shows everything in the figure and opens no box.
`explain_operators.explain_named_arithmetic` is the row for the others: a map whose name
is not its formula, as the square root of the softplus is named $\sqrt{s^{+}}$, and a map
whose formula holds a function the primitive numerics spell out, as a sigmoid is spelt.
The box writes the result $y$ at an input $x$ as the formula the model wrote, so the
doubled sigmoid reads $y = 2 \sigma(x)$. Until 2026-09-19 the spelling
`nm.expand_every_expandable` gives followed it where the two differed. The user ruled that
day, when the Engram gate gained a sign, a magnitude, a maximum and a root, that a box does
not spell such a function out, because every one of them is a function a reader knows, and
that the spelling exists in `nm.Expandable.expand_to_primitives` alone.
`ARITHMETIC_ROLES` in the V4.1 table
says what each of the six such maps of the model is for, by the text of its name, and the
sentence comes first in the box. `nm.Power.to_latex` writes a power of one half as a
root and `nm.Logarithm.to_latex` writes a logarithm to the base $e$ as $\ln$ since the
same day, so the box over the router's score function reads $y = \sqrt{\ln(1 + e^{x})}$
where it would have read `log_{e}(1 + e^{x})^{2^{-1}}`.

An operator standing as the body of a `ParaWrap` is wrapped as every other is. The
explaining block has the operands and the results of the operator it holds, so the
entries of the wrap stand at the same ports, and tsncd builds the box of the body under
the wrap's layout through `broadcasted_box`, which it already did for an operator drawn
in place. The pass left such an operator alone until 2026-09-17, and the top-k selection
of the Reindex layer, which reads the candidate pool from the tape, opened no box.

### A box fed from the tape shows the tape inside its body

The Reindex layer of the V4.1 model grabs the indexer keys beside the Lightning Indexer
box, and the grab feeds that box alone. `para_wrap.to_para_wrap` absorbs it onto the port
of the box, and the block inside the box reads the keys on a wire, so the inspection box
of the indexer, opened from the Reindex layer, showed no grab. The user found that on
2026-09-17, in the Entry Gather as well. `tape_presentation.wrap_inside_boxes` now
rebuilds every plain box a wrap tapes as a `ParaBlockOperator` whose block holds those
tapes as seeds, per [[Para Block Operator]], so the tape is read at the port of the box
and again inside the body an inspection box draws. Nothing new crosses the wire and tsncd
is unchanged, because a `ParaBlockOperator` was drawn that way already. The rebuilt block
takes a tag of its own, because the blocks of a figure are keyed by the uid of their tag
and the indexer of the decoder's Full layer stands under the old tag with no seed.

A registry of explanations keyed by operator class, read by
`auxiliary_information` and sent as an expansion with no diagram, was started first and
abandoned for this design. It needed a second kind of entry on the wire and a second kind
of box in tsncd, and its text could not differ between two models.

### A named reindexing is explained through a block of the stride category

The user asked on 2026-09-17 for "a Block for reindexings, which can also carry hover /
explanation information". A `cat.Block` is generic over the category of its body, so a
block whose body is a `cat.StrideMorphism` is a morphism of the stride category already,
and no new term was added. `notebooks/display/explain_reindexings.py` is the display
pass. It replaces each named `cat.StrideMorphism` that stands in the `reindexings` of a
`cat.Broadcasted`, alone or inside a product beside an identity, by a block whose
aesthetics say `BODY_IN_PLACE`, from a table keyed by the text of the reindexing's name,
`DiagramSettings.reindexing_explanations`.

The formula of the box is written from the rows of the reindexing. A reindexing of a `cat.Broadcasted` maps each position of the result to the position of the operand it reads, so the split of the distances into blocks reads $y[i_{P}, i_{u}] = x[|u|\, i_{P} + i_{u}]$ and the sliding window reads $y[i_{x}, i_{w}] = x[i_{x} - i_{w}]$. The table therefore holds the sentences and the released lines alone. `REINDEXING_EXPLANATIONS` beside the other tables of the V4.1 model explains the window, the two group splits, the count back from an entry, the stride-one renaming and the block split. It explained the three slices of the mixing coefficients until 2026-09-17, when the projection that they cut became three linear maps, per [[Representing Models]].

The pass runs under `AdvancedDisplay.INTERACTIVE` alone, and
`notebook_diagrams.package_auxiliary` applies it after the expansions have been
assembled, because a rule that writes an operator out reads the operator's reindexings
through their `mapping` and their rows, which a block does not have. The information of
the blocks is then read off the wrapped term. The wrapping adds no `cat.Broadcasted`, so
the numbering that keys the expansions is unchanged, which
`check_a_named_reindexing_is_wrapped_in_a_block_drawn_in_place` holds. A reindexing held
in a field of an operator, as an `aops.CovariantView` holds one, is left as it stands,
because the operator's own box draws it and the operator is explained in place.

tsncd draws a block of the stride category whose aesthetics say `BODY_IN_PLACE` as its
body alone and registers the drawn reindexing as the hover region of the block. Wherever
tsncd reads a reindexing to choose how a broadcast is drawn, a block drawn in place
reads as its body.

### A `Broadcasted` is keyed by the order the importer builds it

A block has a tag with a uid, so its information is keyed by that uid. A `Broadcasted`
has no uid, and nothing sent beside the term can name one by identity. The key is the
number tsncd's importer gives the node. `TermJSONConverter.to_json` writes a term with a
uid once, into the `uid_repository`, and every later occurrence as a reference, and
writes a term without a uid in place at every occurrence. tsncd's `to_term` walks the
document depth first in the order the fields were written, follows a reference into the
repository the first time it meets it, and constructs the `Broadcasted` nodes in one
definite order. `data_transfer/broadcast_occurrences.py` reproduces that order from the
Python term, by the same rules: a term with a uid is descended into once, a term without
one at every occurrence, the fields in `Term.dict` order, and a `Broadcasted` numbered on
entry. tsncd's `json.ts` records the number on each `Broadcasted` it constructs in a
`WeakMap`.

A `Block` carries no uid of its own, so the body of a box used twice is written twice
and its operators are numbered twice, once per occurrence, and each occurrence gets its
own expansion. `websocket_transfer/validate_auxiliary_information.py` checks the
numbering against an independent walk over the exported JSON, on a model with a box used
twice inside a repeated block.

Each expansion is exported as a term of its own, with its own auxiliary information, so
an operator inside an expansion is keyed by the numbering of that export and can be
opened in turn. The three standard expansions produce primitives, so at present the
nested information is empty.

### A block carries the code it stands for

`cat.BlockAesthetics` has a trailing field `references`, a tuple of
`cat.CodeReference`, each a label, a url, a path relative to a repository root, and a
line range. `cat.Block.template` takes `references`. The field is last because tsncd
constructs a term positionally, and an older bundle ignores a fourth field, per
[[Terms Mirrored in tsncd]]. A reference with a url is linked as it stands. A reference
with a path and no url is linked under `DiagramSettings.code_link_base`, with `#L<line>`
for a web base and `:<line>` for a `vscode://file/` base, and shown as text where there
is no base. A reference whose url is on `huggingface.co` carries `icon: 'huggingface'`
on the wire, from the table `auxiliary_information.REFERENCE_ICONS`, and tsncd draws the
Hugging Face logo before its link, from a table of its own keyed by the same name.

`term_utilities/code_references.py` writes two kinds. `source_of(function)` reads the
file and the lines of a function in this repository through `inspect`, so a block
carries where it was built without the path being typed. `pinned_link(base, path, line,
end_line)` writes a link to a file at one commit of a repository on a web host, which is
how `notebooks/sota/DeepSeekV41Flash/reference_links.py` writes the links into the
released DeepSeek code. Every block of the V4.1 model carries the links of its mechanism
from the notebook's reference table. The blocks also carried the location of the function
that builds them, through `source_of`, until the user ruled on 2026-09-16 that a code
reference is not to name this package, and the embedding and the output head, which had
carried nothing else, now link the released `F.embedding`, head and softmax.
`validate_deepseek_v41_flash.check_the_code_references` holds the claim.

## The notebook setting

`DiagramSettings.advanced_display` is `AdvancedDisplay.OFF`, `LEGEND` or `INTERACTIVE`,
in `notebooks/display/advanced_display.py`. `OFF`, the default, sends the message an
older page reads, with no `auxiliary` field and neither of the two new settings.
`LEGEND` sets the `legend` setting and sends the field. `INTERACTIVE` sets
`inspectionBoxes` as well. An INLINE figure carries the legend in its image, and the
boxes exist on a page, because a captured image cannot answer a pointer. The page is the
open one under `DiagramMode.BROWSER`. Under `DiagramMode.HTML` it is a file that holds
tsncd's bundle and the message, which opens from a disk with no server and no network and
whose boxes answer the pointer as they do on the open page. The user asked for the file on
2026-09-16, [[Diagram Wire Format]] states its form, and the log is.

Under `LEGEND` or `INTERACTIVE`, `show_diagram` converts the term to the morphism the
transport sends before it assembles the auxiliary information, and passes the converted
morphism on with `recycle` cleared, because the expansions are keyed by the order of the
nodes of exactly what is sent. `notebook_diagrams.package_auxiliary` is that step.

`notebooks/sota/DeepSeekV41Flash.ipynb` draws the V4.1 model this way: the
sliding-window mode with its legend inline, the query RMSNorm and a linear map written
out with their parameters on the tape, the operators explained in place, the references
of every block, the whole model sent to the page with the boxes on, and the
same figure written to `outputs/pages/DeepSeekV41Flash.html`. The notebook's settings carry
`title='DeepSeekV4.1'`, so the page and the file are headed `tsncd - DeepSeekV4.1`.

## See also

- [[Diagram Wire Format]] — the `auxiliary` field and the two settings
- [[Diagram Display]] — how a notebook draws
- [[Terms Mirrored in tsncd]] — `CodeReference` and the trailing field
- [[Compound Axis Labels]] — the assigned sizes the legend evaluates under
