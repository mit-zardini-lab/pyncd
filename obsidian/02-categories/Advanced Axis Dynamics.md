---
tags: [layer/categories, concept]
code: algebra/registries/accumulator.py, advanced_axis_dynamics/data_structure/AffineGuards.py, advanced_axis_dynamics/data_structure/AxisConcatenation.py, advanced_axis_dynamics/data_structure/Operators.py, advanced_axis_dynamics/algebra/mark_sparse_domains.py, advanced_axis_dynamics/algebra/mark_sparse_codomains.py, advanced_axis_dynamics/algebra/disentangle_reindexings.py, advanced_axis_dynamics/algebra/concatenation_expansion.py, advanced_axis_dynamics/algebra/drag_index_backwards.py, advanced_axis_dynamics/algebra/move_reads_backwards.py, advanced_axis_dynamics/algebra/slide_causal_reads_backwards.py, advanced_axis_dynamics/algebra/absorb_linear_maps.py, advanced_axis_dynamics/registries/part_combination.py, advanced_axis_dynamics/registries/derivative.py, advanced_axis_dynamics/validate_advanced_axis_dynamics.py, advanced_axis_dynamics/validate_covariant_broadcast.py
status: partly implemented
agent: Claude Opus 5 (1M context), reasoning effort medium, 2026-09-15; extended by Claude Opus 5 (1M context) at reasoning effort high, 2026-09-15; the concatenated axis by Claude Fable 5.1 at reasoning effort high, 2026-09-15; the index dragged backwards by Claude Fable 5.1 at reasoning effort 80, 2026-09-25
---

# Advanced Axis Dynamics

## What it is

A codomain index of a `StrideMorphism` outside `[0, size)` of its axis names no position,
and a read there yields the universal unit of [[The Universal Unit]]. That rule belongs to
[[Stride Category]] and is one sentence long. What follows from it is a feature of its
own, and `advanced_axis_dynamics/` holds it.

Three things follow. An array read through such a morphism carries an axis some of whose
positions hold the unit, and which positions those are is an affine form of the position,
so the form has to be **derived** from the row that read it. A further read, a fold or a
merge of a derived axis has its own empty positions, so the form has to be **carried**
through the rest of the expression. And an operation that reads two axes with two
different forms at once has no one form to read, so the axes are concatenated and the
form of each is **restored** by rewriting every consumer of the concatenation.

The feature is written so that the core depends on none of it. `data_structure/`,
`graphs/` and `algebra/` import none of it. The folder is reached by `deepseek/`,
`caching/`, the quantisation rule of a concatenation, the listing of a concatenated
axis, and the model packages under `notebooks/`, whose masks read at a negative
stride. A change here therefore leaves the validations of the core untouched, which is
why the requester asked on 2026-09-15 for the machinery to be moved out of
`data_structure/StrideCategory.py`. The feature also holds the crawls that carry an
index or a read backwards through an expression, because a read of an axis at a
shift is the object those crawls move.

## The axis a row states

`AffineSparseAxis` in `advanced_axis_dynamics/data_structure/AffineGuards.py` is an axis
whose position `j`, at positions `i` of its `guides`, holds a value where

$$0 \le \sum_k \sigma_k\, i_k + \sigma\, j + \beta < \text{extent},$$

and the unit elsewhere. The form is the row of the stride morphism whose read produced the
axis. The number of live positions is a floor of the form and is not affine, so the axis
carries the form and derives the count. It prints as `w|x`, the axis letter, a bar and the
guide letters. Since 2026-09-27 its name carries the code form of the axis it replaces,
so the legend of a figure names `w|x` by the code name of `w`. A merge that carries such
an axis past the axis guiding it writes the axis onto a fresh one of the same body, code
form and size, so the slots `r|b` of DeepSeek-V4.1-Flash carried into the queries come
out as `r|x` under the code name of `r`. [[Padding and Masks as Sparse Axes]] states the
three reads that produce one and the rules the unit's laws give.

| name | what it is |
|---|---|
| `AffineSparseAxis` | the axis, with `guard_form`, `empty_end`, `selected_slots` and the two `crosses_` tests |
| `EmptyEnd` | which end of the axis holds the unit: `FIRST`, `LAST` or `BOTH` |
| `reads_before_start`, `reads_past_end` | whether the form leaves its range at some position of the domain box, evaluated at the corner most favourable to leaving it, every size symbol taken as a positive integer |
| `extent_is_independent` | whether a codomain axis's size shares no symbol with the row and its domain sizes, in which case the row sizes that axis and never reads past its end |
| `marked_position`, `sparse_axis_for_row` | the last domain axis a row reads, and the axis that replaces it |

The tests sit beside the axis rather than in `algebra/` because the axis's own
`crosses_start`, `crosses_end` and `empty_end` are written in them.

`advanced_axis_dynamics/algebra/write_guard_ranges.py` writes the positions of the axis
that hold a value at given indices of its guides, as an interval where the stride is 1
or -1 and as a condition for every stride, for the line of indices an inspection box
draws under a formula. [[Indices of Inspection Box Formulas]] states the rules, which
the user set on 2026-09-28.

## Deriving and carrying a form

`advanced_axis_dynamics/algebra/mark_sparse_domains.py` derives the form from a read.
`rows_reading_outside` finds the rows whose form leaves its axis, each marks the last
domain axis it reads, and `mark_sparse_domain` performs the replacement, descending a
composite from its last factor back to its first. `guarded_view` is the `ops.View` of a
reindexing with its domain marked, and is what a model calls.

`ops.Elementwise.template` applied the marking to every reindexing it built until
2026-09-15, so every view in the repository acquired a guard whether its model wanted one
or not. It now composes its reindexing with the base shape and marks nothing. The guard
is opt-in, through `guarded_view`, and the consequence is recorded under *What the move
changed* below.

Two rules carry a form further. `pulled_back_sparse_axis` substitutes a row into the form
of a codomain axis that already carries one, so a read of a guarded axis guards its own
domain, and a block split reading the reachable entries marks its offsets.
`sparse_axis_after_fold` substitutes the folded axis's most favourable position, so a fold
over a guarded axis leaves the form on its last guide. `live_positions` evaluates a form
at concrete positions and sizes, which is how a check compares a derived set against a
reference implementation's.

## A merge read covariantly

`CovariantView` in `advanced_axis_dynamics/data_structure/Operators.py` reads a stride
morphism the other way: the output at the position the rows compute is the input at the
domain position. The read is a function on positions when the morphism is an injection
from its domain box, and `merge_groups` in
`advanced_axis_dynamics/algebra/mark_sparse_codomains.py` tests that by finding an order
of the rows in which each row writes its own domain axes as a signed mixed-radix number,
with a shift, beside the axes the rows before it determine.

A codomain position no live domain position writes holds the unit, and
`mark_sparse_codomain` derives the `AffineSparseAxis` each partly written codomain axis
becomes. It writes every constraint on the domain, the range of each group and the form of
each domain guard, over the codomain. A constraint reading a group's digits in proportion
to their signed radices reads a multiple of the group's number, which is exact, and a
constraint reading the coarsest digit alone at plus or minus one reads a floor of it, and
$\lfloor y \rfloor \ge m$ is $y \ge m$ for an integer $m$. A constraint that holds over
the whole box is dropped, and so is one another implies. `merge_carrying` lets the axes a
merge is broadcast over enter the marking, and an axis guided by one the merge consumes is
written onto a fresh axis of its body, so the slots `r|b` of the lightning indexer leave
the merge of `(b, a)` into `x` as `r|x`.

A codomain position the image misses is left dense where a re-guided broadcast axis's form
is negative throughout that position. The constraint the image states is then dropped for
the constraint the slots state, which implies it, and the emptiness is recorded on the
slot axis rather than on the axis the merge produces. The indexer's merge writes no query
below $|a| - 1$ and writes $|a| - 1$ positions past the last query, and every slot of an
unwritten query is empty at every distance, so `x` stays dense and `r|x` carries the
condition. The same merge with no slot axis to carry the form marks `x`, which
`advanced_axis_dynamics/validate_covariant_broadcast.py` checks beside the rest, so the
slots are the reason the query axis stays dense.

The input need not carry every axis the reindexing merges.
`CovariantView.broadcast_over_absent_axes_and_merge` takes the axes the input does carry,
which stand at the head of the domain, and broadcasts the input over the rest, so one
input position is written to one output position per index of the absent axes. The image
of the whole domain box is what the marking reads, so the absent axes enter `merge_groups`
and `mark_sparse_codomain` beside the axes the input carries, and
`check_input_axes_lead_the_domain` rejects input axes that are not the leading domain
axes. `CovariantView.template` merges an input carrying one axis per domain axis and
raises `InputAxesDoNotMatchTheDomain` for an input carrying fewer, so a call says which of
the two it is. The lightning indexer is the case: its merge reads `[b, r|b, d]`, its
reindexing merges `(b, a)` into `x`, and the $|a|$ queries of a group all read the same
entry at the same distance.

[[Stride Category]] states the category the morphism lives in.

## Splitting a reindexing into independent maps

A row of a `sc.StrideMorphism` reads the domain axes it takes a stride along and no
others, so a morphism whose rows read disjoint sets of axes is several maps written as
one. `disentangle_reindexings.disentangle_reindexing` returns the product of them. It
partitions the rows and the domain axes into connected components, a row joined to every
axis it reads, and emits one `sc.StrideMorphism` per component, an identity where a
component is the identity on its axis, and the `pc.Rearrangement`s that carry the domain
and the codomain between their own order and the grouped order where the components
interleave. The result is equal to the input as a map. It draws as one pentagon per
factor with a straight wire through every axis that factor does not touch, where the
whole morphism drew as one hexagon reaching across every wire at once.

The recurring case is a map that mixes one real reindexing with axes that pass through.
`aops.degree_reindexing` writes one unit-stride row per degree position, so a merge
broadcast over the slots and the key width re-guides the slots and names the key width on
a row of its own. Disentangled, the re-guiding is a one-row morphism on the slot wire and
the key width is an identity, which is what the lightning indexer's figure shows.
`CovariantView.template` and `CovariantView.broadcast_over_absent_axes_and_merge` build
their degree reindexing through `degree_reindexing`, so both are disentangled.

| name | what it is |
|---|---|
| `disentangle_reindexing` | the product of the independent factors of a `sc.StrideCategory` morphism, recursing through a composite and leaving a `pc.Rearrangement` alone |
| `connected_components`, `ReindexingComponent` | the domain positions and the rows grouped, a row joined to every axis it takes a stride along, ordered so that a morphism whose components do not interleave needs no rearrangement |
| `states_the_identity` | whether a component's rows are the identity on its axes, which is the condition for emitting an identity |
| `sliced_component`, `factor_of_component` | one component's axes and rows, and the morphism or identity they state |

The split is applied to a merge's degree reindexing and nowhere else.
`ops.View.template` composes its reindexing with the shape it reads and is left alone,
because a split changes the term every later pass reads. Whether a view should be
disentangled too is the gap below.

## A concatenation restores two forms

Two axes with two forms cannot be read as one axis with one form. The attention core of
DeepSeek-V4.1-Flash scores each query against the window slots `w|x` and against the
selected entries' slots `s|x`, and one softmax runs over the two together. The union of
two runs under two forms is no affine form of a position of either axis, so no
`AffineSparseAxis` states it.

`ConcatenateAxes` states the pair instead. It is the multi-input `CovariantView`: one
stride morphism per input, each reading one axis covariantly into one codomain axis at
unit stride and at the offset the sizes of the parts before it sum to. The images are
therefore disjoint runs in order, and they fill the codomain axis because its size is the
sum of the parts' sizes. `check_parts_fill_the_axis` states exactly that and raises
`PartsDoNotFillTheAxis` otherwise. The concatenated axis carries no guard, and the
expansion restores each part's own guard rather than summing them.

A concatenation may also fill a declared axis. `ConcatenateAxes.template(...,
concatenated=c)` accepts a raw axis whose size equals the sum of the parts, as
`DeconcatenateAxes.template` already did, and the result then carries that axis rather
than a fresh `ConcatenatedAxis`. The rotation boxes of the integrated DeepSeek-V4.1-Flash
cut the channel axis `c` into the channels left alone and the channels rotated and join
the two runs back onto `c`, so a box over `c` returns `c` and every wire signature of the
model is kept. Without the argument the fresh axis replaced `c` across a whole attention
mode, which is the negative result of.

### The concatenated axis references its parts

The axis a concatenation produces is `AxisConcatenation.ConcatenatedAxis`, in
`advanced_axis_dynamics/data_structure/AxisConcatenation.py`. It is an `sc.Axis` with one
field of its own, `parts`, the axes laid end to end in order. Its `_size` is written by
`__post_init__` as the sum of the parts' sizes, on construction and on every
reconstruction, so the field the package reads through `local_size()` is always the sum,
and a configuration that sizes the parts sizes the concatenated axis to the integer they
sum to. Its uid carries no name. Its label is the labels of the parts joined by
`PART_SEPARATOR`, which is a plus sign between spaces, so the window slots `w|x` beside
the selected slots `s|x` concatenate into `w|x + s|x`.
`agent_display.morphism_ir.axis_name` derives that label for a listing, and tsncd
derives it for a wire through a processor registered on the class, per
[[Terms Mirrored in tsncd]].

The parts are references because composition rewrites them. A model writes the
concatenation over the axes it has in hand, and `construction_helpers.composition.align_axis`
then replaces a part by the axis it meets, so the raw window axis `w` the V4.1 core is
written with becomes the guarded `w|x` the window view produces when a mode composes the
two. A `fd.Context` rebuilds every field, so the concatenated axis is rebuilt with its parts
and the label follows. Until 2026-09-15 the axis was a fresh `cat.RawAxis` named `w+s` at
construction, and that name stayed after composition had replaced both parts, so a mode
drew `w+s` where its slots were `w|x` and `s|x`. The requester ruled that a concatenated
axis should show its parts, `w|x + s|x`, reference the axes it is made from, and take its
size from theirs.

| name | what it is |
|---|---|
| `ConcatenatedAxis` | the axis, with `parts` and a `_size` written from them |
| `PART_SEPARATOR` | the string between two part labels, read by `agent_display` and mirrored in tsncd |
| `concatenated_axis_over` | the axis `ConcatenateAxes.template` uses: a fresh one over the parts, or the one a caller hands it, checked by uid to be over the same parts |
| `ConcatenatedAxisHasOtherParts` | raised for an axis handed over that is not a `ConcatenatedAxis` over the shapes' parts |

`ConcatenateAxes.template` takes the axis to reuse as `concatenated`, because the streamed
case of the expansion builds a second concatenation onto the wire the first one produced,
and the consumers downstream name that axis by uid.

`concatenation_expansion.expand_concatenations` is the rewrite

$$F(\mathrm{Concat}(x, y)) = B_F(F(x), F(y)),$$

which holds whenever `F` treats the positions of the concatenated axis one at a time.
`ConcatenatedAxisRole` names the two cases the rewrite recognises.

| what `F` does with the concatenated axis | how it is recognised | `B_F` |
|---|---|---|
| computes at each position on its own | the axis stands in the degree, and every operand reads that degree position at its own index | a `ConcatenateAxes` again, along the same axis |
| folds the axis away | the axis stands in the target of an operand and in no output | the fold `part_combination.combination_for` reads from `algebra.registries.accumulator`: `ops.AdditionOp` for an `Einops`, `ops.Maximum` for a `Maximum` |
| reads two positions at once | the axis stands in the target on both sides, or twice in one operand's target | nothing: `ConcatenatedAxisIsNotStreamed` is raised, which is what a `SoftMax` over the concatenated axis gets |

## The combination of the parts is the accumulator of the fold

There is one question here rather than two: how the results of an operator over part of
an axis combine into its result over the whole of it. A fold run in pieces divides the
axis into runs, and a concatenation divides it into the parts that were laid end to end,
and in both the operator combining two partial results is the same. `+` is the accumulator
of a sum, so a contraction over a concatenated axis adds the contractions over the parts.

`algebra/registries/accumulator.py` holds that table, one layer below its callers, with
three rows: `ops.Einops` and `ops.Linear` to `ops.AdditionOp`, and `ops.Maximum` to
`ops.Maximum`. `accumulator_for` walks the type's MRO and returns `None` where no rule
declares a fold. `part_combination.combination_for` raises `FoldHasNoCombination` on that
`None`, because a rewrite that cannot name the fold would produce an expression computing
something else. `ops.SoftMax` and `ops.Normalize` have no row anywhere, each reading
every position of the axis to write every position, so neither has a partial result at
all.

`part_combination` refuses a `ops.Linear` although the table declares its fold. The
operator holds its weight rather than naming it, so cutting the concatenated axis into
parts would cut the weight with it and the two parts would carry one name. A fold run in
pieces over a `ops.Linear` divides the same axis and keeps one weight, which is why the
accumulator stands in the registry and the concatenation is the case that refuses.

The requester asked for the single source on 2026-09-15, and the two rows
`part_combination` held of its own until then are deleted.

Every consumer of the concatenated wire changes, so the rewrite works on the hypergraph,
as [[Sparse Expansion]] does. One pass replaces each consumer by one root per part and one
root for the combination, on the consumer's own wires, and leaves out every concatenation
nothing reads. A streamed consumer makes a new concatenation, so a chain of streamed
operations between the concatenation and the fold unwinds one operation per pass. A
consumer one of whose operands carries the concatenated axis without arriving from a
concatenation is left for a later pass, which is how the contraction against the
concatenated values waits for the exponential's own concatenation to exist.

The attention core is the worked case. The requester wrote it on 2026-09-15 as three
operands, the queries `[x, c]`, the window latents `[x, w|x, c]` and the selected latents
`[x, s|x, c]`, with `hold` on the queries beside the concatenation of the two kinds of
latent, so that the core itself reads `[x, c]` and `[x, w|x + s|x, c]`. It holds one
contraction `x c, x t c -> x t` of the queries against the concatenated latents, one
exponential, one denominator sum `x t -> x` on one copy of the result, one contraction
`x t, x t c -> x c` against the same latents on the other copy, and the reciprocal of the
denominator against that.

Every one of those is a consumer the rewrite recognises. The slot axis `t` stands in the
degree of the contraction that produces the scores, because an `ops.Einops` writes the
axes it does not contract as degree positions, so that contraction streams over `t` and
its results concatenate again. The axis stands in the target of the sum and of the second
contraction, which consume it, so those two fold and combine by addition. Expanded, the
core is the two-branch core `attend_over_all_heads_with_entries` writes by hand in
`notebooks/sota/DeepSeekV41Flash/attention_core.py`: each branch contracts the queries
against its own latents, exponentiates its own scores, sums its own denominator and
contracts against its own latents, and an addition joins each pair.
`advanced_axis_dynamics/validate_advanced_axis_dynamics.py` builds both forms and
compares their `agent_display` listings.

## A concatenation crosses a block by writing it out

The requester's core is computed once per head. The queries carry the head axis and the
latents do not, because one latent per slot is shared by every head, so the core is a
`ops.BlockOperator` whose degree is the head axis and whose reindexings read the queries
at the head and the latents whole, which is the form [[Discovering Broadcasts]] builds
and confirms. The concatenation stands outside that box.

It has to stand outside. A box computes its body once per index of its degree, so a body
holding the concatenation would concatenate the latents once per head, and
`discovering_broadcasts.confirm_broadcast_expansion` refuses such a body against the core
written out over every head. That refusal was before this rewrite existed.

`expand_concatenations_through_blocks` therefore leaves the concatenation where it is and
writes the block out. `expand_blocks_reading_concatenations` replaces every broadcast of
a `ops.BlockOperator` one of whose operands carries a concatenated axis by
`discovering_broadcasts.expand_broadcast_of_block`, which lifts the body over the degree
and reads each operand through the `ops.View` of its own reindexing, and drops the block
that held the body. The concatenation and its consumers then stand in one scope, and
`expand_concatenations` rewrites them.

The rule being applied is the one the requester stated,
$x \mathbin{\vdots} y \gg F = (x \gg F \times y \gg F) \mathbin{;} \mathrm{concat}$,
read from the concatenation towards its consumers. The left side is the box reading the
concatenated array once per index of the degree. The right side is the body reading each
part, once per index, with the concatenation of the results standing where the box's
result stood. Writing the box out is what puts one copy of the concatenation in front of
the consumers of each part, and the rewrite then unwinds the chain of streamed
operations one at a time as it does in any other scope.

An operand shared by every index of the degree is read through a repeat, which is the one
shape a lift gives it, so the expansion of a box carries one repeat per part. The
comparison against the two-branch core therefore runs after
`discovering_broadcasts.normalise_for_comparison`, which absorbs a repeat into every
operation that reads it. The flat form, where the core is written out over every head and
no box stands, needs no such absorption and matches the two-branch core listing for
listing.

## The reverse derivative

`advanced_axis_dynamics/registries/derivative.py` registers the three operators into
`para.registries.derivative.RULES`, the way `deepseek.registries.derivative` registers a
selection's, and `deepseek.registries.derivative` imports it because a merged selection
holds a `CovariantView`. The three operators are linear, so none declares a residual. A
merge reverses into the `ops.View` of its reindexing, lifted over the degree. A
concatenation reverses into the `DeconcatenateAxes` of the same part reindexings, so
part `k` receives the positions `[offset_k, offset_k + |part_k|)` of the one incoming
cotangent. A deconcatenation reverses into the `ConcatenateAxes` of the same part
reindexings.

## A deconcatenation cuts an axis into its parts

`aops.DeconcatenateAxes` is the operator of the cut. It has one input, whose target is
the axis being cut, and one output for each part, and it holds the part reindexings a
`ConcatenateAxes` of the same parts holds. A concatenation reads those reindexings
covariantly, from each part onto the whole axis. A deconcatenation reads them
contravariantly, from the whole axis into each part, which is the direction an
`ops.View` reads. The user asked for it on 2026-09-17, for the 512 channels of a
DeepSeek-V4.1-Flash latent, which are the 448 channels the rotary embedding leaves
alone followed by the 64 it rotates.

`DeconcatenateAxes.template` takes the shapes of the parts, as `ConcatenateAxes.template`
does, and the axis being cut as `concatenated`. The axis defaults to a fresh
`ConcatenatedAxis` over the parts. A model that holds its array on a declared axis
passes that axis, and `check_parts_fill_the_axis` raises where the sizes of the parts do
not sum to its size, so the first part of the latent is declared at the size
`|c| - |z|`.

`advanced_axis_dynamics/registries/standard_expansions.py` registers the standard
expansion into `algebra.registries.standard_expansions`. The expansion is a copy of the
input followed by one `ops.View` per part, each reading the whole array through that
part's row beside the identity on the degree. That expansion was the body of the
concatenation's reverse rule until 2026-09-17. `algebra.define_by_expansion`
pairs the operator with it in a `cat.DefinedExpression`, which a figure draws as the
operator, `:=` and the expansion, per [[Product Categories]].

## An index dragged backwards

A morphism $F$ lifted over an axis $x$, written $[F; x]$, computes $F$ once at every index of $x$, so its result read at one index $i_t$ of $x$ is $F$ computed on the input read at $i_t$: $[F; x](z)[i_t] = F(z[i_t])$. The requester asked on 2026-09-25 for that rule to be applied by a reverse crawl, so that one index of one axis of a result is carried back through an expression to its inputs, and `advanced_axis_dynamics/algebra/drag_index_backwards.py` is the crawl. The rule holds for every `Broadcasted`, because the operator is broadcast over its degree and each reindexing maps a degree index of the result to the degree index of the operand read by it. A fixed index of the result therefore fixes the index of every operand read through a row of that index.

An axis of an array is pinned at an index when the result being computed reads that
axis at that one index. The pins of a wire are one entry per axis, the index or `None`.
`IndexPinCrawler` is a `ReverseCrawler` over **Br** whose guide is the pins of each
wire, per [[Crawlers]]. At a `Broadcasted` it drops the pin of every target position,
because the operator reads the whole of the array it receives, and carries the pins of
the degree through each reindexing with `ReindexingPinCrawler`, a `ForwardCrawler` over
**St**, because a reindexing maps the degree of the result to the degree of the operand.
The two directions are the contravariance of an operand in its reindexing. The crawl
over the expression runs from the codomain to the domain, and the crawl through each
reindexing runs from its domain to its codomain. A row of a stride morphism pins its
codomain axis at the value taken by the row where every domain axis read by the row is pinned,
and leaves the codomain axis free otherwise. A row reading no axis pins its codomain
axis at its shift. A `ops.BlockOperator` computes its body on the targets of its
operands, so the pins of its target positions are carried through the body and out to
the operands, and only a pin on a target the body itself consumes is dropped. Every
box of GLM-5.3 is built with every axis of the box in its target, so without that rule
the index would stop at the gathers, the indexer and the rotations.

A wire read by several operations carries the pins those operations agree on and is
free wherever two of them differ. The requester stated the rule: an index that meets an
expression at several points, at several result slots or through a copy, continues only
where every one of them holds the same value. `Crawler.merge_guides` in
`graphs/processing/hypergraph_crawler.py` was added for it. Every crawler had merged the
guides of a wire read twice with `util.iallequals`, which raises on a disagreement, and
the merge is now a method with that default, which `IndexPinCrawler` overrides with
`agreed_pins`.

The crawl rebuilds the expression. A pinned position carries the axis `AffineGuards.axis_pinned_at` builds, an `AffineSparseAxis` with no guide, unit stride, the negative of the index as its shift and an extent of one, which is the form $0 \le j - i_t < 1$ and holds a value at $i_t$ alone. It keeps the size of the axis it replaces and is named `x[i_t]` as one body, so a wire the index reached draws with that label and tsncd needs no new term. `PinnedAxes` makes one such axis per axis and index, so every wire pinned at one index of one axis carries one term and the rebuilt expression composes. Where an operation asks for a pin the wire before it does not carry, which happens after a copy whose branches disagree and after an operator whose target the index does not cross, the crawl writes an `ops.View` between the two, with the pinned axis on its domain, the dense axis on its codomain and the identity row between them, named `[i_t]`. `DraggedIndex.stops` lists those views in the order they were written, and `DraggedIndex.domain_pins` the pins of the inputs.

| name | what it is |
|---|---|
| `Pins`, `free_pins`, `agreed_pins`, `pins_carried` | the pins of one array, every axis free, the pins several readers agree on, and the pins read off a rebuilt array |
| `row_pin`, `pins_over_degree` | the index computed by a row from pinned domain axes, and the pins of an array from the pins of a degree |
| `PinnedAxes` | one pinned axis per axis and index |
| `ReindexingPinCrawler` | the forward crawl over **St** |
| `IndexPinCrawler` | the reverse crawl over **Br**, with `read_pinned_positions` writing a stop |
| `drag_index_backwards`, `codomain_pinned_at`, `DraggedIndex` | the entry point, the pins of a codomain with one axis pinned, and the result with the stops and the pinned axes |
| `TargetPins` | what an operator does with the pins asked of its targets: a box carries them through its body, and every other operator drops them |
| `pin_guards`, `guard_pinned_at` | the step after the crawl that substitutes the index into every guard whose guide it passed through, and the guard written by the substitution |

On attention without a causal mask, the result pinned at the query index $i_t$ drags the
index through the output projection, the contraction against the values, the softmax,
the scale, the score contraction and the query projection, because $x$ stands in the
degree of every one of them. The keys and the values stand at $x'$, which the softmax
and the contraction against the values consume, so nothing pins them and their
projections read the whole state. At the copy the query branch asks for $i_t$ and the
two other branches ask for every position, so the index stops there and one view reads
position $i_t$ of the state for the queries. The rebuilt expression is attention for
one query against every key and value, which is the form a decoding step computes.

### A pinned guide

A guarded axis names the axes read by its form as its `guides`, so a pin on one of those axes reaches the form. The causal mask of `notebooks/classic/shared_mechanisms.py` reads the keys at $i_x - i_w$ and marks the slot axis `w|x`, which holds a value where $0 \le i_x - i_w < |x|$. With $x$ pinned at $i_t$ the slot axis becomes `w|x[i_t]`, the same form with $i_t$ substituted for $i_x$, which holds a value where $0 \le i_t - i_w < |x|$ and reads no guide. The name says that the axis is pinned up to the location of $x$. The requester asked for that reading on 2026-09-25, and later the same day for it to be written as a step after the crawl, because a covariant view relates its axes by a map the pin cannot state, and a step that reads the rebuilt expression can be told what it needs.

`drag_index_backwards.pin_guards` is that step. It reads the domain and the codomain of every operation of the rebuilt expression. A guarded axis that stands on no array without the pinned form of one of its guides beside it is replaced everywhere, through an `fd.Context`, by `guard_pinned_at`: the index substituted into the form, the guide dropped, and the name `w|x[t_x]`. On the causal attention the slot axis reads no guide, has the stride $-1$, the shift $t_x$ and the extent $|x|$, and at five positions with $t_x$ bound to 2 its live slots are 0, 1 and 2. The crawl itself leaves a guide alone. The row of the mask view reads the pinned $x$ and the free $w$ together, so the key and value projections read the whole state, which is a gap listed below.

Two things still stand between the substitution and a rule for every guard, and the
step leaves such a guard alone.

The guide need not be the pinned axis. An expression may batch $x$ before the guarded read, as the group view of DeepSeek-V4.1-Flash reads the queries by group and offset, $i_x = |a|\, i_b + i_a$, so the slot axis is guarded by $b$ and $a$ and not by $x$. A pin on $x$ then reaches the guides only through the reindexing that relates them, and that relation is a floor and a remainder, which no affine form states. The guides of a guarded axis are the axes its row read when the axis was marked. The axis carries no record of how those axes relate to the axes of the arrays it later stands in, so the substitution cannot be made from the axis alone.

The form need not be a window. A guard may read several guides at several strides, and a guide may be pinned at an expression rather than at an index, as $2 i_t + 1$ after a strided read. Substituting a pin into such a form gives a form in the remaining guides that is still affine. Whether that form is a window, a prefix or a condition with no name the display can give it is not decided.

## The read moved backwards

The crawl above carries an index and leaves every operator where it stands. The
requester asked on 2026-09-25 for a second process that carries the read itself, so
that the index morphism slides over the expression, composes with every reindexing it
meets and is written once where the branches of a copy agree.
`advanced_axis_dynamics/algebra/move_reads_backwards.py` is that crawl, and it is the
Yoneda trick of [[Yoneda and Cartesian Tricks]] run mechanically, from the result of
an expression towards its inputs. The requester calls the whole process Yoneda sliding.

A read is an `ops.View`, and its reindexing maps the axes of its result to the axes of its operand. The pending read of a wire is the reindexing of the view that would stand on it, an `sc.StrideMorphism` whose codomain is the wire's axes in order, or `None` where nothing is read. The index morphism is the read with no domain axis whose one row is the shift $t_x$, where $t_x$ is the index of the axis $x$, beside the identity on every other axis of the result, and `index_read` builds it. `ReadCrawler` is a `ReverseCrawler` over **Br** whose guide is the pending read of each wire, per [[Crawlers]].

At a `Broadcasted` the read splits by the output weave. A row onto a target position
must be the identity on that axis, because the operator reads the whole target, and
the rows onto the degree positions form the read of the degree, from the axes the
read returns at its tiled positions to the degree. That read composes with each
operand's reindexing, by `compose_stride_morphisms`, after `as_stride_morphism` has
written the reindexing as one stride morphism. A composite whose every row selects one
degree axis at unit stride, and reads that axis as itself, stays in the operator as its
reindexing, as a rearrangement. A row that reads one axis as another is a renaming, and
since 2026-09-26 it moves onto the operand like any other read, because a rearrangement
left in the operator would read an axis the operand's wire does not carry. Where any
row does more, the composite moves onto the operand's wire as its pending read, and the
operator keeps the projection onto the axes needed by the read, so a repeat stated by
the operator stays in the operator. The operator is rebuilt over the read's domain, with the
tiled positions of its weaves counted afresh, and a view whose reindexing is now the
identity is dropped. A read the operator cannot pass, because a target row is not the
identity, is written after it as a view, and the operands are read as they were.

A composite is named after the reindexing of the view it composed into, and after the
read where that reindexing carries no name. A view written with a `cat.Rearrangement`,
such as the diagonal of a mixture of experts, holds its name on the `ops.View` alone, so
since 2026-09-27 `reindexing_named_after_its_view` gives the rearrangement the view's
name before the composition, where `ReadCrawler.read_yields_its_name` says the incoming
read yields. A read carrying no name yields, and the derivation of a cached pass lets
its reads `New` and `Cached` yield too. `view_of` writes a named read that only copies,
permutes or deletes axes as rearrangements under the view's name, so a diagonal written
out again still draws as a dot on its wire.

The domain of a pending read is ordered by the first row that reads each axis, in
`in_reading_order`, so two branches that read one wire the same way carry equal reads
whatever the order of the degrees they came through. At a `Rearrangement` the reads
asked of the wires copied from one wire are grouped by that equality, in `read_groups`.
A wire whose branches all ask for one read carries it further. A wire whose branches
disagree stops the reads, and each distinct read is written once after the copy, as a
view whose result is copied to the branches that asked for it, which is the Cartesian
trick of the same note.

Two operators pass more, per the requester's second request of 2026-09-25. A
`ops.BlockOperator` computes its body on the targets of its operands, so the rows onto
its target positions are a read of the body's result. `through_box` splits the read
into the read of the degree, carried as for any operator, and the read of the target,
carried through the body by the same crawl and out to the operands, where
`carry_through_operand` joins the two parts into one read on the operand's wire. An
operator with several results passes a read where every result is read and the reads
of the degree agree, which the channel cut of the indexer's rotation needs. An
`ops.Arrange` writes the index of every position of its axis, so its result read
through a row is the value of that row, and `index_values` writes it. A row with no
domain axis is its shift, from an `ops.ConstantOp`, and a row over one axis is the
arrangement of that axis followed by the affine map of the row, from an
`ops.Arithmetic`. The requester stated the rule as an index that goes through an
arrangement becoming the covariant view of the index, and noted that an arrangement is
over one axis, so a row over several axes stops the read. `written_out` writes the
operators of given classes out by their standard expansions inside every box, so that a
rotary table is met as the arrangement of its positions.

| name | what it is |
|---|---|
| `Read`, `index_read` | the pending read of a wire, and the index morphism on one axis of one result |
| `compose_stride_morphisms`, `as_stride_morphism` | two stride morphisms composed by position, and any morphism of **St** written as one |
| `in_reading_order`, `same_read` | the domain order the reads are compared in, and the comparison, which ignores names |
| `split_at_weave`, `SplitRead` | the read of the degree and the weave of the result of the read |
| `carry_through_operand`, `CarriedRead` | the composite kept or moved, the operand's weave, and the projection kept by the operator |
| `read_groups`, `ReadGroup` | the branches of a copy that ask one wire for one read |
| `index_values` | the values of a read's one row over its domain, which an arrangement read through it returns |
| `written_out`, `WriteOutOperators` | the operators of given classes written out by their standard expansions, inside every box |
| `ReadCrawler`, `move_reads_backwards`, `MovedReads` | the crawl, with `through_box` and `splits_agreeing_on_the_degree`, the entry point, and the result with the reads that reached the domain and the stops |

On the attention with the causal mask the index morphism slides through the output
projection, the contraction against the values, the softmax, the scale and the score
contraction, dropping the axis $x$ from every wire, and reaches the mask, which
composes into it as the read $x = t_x - j_w$ of one axis by one axis. That read moves
in front of the key and value projections, which then run over the slot axis. At the
copy the query branch asks for $t_x$ while the key and value branches ask for the
composed mask, so the state is read twice and the mask's result is copied to both
projections. The result is one vector of width $m$, the attention of one query against
the positions before it.

On the Full mode of GLM-5.3, with its rotary tables written out first, the read enters
the query path, where the copy of the low rank agrees and the state is read once at
$t_x$. It enters the two gathers, where the relative read of the distances composes into
it as the read named Back, $x = t_x - j_r$, which moves in front of the key and value
projections. It enters the indexer, whose scoring box is broadcast over the queries and
passes it, and whose own copy of the state disagrees, the keys asking for Back and the
head weights for $t_x$, so both are written inside the box and the indexer reads the
state whole. In the rotation bodies the read meets the arrangement of the positions: the
query rotation holds the constant $t_x$ where the positions stood, and the key rotation
holds the arrangement of the distances followed by $t_x - x$.

The slot axis the composed mask returns is still `w|x`, guided by the $x$ the read has
read away, which is the pinned guide above in another form. The composite's own row
states the guard the axis should carry, $0 \le t_x - j_w < |x|$, and
`mark_sparse_domains.mark_sparse_domain` applied to the composite would derive it as a
form with no guide, because $w$ is the one domain axis the row reads. That rule is the
candidate, and it is not applied.

The derivation of a cached pass is the same crawl carrying the read of the new tokens of
a pass, per [[Deriving Caches by Dragging the New Tokens]]. `ReadCrawler.through_tape_seed`
passes a bare tape seed, which carries no read where its array holds no axis the read
names.

## The causal reads slid backwards

`advanced_axis_dynamics/algebra/slide_causal_reads_backwards.py` starts the read crawl at
every causal read of an expression, a view whose row reads its own axis at stride one
and other axes at strides of zero or less, and carries the read back until a copy whose
other branches read the operand unmasked. The result is the CausalSlide, which the user
ruled on 2026-09-26 to be the standard form of a displayed expression, per
[[Representing Models]] and [[Yoneda and Cartesian Tricks]]. Every page of
[[Website Notebooks]] draws its model in that form.
`slide_causal_reads_back_past` carries each read past a given set of operators alone, a
box among them passing its whole body. The position of the read is the placement of a
cache, per [[Deriving Caches by Dragging the New Tokens]].

The slide enters the body of every box and every `ParaWrap` since 2026-09-27. A causal
read inside a body slides to a copy inside it or to the domain of the body, and a read
at the domain of a box's body leaves the box as the pending read of the operand, so the
box then reads the result of the read. A read at an operand grabbed by a wrap from the
tape stops at the grab. Before the change the slide left a model whose causal reads sit
inside boxes unchanged, which every model of `notebooks/sota/` is. Two reads meet at a
copy only when they are equal, and each call of `mark_sparse_domains.guarded_view` mints
a fresh sparse axis. GLM-5.3 therefore marks its read back once, as
`lightning_indexer.READ_BACK`, so the key and value branches of a layer ask the latent
for one read and the read passes the copy of the latent.

A repeated block that receives no read from its results and holds no causal read passes
the slide whole since 2026-09-28, and no read reaches its operands. The loop of a scan
over the tokens is the case: each iteration reads its token at the counter, which is no
causal read, and the loop reads more arrays than it returns, so the crawl had refused it
with `RepeatedBlockChangesTheGuide`. The whole Mamba layer of
[[Carrying the State of a Scan Between Passes]] and the 69 delta layers of Kimi K3 now
slide, per [[SOTA Model Notebooks]].

A read that leaves a box whose degree is empty takes the order of the axes the read
returns, and the weave of the box is rewritten in that order. When the body reads its
operand with the axes in another order, the weave and the domain of the body then
disagree, and nothing raises. Kimi K3 met this with a convolution whose view put the taps
last. Its convolution reads `[x, w|x, *channels]` with the taps second and weighs them
with a `Linear` whose weave is written by position, so the read leaves the box with the
order the body reads, and [[Open Gaps]] records the mismatch.

A body the crawl rebuilds is a different block, and `through_box` gives it a tag derived
from its old tag and its new body with `with_the_tag_of_its_body`, as the quantisation
pass tags a quantised block. A figure draws one body per tag and a page opens one body
per tag. Before the change the rotation of the window latent of DeepSeek-V4.1-Flash,
which reads its table through the window, shared the tag of the rotation of the query
and was drawn and opened as that rotation.

## A linear map absorbed into the contraction that reads it

`advanced_axis_dynamics/algebra/absorb_linear_maps.py` finds a chain of two
contractions, the second reading the result of the first directly or through a view,
and rewrites it into the order of its operands that costs the fewest operations at
bound sizes. The view is carried onto the first contraction's operands by the read
crawl, the two are merged by `einops_rearrange.merge_einops`, and `contract_pair_first`
splits the merged contraction with a chosen pair contracted first. The operations of an
order are counted by `morphism_work.read_symbolic_work`, per
[[Operation Counts and Machine Rates]]. The search runs inside each scope of the
hypergraph, so a producer and a consumer in different blocks are not paired, and a
`Linear` whose weight is inside the operator is written as a contraction first by
`linear_expansion.expand_linear_root`. The module sits here rather than in `algebra/`
because it uses the read crawl. [[Einops Rearrangement]] states the merge, and
[[Deriving Caches by Dragging the New Tokens]] applies the rewrite to the cached pass of
multi-head latent attention, where it derives the absorb mode of DeepSeek-V3.

## What the move changed

The code was `data_structure/StrideCategory.py`, which had grown past 1100 lines, and
`CovariantView` was in `data_structure/Operators.py`. Nothing about the derivation changed
in the move, and two things about the package did.

`ops.View.template` no longer marks. Four models acquired a guard through
`ops.Elementwise.template` and no longer do: the multi-token-prediction shift of GLM-5.2,
the sliding windows of Kimi-K3 and of DeepSeek-V4-Flash, and a padded convolution.
Each states a read that does leave its axis, and each
would carry its guard again by calling `mark_sparse_domains.guarded_view` in place of
`ops.View.template`. None of them asserts anything about a guard and all four notebooks
execute, so the change is recorded here rather than applied, and the requester's ruling
that only the V4.1 notebook depends on the feature holds as long as they are left dense.

`sparse_axis_after_merge` is deleted. It was written on 2026-09-14 for a merge whose guard
reads both digits of one group at equal strides, using the identity
$\lfloor x/2 \rfloor + (x \bmod 2) = \lfloor (x+1)/2 \rfloor$, and `mark_sparse_codomain`
superseded it the next day with a general elimination that has no such case. It had no
caller from the day it was written, and `eliminate_group` raises
`GuardNotAffineAfterMerge` for the case it handled, which is the gap below.

## Gaps

- **A row reading a pinned axis and a free axis together leaves its codomain axis
  free.** The causal read $i_x - i_w$ at a pinned $i_x$ reads the positions $i_t - i_w$
  for every $i_w$, which is a window, and a window is an `AffineSparseAxis` with an
  extent rather than a pin. `drag_index_backwards.row_pin` returns `None` for such a
  row, so a drag through a causal or windowed attention frees the keys where the window
  would be right. A repeated block asserts that the pins leaving it equal the pins
  entering it, so a loop whose body drops the pin raises, and the fixed point a loop
  needs is not computed. The crawl runs on the morphism form, because the hypergraph
  form merges the readers of a wire by identity and the view needed by a stop would have to
  be spliced in. A pinned guide is left alone, per *A pinned guide* above, so the read
  crawl leaves the slot axis it moves in front of the projections guided by the $x$ it
  has read away, although the composed mask states the guard. `pin_guards` leaves a
  guard whose guide is not the pinned axis, a guard by a form the substitution does not
  simplify, and a `deepseek.SparseAxis`, which carries no form. The two gathers of
  GLM-5.3 each mark a distance axis of their own, so the latent is read twice through
  two reads that state one map over two axes, and the crawl, which compares reads by
  their axes, writes both. [[Open Gaps]] lists these gaps.
- **No public validator runs the index crawl.** `validate_advanced_axis_dynamics.py`
  checks the guarded reads, the concatenation and the reverse rules, and nothing public
  calls `drag_index_backwards`. The read crawl is checked where it is used, by the
  validators of [[Website Notebooks]], which check the CausalSlide of each model and the
  cached passes derived through the crawl.
- **Nothing reads a guarded axis as anything but a dense one.** A pass that divided such
  an axis would have to classify each part as wholly live, wholly empty or straddling, and
  none does, which is the account in [[Padding and Masks as Sparse Axes]].
- **A merge whose guard reads two digits of one group at equal strides raises.**
  `eliminate_group` handles a constraint proportional to the radices and one reading the
  coarsest digit alone, and `GuardNotAffineAfterMerge` for anything else. The halving
  identity `sparse_axis_after_merge` carried is in the git history.
- **The reverse of a `CovariantView` ignores the degree reindexing.** The rule builds the
  `View` of the merge lifted over the output degree, so the cotangent of a re-guided axis
  arrives on `r|x` where the forward pass read `r|b`. The rule also reads the reindexing's
  whole domain, so the reverse of a merge that broadcasts over an absent axis carries that
  axis where the forward pass read nothing at it, and the cotangent of the indexer's keys
  would have to be summed over the offsets of each group. No model differentiates the
  indexer yet.
- **A view's reindexing is not disentangled.** The split is applied to a merge's degree
  reindexing alone, so a view that mixes one real read with axes passing through still
  draws as one hexagon over all of them. Splitting a view changes the term every later
  pass reads, and none of them has been examined for it.
- **A concatenation crosses no loop.** `expand_concatenations_through_blocks` reaches a
  consumer inside a block by writing the block out, per *A concatenation crosses a block
  by writing it out* above, and a block whose repetition is not one is a loop that no
  rule writes out. The part wires are still never routed across a scope the way
  `deepseek.sparse_expansion` routes an index wire, so a concatenation outside a loop
  whose consumer is inside it is left alone.
- **Only `Einops`, `Linear` and `Maximum` declare a fold.**
  `algebra.registries.accumulator` has three rows, and a `SoftMax` or a `Normalize` over
  a concatenated axis refuses the rewrite for want of one. A `Linear` refuses it for the
  reason stated above.

## See also

- [[Stride Category]] — the category the reads live in, and the universal-unit rule
- [[Padding and Masks as Sparse Axes]] — the three reads and the rules
- [[Operators]] — every operator in the package, including these two
- [[Sparse Expansion]] — the rewrite a selection's index wire needs, which a guard does
  without
- [[Discovering Broadcasts]] — the box the concatenation stands outside of, and the
  expansion that writes it out
- [[Representing Models]] — how a model is written with these reads
- [[The Universal Unit]] — what an empty position holds
- [[Compound Axis Labels]] — how a guarded axis and a concatenated axis carry their assigned sizes
- [[Crawlers]] — the reverse crawler both crawls extend
- [[Deriving Caches by Dragging the New Tokens]] — the read of the new tokens of a pass,
  carried by the read crawl, and the causal read slid back as the placement of a cache
