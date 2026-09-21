---
tags: [layer/categories, concept]
code: algebra/registries/accumulator.py, advanced_axis_dynamics/data_structure/AffineGuards.py, advanced_axis_dynamics/data_structure/AxisConcatenation.py, advanced_axis_dynamics/data_structure/Operators.py, advanced_axis_dynamics/algebra/mark_sparse_domains.py, advanced_axis_dynamics/algebra/mark_sparse_codomains.py, advanced_axis_dynamics/algebra/disentangle_reindexings.py, advanced_axis_dynamics/algebra/concatenation_expansion.py, advanced_axis_dynamics/registries/part_combination.py, advanced_axis_dynamics/registries/derivative.py, advanced_axis_dynamics/validate_advanced_axis_dynamics.py, advanced_axis_dynamics/validate_covariant_broadcast.py
status: partly implemented
agent: Claude Opus 5 (1M context), reasoning effort medium, 2026-09-15; extended by Claude Opus 5 (1M context) at reasoning effort high, 2026-09-15; the concatenated axis by Claude Fable 5.1 at reasoning effort high, 2026-09-15
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

The feature is written so that nothing else depends on it. `data_structure/`, `graphs/`
and `algebra/` import none of it, and the folder is reached only by `deepseek/`, two
modules of `para/` and
`notebooks/sota/DeepSeekV41Flash.ipynb`, which is the one model in the repository that
reads at a negative stride. A change here therefore leaves every other validation
untouched, which is why the requester asked on 2026-09-15 for the machinery to be moved
out of `data_structure/StrideCategory.py`.

## The axis a row states

`AffineSparseAxis` in `advanced_axis_dynamics/data_structure/AffineGuards.py` is an axis
whose position `j`, at positions `i` of its `guides`, holds a value where

$$0 \le \sum_k \sigma_k\, i_k + \sigma\, j + \beta < \text{extent},$$

and the unit elsewhere. The form is the row of the stride morphism whose read produced the
axis. The number of live positions is a floor of the form and is not affine, so the axis
carries the form and derives the count. It prints as `w|x`, the axis letter, a bar and the
guide letters. [[Padding and Masks as Sparse Axes]] states the three reads that produce one and the
rules the unit's laws give.

| name | what it is |
|---|---|
| `AffineSparseAxis` | the axis, with `guard_form`, `empty_end`, `selected_slots` and the two `crosses_` tests |
| `EmptyEnd` | which end of the axis holds the unit: `FIRST`, `LAST` or `BOTH` |
| `reads_before_start`, `reads_past_end` | whether the form leaves its range at some position of the domain box, evaluated at the corner most favourable to leaving it, every size symbol taken as a positive integer |
| `extent_is_independent` | whether a codomain axis's size shares no symbol with the row and its domain sizes, in which case the row sizes that axis and never reads past its end |
| `marked_position`, `sparse_axis_for_row` | the last domain axis a row reads, and the axis that replaces it |

The tests sit beside the axis rather than in `algebra/` because the axis's own
`crosses_start`, `crosses_end` and `empty_end` are written in them.

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

## What the move changed

The code was `data_structure/StrideCategory.py`, which had grown past 1100 lines, and
`CovariantView` was in `data_structure/Operators.py`. Nothing about the derivation changed
in the move, and two things about the package did.

`ops.View.template` no longer marks. Four models acquired a guard through
`ops.Elementwise.template` and no longer do: the multi-token-prediction shift of GLM-5.2,
the sliding windows of Kimi-K3 and of DeepSeek-V4-Flash, and the padded convolution of
a padded convolution. Each states a read that does leave its axis, and each
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
