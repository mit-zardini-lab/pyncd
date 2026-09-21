---
tags: [layer/categories, concept]
code: algebra/discovering_broadcasts.py, algebra/validate_discovering_broadcasts.py
status: working
agent: Claude (Opus 5, 1M context, high effort, 2026-09-15)
---

# Discovering Broadcasts

## What it is

An expression that computes one body once per index of an axis can be written two ways.
It can be written out, with the axis standing in every array and in the degree of every
operation, or it can be written as one [[Broadcasted Category|Broadcasted]] over an
`ops.BlockOperator`, whose degree is that axis and whose body carries it nowhere. The
second form is one operation where the first is a dozen, and it says in the reindexings
which operands vary with the axis and which are shared by every index of it.

`algebra/discovering_broadcasts.py` builds the second form from the first and proves the
two are the same expression. The attention core of `notebooks/sota/DeepSeekV41Flash/attention_core.py`
is the case it was written for. That core scores 64 query heads against one 512-wide latent
per slot, so its queries and its learned sink carry the head axis `h` and its window
latents and its selected latents do not, and the reindexings of its broadcast form are
`((0,), (0,), (), ())`.

## The mathematics

A `Broadcasted` carries an operator, one weave per operand and per result, and one
reindexing per operand. The degree is the common domain of the reindexings, and a weave
holds a `WeaveMode.TILED` slot at each degree position it carries and an axis at each
position the operator consumes, per [[Weaves and Degree]]. Written out, the three pieces of
the broadcast form of a body $f$ over axes $\vec{h}$ are:

- each operand's weave tiles as many positions as that operand's reindexing names, in front
  of the target $f$ receives, so an operand that varies with $\vec{h}$ has
  $\lvert \vec{h} \rvert$ tiled slots and a shared one has none;
- each result's weave tiles all of $\vec{h}$, because an operation is computed once per
  index of the degree and every result of that computation is indexed by it;
- each operand's reindexing is the rearrangement of the degree onto the positions that
  operand reads, meaning the identity $(0, \dots, n-1)$ for an operand that varies and the
  deletion $()$ for one that is shared.

The expansion runs the construction the other way. `lift.morphism_object_lift` broadcasts
the body over the degree, which prepends the degree axes to every array inside it, and one
`ops.View` per operand carries that operand's reindexing composed with the rearrangement
that separates the weave's degree slots from its target. An operand whose reindexing
deletes the degree is therefore read through a repeat, because a lifted body reads every
operand at the degree and a lift gives it no other shape. A repeat is a `View` whose
reindexing does not name the repeated axis, and [[Expression Simplification]] absorbs one
into each operation that reads it, so the repeats the expansion writes disappear again into
the reindexings the original expression already had.

Deleting the degree axes from a body is the inverse lift, and it is the step that can fail.
For each operation, the axes have to stand at the head of the degree, each removed degree
position may be read at most once by one operand, and the reindexing has to be a
rearrangement. An operand that reads one removed position twice is a diagonal rather than a
broadcast, and a reindexing that computes holds the position in a row whose meaning a
deletion would change. An operation that carries a degree axis at a later position is
refused as well, because a deletion takes a leading degree alone, where an operation that
carries none of them is returned untouched. `DegreeAxesNotRemovable` reports all three,
naming the operation, its arrays and its degree.

## The confirmation is what the derivation rests on

The removal reads a block's domain and codomain, and it decides that an array carries the
degree by comparing its leading axes against the declared ones. An array whose head happens
to be those axes and which the model does not broadcast along would be read as carrying
them. The confirmation is what settles it. `confirm_broadcast_expansion` expands the
candidate, normalises the expansion and the original the same way, and compares the two
terms, so a misread array shows up as a domain that differs or as an operation that differs
and the candidate is refused with the position named. Nothing about the discovery has to be
trusted, because a candidate is only ever reported beside the comparison against the
expression it came from.

Normalising is four steps, in `normalise_for_comparison`:

1. `h2m.recycle` converts to a [[Hypergraphs|hypergraph]] and back, so the two sides hold
   their wiring in the same rearrangements.
2. Every block whose repetition is one is replaced by its body. A block whose repetition is
   not one denotes a loop and stays. The blocks go because an expansion prepends its
   repeats outside the block it lifts and `reindexing_absorption.absorb_nodes` merges
   siblings alone, so a repeat and the operation inside the block that reads it never meet
   while the block stands.
3. `absorb_nodes` folds each repeat into every operation that reads it.
4. Every reindexing that permutes, copies and deletes is rewritten as one
   `cat.Rearrangement` over the degree. Two morphisms that read the same positions may hold
   the reading as a product, as a composition or as one rearrangement, and an equality test
   on the terms answers no on a pair that states one map.

## Where it lives

| | |
|---|---|
| `algebra/discovering_broadcasts.py` | `discover_broadcast_over_axes(block, degree_axes, name)`, which removes the axes, builds the candidate and confirms it; `broadcast_block_over_axes(body, degree_axes, degree_readings, name)`, which builds a candidate from a body that already carries none of the axes; `confirm_broadcast_expansion(original, candidate)`; `expand_broadcast_of_block(candidate)`; `remove_leading_degree_axes`, `remove_grouping_blocks`, `canonicalise_rearranging_reindexings`, `normalise_for_comparison`, `name_difference` and `operations_of` |
| `algebra/validate_discovering_broadcasts.py` | the attention core written by hand with and without its head axis, discovered, and three refusals: a candidate that drops the head from the queries, a candidate with an empty degree, and an operation carrying the head at a later degree position |
| `construction_helpers/lift.py` | `morphism_object_lift`, the lift the expansion runs, per [[Construction Helpers]] |

`broadcast_block_over_axes` takes `degree_readings`, one tuple of degree positions per
operand of the body, because the body alone does not say which operands vary with the
degree. It is the user's own spelling of the core, `((0,), (0,), (), ())`.

`expand_broadcast_of_block` has a second caller.
`advanced_axis_dynamics.algebra.concatenation_expansion.expand_blocks_reading_concatenations`
writes out every box one of whose operands carries a concatenated axis, so that the
concatenation and the consumers inside the body stand in one scope, per
[[Advanced Axis Dynamics]]. A box cannot hold the concatenation instead: it computes its
body once per index of the degree and the parts are one array at every index, so
`confirm_broadcast_expansion` refuses such a body.

## `ops.BlockOperator.expand` states the same expansion and builds it wrongly

`ops.BlockOperator.expand` in `data_structure/Operators.py` composes the operand
reindexing with the weave's rearrangement, which is the right index map, and then hands
both that map and the weave's target to `ops.View.template` as the base.
`Elementwise.template` products the base's shape onto the reindexing a second time, so the
operand of the expansion arrives carrying its target axes once more than it should: the
queries of the core, whose target is `R[x, c]`, come out as `R[h, x, c, x, c, x, c]`. It
also closes with `weave.inverse_rearrangement(degree)`, which is a rearrangement of axes
standing in a product of morphisms over arrays, and whose domain is the weave's own shape
where the lifted body hands out the degree followed by the target.

Nothing in the repository calls `expand`, so neither fault has ever been reached. The
degree-free case that `ops.BlockOperator.template` builds comes out correct by accident,
because every operand view is then an identity and `make_composed` drops identities before
the doubled shape is used. `expand_broadcast_of_block` is the expansion written out
correctly, and `para.data_structure.ParaBlockOperator.expand` calls it.

## Gaps

- **A degree axis has to stand at the head.** `remove_leading_degree_axes` deletes a prefix
  of the degree and a prefix of an array's shape, and raises where an operation carries a
  degree axis at a later position. A core whose result is `R[x, h, c]` has to be rewritten
  so that the head axis leads in every array and in every `Einops` result before the
  removal will take it. Deleting an axis from the middle needs, per wire, the answer to
  whether that wire carries the axis, which is a dataflow question rather than a local
  one.
- **A tape seed inside the block is removed by no rule.** `remove_leading_degree_axes`
  raises `DegreeAxesNotRemovable` on a `Para.Grab`, where `morphism_object_lift` lifts one
  through `para.registries.object_lift`. The un-lift has no registry of its own, so a Para
  body is written without the degree axes by hand and handed to
  `broadcast_block_over_axes`, per [[Para Block Operator]].
- **The comparison is an equality of terms after four normalising steps.** Two expressions
  that compute the same thing and differ in an operator's own fields, such as an `Einops`
  signature written in a different grouping order, are reported as different. The
  confirmation therefore refuses more candidates than are wrong, which is the safe
  direction.

## See also

- [[Broadcasted Category]] — the morphism a candidate is
- [[Weaves and Degree]] — the weaves and reindexings a candidate is built out of
- [[Expression Simplification]] — the absorption that removes the repeats an expansion writes
- [[Construction Helpers]] — `morphism_object_lift`, and the registry a seed's lift is declared in
- [[Operators]] — `ops.BlockOperator` and `ops.View`
- [[Para Block Operator]] — the same broadcast over a body that reads and writes a tape
- [[Advanced Axis Dynamics]] — the concatenation that stands outside a box, and the
  second caller of the expansion
- [[Representing Models]] — where a ruling on how a model is written is recorded
