---
tags: [layer/para, concept]
code: para/data_structure/ParaBlockOperator.py, para/registries/object_lift.py, para/validate_para_block_operator.py
status: working
agent: Claude (Opus 5, 1M context, high effort, 2026-09-15)
---

# Para Block Operator

## What it is

`ParaBlockOperator(name, block, grabs, drops)` is an `ops.BlockOperator` whose tape stands
at its own ports. Each `Para.Grab` of the body becomes a leading operand of the box and
each `Para.Drop` a trailing result, so the box reads
`(*grabbed, *apparent_dom) -> (*apparent_cod, *dropped)`, and `template` returns the box
inside a [[Para Wrap]] naming the slot at each of those positions. From outside, the wrap
has the apparent domain and the apparent codomain of the block, which is what the rest of
the model composes with. `grabs` and `drops` stay on the operator as the record of which
port stands for which slot.

`block` is the block as the model wrote it, seeds included, because that is what a diagram
draws beside the box. `expose_tape_as_ports` derives the ported body from it, and the
weaves, the recorded seeds and the body `expand` lifts all come from that derivation, so
the box and the block state the same morphism and the tape is read twice over: at the
ports of the box, and inside the body drawn beside it.

A grab and a drop have the empty product on one side, per [[Para Category]], so a block
holding them has the domain and the codomain it would have without them. That is what
makes a tape a good way to pass a value between the layers of a model, and it is also what
made a box over such a block say nothing about the tape: `ops.BlockOperator.template`
draws the box and the two seeds vanish into it.

The user asked for the seeds on the operator on 2026-09-15: "The Para Drops / Grabs should
occur inside the operation boxes. We could implement a ParaBlockOperator, which notes
externally the Para Drops / Grabs that occur internally, and properly broadcasts them."

Recording the seeds on the operator alone was the first form, and the user replaced it the
same day with the ports: "Inside the wrapped block, there are some number of Para Grabs and
Drops. The domain of the ParaOperationBox is (\*internallygrabbed, \*apparentdomain) and
the codomain is (\*apparentcodomain, \*internallydropped). Then, this is placed in a
ParaWrapped which imitates this structure, placing the tapes at the appropriate locations.
So, the ParaOperationBox is nearly always ParaWrapped. This means we don't need to look
inside the ParaOperationBox to see the grabs / drops it uses."

## A plain box taped by a wrap

`para_wrap.to_para_wrap` writes a grab that feeds a plain `ops.BlockOperator` onto the
port of the box, as a `ParaWrap` over the box, and the block inside still reads the
operand on a wire. The user found on 2026-09-17 that the body drawn for such a box shows
no tape, in the Lightning Indexer of the Reindex layer of DeepSeek-V4.1-Flash.
`box_with_wrapped_tapes_as_seeds` rebuilds the wrap as the wrapped `ParaBlockOperator`
that states the same morphism. The block of the rebuilt box is
`ParaWrap(block.body, grabs, drops).to_base()`, which is the grabs, the body and the
drops composed, so it holds each tape as a seed. The grabbed operands are moved to the
front of the box and the dropped results to the end, which is the order of ports
`wrap_box` reads, and the weaves and the reindexings move with them. A seed inside the
block carries the array of the block at that port, and where the box is broadcast over a
degree the array at the port of the box carries the degree as well, as the section on
broadcasting below states. The block takes a tag of its own, whose id is
`fd.hash_id` of the old id and the entries of the wrap, so rebuilding the same wrap twice
gives the same block and the block standing elsewhere under the old tag is a different
one. `notebooks/display/tape_presentation.py` applies the rebuild when it presents a
term, per [[Para Wrap]], and `para/validate_para_block_operator.py` checks it, with and
without a degree.

## The mathematics

A tape slot names a value, and a `Grab<s> : I -> A` reads it while a `Drop<s> : A -> I`
writes it. Moving a grab to a port is the composition read the other way: a body
`g : X -> Y` holding `Grab<s>` is `(Grab<s> * id_X) ; g'` for the body `g' : A x X -> Y`
that reads the grabbed array on a wire, and a body holding `Drop<s>` is
`g'' ; (id_Y * Drop<s>)`. `expose_tape_as_ports` performs that factorisation on the body's
graph, where a seed is a subgraph of its own and the wire it carries becomes the port, and
`wrap_box` writes the two seeds back as the wrap's entries. `ParaWrap.to_base` therefore
gives the original taped morphism, so the wrapped box and the boxed body state the same
expression. The factorisation is derived rather than stored: `block` keeps the seeds and
`expand` runs `expose_tape_as_ports` again to get the body it lifts.

The factorisation opens no scope. A seed standing inside a nested block of the body would
have to move its wire across that block, so `expose_tape_as_ports` raises
`TapeBelowTheTopLevel` instead, naming the slots it found.

Broadcasting the box over a degree $\vec{h}$ is the second half of what the user asked
for. `expand` lifts the body over the degree through `lift.morphism_object_lift`, per
[[Discovering Broadcasts]], which prepends the degree axes to every array inside the body.
A grabbed array therefore arrives as `R[h, x, d]` where the unlifted one was `R[x, d]`,
and the slot is unchanged, because the slot names the value and the degree names which
member of it one index of the broadcast reads. The whole broadcast reads one array
carrying the degree, which is what a kernel would load. A grabbed operand is read at every
position of the degree for that reason, so
`broadcast_para_block_over_axes` gives it the whole degree as its reading and takes one
reading per apparent operand from the caller, as
`discovering_broadcasts.broadcast_block_over_axes` does.

`broadcast_grabs(parent)` and `broadcast_drops(parent)` report the seeds at the arrays the
ports carry, reading either the wrapped box or the box.

A pass that rewrites `block` has to leave the seeds in it, and their arrays alone, because
the weaves of the box were built from the factorisation of the block as it stood and
`expand` factors it again. A pass that changes what the seeds carry has to rebuild the box
through `bare_box`.

## Reading the slots of a term

`slots_grabbed(target)` and `slots_dropped(target)` answer which slots a term reads and
writes. A term states its tape in one of two forms, and both are read: a seed of its own,
which is the form the algebra works on, and an entry on the grab or drop side of a
`ParaWrap`, which is the form a boxed block is built in and the form
`para_wrap.to_para_wrap` puts an ordinary operation into. A notebook therefore asks the
same question of a model, of a block and of a drawn figure.

`tutil.type_search(Para.Grab, term)` finds the seeds the operator records and no longer
finds a seed inside the body, because there is none there. Ask through `slots_grabbed`
rather than searching for the class.

## `morphism_object_lift` reached no tape seed before 2026-09-15

`lift.morphism_object_lift` matches on `cat.Block`, `cat.Rearrangement`, `cat.Composed`,
`cat.ProductOfMorphisms` and `cat.Broadcasted`. A `Para.Grab` is a `cat.Morphism` and none
of those, so the match fell through every case and the function returned `None`. Lifting a
block that held a grab returned a block whose body held `None` in place of the seed, and
the failure surfaced later as `AttributeError: 'NoneType' object has no attribute 'dom'`
from whatever next read the term.

`lift.OBJECT_LIFTS` is now a registry of one rule per class that carries its objects in a
field of its own, `lift.register_object_lift` declares one, and `morphism_object_lift`
raises `lift.MorphismNotLiftable` for a class that has none rather than returning `None`.
`para/registries/object_lift.py` registers `Para.Grab` and `Para.Drop`, and the rule is one
line: the seed with `lift.object_object_lift` applied to its `size`. The lookup walks the
MRO, so the stream, loop and reduction seeds take the rule of the seed they subclass. The
rule is no longer on the path a boxed block takes, since the body it lifts holds no seed,
and it is what lifts a seed standing beside one.

## Where it lives

| | |
|---|---|
| `para/data_structure/ParaBlockOperator.py` | `ParaBlockOperator` and its `template(block, name)`, `bare_box`, `bare_box_of` and `wrap_box`; `expose_tape_as_ports` and the `TapeAsPorts` it returns; `tape_seeds_of`; `broadcast_para_block_over_axes`, which is `discovering_broadcasts.broadcast_block_over_axes` with the tape at the ports; `broadcast_grabs`, `broadcast_drops`, `slots_grabbed` and `slots_dropped`; and the types `ParaBBlock`, a `cat.Block` over `Para.Para[cat.Array, cat.BroadcastedCategory]`, and `WrappedBox` |
| `para/registries/object_lift.py` | `lift_tape_seed_over_axes`, registered for `Para.Grab` and `Para.Drop` |
| `construction_helpers/lift.py` | `OBJECT_LIFTS`, `register_object_lift`, `object_lift_for` and `MorphismNotLiftable` |
| `para/validate_para_block_operator.py` | a body that grabs a gain, adds it to a per-head operand and a shared one, and drops the total, boxed, broadcast over a head axis and expanded |

The fields are declared in this order, which is the order a term is constructed in and the
order `tsncd` reads: `name`, `block`, `grabs`, `drops`. The first two are inherited from
`cat.Operator` and `ops.BlockOperator`.

`notebooks/base_features/BuildingAModel.ipynb` states the rule with a worked block, and
the five attention modes of `notebooks/sota/DeepSeekV41Flash/attention_modes.py` are the
model that uses it, through `construction_idioms.para_boxed`.

## How it draws

The box draws with the slot ports a wrapped operation draws with, and nothing in `tsncd`
knows about the class beyond the mirror. `ParaWrapBroadcastedBox` puts each grabbed array
on a row along the top edge of the box and each dropped array on a row along the bottom,
and runs a tape from each row to a free end with the slot's label, per [[Para Wrap]].

The body drawn beside the figure is the block as the model wrote it, so it shows its own
grabs and drops, under whatever presentation the rest of the figure is drawn in.
`tape_presentation.wrap_inside_boxes` applies `to_para_wrap` to the block of every
`ops.BlockOperator` for that reason, so under `TapePresentation.ABSORBED` the drop inside
the Reindex body sits on the `TopK` result it saves and the three grabs on the operands
they feed, and under `BOXED` each is a box of its own. The block keeps its tag through the
rewrite, so `remember_drawn_blocks` still recognises a body it has delivered.

A reader of the figure therefore meets each slot twice, at the ports of the box and inside
the body, and that is what the user asked for: "The ParaBlockOperator (eg FullAttention)
you should still show the drops internally.

`tsncd` mirrors the classes of every `data_structure` folder, folder for folder, and a
class with no mirror stops the whole diagram from transporting wherever in the term it is
nested, per [[Terms Mirrored in tsncd]]. `src/para/data_structure/ParaBlockOperator.ts` is
the mirror, constructing positionally in the field order above, and
`display/Framework/para/ParaCategoryRenderer.ts` registers `BlockOperatorBox` for it,
because the operator registry keys on the exact constructor name and does not walk
superclasses.

[[Agent Display]] prints the wrapped box as the operation it is, with the tape in place:
`%9, <sel_B> = ParaBlockOperator<Rex>(<pool>[{x}, {C}], <ik_B>[{B}, {d}], <ckv_B>[{B},
{c}], %8[{x}, {m}]) : R[{x}, {m}], Nat[{x}, {s|x}]`. The listing rule for a `ParaWrap`
needed nothing added.

## Gaps

- **A derivative rule for the box does not exist.** `para/registries/derivative.py` has no
  entry for `ops.BlockOperator`, so a boxed body is differentiated by the opaque rule. A
  body that reads a slot has a reverse pass that writes one, and nothing derives it.
- **Removing a degree axis from a Para body has no rule**, per the gap in
  [[Discovering Broadcasts]], so a Para body is written without the degree axes by hand and
  handed to `broadcast_para_block_over_axes` rather than discovered from the written-out
  form.
- **A pass that rewrites the block of a box can put it out of step with the ports.** The
  hazard is stated under the mathematics above, and nothing checks it.
- **A pass that matches a root against `cat.Broadcasted` meets the wrap first.** Every box
  of a model built this way stands inside a `ParaWrap`, so a pass that rewrites a box has
  to enter the wrap. `para_sparse_expansion` was the case checked, and it needed nothing,
  because the one compressed selection of DeepSeek-V4.1 stands in a box that touches no
  slot. The next pass to run over a wrapped box may not be so lucky.
- **The slot labels of several grabbed arrays on one box crowd each other.** A label is
  placed to the left of its own tape and the axis names to the right of each tape, so with
  three grabbed arrays on one box the second label lands over the first array's axis
  names. It is the gap [[Para Wrap]] records for labels in general, now reached by a box
  with four ports.
- **The seeds recorded on the operator are not rewritten with the box.** A pass that
  substitutes the weaves of a box, as `_expand_inside_the_box` does, leaves `grabs` and
  `drops` carrying the arrays the ports had when the box was built. The ports and the
  wrap's entries are what a reader should use.

## See also

- [[Para Category]] — the two seeds, and why they are enough
- [[Para Wrap]] — the wrap the box is returned in, and how a tape is drawn on an operation
- [[Discovering Broadcasts]] — the broadcast of a block, and the expansion `expand` calls
- [[Weaves and Degree]] — the weaves the broadcast form is built out of
- [[Training]] — what writes a slot
- [[Terms Mirrored in tsncd]] — the mirror the box needs before it draws
