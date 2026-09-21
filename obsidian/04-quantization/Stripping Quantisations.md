---
tags: [layer/quantization, concept, algorithm]
code: quantization/algebra/strip_quantisations.py
status: stable
written: Claude Opus 5 (1M context), effort high, on 2026-09-20.
---

# Stripping Quantisations

## What it is

`quantise_model` writes a quantisation onto every wire of a model and puts a
`TypeConvert` named `cast` wherever an operation requires another. `strip_quantisations`
is the other direction. It takes a quantised model back to the expression in the reals
it was made from: every `Quantified` wrapper is taken off every datatype of every wire
and every weight, the `BlockScale` carried by such a wrapper goes with it, and every
`TypeConvert` reading one quantisation of a value into another quantisation of the same
value is deleted, with the operations reading its result redirected onto the wire its
operand came from.

A reader of a quantised figure sees two things at once: the arithmetic the model
performs, and the format each value is held in. The released code settles the formats,
and [[Quantization]] states how they reach the expression. A derivation runs on the
arithmetic alone, because a cast changes the representation of a value and performs no
arithmetic. Drawing one mechanism twice, once with the formats and once without them,
shows which marks in the figure belong to which of the two, and the stripped figure is
the smaller of the two by the count of the casts.

The functor is also the test that the pass adds and removes the same thing. Stripping
the quantised text-only DeepSeek-V4.1-Flash and reading the result gives the listing of
`text_only_model.v41_flash_text_only` after the same recycling, so the 190 casts and the
quantisation on every wire are the whole of what the pass wrote.

## Where it lives

| module | what it holds |
|---|---|
| `quantization/algebra/strip_quantisations.py` | `without_quantisations`, `converts_between_two_quantisations`, `is_a_conversion_between_two_quantisations`, the functor `StripQuantisations`, `holds_a_quantisation` and `strip_quantisations` |
| `quantization/validate_quantization.py` | seven `check_*` cases, from `check_stripping_the_model_leaves_no_quantisation` to `check_a_conversion_that_is_not_between_two_quantisations_is_kept` |
| `notebooks/sota/DeepSeekV41Flash/quantised_text_only_model.py` | `v41_flash_text_only_without_quantisations`, the stripped model a figure draws |

`notebooks/sota/DeepSeekV41Flash.ipynb` draws the stripped model beside
the quantised one, and [[SOTA Model Notebooks]] lists that notebook.

## The mathematics

A quantisation is a wrapper around the mathematics of a value, so taking it off is a map
on datatypes and the functor it induces on the expression. `without_quantisations` is the
map on datatypes. It replaces `Quantified(wraps=X)` by `X` wherever it stands, so a
quantisation of the reals leaves `cat.Reals`, a quantisation of an index leaves the
`cat.Natural` with its own bound, and a wrapper around a quantised value, as
`deepseek.data_structure.Complex` is, keeps its own form around the datatype now under
it. The map is `fd.deep_reconstruct` memoised on object identity, which visits each node
of the term once and returns the original object for a subterm holding no quantisation,
per the sharing invariant of [[Invariants]].

`StripQuantisations` is a `graphs.processing.hypergraph_functor.Endofunctor`. Its
`apply_object` is the map above on the array a wire carries. Its `apply_root` answers one
question of every operation: whether the operation is a conversion whose source and
target both carry a quantisation and are the same datatype once the quantisations are
taken off them. An operation answering yes becomes the identity `cat.Rearrangement` on
the wire it reads, and `construction_helpers.simple_helper.make_composed` drops an
identity out of the composition it stood in, so the operations after it read the wire in
front of it. Every other operation keeps its operator and loses the quantisations of its
weaves, of the datatypes its operator declares and of the slot of the tape it reads or
writes.

The functor descends into the block of an `ops.BlockOperator` and into the block of a
`para.data_structure.ParaBlockOperator.ParaBlockOperator` inside a `ParaWrap`, because a
box is a morphism of its own and the pass writes casts inside one. One `_Unquantified` is
shared by the whole walk, so a `Para.Grab` written inside a box and recorded a second
time on the operator of that box is rewritten once and the two occurrences stay one term.

## The rules

**A conversion is removed when both of its sides carry a quantisation of one value.**
Neither test on its own is the rule. A `TypeConvert` reading a quantised index into a
real number carries a quantisation on each side, and the two sides differ once the
quantisations are taken off them, so it converts one value into another and stays. A
conversion into a datatype carrying no quantisation is a conversion the pass left
unwritten, and it stays for the other reason.

**A conversion written by hand is removed as well as one written by the pass.** The three
cache round trips of DeepSeek-V4.1-Flash are written in
`notebooks/sota/DeepSeekV41Flash/quantised_caches.py` and stand in
`text_only_model.v41_flash_text_only` before any quantisation pass has run. Each reads a
channel divided by its scale into E2M1 or E4M3 and reads it back, which is one
quantisation of one value into another, so the functor removes it and the round trip is
left as the scaling it wraps: the group view, the largest magnitude, the scale, the
division, the clamp and the multiplication back. Stripping the quantised model is
therefore not the identity on the source model, and the difference is these three boxes
alone.

**A model carrying no quantisation is returned as the object it was given.** The functor
is applied only after `holds_a_quantisation` has found one, and every recursion inside it
returns the original object where every part came back unchanged. Stripping is idempotent
for that reason: the model returned carries no quantisation, so stripping it again
returns it.

**No rule differs by operator, so the package registers nothing.** Every operation is
asked one question, and the answer is read off the source and the target the conversion
carries rather than off the class of the operator, which is why there is no
`registries/` folder beside this module.

**The tag of a block is left as it stands.** `quantise_model` derives the tag of a
quantised block from the earlier tag and the body now held by it, through `fd.hash_id`,
so that one block composed at two quantisations has two bodies and two tags. The
derivation cannot be inverted, so the stripped model carries the tags the quantised model
carried and not the tags of the source model. Two boxes whose bodies became equal under
stripping therefore keep two tags, and a figure recycling blocks draws a body for each.
The listing does not read a tag, which is why the listings agree.

## Gaps

- The stripped model and the source model have the same listing and are not the same
  term, because of the block tags above and because the pass settled the order of two
  independent operations at the head of the model. Both differences go once the two are
  recycled through a hypergraph.
- A weight carries no wire, so the table of weight quantisations that `quantise_model`
  returns beside the model is not part of the expression and the functor does not touch
  it. A caller stripping a `QuantisedModel` drops that table itself.

## See also

- [[Quantization]] — the pass this functor inverts, the policy and the per-operator rules
- [[Functors]] — the hypergraph functor this one subclasses
- [[SOTA Model Notebooks]] — the notebook that draws the stripped model
