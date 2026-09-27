---
tags: [layer/quantization, concept, algorithm]
code: quantization/algebra/strip_quantisations.py, algebra/remove_identities.py
status: stable
written: Claude Opus 5 (1M context), effort high, on 2026-09-20.
---

# Stripping Quantisations

## What it is

`quantise_model` writes a quantisation onto every wire of a model and puts a
`TypeConvert` named `cast` wherever an operation requires another. `strip_quantisations`
is the other direction. It takes a quantised model back to the expression in the reals
it was made from, in three steps. Every `Quantified` wrapper is taken off every datatype
of every wire and every weight, with the `BlockScale` it carries. Every cast that then
reads and writes one datatype is turned into the identity on its operand. Every identity
is then removed wherever the laws of the category remove it.

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
quantisation on every wire are everything written by the pass.

A page with variants, stated in [[Diagram Wire Format]], draws the unquantised form of a
model by applying the dequantisation functor in the browser to the term of the quantised
form. tsncd names the functor `dequantise`, and `notebooks/display/notebook_diagrams.py`
names it `PageFunctor.DEQUANTISE`. `strip_quantisations` is the Python statement of the
functor. `notebook_diagrams.PYTHON_STATEMENT_OF_PAGE_FUNCTOR` maps
`PageFunctor.DEQUANTISE` to it, and `show_page_variants` prints a derived variant through
it under `DiagramMode.LISTING`. The user asked for the variants on 2026-09-27, and on the
same day set out the steps of the functor, so that the Python statement and the
statement in tsncd agree:

> Remove the quantization wrapper, turn casts into identities, have a check over Composed expressions that (a) removes identities (b) if all the constituent morphisms are identities, turn the expression into that identity. This layers upwards, so that identities are removed if at all possible. This needs an identity check. This is so that we are not left with excess identities in expressions, but maintain categorical equalities.

The validator of each quantised page of the lab website checks that
`strip_quantisations` turns the quantised model into the unquantised one, and
[[Website Notebooks]] lists the pages and their validators. The page of
DeepSeek-V4.1-Flash is the exception, and it embeds its unquantised variant as a term of
its own. The released
code rounds three caches through FP8 and FP4 in place, so the model in the reals keeps
three round trips of casts, and the functor would turn every one of those casts into an
identity. `notebooks/website/modern/validate_deepseek_v41_flash.py` checks that the
stripped model and the model in the reals differ in the bodies of those three round
trips alone.

## Where it lives

| module | what it holds |
|---|---|
| `quantization/algebra/strip_quantisations.py` | the three steps: `without_quantisations`, `turn_casts_into_identities` with `converts_nothing_once_dequantised` and `reads_and_writes_one_datatype`, and `strip_quantisations`, which applies them after `holds_a_quantisation` has found a quantisation |
| `algebra/remove_identities.py` | the third step, general to any morphism: `is_identity_on_its_domain`, `identity_on_the_domain_of`, `written_as_identity`, `with_identities_taken_out` and `remove_identities` |
| `quantization/validate_quantization.py` | eight `check_*` cases, from `check_stripping_the_model_leaves_no_quantisation` to `check_stripping_a_presented_figure_leaves_no_cast_and_no_empty_box` |
| `notebooks/sota/GLM53/validate_quantised_glm53.py` | `check_stripping_the_quantisations_returns_the_unquantised_model`, the same comparison for GLM-5.3 |
| `notebooks/display/notebook_diagrams.py` | `PageFunctor.DEQUANTISE` and `PYTHON_STATEMENT_OF_PAGE_FUNCTOR`, which name this functor for a page with variants |
| `notebooks/sota/DeepSeekV41Flash/quantised_text_only_model.py` | `v41_flash_text_only_without_quantisations`, the stripped model drawn by a figure |

`notebooks/sota/DeepSeekV41Flash.ipynb` draws the stripped model beside
the quantised one, and [[SOTA Model Notebooks]] lists that notebook.

## The mathematics

A quantisation is a wrapper around the mathematics of a value, so taking it off is a map
on datatypes and the functor it induces on the expression. `without_quantisations` is the
map on datatypes. It replaces `Quantified(wraps=X)` by `X` wherever it stands, so a
quantisation of the reals leaves `cat.Reals`, a quantisation of an index leaves the
`cat.Natural` with its own bound, and a wrapper around a quantised value, as
`deepseek.data_structure.Complex` is, keeps its own form around the datatype now under
it.

The map takes a cast to a conversion whose source and target are one datatype, which is
an identity written as an operation. `turn_casts_into_identities` writes the identity
`cat.Rearrangement` on the arrays read by the operation in its place, wherever it
stands: in a composition, in the body of a box, or as the body of a `ParaWrap`. The
functor so far keeps every categorical equality, because it maps each operation to one
with the same domain and codomain.

`remove_identities` then takes the identities out. An identity is a rearrangement taking
every wire to itself, a composition or a product of identities, a block whose body is
one, a box whose block is one and which reads and writes the same arrays through
identity reindexings, or a `ParaWrap` of one that grabs and drops nothing.
`is_identity_on_its_domain` reports the condition. `tutil.is_identity` recognises the
first three and `ops.is_identity` adds a view with identity reindexings. Neither
recognises a box, so the check is written anew beside the removal. The removal runs
from the leaves upwards, simplifying the parts of a term before the term. A block left
holding the identity therefore becomes the identity, and the block around it may then
hold the identity in turn.

Each of the three steps is `fd.deep_reconstruct` memoised on object identity, which
visits each node of the term once and returns the original object for a subterm left
unchanged by the step, per the sharing invariant of [[Invariants]]. A `fd.Context`
applies equality classes keyed by uid, and none of the three steps identifies terms by
uid, so none uses one.

## The rules

**A cast becomes the identity when it reads and writes one datatype once the
quantisations are taken off.** A cast between two quantisations of one value does, and
so does a conversion from a quantised value into the same value unquantised. A
conversion between two different values, such as a quantised index read into a real
number, still converts once the quantisations are taken off, and it stays.
`converts_nothing_once_dequantised` reports the condition on a conversion. Until
2026-09-27 the functor also required both sides to carry a quantisation, and it deleted
the cast by redirecting its readers onto the wire carrying its operand.

**A composition drops each identity among its members, and becomes the identity when
every member is one.** A product of identities becomes the identity on its domain, and
an identity standing in a product beside other factors is written as the identity
rearrangement.

**A block whose body is the identity is the identity, whatever its tag, repetition,
title or colour.** A block only groups, and a repeated identity is the identity. A box,
meaning an `ops.BlockOperator` operation, whose block is the identity is the identity
when it reads and writes the same arrays through identity reindexings. A figure drawn
with inspection boxes wraps every cast in such a box, drawn `BODY_IN_PLACE` by
`notebooks/display/explain_operators.py`, and the box goes with its cast by this rule.

**A `ParaWrap` that grabs or drops keeps the identity as its body.** The wrap acts on the
tape, so it is not the identity. A figure drawn under `TapePresentation.ABSORBED` puts
each grab on the port of the operand fed by the grab, and each drop on the port of the
result saved by the drop. A cast whose operand is grabbed from the tape, or whose result
is dropped onto it, therefore becomes the body of a `ParaWrap`. The four casts of
GLM-5.3 converting the selection between INT64 and INT32 stand that way, and so does
the one cast of DeepSeek-V4.1-Flash converting the positions of the candidate pool. The
body becomes the identity rearrangement on the wire, which is the form of a bare grab or
drop drawn as a wrap, so the wrap becomes the grab and the drop on one wire.

**A conversion written by hand becomes the identity as one written by the pass does.**
The three cache round trips of DeepSeek-V4.1-Flash are written in
`notebooks/sota/DeepSeekV41Flash/quantised_caches.py` and stand in
`text_only_model.v41_flash_text_only` before any quantisation pass has run. Each reads a
channel divided by its scale into E2M1 or E4M3 and reads it back, so the functor turns
both conversions into identities. The round trip keeps its scaling alone, which is the
group view, the largest magnitude, the scale, the division, the clamp and the
multiplication back. Stripping the quantised model is therefore not the identity on the
source model, and the difference is these three boxes alone.

**A model carrying no quantisation is returned as the same object.** The functor is
applied only after `holds_a_quantisation` has found one, and every step returns the
original object where every part came back unchanged. Stripping is idempotent for that
reason. The model returned carries no quantisation, so stripping it again returns it.

**No rule differs by operator, so the package registers nothing.** Every operation is
asked whether it is a conversion reading and writing one datatype. The answer is read
off the source and the target carried by the conversion, and the class of the operator
is not consulted, so there is no `registries/` folder beside this module.

**The tag of a block is left as it stands.** `quantise_model` derives the tag of a
quantised block from the earlier tag and the body now held by it, through `fd.hash_id`,
so that one block composed at two quantisations has two bodies and two tags. The
derivation cannot be inverted, so the stripped model carries the tags of the quantised
model. The tags of the source model are not recovered. Two boxes whose bodies became
equal under stripping therefore keep two tags, and a figure recycling blocks draws a
body for each. The listing does not read a tag, so the listings agree. In the browser,
tsncd gives one tag to the blocks made equal by its functor, per
[[Diagram Wire Format]], so a derived variant draws each body once.

## Gaps

- The stripped model and the source model have the same listing and are not the same
  term, because of the block tags above and because the pass settled the order of two
  independent operations at the head of the model. Both differences go once the two are
  recycled through a hypergraph.
- The tags also reach a figure drawn from Python. The stripped DeepSeek-V4.1-Flash writes
  the body of the attention kernel `Core` under eight tags, one for each quantisation
  given to it by the pass, where the source model writes it under one. A walk entering
  each tag once therefore counts 28 more `Arithmetic`, 28 more `Einops` and 7 more
  `AdditionOp` operations in the stripped model, and a figure of the stripped model sent
  from Python draws that body seven more times. GLM-5.3 has no such box, and its
  stripped model counts the operations of `whole_model.glm53` exactly.
- Stripping a figure presented under `TapePresentation.ABSORBED` gives a different figure
  from presenting the stripped model. In GLM-5.3 the selection is dropped onto the tape
  from the cast, so the stripped figure draws the grab and the drop of the slot `sel` on
  one wire at four sites, and the gather box reading the selection stays an
  `ops.BlockOperator`. The figure of `whole_model.glm53` absorbs the drop into the top-k
  and the grab into the gather box, and draws the gather box as a `ParaBlockOperator`.
  DeepSeek-V4.1-Flash draws the pool slot the same way at its one site. The listings
  agree, and the figures differ in where the slot is drawn.
- A weight carries no wire, so the table of weight quantisations returned by
  `quantise_model` beside the model is not part of the expression and the functor does
  not touch it. A caller stripping a `QuantisedModel` drops that table itself.

## See also

- [[Quantization]] — the pass this functor inverts, the policy and the per-operator rules
- [[Diagram Wire Format]] — the page whose unquantised variant tsncd derives by this functor
- [[Website Notebooks]] — the pages drawing a model with and without its quantisations
- [[SOTA Model Notebooks]] — the notebook that draws the stripped model
