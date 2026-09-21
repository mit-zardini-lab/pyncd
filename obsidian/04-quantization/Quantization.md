---
tags: [layer/quantization, concept, algorithm]
code: quantization/data_structure/Quantization.py, quantization/processing/conversion_insertion.py, quantization/processing/quantise_model.py, quantization/registries/operator_quantisations.py, quantization/validate_quantization.py
status: evolving
written: Claude Opus 5 (1M context), effort high. Rewritten by Claude Fable 5.1, effort 80, on 2026-09-20.
---

# Quantization

## What it is

A quantisation is the number format a value is held in, BF16, FP32, E4M3 or E2M1,
together with the size of that format in bits. A model is written in the reals and a
released model runs in a handful of quantisations. The `quantization/` package holds the
quantisation of a value, the conversion between two quantisations, and the pass writing
a quantisation onto every wire of an expression. Nothing in it reads a description of
hardware, so a model carries its quantisations as a property of the expression.

A `Quantified` datatype wraps the mathematics of a value with its quantisation: the
number of values packed into one 32-bit word, the size in bits, the `Encoding`
naming the format, and the `BlockScale` its elements are multiplied by where the
format carries one. A block scale is one scale, held in its own format, shared by a
group of consecutive channels, and by a group of rows as well for a weight. The value
denoted by an element of such a format is the element multiplied by the scale of its
group. `MXFP8` is E4M3 elements with one UE8M0 scale per 32 channels and `MXFP4` is
E2M1 elements with the same scale, the two names given by the Open Compute Project's
Microscaling specification, and `format_name` prints them and spells out any other
scale. An index carries an integer quantisation, `INT64` or `INT32`, over its own
`cat.Natural` bound, and `quantisation_of` returns the quantisation of a wire with
the bound replaced by a fixed one, so two index wires compare equal. The user asked
for the block scale, the integer formats and the packing in front of every label on
2026-09-20. A
`TypeConvert` reads a value of one datatype into another, and between two
quantisations it is a cast, which rounds where the target holds fewer numbers and
changes nothing where it holds more.

## Where it lives

| module | what it holds |
|---|---|
| `quantization/data_structure/Quantization.py` | `Encoding`, `BlockScale`, `Quantified`, `TypeConvert` with `template` and `over_shape`, the declared quantisations `FP32`, `FP16`, `BF16`, `E5M2`, `E4M3`, `E2M1`, `UE8M0`, `INT64` and `INT32`, each packed into a 32-bit word, `block_scaled` with `MXFP8` and `MXFP4`, `INDUSTRY_NAMES`, `format_name` and `describe_quantisation`, and `quantisation_of`, `holds_real_numbers`, `holds_natural_numbers`, `holds_a_number`, `with_quantisation` and `is_cast` |
| `quantization/processing/conversion_insertion.py` | reading a hypergraph wire by wire with `leaves_in_dataflow_order`, `producers`, `consumers`, `containers`, `wire_array` and `all_wires`, writing a datatype onto one operation with `with_datatypes` and onto every wire with `rewrite_datatypes`, and the two placements `insert_conversions_beside_the_producer` and `insert_conversions_beside_the_consumer` |
| `quantization/processing/quantise_model.py` | `QuantizationPolicy`, `BoxPolicy`, `QuantisedModel` and `quantise_model`, with `operations_of`, `cast_counts`, `casts_of`, `unquantised_weaves` and `leaves_without_a_rule` for reading a quantised model |
| `quantization/registries/operator_quantisations.py` | `OperatorQuantisation`, `QuantisationQuestion`, `ArithmeticQuantisation`, `ContractionQuantisation`, `BoxPolicy`, the `register` decorator and one rule per operator class |
| `quantization/validate_quantization.py` | the checks, one `check_*` per claim |
| `notebooks/sota/DeepSeekV41Flash/quantised_text_only_model.py` | `RELEASED_POLICY`, the quantised text-only DeepSeek-V4.1-Flash, and the tables taken by its figures |

`Encoding`, `Quantified` and `TypeConvert` are mirrored in tsncd, per
[[Terms Mirrored in tsncd]]. The registry was named `operator_precisions.py` until 2026-09-20, when the reviewer
rejected "width" for a quantisation and the package settled on one word.

## The policy

`QuantizationPolicy` is a frozen dataclass of the quantisations followed by the released
code and the few places departing from them.

| field | what it applies to |
|---|---|
| `inputs` | the operands read by the model itself |
| `activations` | the quantisation of a tensor held between two modules or kernels: the value returned by a box, a slot of the tape, the result of a projection with a rounded weight, and the result of an RMSNorm |
| `scalars` | the quantisation of arithmetic, inside a module upcasting on entry and inside every kernel |
| `rounded_operands` | the quantisation written by the rounding kernel in front of a projection, when the weight of the projection has fewer bits than an activation, which is `MXFP8` for the released code because the kernel writes the elements and one scale per 32 channels |
| `integers` | the quantisation of an index: the model's token identifiers on arrival, every index computed by an operation, a slot holding one, and a table of integers not named in `weights` |
| `slots` | the quantisation held by a slot of the tape, by the text of the slot's name, for a slot dropped inside one box and grabbed inside another, which the pass cannot follow through the ports; the value dropped is cast to it in front of the drop |
| `results` | the quantisation required of the model's own results, and `None` requires nothing of them |
| `weights` and `weights_by_default` | the read quantisation of a weight, by the text of the name carried by the operator holding it |
| `boxes` | a `BoxPolicy` by box name: the quantisation returned by the box for a real number and for an index, the quantisation required of every index read by the box, whether its arithmetic computes at the scalar quantisation or at the quantisation carried by its operands, and whether its contractions promote or accumulate |
| `block_results` | the quantisation returned by a block at each position of its codomain, by its title, with `None` at a position returned as computed |

The released DeepSeek-V4.1-Flash is eager PyTorch around a few kernels, and its
quantisations follow four facts, established line by line from the released code. A
tensor passed
from one module or kernel to the next is held in BF16. Arithmetic inside a module
upcasting on entry, and inside every kernel, runs in FP32. An activation is rounded to
E4M3 in one place, in front of a projection whose weight is FP8 or FP4, by the rounding
kernel called by that projection. Everything else computes at the quantisation carried
by its operands and promotes to the one with the most bits where they differ. The first
policy, written on 2026-09-19, gave every contraction E4M3 operands and an FP32 result
and did not match the code.

A weight has no wire, because an `ops.Linear` and an `ops.Embedding` hold theirs rather
than reading it, so `quantise_model` returns a `QuantisedModel` carrying the model and a
table from weight name to quantisation. The keys are the keys of
`websocket_transfer.auxiliary_information.OperatorRole`, so a notebook writes the
quantisation into the sentence shown by an inspection box over the weight. A table whose
entries are integers holds no quantisation and is left out.

## The rules

**One rule per operator class says which quantisations an operation uses.** A rule is
given the operation, the policy and the quantisation carried by each operand, and
returns the quantisation of each result and the quantisation required of each operand.
`None` at a result says the wire keeps its datatype, and `None` at an operand says any
quantisation is accepted. The lookup walks the operator's method resolution order, as
`algebra/registries/standard_expansions.py` does, so a subclass takes the rule of its
parent unless it declares one.

| rule | operators | what it says |
|---|---|---|
| `projected` | `ops.Linear` | a weight with fewer bits than an activation: the operand at the activation quantisation and then at the quantisation for rounded operands, which an operand already carrying it, a row of a block-scaled table, needs no cast for, and the result at the activation quantisation; any other weight: the operand and the result at the quantisation of the weight |
| `contracted` | `ops.Einops` | the operands and the result at the operand quantisation with the most bits; inside a box whose contractions accumulate, a contraction reads its operands at the operand quantisation with the fewest bits and returns the scalar quantisation, and a broadcast product still promotes; a product of indices returns the integer quantisation |
| `computed_at_the_scalar_quantisation` | `ops.Elementwise`, `ops.Arithmetic`, `ops.SoftMax`, `ops.L1Norm`, `ops.L2Norm`, `ops.Maximum`, `ops.ConstantOp`, `ops.FixedArray`, `ops.GenericOperator`, `dst.Rotary`, `dst.YarnRotary` | the operands and the result at the scalar quantisation; inside a box whose arithmetic is carried, the result at the quantisation of the operand, and a constant at the activation quantisation |
| `normalised` | `ops.Normalize` | with a gain, the operand and the result at the activation quantisation, because that is the RMSNorm module; without one, at the scalar quantisation, because that is arithmetic written inline |
| `added` | `ops.AdditionOp` | the operands and the result at the operand quantisation with the most bits |
| `moved` | `ops.View`, `aops.CovariantView`, `aops.ConcatenateAxes`, `aops.DeconcatenateAxes`, `dst.MergedPositions`, `dst.PairsAsComplex`, `dst.Decomplex` | every operand and the result at the operand quantisation with the fewest bits, and the scalar quantisation for a constant; indices pass through at the index quantisation with the fewest bits among them, and every index operand is required at it |
| `ranked` | `dst.TopK` | the values at the quantisation of the values, and the positions at the integer quantisation of the policy, whatever the candidate positions carry, because `torch.topk` returns `int64` |
| `selected` | `dst.Select`, `dst.IndexSelect` | the quantisation of the payload, which is the last real operand |
| `embedded` | `ops.Embedding` | the quantisation the policy names for the table, so a row of an MXFP8 table is MXFP8 and reaches a rounded projection with no cast, and the integer quantisation for a table of integers |
| `integer_arithmetic` | `ops.BitwiseXor`, `ops.Modulo` | the integer quantisation of the policy |
| `cast_to_a_declared_datatype` | `ops.Cast` | the scalar quantisation where the declared datatype holds a real number, and the integer quantisation where it holds an index, because the cast narrows a bound and not a format |
| `converted` | `TypeConvert` | the quantisation named by its target, or the scalar quantisation where the target names none, because a kernel reads a rounded value back into its compute quantisation |
| `boxed` | `ops.BlockOperator`, and `ParaBlockOperator` through it | the quantisation named by the policy for a box of that name, and the activation quantisation otherwise; for an index, the integer quantisation named for the box or of the policy, and the index operands at the quantisation the box requires where it names one |

**An index carries an integer quantisation, and a value holding no number is left
alone.** `holds_real_numbers` and `holds_natural_numbers` are the tests made by every
rule, through `QuantisationQuestion.results_of` and `operands_of`, which give one
quantisation to the real positions and another to the index positions. A complex
number is quantised by quantising the reals wrapped by it, which `with_quantisation`
does by rebuilding the wrapper around a quantised base, and an index is quantised by
wrapping its `cat.Natural`, which keeps its bound.

**The pass assigns by dataflow and then converts.** `quantise_model` converts the model
to a hypergraph, assigns a quantisation to every wire, collects each operand required by
a rule at another quantisation, inserts a `TypeConvert` named `cast` for each, and
repeats until nothing is required. It then writes the assignment onto every wire and
converts back. The steps are in `quantization/processing/conversion_insertion.py`.

**A cast stands in front of the operations reading it.** The layout pipeline puts its
conversion beside the producer of the wire, because it converts one operand at a time
and the readers redirected by it are the whole of the readers of the value. A pass
converting a whole model cannot: a value produced inside a loop and read at another
quantisation outside it would have its cast written inside the loop, and the wire
written by the cast would never leave. `insert_conversions_beside_the_consumer` is the
other placement, and `quantise_model` groups the consumers of a wire by their scope, so
the operations of one scope share a cast and the operations of another take one of
their own.

**A titled block returns the quantisations named by the policy.** A released module
ending in `.to(dtype)` on some of its returns need not be a box of the expression. The
sublayer of the four-stream residual is a block, and it returns the residual in BF16 and
the collapse vector for the next sublayer in FP32. `_cast_block_results` writes a cast
beside the operation computing each named result, inside the block, and the cast writes
the wire returned by the block, so every reader of that wire reads the converted value
and the operation writes a fresh wire read by the cast alone.

**A box is descended into, and its tag is derived from the body now held by it.** The
body of an `ops.BlockOperator` is a morphism of its own, quantised with the
quantisations carried by the operands of the box and required to return the
quantisations of its results, which puts a cast at the end of a body computing
something else. A `ParaBlockOperator` inside a `ParaWrap` is descended into the same
way, and the seeds recorded by it are re-derived from the quantised block so that the
box and the block state one morphism. One block written once and composed at two
quantisations has two quantised bodies, so the tag is derived from the earlier tag and
the body now held, and a figure recycling blocks draws one body per tag.

## The quantised text-only DeepSeek-V4.1-Flash

`notebooks/sota/DeepSeekV41Flash/quantised_text_only_model.py` applies the
released quantisations to `text_only_model.v41_flash_text_only`. The policy names BF16
for the activations, FP32 for the scalars, MXFP8 for rounded operands, INT64 for
indices and no quantisation for the results of the model, because the released
probabilities are FP32. The three projections of a routed expert are read in MXFP4,
the token embedding, the first output projection and the key and head-weight
projections of the indexer in BF16, the mixing projections, the compressor
projections, the router weight, the stream weights of Engram, the correction bias, the
sink logit and the output head in FP32, the two n-gram tables of Engram in MXFP8, the
token map of Engram in INT64, and every other weight in `FP8_WEIGHT`, which is E4M3
with one UE8M0 scale per block of 32 rows by 32 channels. `FILE_QUANTISATIONS` names
the five weights held by the file in another quantisation. `Core` is a fused kernel
whose contractions accumulate, `Sco` and `Pool` compute at the quantisation carried by
their operands, `Pool` returns its positions in INT32, `Gth` reads its positions in
INT32, `Gate` and `Coef` return FP32, the title of the sublayer returns its residual
in BF16 and its collapse vector as computed, and the pool and selection slots of the
tape hold INT32.

The rounded operand carries its scale inside its format because the released
rounding kernel writes the E4M3 elements and one UE8M0 scale per 32 channels, and the
GEMM multiplies its FP32 accumulator by the scale of the operand and the scale of the
weight. A cast to E4M3 with no scale would saturate above 448 and flush to zero below
2^-9. The three cache round trips keep their bare E2M1 and E4M3 casts, because the
value they round is the channel already divided by its scale and the scale stands
beside them in the expression; the kernel called in place multiplies the scale back
before the cache is written, so the cache holds BF16 numbers carrying the rounding and
no scale beside them. The n-gram table of Engram is MXFP8 and its rows reach the key
and value projections with no cast, because the released lookup multiplies a row by
its scales into BF16 and the projection rounds it back with the same groups of 32,
which changes no value. The user ruled on the rows on 2026-09-20, and
[[Representing Models]] records the ruling.

The model holds 190 cast operations, counted once per written operation. Twenty-five
round BF16 to MXFP8, each in front of a projection whose weight is FP8 or FP4.
Seventy-three read BF16 into FP32 where the released code writes `.float()` or
promotes. Seventy-nine round FP32 to BF16 at the end of a module, at the end of a
sublayer and in front of an RMSNorm. Eight stand inside the round trips of the caches.
Five convert positions from INT64 to INT32 where the released code writes `.int()`,
on the result of the candidate pool and on the positions picked by each indexer.
Every row of the policy table cites its released lines, pinned to one commit by
`reference_links.py`. `notebooks/sota/DeepSeekV41Flash.ipynb` draws the
model and its parts, and [[SOTA Model Notebooks]] lists it.

A cast is drawn as no glyph, on a box of no size, and the rounding is read from the
quantisation labelled on each wire. The format written by the cast is blue, and the gap
holding it names no axis. `DiagramSettings.casts`, set to `CastPresentation.DRAWN`,
draws it as a chevron whose two halves are as tall as the two quantisations of the cast
instead. Resting the pointer on the blue format opens an inspection box naming the
quantisation the cast reads and the one it writes, which the user asked for on
2026-09-20, because a cast drawn as no glyph is two pixels wide and the label is the
mark a reader sees. [[Diagram Display]] states the pass, added on 2026-09-20 and made
the default by the user the same day.

## Gaps

- A weight is not drawn with its quantisation. The table and the inspection box over
  the operator carry it, and a figure drawing the weight array of a projection under
  `ExpandedParameters.WEIGHT_ARRAYS` draws that array from the standard expansion of
  the operator, written in the reals.
- A slot of the tape dropped inside a box and grabbed outside it takes the activation
  quantisation for a real number and the integer quantisation for an index, because a
  box states the slot carried by each of its ports and says nothing about the
  quantisation behind the port. The policy names such a slot by its name where the
  released code holds it in another format, as it names the pool and selection slots
  in INT32, and the pass does not derive it.
- `bits` orders quantisations by the size of an element and counts no scale, so MXFP8
  has as few bits as E4M3 for the purpose of choosing a rounded projection.
- The policy keys a weight by its name. The compressor projection is named `W^{C}` at
  both of its sites, and the released ratio-one compressor reads it in BF16 where the
  ratio-two compressor reads it in FP32, so the ratio-one site carries FP32 here.
- The pass assigns a quantisation to every wire from one policy, and a model already
  carrying quantisations is assigned again from that policy rather than having the two
  reconciled.
