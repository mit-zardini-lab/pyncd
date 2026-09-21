---
tags: [layer/practice, index]
status: evolving
---

# Open Gaps

Written by Claude Opus 5 (1M context), effort high.

The ranked worklist. Each item records what is unfinished, why it is hard, and, where the
question has been attempted, what was learned. When an item is closed, move it to *Closed*
with a link to the log that closed it, so that the list is also a history.

## The ranked list

### 1. Data-dependent reindexing with static structure

The cases are top-k gathers, meaning mixture-of-experts dispatch and the block selection of
DSA, NSA and MSA, and paged key-value caches. Each needs a gather whose index map is a
runtime value with static bounds: a fan-out of at most `k`, a factorisation through the
hierarchy, and dispatch and combine being adjoint.

Every mechanism that shipped in 2025 and 2026 is a cheap and possibly non-affine index
plane feeding a dense affine compute plane, so the gap is the one worth closing.

Two presentations of a selection already exist: compressed onto a `SparseAxis`, or expanded
into values plus `Natural` indices, described in [[Sparse Axes]]. The second is where a
gather with static bounds would be read from. Since 2026-08-20 the expanded form is
reachable by rewriting, described in [[Sparse Expansion]], so every selection can be put in
index-wire form before anything analyses the dependency it introduces.

The reverse direction needs the same rewrite for a different reason. A gradient cannot be
sent back through a compressed selection, because the compression is the loss of which of
the `n` was chosen. [[Selection and the Reverse Pass]] covers both reasons.

*Touches: [[Stride Category]], [[Operators]], [[Sparse Axes]], [[Training]].*

### 2. The reverse functor has no rule for `IndexSelect`, and there is no loss or optimiser

The duals of the seed operators exist, described in [[Backpropagation]]. The affine half of
the scatter problem closed on 2026-08-20 as `transpose.ReindexTranspose`, and the
data-dependent half closed on 2026-09-04 as `Inject` followed by a sum over the degree,
for a complete `TopK` and for a `Linear` that selects a weight, per
[[Selection and the Reverse Pass]] and [[DeepSeek-V3 Backward Pass]].

`IndexSelect` has no rule, and its rule is the same pair at the payload's degree. A `TopK`
in the `ONLY_WEIGHTS` or `ONLY_SELECTION` form of `ds.SelectionForm` has no rule either.
`ONLY_WEIGHTS` hands out no positions to say where its cotangents land. `ONLY_SELECTION`
hands out positions alone, which carry no cotangent, so its rule would return zero to the
scores. `deepseek.registries.derivative` raises `SelectionFormHasNoReverseRule` for both,
and the entry selections of
`notebooks/sota/DeepSeekV41Flash.ipynb` are in the second form since
2026-09-14. A `Select` at positions has no rule either, and its rule is the one
`IndexSelect` needs. A `TopK` taking the positions of the entries it selects over as an
operand, per `ds.TopK.template(positions_of=)`, has no rule either: its cotangent returns
to the entries, and the slot holds the positions of the wider axis it reported, so the rule
needs the index over the entries, which the forward expansion composes away. The registry
raises `SelectionOverPositionsHasNoReverseRule` for it.

Whether the reverse functor steps over a node whose inputs and outputs are all `Natural`,
meaning `deepseek.MergedPositions` and the `IndexSelect` that composes two selections, has
not been run. Neither rule is checked numerically, because `torch_compile` has no module
for a complete `TopK`, for `Inject`, or for a `Linear` with an index operand. There is no
loss function and no optimiser.

*Touches: [[Selection and the Reverse Pass]], [[Training]], [[Torch Compile]].*

### 3. Axis sizes are declared rather than derived

A `StrideMorphism`'s `x = r*b + w + shift` already determines that `w` has size `r` and `b`
has size `x / r`, and nothing reads it. `nm.Equality` and `Operator.sizing_rules` have one
producer, `Decomplex`, and no consumer. Since 2026-09-15 a relation can be declared: an
axis may carry a size written as a product of other sizes, `composition.size_class` carries
it through every alignment, and the marking of a view onto such an axis reads it, which is
how the V4.1 model's query axis, sized `|a| |b|`, lets the last query group's overshoot mark
the offsets. Nothing derives the relation from the view that states it.

The truth is `ceil(x / r)`, and `nm` has no ceiling. The phase-`b` window of entry 0 is
empty, which is the edge case an exact form has to account for.

*Touches: [[Stride Category]], [[Numerics]].*

### 4. A `Transpose` with an implicit weight does not share the forward weight

`expand_linear_root` leaves a `Linear` with no inputs or with several inputs unchanged, so
the parametrised backward pass of [[Show Grabbed Parameters]] gains no second weight under
`ExpandLinear`. What remains is the unparametrised `Transpose`, whose weight is inside the
operator. `ExpandLinear` leaves it as it finds it, and nothing reads the forward operator it
keeps in its `operator` field to write the weight array the forward pass would share. A
pair has to go through `show_grabbed_parameters` before its weights can be shared.

*Touches: [[Linear Expansion]], [[Backpropagation]].*

### 5. Unifying two contractions can leave a form nothing benefits from

[[Einops Rearrangement]]'s rule is to unify first and then disentangle. Disentangling only
splits a merged einsum whose operands fall into independent components, and $Q$, $K$ and
$V$ do not.

The unified einsum with three operands names no evaluation order, and by the counting in
[[Einops Rearrangement]] it is worse than the pair. A derived attention never reaches it,
because a `SoftMax` sits between the two contractions, so nothing is broken. Deciding it in
general is the einsum-path problem and needs an estimate of what each order costs.

*Touches: [[Einops Rearrangement]], [[Expression Simplification]].*

### 6. A `ReindexTranspose` does not factor its context out

It is built with an empty degree and the whole of both shapes in its weaves' targets, so a
batched convolution's transpose consumes the batch axis where the forward node was
broadcast over it. The result is correct, and harder to read than it needs to be.

Splitting the reindexing into a context and a smaller map is integer linear algebra over
the affine maps the expression already carries, rather than a pattern match.

*Touches: [[Derivatives]].*

## Smaller and local

### Five gaps the integrated DeepSeek-V4.1-Flash opened

- `para_sparse_expansion.apply_root` does not enter a `ParaWrap` whose body is a box, so a
  para-boxed mixture sublayer keeps `k/e` compressed. The integrated sublayer stands the
  grab beside the plain box.
- The DSpark chain of five draft steps cannot be a repeated block: a loop whose every
  result is on the tape is deleted by `h2m.recycle`, and a `LoopDrop` inside the loop
  raises `TapeBelowTheTopLevel` under `para_boxed`.
- `torch_compile` has no module for `dst.Rotary`, `ops.Arrange`,
  `Quantization.TypeConvert` or a forward `Inject`, so the integrated validator is
  structural throughout.
- A constant member index, `Para.LoopSlot(slot, nm.Integer(2))` for a group written alone,
  is accepted by `tape_members` and untested in tsncd's tape drawing.
- The gate of a chosen expert whose biased score is at or below zero is zero in the
  expression, where the release reads the unbiased score.

### An axis read through a guarded view carries one affine form

A second guard on the same axis raises `AxisMarkedTwice`. The prefix of the Engram hash
reads the lookback slots `L|x`, guarded by the token, through a row guarded by the order,
and the result `L'|G` records the order's guard alone. The values at the slots before the
first token are still the unit, because a view moves the unit, but the label and
`live_positions` do not say so, and a fold over `L'` cannot tell a token near the start from
any other. Two affine forms on one axis, or a conjunction of forms, would state it. Found on
2026-09-18 when the user asked whether `G` should be guarded by `x`.

*Touches: [[Advanced Axis Dynamics]], [[Padding and Masks as Sparse Axes]].*

### The mechanisms the model leaves out each need one thing the operator set lacks

`validate_omitted_mechanisms.py` writes each of them with one `ops.GenericOperator` at the
operation the standard set cannot state. What each needs: a rewrite that replaces a generic
operator by the right-hand side of the `cat.DefinedExpression` that defines it, which the
rotary embedding has had since 2026-09-17, over every position through an `ops.Arrange` for
the rotation, the YaRN frequencies and the YaRN ramp, and a floor numeric, for the pinned
block; a numeric for infinity or a selection with a forced member, for the pin under the
model's anchoring; nothing further for Engram, whose token map is an `ops.Embedding` that
returns a `Natural` and whose hash is written out over `ops.BitwiseXor`, `ops.Modulo`,
`ops.Cast` and `ops.FixedArray`; nothing further for the image span, whose write is `Inject`
twice and a product; an operator that returns a random number and a loop variable a boxed
loop can drop, for DSpark, whose Gumbel-max sampler is otherwise written out; a numeric for
the ceiling, for the scales of the FP4 indexer cache and the FP8 window cache; and a
parameter array between the normalisation of the window latent and the normalisation of the
compressed entry, because the ratio of their two RMSNorm gains is part of no linear map.

*Touches: [[SOTA Model Notebooks]], [[Operators]].*

### An inspection box opens no nested expansion and no box over a `Linear`

The `auxiliary` field of a message packages each operator expansion with its own
auxiliary information, so an operator inside an expansion could open a box of its own.
The three registered expansions, of the softmax, the L1 norm and the RMSNorm, produce
primitives, so no nested box opens yet. The expansion of a `Linear` lives in
`algebra/linear_expansion.py`, which the display layer does not import, so a weight's
contraction opens no box. A `ConcatenatedAxis` is left out of the legend because its uid
carries no name, where `agent_display` labels it by its parts.

*Touches: [[Advanced Display]].*

### The rest

| gap | where |
|---|---|
| `ops.Normalize` always carries a gain, and the normalisation of the four streams in the mixing coefficients of DeepSeek-V4.1-Flash has none in the released code, which computes a bare reciprocal root mean square ([model.py L951-L954](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/dba1be0a40aa45a94ad051997016db3960a90277/inference/model.py#L951-L954), read 2026-09-17). The inspection box over that RMSNorm therefore draws a gain $\gamma$ of 4 × 5120 entries that the released model does not hold. An RMSNorm with no gain needs a field on the operator or an operator of its own | [[Operators]], [[SOTA Model Notebooks]] |
| A quotient in a formula prints as a product with a negative power, so the inspection box over the route scale of the DeepSeek-V4.1 router reads $y = 3 x 2^{-1}$. `Multiplication.to_latex` also writes every stride and size, so a fraction there has to be checked against those | [[Numerics]], [[Advanced Display]] |
| A grab drawn as a box of its own reserves no room for its labels, so beside other wires its slot name and its turned axis names are drawn over the neighbouring wire labels. The body of the Lightning Indexer of a Reindex layer and the expansion of the projection $H$ of DeepSeek-V4.1-Flash show it since 2026-09-17, when an inspection box began to draw both | [[Para Wrap]], [[Advanced Display]] |
| `UnitOfMeasure` and `DimensionOfMeasure` in `data_structure/Numeric.py` have no mirror in tsncd, so a numeric carrying a unit of measure cannot be drawn | [[Terms Mirrored in tsncd]], [[Units of Measure]] |
| The tape of a shifted softmax holds `m` and `r` as two slots where FlashAttention holds one `L = m + log l`. Folding them needs a collapse rule for a maximum and a reciprocal feeding one exponential | [[Recomputing the Exponent in the Backward Pass]] |
| The reverse of a repeated block repeats the same number of times and reads the tape in the order the forward pass wrote it, where a loop's tape is read in reverse. `tape_members` indexes the members of both passes by the same loop index, which is right for the members and says nothing about the order | [[Code Forms]], [[Backpropagation]] |
| `torch_compile` names a generated module's parameters after `to_bodies` and reads no code form. A name's code form is the identifier the module should write | [[Code Forms]], [[Torch Compile]] |
| Nothing in the accumulator registry declares that an operator is symmetric, and a fold run in pieces needs it to be | [[Design Space]] |
| Add every operation present in transformers, make the operator constructors consistent, and give each operator a formal-name characteristic | [[Operators]] |
| `Multilinear.forward`'s per-axis contraction is commented out and unverified | [[Torch Compile]], `PublicCodeTODOs.md` |
| `ReverseCrawler.crawl` creates a fresh root. Whether the UID should carry over is undecided | [[Crawlers]], `PublicCodeTODOs.md` |
| Reindexing names are truncated to three characters in the ASCII renderer, which needs a real layout | [[Diagram Display]], `PublicCodeTODOs.md` |
| The pass of [[Quantization]] assigns a quantisation to every wire from one policy, and a model already carrying quantisations is assigned again from that policy rather than having the two reconciled | [[Quantization]] |
| A weight carries no wire, so the table of weight quantisations `quantise_model` returns beside the model is not part of the expression, and [[Stripping Quantisations]] does not touch it | [[Stripping Quantisations]] |
| Whether running [[Expression Simplification]] before a derivation helps or hurts is untested | [[Expression Simplification]] |

## Closed

### Every sentence of a page reads from a wording file

The three wording files, the model's, the expansions' and the display's, hold every
sentence a box of the quantised text-only page shows, and a second file changes any of
them. Closed 2026-09-20, the same day the gap was opened.

### Every diagram of a taped loop drew the initializer of each loop variable

Closed 2026-09-13. `DiagramSettings.loop_initializers` defaults to
`LoopInitializers.HIDDEN`, which leaves the initializer and the drop that starts its
variable out of every diagram.

### The derivative rules read a shape as a set of axes, so a copied token axis had no backward pass

Closed 2026-09-05. `einops_simplification.einsum` takes shapes of `IndexVariable`s, and
`index_shapes` reads a morphism's variables off its weaves, reindexings and signature, per
[[Weaves and Degree]].

## See also

- [[Agent Log Protocol]] — a log that closes a gap must say so
- [[Home]]
- [[Design Space]] — the operations the rules are not written for at all
