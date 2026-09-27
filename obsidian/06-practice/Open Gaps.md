---
tags: [layer/practice, index]
status: evolving
---

# Open Gaps

Written by Claude Opus 5 (1M context), effort high.

The ranked worklist. Each item records what is unfinished, why it is hard, and, where the
question has been attempted, what was learned. When an item is closed, move it to *Closed*
with the date it closed and the change that closed it, so that the list is also a
history.

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

### A cached pass is not checked by value, and a state carried between passes is not derived

`caching.algebra.derive_cached_pass` places a cache wherever a causal read demands an
earlier token, per [[Deriving Caches by Dragging the New Tokens]], and keeps the last
`|w| - 1` earlier tokens for a sliding window of `|w|` slots. Five things it could state
are not written.

- No derived pass is compared by value with its model. `torch_compile` compiles every
  `ops.View` as the identity and has no rule for a `Caching` or an
  `aops.ConcatenateAxes`, so `caching/validate_caching.py` and the validators of
  [[Website Notebooks]] check the cached passes by their structure and by the stripped
  quantised pass. Rules for the
  concatenation, for a view that reads zero outside its operand, and for carrying a cache
  slot between passes would check every derived pass against its model.
- A state carried from one token to the next, as the running sums of linear attention and
  the state of a scan carry one, is not derived. Each is a wire over the tokens that its
  own recurrence reads one token back, so each would be cached on a kept axis of one token
  with the existing `CacheGrab` and `CacheDrop`. The derivation stops at a block repeated
  over the token axis with `ReadsTheTokenAxisWhole`, because the view at the loop counter
  holds no token axis.
- `cost_cache_placements.cost_of_a_pass` counts the `Caching` operators of a pass alone,
  so a `CacheGrab` and a `CacheDrop` written outside one cost nothing, and
  `cache_contents.caches_in_the_order_they_run` raises `RepetitionIsNotAnInteger` at a
  loop over the new tokens.
- The attention core over the cached axis splits into an attention over the earlier
  tokens and one over the new tokens, merged by their maxima and sums, which
  `concatenation_expansion` states. The derivation does not split it.
- `absorb_linear_maps` chooses an order from bound sizes and does not solve for the
  region of sizes where each order is cheaper. The absorbed pass reads the cached latent
  through two equal views, which nothing merges.

`notebooks/sota/DeepSeekV41Flash.ipynb` draws its model as built, where the user ruled on
2026-09-26 that the CausalSlide is the standard form of a displayed expression. The
notebooks under `notebooks/website/` draw every model in the CausalSlide.

*Touches: [[Deriving Caches by Dragging the New Tokens]], [[Caching Between Passes]],
[[Torch Compile]], [[Yoneda and Cartesian Tricks]], [[Advanced Axis Dynamics]].*

### The gather before the expansion of a cached latent is not written as an expression

The cached GLM-5.3 of [[Caching Between Passes]] follows the reference and expands the
latent of every cached token by `W^{Kb}` and `W^{Vb}` in every pass. A decode pass at a
context of 1,048,576 tokens then spends 2.4 PFLOP on the expansion, more than a hundred
times the rest of the attention, and expanding only the 2,048 latents each query selects
would take 4.69 TFLOP. That order, with the gather before the expansion, is counted from
the expression and not written as one. The gather is an `IndexSelect` at a position held
as data, so `move_reads_backwards`, which moves an affine read, does not move it past the
expansion.

The order that multiplies every query by the key up-projection and applies the value
up-projection after the core is derived since 2026-09-26.
`advanced_axis_dynamics.algebra.absorb_linear_maps` chooses it on a derived pass by its
operation count, per [[Deriving Caches by Dragging the New Tokens]], and on
DeepSeek-V3's attention it gives the absorb mode of the released code. It has not been
applied to GLM-5.3, whose gathers stand between the latent and the core.

A `Caching` states the dynamic cache alone. The static cache of `transformers` writes
each pass into a buffer of the longest sequence in place.

*Touches: [[Caching Between Passes]], [[Advanced Axis Dynamics]], [[Einops Rearrangement]].*

### A value computed once per sequence is recomputed in every pass

`caching.algebra.derive_cached_pass` carries the read of the new tokens of one token
axis, and it leaves every operation that reads no token of that axis as the model writes
it. The encoder of the transformer of *Attention Is All You Need* reads the source
sentence and no target token. The pass derived from the whole model therefore runs the
encoder over the whole source sentence in every step, and the pass derived from the
decoder projects the encoded input `A` into the keys and the values of every
cross-attention in every step. tensor2tensor runs the encoder once per sentence, and
computes those keys and values once per sentence in `_init_transformer_cache`
(lines 967 to 988 of the `transformer.py` of tensor2tensor at `bafdc1b`, read on
2026-09-27).

Stating the reference needs a value computed in the first pass of a sequence and loaded
by every later pass, and no operator states that, because a `Caching` appends the tokens
of each pass to what it holds. The keys and the values of the cross-attention are
computed inside the repeated decoder block, one pair per layer, so writing them to the
tape beside `A` also needs one slot per layer, which a repeated block cannot name.
`notebooks/website/classic/AttentionIsAllYouNeed.ipynb` draws the pass of the decoder,
whose cross-attentions project `A` in every step, and says so.

*Touches: [[Deriving Caches by Dragging the New Tokens]], [[Caching Between Passes]],
[[Para Category]].*

### An index dragged backwards stops at a window, a loop and a hypergraph

`advanced_axis_dynamics/algebra/drag_index_backwards.py` carries one index of one axis of a result back to the inputs, and three things stop it short of where the rule $[F; x](z)[i_t] = F(z[i_t])$ reaches. `move_reads_backwards.py` carries the read itself, which closes the window for it: the mask composes into the read, so the keys of the causal attention are read at $t_x - j_w$. The loop and the hypergraph stand for both crawls, and the guide below stands for both in different forms.

A row that reads a pinned axis and a free axis together leaves its codomain axis free. The causal read $i_x - i_w$ at a pinned $i_x$ reads the positions $i_t - i_w$ for every $i_w$, which is a window, and a window is an `AffineSparseAxis` with an extent rather than a pin. `row_pin` returns `None` for such a row, so a drag through the causal attention of `notebooks/classic/` frees the keys where the window would be right. The image of a pinned index under a row with unit stride on one free axis is such an axis, and stating it is the first move.

A repeated block asserts that the pins leaving it equal the pins entering it, so a loop
whose body drops the pin raises. The pins a loop's input carries are a fixed point, the
pins its body carries back agreed with the pins asked of its output, and nothing
computes it.

The crawl runs on the morphism form. `ReverseCrawler.propagate_multigraph` merges the
readers of a wire by identity through the same `merge_guides`, but the view a stop
needs would have to be spliced into the graph with
`graphs/processing/leaf_splicing.py`, and that is not written.

A guide is pinned only where it is the pinned axis. `drag_index_backwards.pin_guards` substitutes the index into a guard whose guide is the pinned axis, so the causal attention's slot axis becomes `w|x[t_x]`, and the two concerns under *A pinned guide* in [[Advanced Axis Dynamics]] remain for the rest: a guide that is not the pinned axis, because $x$ was batched before the read, and a form the substitution does not simplify. A `deepseek.SparseAxis`, as the selection of GLM-5.3 hands out, carries no form to pin. The read crawl leaves the slot axis it moves in front of the projections guided by the $x$ it has read away, and there the composed mask's own row states the guard, so `mark_sparse_domains.mark_sparse_domain` on the composite is the candidate rule.

Two reads that state one map over two axes are written twice. The two gathers of
GLM-5.3 each mark a distance axis of their own, so the latent is read through two reads
whose rows agree and whose domain axes differ, and `same_read` compares the axes. A
comparison up to a renaming of axes of one size and one form would write the read once
and copy it, and is not written.

*Touches: [[Advanced Axis Dynamics]], [[Crawlers]], [[Padding and Masks as Sparse Axes]].*

### The arrow forms leave details of their arrows open

- A tape reaching an arrow lights with its slot alone, because
  `ParaWrapDisplay.axis_highlight_tokens` recognises `scr.AxisAnchor` and no arrow
  anchor.
- A result whose array has a datatype anchor, as the `Natural` identifiers of an
  embedding do, carries a small triangle on the branch of the fan that anchor paints,
  because `bb.DatatypeAnchor.update` puts one on every wire it paints. Removing it needs
  a hook in `BroadcastedCategoryRenderer.ts`.
- The second render of the boot figure in one headless session is 21 px wider than the
  first, under the arrow form and under the axes form alike, so the difference predates
  the form.
- A `Quantization.TypeConvert` has no name, so under `ARROWS_AND_BOXES` it is a box
  labelled `TypeConvert`, and a quantised page holds many of them. A face registered for
  the class in `operatorFaces.ts`, naming the format written, would replace the label. A
  few numbers of `additionalOperationBoxes.ts` are copied into `operatorFaces.ts` and the
  box settings, and exporting them would remove the copies.
- `DiagramSettings.clean_quantisation_labels`, true by default, takes the quantisation
  off every top-level wire written at the quantisation of every operand of the operation
  writing it, so an arrow form labels such a wire with its bare datatype. The pages of
  [[Website Notebooks]] set the field false. The user has not ruled on whether the
  default should keep the quantisation.

*Touches: [[Diagram Display]].*

### A clean cut of a figure depends on a narrow window of widths

- `Multiline.ts` in tsncd places the first members of the next block whenever they fit in
  the room left in a row, so a width that leaves every block whole lies in a window a few
  tens of pixels wide, per the paragraph on `width` in [[Diagram Display]]. The window of
  a figure moves with any change to one of its blocks. A block setting that keeps the
  body of a block on one row would make the cut hold at every width.
- tsncd draws the name of an `ops.Arithmetic` above the top wire of its operand, and when
  that wire is the top wire of a block the name overlaps the title of the block. The
  classic models avoid it by writing the branch without the name first.
- `broadcast_between_positions_and_channels` has three copies, in
  `notebooks/sota/GLM53/rotary_embedding.py`,
  `notebooks/sota/DeepSeekV41Flash/rotary_embedding.py` and
  `notebooks/classic/shared_mechanisms.py`. It belongs in `construction_idioms.py`.
- `SubBlocks.BODIES_NOT_YET_DRAWN` drew the body of the box `PE` over `x` beside the whole
  transformer after the figure of the input embedding had drawn it.

*Touches: [[Diagram Display]], [[Website Notebooks]].*

### Two gaps the outer and inner tape slots opened

- A `ParaBlockOperator` broadcast over a degree that drops onto an outer slot is refused
  with `OuterDropInBroadcastBox`, because nothing sums over the degree between the box's
  result and the drop of the wrap.
- `ops.Dropout` holds its randomness inside the operator. No expression in the package
  grabs a uniform sample from an inner slot, so the inner slot of a random variable is
  stated and not yet used.

*Touches: [[Outer and Inner Tape Slots]], [[Para Block Operator]].*

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
