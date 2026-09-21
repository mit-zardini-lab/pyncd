---
tags: [layer/practice, concept]
code: notebooks/sota/, notebooks/base_features/BuildingAModel.ipynb
status: evolving
aliases: [Building a Model]
agent: Claude (Opus 5, high effort, 2026-09-10)
---

# Representing Models

This note holds every ruling the author has given on how a deep learning architecture is
written as a morphism in [[Broadcasted Category|Br]]. It was consolidated on 2026-09-10
from the logs under `obsidian/00-meta/logs/`, from the notes those logs corrected, and
from `notebooks/base_features/BuildingAModel.ipynb`. Each ruling states the rule, gives
the form that was rejected beside the form that replaced it where the source gives both,
and names the log the ruling was recorded in. A reader who finds an old form in a log can
check the *Reversed rulings* section below for the date it was superseded.

A new correction on how a model is expressed is added to this note, in the section it
belongs to, with the rejected form beside its replacement and a link to the log that
carries it. `notebooks/base_features/BuildingAModel.ipynb` holds the asserted code for
every rule that has a code form, so a rule with code is added there in the same session,
in the section it belongs to. A rule about presentation or about a modelling choice with
no code form lives in this note alone.

## The rules in short

The rules a reader needs before writing a model, in the order a construction meets them.
Each has a subsection under *The rulings* with the log and the rejected form.

1. **Declare the axes explicitly.** Predeclare every structural `RawAxis` and pass the
   axis objects rather than strings. Composition aligns by position and never by name, so
   exact identity beats guessing at what `@` will align, per [[UIDs and Names]].
2. **Reach for a standard operator first, and add a custom one only for new semantics.**
   `Einops` and `Linear` come before `GenericOperator`. An elementwise product is
   `Einops.template('a, a -> a')`, because the unit-axis spelling parses and `align_axis`
   cannot merge an axis whose size is a literal.
3. **A weight rides a `Linear`, and a per-index weight produces its axis.** A `Linear`
   with no inputs is a parameter array. An up-projection in a mixture of experts is
   `Linear (m,) -> (e, f)` broadcast over the tokens, and the down-projection produces the
   expert axis again and is followed by a diagonalisation.
4. **The target of a weight array is what makes a map act per channel.** A tap table whose
   target carries the window, the heads and the channels is a depthwise convolution. The
   same table read as a `Linear` over the window alone, tiled over the channels, is one
   shared filter and draws the same shape.
5. **Selection is the `TopK` in `deepseek/data_structure.py`, drawn compressed.** Pass
   `template(axis=)` so the caller's axis is consumed and `template(name=)` so two
   selections in one model are distinguishable. What consumes the sparse axis decides how
   far the annotation travels, per [[Sparse Axes]].
6. **A selection sits between the router's projection and the softmax.** The softmax then
   normalises over the experts that were chosen, which is the order DeepSeek-V3,
   DeepSeek-V4 and Mixtral use.
7. **Factor a reindexing.** A reindexing acting on some axes is the acting
   `StrideMorphism` times the identity on the rest, written as
   `View.template(reindexing=(stride, id_c))`, per [[Stride Category]].
8. **A stride morphism may name one axis in its domain and in its codomain.** That is the
   statement that a convolution preserves length, and a padded convolution needs it, or the
   gathered array carries a fresh output axis that nothing else names.
9. **Write a window's domain with the offsets and the channels last.** Composition aligns
   a `Linear` against the last axes of what it is handed, so the order of a window's domain
   decides which axes the following contraction takes.
10. **A position-dependent map consumes its axis.** Positional encoding, rotary embeddings
    and a compressor's positional bias are each a function of the index, and a function of
    the index cannot be tiled over it.
11. **Put the operand a composition supplies first in a signature.** The surplus wire is
    then the one the composition does not carry, so `'e f, e -> e f'` reads the gates as the
    surplus and `'e, e f -> e f'` lifts the projection against them.
12. **Assert on the domain and the codomain of a chain.** `cat.Composed` reads its domain
    off its first entry and its codomain off its last and compares no neighbouring pair, so
    a composition that does not compose raises nothing.
13. **A skip connection is a tape slot.** A compression path and an expansion path then
    carry none of each other's wires, and the contravariant path has the domain and the
    codomain the covariant one has.
14. **Present compactly with `ops.BlockOperator`.** A block wrapped as a bold named
    operator draws as one box whose body renders once beside the main figure, and the
    definition stays attached. Reserve `GenericOperator` for a body that cannot be written.
15. **A value one layer publishes and later layers read is a tape slot.** The publishing
    layer ends in a `Para.Drop` onto the slot and each reading layer begins with a
    `Para.Grab` from it. Both are composition-neutral, so every sublayer keeps the
    signature the reference's layer has, and a repeated `cat.Block` carries the residual
    alone. The assertions that decide a layer plan are `block.dom() == block.cod()` on
    every repeated block and that every slot dropped in the model is grabbed in it.
16. **Every block carries a fill colour, and a derivative rule never invents a block.** An
    uncoloured block has no drop shadow and cannot be seen, and the only block in a backward
    pass is the one a forward block produced.
17. **Every size is a free symbol, assigned by name in one `NumericConfig`.** The selection
    counts and the window widths are included, so no `Integer` literal appears in a
    construction. Keep the name bodies to a single letter.
18. **Draw through `notebooks/display/notebook_diagrams.py`.** Declare a `DiagramSettings`
    at the top of the notebook and pass it to every call, and deliver each diagram in the
    notebook's INLINE output, per [[Notebooks]].
19. **Simplify the drawing with the [[Yoneda and Cartesian Tricks]].** Draw the smaller
    form and leave the schedule to the cost tooling, since choosing a presentation to look
    cheap hard-codes a decision the algebra exists to derive.
20. **Expand onto the tape, and draw the implicit or the wrapped form.** An expansion into
    Para, of a selection's index or of an operator's parameters, is the full form, and
    `expand_sparse_onto_tape` is the default expansion of a selection. The tied form that
    `tie_tapes` and `expand_sparse` reach is available and is not worked in, and a figure
    shows the implicit form or the `ParaWrap` form unless the tied form is asked for.
    `show_diagram` draws the `ParaWrap` form by default since 2026-09-12, for a loop's
    variables as for a model's parameters.
21. **Every axis of the result of a contraction is a degree position.** An `Einops` writes
    no axis of its own, so a hand-built one has an output weave of tiled positions alone
    and reads an axis it shares with an operand through that operand's reindexing. Build a
    contraction over declared axes with `einops_simplification.einsum`, which places every
    axis from the shapes.
22. **A negative index is the unit, and causality is the sign of an index.** A causal
    window is read back from the current token, at `i_x - i_w`, built with
    `mark_sparse_domains.guarded_view` so that its output carries the slot axis as an
    `AffineGuards.AffineSparseAxis` and the guard arises from the negative stride on the
    window axis. A condition no read states is a relative read followed by an
    `aops.CovariantView` merge. Put the relative read on the array the condition
    restricts, which for the reachability of a compressed entry is the keys, and let
    `aops.CovariantView.broadcast_over_absent_axes_and_merge` broadcast the input over
    the axes the merge reads and the input does not carry. Write no mask
    operator and no guard field, and write no row onto an axis a `Rearrangement` deletes,
    per [[Padding and Masks as Sparse Axes]] and [[Advanced Axis Dynamics]]. Write the
    part of a reindexing that changes nothing as a rearrangement, a product with the
    identity and a composition, and keep the row that makes an axis sparse in the stride
    morphism.
23. **Two axes carrying two forms are concatenated, not joined into one axis.** The
    window slots `w|x` and the selected slots `s|x` of a V4.1 attention core hold two
    affine forms, and their union is no affine form, so `aops.ConcatenateAxes` states the
    pair. The concatenated axis is an `AxisConcatenation.ConcatenatedAxis`. It references
    its parts, its size is the sum of theirs, its label is theirs joined, `w|x + s|x`,
    and it carries no form.
    `concatenation_expansion.expand_concatenations` rewrites every consumer that treats
    the concatenated positions one at a time into the consumers of the two parts, per
    [[Advanced Axis Dynamics]].
24. **A body computed once per index of an axis is one `ops.BlockOperator` broadcast over
    that axis, and the form is confirmed rather than trusted.** The axis leads every array
    of the written-out form, `discovering_broadcasts` deletes it from the body, and the
    confirmation expands the candidate and compares, per [[Discovering Broadcasts]].
25. **A boxed layer that reads and writes a tape is a `ParaBlockOperator`, and it carries
    the tape at its own ports.** Each grab of the body becomes a leading operand of the
    box and each drop a trailing result, and the box is returned inside a `ParaWrap`
    naming the slot at each of those positions, so the wrap has the apparent domain and
    codomain of the block and nothing looks inside the body to see which slots it
    touches. The block itself keeps its seeds, so the body drawn beside the box shows
    them, per [[Para Block Operator]].
26. **A contraction against a parameter array is a `Linear` that produces the array's
    axes, followed by a diagonalisation.** The weight then sits inside the operator, which
    is the form every other weight takes.
27. **A block carries a long title naming the thing in words, and the box over it carries
    a short name.** A title is read in a contents list and a box name inside a box a few
    characters wide.
28. **A fixed one-hot is a covariant read of one index with no operands.** A reindexing is
    a linear map, so the row that selects an index read the other way injects a scalar at
    it, and no opaque operator is needed for a constant vector.
29. **A part the reference implements as one module is one box, and a box may stand inside
    a box.** V4.1's gate is the `Gate` box inside the `MoE` box, running from `W^{R}` to
    the six normalised gates. Recycling the block keeps the sparse axis the box hands out,
    the broadcast confirmation and the tape expansion both cross a nested box, and
    `node_with_box_named` reads a morphism holding two of them.
30. **A payload read at a selection's positions is a `ds.IndexSelect` broadcast over the
    selection axis.** A gather reads one entry per slot, so the slot axis is a degree
    position and the targets are a rank-0 `Natural` index and the payload's parent axis. A
    wire that passes through an operator says the operator is broadcast over that axis, and
    a wire that ends at one says the axis is part of a target, so the slot axis in a target
    draws a gather as a consumer of the slots.

31. **An axis cut into the parts that fill it is a deconcatenation.**
    `aops.DeconcatenateAxes` holds the part reindexings an `aops.ConcatenateAxes` of the
    same parts holds. Each is the reverse derivative of the other, and the standard
    expansion of a deconcatenation is a copy followed by one `View` per part.
32. **Reals that are rotated in pairs are complex numbers.** `dst.PairsAsComplex` reads
    `R[z]` as `Complex(R)[t]`, a rotation is a product with a complex exponential, and
    `dst.Decomplex` writes the pairs back.
33. **An operator whose value depends on an index is a `GenericOperator` with a
    definition over every position.** The left-hand side of a `cat.DefinedExpression`
    is the operator and the right-hand side is the expression that computes it, with
    the index supplied as a value by an `ops.Arrange`, so neither side reads one
    position. A figure draws the two sides with `:=` between them, and the same term
    states that an operator equals its expansion.
34. **A value that is proportional to a position is a product with an `ops.Arrange`,
    and a formula in a position is an `ops.Arithmetic` applied to one.** An
    `ops.Arrange` over the axis `x` has no operands and returns `Nat(x)[x]`, the array
    that holds `i_x` at index `i_x`. `ops.multiply_by_positions` writes the product of
    that array with a real array, because the two operands carry two datatypes, and
    `ops.Arithmetic.template` takes the datatype of its result as `output_datatype`.
    The formula of an `ops.Arithmetic` holds no free index. The axis of the result is
    the axis the indices are arranged over, and two formulas that read the index of
    one axis share one `ops.Arrange`, whose result is copied.
35. **A clamp between two bounds is an `nm.Clamp`.** The clamp to the unit interval is
    the default and prints between corner brackets, and it expands to `nm.IsPositive`
    terms.

## The rulings

### Operators and what they stand for

- A root-mean-square normalisation is an `ops.Normalize` whatever it carries. The
  operator declares `gain`, `bias` and `epsilon`, so the four combinations of a learned
  gain and a learned bias, and every epsilon a released kernel adds, are one operator.
  A model no longer writes the normalisation out in primitives because of its epsilon,
  and no longer writes an `ops.L2Norm` beside a scale because it has no gain.
  `ops.L2Norm` stays for a normalisation stated as a division by the root of a sum,
  whose result has length one.
  Rejected: `ops.L2Norm.template((n, m))` followed by the root of the number of values,
  and a copy, a square, a contraction and an `ops.Arithmetic` named
  `(x / |n| |m| + \varepsilon)^{-1/2}`.
  Replacement: `ops.Normalize.template((n, m), gain=False, bias=False, epsilon=...)`.

- A division by a sum plus an epsilon is an `ops.L1Norm` carrying that epsilon. The norm
  divides by the sum and never by the mean, and `ops.NO_EPSILON` is the unit of addition,
  so the exact form and the form the released kernel runs are one operator at two values.
  Rejected: a copy, a contraction and an `ops.Arithmetic` named `(x + \varepsilon)^{-1}`,
  for a row of the Sinkhorn normalisation or for the gate normalisation of the mixture.
  Replacement: `ops.L1Norm.template(epsilon=...)`, or `l1_norm_over(shape, position,
  epsilon=...)` where the normalised axis is named by its position.

- An inverse rotary embedding is the counterclockwise table followed by an elementwise
  conjugate. A table of factors of length one is turned back by conjugating it, so one
  table and one `ops.Arithmetic` named `\overline{x}` say what a second table said, and
  the `RotationDirection` of a `dst.Rotary` is gone.
  Rejected: `dst.YarnRotary.template(x, t, direction=CLOCKWISE)`, whose name carried a
  conjugate bar.
  Replacement: `table @ dst.conjugate_complex_values(table.cod()[0])`.

- Every rotary table writes its own name above its circle. `\mathrm{RoPE}`,
  `\mathrm{YaRN}` and the conjugated form are all rotaries and are told apart by
  reading the name, not by a mark inside the glyph.
  Rejected: the name in a cap inside the circle, and only for a `YarnRotary` or a
  clockwise table.
  Replacement: the name in a strip above the circle, for every table.

- Every block title and every block description of a model lives in one module, which
  the construction modules import as `text`. A formula stays beside the algebra that
  builds it, because a formula is written from the axes of the morphism. A description a
  module composes from a layer number or a size is a template in that module, filled in
  at the call.
  Rejected: a description written as a literal at the `cat.Block.template` that carries
  it.
  Replacement: `description=text.ENCODER_GROUP_DESCRIPTION`, and
  `text.WINDOW_LAYER_DESCRIPTION.format(layer_number=...)` where it is composed.

- Reach for a standard operator first and add a custom one only where the semantics are
  new. `Einops` and `Linear` come before `GenericOperator`.
  Rejected: a `GenericOperator` for an operation the standard set can state.
  Replacement: the standard operator.

- An elementwise product is an einsum along one shared letter, broadcast over the rest of
  the shape. The unit-axis spelling parses and gives a `RawAxis` sized `Integer(1)`, and
  `align_axis` cannot merge an axis whose size is a literal.
  Rejected: `ops.GenericOperator.template('⊙', 'f, f -> f')` and
  `ops.Einops.template('1, 1 -> 1')`.
  Replacement: `ops.Einops.template('a, a -> a')`.

- A normalisation that divides by a sum is one `ops.L1Norm`, not the copy, contraction,
  reciprocal and product that write it out. The four appear in the Sinkhorn iteration
  twice per round, in the mixture of experts once, and inside every `SoftMax`, and one
  operator states the mathematics where the four leave the reader to recognise it.
  `L1Norm` carries the `contracted` flag a `SoftMax` carries, since both read every
  position of the axis to write every position.
  Rejected: `route((0, 0), (MIX,)) @ (hold(MIX) * (ops.Einops.template('n N -> n') @
  reciprocal())) @ ops.Einops.template('n N, n -> n N')`.
  Replacement: `l1_norm_over((n, N), 1)`.

- A `SoftMax` is an exponential followed by an L1 norm. `expand_softmax` composes
  `rewrite_softmax_as_l1_norm` with `expand_l1_norm`, so the five primitives it has always
  returned are now the statement of two identities rather than one block of construction.
  Rejected: writing the normalisation out a second time inside `expand_softmax`.
  Replacement: `chsh.make_composed(exponential_of_softmax(target),
  expand_l1_norm(l1_norm_of_softmax(target)))`.

- A `SoftMax` that stands before a repeated normalisation of the same axis is written as
  the exponential, and its normalisation joins the repetition. The Sinkhorn normalisation
  of the released DeepSeek-V4.1-Flash takes a row softmax, a column normalisation and then
  nineteen rounds of a row and a column normalisation. The softmax is the exponential
  followed by the first row normalisation, so the same function is the exponential and
  twenty equal rounds, which is the `hc_sinkhorn_iters` of the released configuration.
  The equality needs the epsilons left out, which the model does, because the released
  kernel adds one after the softmax and one to every later sum.
  Rejected: `over((n,), ops.SoftMax.template()) @ columns @ cat.Block.template(rows @
  columns, repetition=19)`.
  Replacement: `over((n, N), exponential()) @ cat.Block.template(rows @ columns,
  repetition=20)`.

- Name the axis a normalisation consumes by its position in the shape, not by the axis.
  A weave is a sequence of positions and the position is what says which of them the
  operation consumes, per the invariant on manipulating a `Broadcasted` by position.
  `over` prefixes its axes onto the degree, so it cannot state the Sinkhorn column
  normalisation, which consumes the first of two positions.
  Rejected: `over((N,), ops.L1Norm.template())` for a column of an `(n, N)` matrix, which
  builds the shape `(N, n)`.
  Replacement: `l1_norm_over((n, N), 0)`.

- A gather is an operator of its own. An `Einops` signature describes a contraction and a
  reindexing is affine, so neither states a selection. The weave layout that pairs the
  sparse axis with its parent as two targets was correct before the operator existed, per
  [[Sparse Axes]].
  Rejected: `ops.Einops` with `signature=((), ())` under rank-1 weave targets, and its
  attempted repair `((), (0,))`, which means a sum over the parent axis.
  Replacement: `ds.Select`, whose targets are `(k/n), (n) -> (k/n)`.

- A one-hot indicator contracted against the payload was built as a way to write a gather
  and then withdrawn. It reads as an ordinary contraction and materialises `n` times `k`
  values to keep `k`.
  Rejected: `TopK` emitting a 0/1 tensor `(q, k, n)` with the gather written
  `'q k n, n c -> q k c'`.
  Replacement: the pre-existing `ds.TopK` on a `SparseAxis`, consumed by `ds.Select` or by
  an identifying reindexing.

- A repeat is a `View` whose reindexing does not name the repeated axis, and no operator
  may introduce one. A degree axis that no operand's reindexing names is one the output is
  repeated along, so `einsum` has an absorbed case and a degree case and no produced case.
  Rejected: `para.data_structure.contraction.Broadcast`, with its own derivative rule, its
  own absorption rule and its own glyph.
  Replacement: `ops.Einops.template('q -> q v')`, which writes a `View`.

- Every axis of the result of an `Einops` is a degree position. An `Einops` writes no axis
  of its own, so a hand-built one has an output weave of tiled positions alone. An axis the
  result shares with an operand is a degree axis, and the operand's reindexing names it. The
  target positions of an operand are the axes it contracts, and the signature gives each of
  them a contraction group. An axis written as a target of an operand and again as a target
  of the result is related across the two positions by its uid alone. Read by position, the
  operand then has a contracted position with no group, and the result has a position that
  no operand supplies. `agent_display` prints the axis in braces on both sides, and the
  renderer draws the two wires unconnected. `einops_simplification.index_shapes` raises
  `RuntimeError: generator raised StopIteration` on the operand whose position has no
  group. The domain and the codomain are those of the correct form, so an assertion on the
  ends of a chain passes. `einops_simplification.einsum` over the declared axes places every
  axis from the shapes, and it is the replacement that the ruling on hand-built
  contractions under *Composition traps of `@`, `*` and `>>`* already names.
  Rejected: the grouped output projection of `notebooks/sota/DeepSeekV41Flash.ipynb` as a
  hand-built `Broadcasted` whose weight weave is `(T, j, c, o)` and whose result weave is
  `(T, T, o)`, under the signature `((0, 1), (0, 1))`, with both reindexings read from the
  degree `(x, g)`, where `T` is `cat.WeaveMode.TILED`.
  Replacement: `einops_simplification.einsum(((x, g, j, c), (g, j, c, o)), (x, g, o))`,
  whose degree is `(x, g, o)`, whose weight weave is `(T, j, c, T)` read at `(g, o)`, and
  whose result weave is `(T, T, T)`.

- The operator that carries a reindexing is called `View`, because it is always coupled
  with one.
  Rejected: `ops.Identity`.
  Replacement: `ops.View`.

- Affine geometry never justifies an opaque operator. A window, a convolution and a
  one-position shift are one factored reindexing at three strides, so each is an
  `sc.StrideMorphism` times the identity on the untouched axes, carried by an `ops.View`.
  Rejected: a `GenericOperator` named `Shift` for the map `x` to `x + 1`, and `Conv₄` drawn
  opaque for compactness.
  Replacement: the factored reindexing, wrapped in `ops.BlockOperator` where a box is
  wanted.

- A `GenericOperator` is reserved for a body that cannot be written. After 2026-08-18 the
  only ones left under `notebooks/sota/` are Kimi K3's `KDA`, its `AttnRes` and its `0`
  source, all of them unpublished.
  Rejected: `GenericOperator` for `Compress`, for `Conv₄`, for `Shift` and for the sparse
  applies.
  Replacement: `ops.BlockOperator` over an expressible body.

- An operation the operator set cannot state is defined in full beside the term that
  stands for it, in a markdown cell of the notebook that draws it, giving its value at a
  generic index and the function it computes. A `GenericOperator` carries a name and
  shapes and no formula, so the term alone does not say what the operation does, and the
  user reads the definition to choose the operation's representation.
  Rejected: `custom('\mathrm{blk}', (), ((B,),), datatype=cat.Natural(P.local_size()))`
  with its value stated only as "the block a compressed position belongs to" in a table.
  Replacement: the same term beside a cell that defines it, *Defining the block table*
  on that day and *Defining the merge* since the table became a merge, which states
  the split it inverts, $i_B = |u| \cdot i_P + i_u$ with $0 \le i_u < |u|$, its value at
  a generic position, $\mathrm{blk}[i_B] = \lfloor i_B / |u| \rfloor$, and what its
  consumer computes, $\mathrm{mask}[i_x, i_B] = \mathrm{indicator}[i_x, \mathrm{blk}[i_B]]$.

- Addition is `ops.AdditionOp.template()`, RMS normalisation is `ops.Normalize.template()`,
  and `ops.SoftMax.template()` consumes the last axis of the array it is composed after.

- **A normalisation with no learned weight is an `ops.L2Norm` and the root of the number
  of values.** The user asked on 2026-09-18 for "an L2 norm operator (that doesn't take
  values)" that "should form part of the RMSNorm, Mixing Coefficients etc". An
  `ops.Normalize` carries a gain of one learned weight per position, and a model whose
  normalisation has no weight says so by writing the operator that has none.
  `ops.L2Norm.template(shape)` divides an array by the root of the sum of the squares
  over every axis in `shape`, so the root mean square is the `L2Norm` followed by an
  `ops.Arithmetic` that multiplies by the root of the number of values. The four streams
  of a token in Single-Pass mHC are normalised that way. Where the released code adds an
  epsilon under the root, as `mhc_with_epsilons` does, the normalisation is not a
  division by the root of a sum of squares and stays written out in primitives.
  Rejected: `ops.Normalize.template((n, m))` for the four streams of a token, whose gain
  the released code does not have.
  Replacement: `ops.L2Norm.template((n, m)) @ over((n, m), scale_by_the_root)`.

- `ds.TopK` is the only operator that marks an axis sparse. `ds.Select` reads a compressed
  payload and `ds.IndexSelect` reads an expanded one, so the three are one mechanism in
  three positions, per [[Sparse Axes]].

- The reverse of a selection is an operator of its own, `Inject`, whose signature runs from
  the sparse axis to its parent and which takes the selector as an operand so that it states
  where the cotangent goes.
  Rejected: the identity of the selection's domain, which typed only because a `SparseAxis`
  carries its parent's extent, and which unified the sparse axis with its parent in the
  reverse graph.
  Replacement: `para/data_structure/inject.py`, with the index grabbed from the slot the
  selection dropped.

- The operator that seeds a fold is `ops.ConstantOp`, so that the operator and the numeric
  `nm.Constant` do not collide. A class name is the wire format, so the two cannot share
  one.
  Rejected: `ops.Constant` beside `nm.Constant`.
  Replacement: `ops.ConstantOp`.

- A fixed one-hot vector is the covariant reading of the row that selects one index, with
  no operands. Every reindexing is a linear map. A row with an empty domain and the shift
  `i_x` sends the one index of the empty product to position `i_x`, so an `ops.View` of it,
  which reads a reindexing contravariantly, selects position `i_x` of the array it reads.
  Read covariantly the same row writes its operand at `i_x` and leaves every other
  position holding the universal unit, which is the one-hot vector at that position. The
  collapse coefficients V4.1's first sublayer starts from are that vector on the first
  stream. `aops.CovariantView.template` cannot build it, because it asks
  `mark_sparse_codomains.merge_groups` for a bijection between the two index boxes and an
  injection out of the empty product is not one, so the `cat.Broadcasted` is written
  directly, as a parameter array is.
  Rejected: `custom('A^{0}', (), ((x, n),))`, a nullary `GenericOperator` over the
  positions and the streams.
  Replacement: a `cat.Broadcasted` over `aops.CovariantView` of the row
  `_dom=(), _cod_stride_shift=((n, (), 0),)`, with no input weaves and
  `backup_degree=(x,)`.

- A `cat.DefinedExpression` states that one expression is defined to be another. It has
  a `left_hand_side` and a `right_hand_side` with one domain and one codomain, and a
  figure draws the two sides with `:=` between them. It is a term that is drawn and is
  never composed, so it is no morphism.
  `algebra.define_by_expansion.define_by_standard_expansion` pairs an operator with the
  expansion `algebra.registries.standard_expansions` registers for it.
  Rejected: an operator drawn in one figure and its expansion in a second figure, with
  the equality of the two stated in the caption.
  Replacement: `define_by_expansion.define_by_standard_expansion(cut_rotated_channels())`.

- An operator whose value depends on an index stays a `GenericOperator` in the model,
  because no broadcast operator is a different function at every index of its degree,
  and a `cat.DefinedExpression` gives it its meaning. Both sides of the definition are
  written over every position. The right-hand side takes the index as a value from an
  `ops.Arrange`, per the three rulings below, so neither side reads one position. The
  first form of this ruling defined the operator at one index, and it is listed under
  the reversed rulings.
  Rejected: `generic_operator('\\Theta', (), (ANGLES,))`, a table with no operands whose
  dependence on the position is stated in prose alone.
  Replacement: `cat.DefinedExpression.template(yarn_ramp(), pair_positions() @
  clamp_pair_index_to_ramp())`.

- A value that is proportional to a position is the product of the position and a
  factor, and the position of every index of an axis is an `ops.Arrange`. The operator
  has no operands. Its result over the axis `x` is `Nat(x)[x]`, the array that holds
  `i_x` at index `i_x`, so it is `[0, 1, ..., |x| - 1]`, and its datatype is the datatype
  of an index of `x`. The axis stands in the target of the output weave, because a
  tiled operator is the same function at every index of its degree. The angle of a
  rotary embedding is the product of the positions and the frequencies, and the
  exponential `e^{i x}` is then one `ops.Arithmetic` applied to the angle, so the
  formula holds no index and the definition of the rotation reads no position. The
  operands of the product carry two datatypes, a `Natural` and the reals, and
  `ops.Einops.template` gives one datatype to every weave, so
  `ops.multiply_by_positions` writes the weaves out.
  Rejected: an `ops.Arithmetic` with the formula `e^{i i_x x}`, where `i_x` is a free
  numeric, applied to the frequencies on the right-hand side of a definition at `i_x`.
  Replacement: `(ops.Arrange.template(x) * yarn_frequencies()) @
  ops.multiply_by_positions(TOKEN_POSITIONS, VALUE_OF_EVERY_PAIR)`, followed by the
  `ops.Arithmetic` of `e^{i x}` from the reals to `Complex(R)`.

- A formula in a position is an `ops.Arithmetic` applied to an `ops.Arrange`, so that
  the formula of an `ops.Arithmetic` is a function of `nm.x` and holds no free index.
  The formula reads a `Natural` and returns a real, and `ops.Arithmetic.template`
  reads the first datatype from `base` and takes the second as `output_datatype`. An
  expression with two such factors is assembled in two steps and multiplied. The YaRN
  frequencies are the power of the base, assembled from the index of every pair, times
  a formula of the ramp.
  Rejected: `ops.Arithmetic.template((1 - nm.x + nm.x / s) * beta ** (-2 * PAIR_INDEX /
  |z|), name='(1 - x + x / s) \\beta^{-2 i_t / |z|}')` applied to the ramp read at `i_t`,
  whose formula holds the free numeric `i_t` and whose name is written by hand.
  Replacement: `((ops.Arrange.template(t) @ ops.Arithmetic.template(beta ** (-2 * nm.x /
  |z|), base=PAIR_POSITIONS, output_datatype=R)) * (yarn_ramp() @
  ops.Arithmetic.template(1 - nm.x + nm.x / s, base=VALUE_OF_EVERY_PAIR))) @
  ops.Einops.template('t, t -> t')`.

- A value that is a formula in the index of an axis is that formula applied to the
  `ops.Arrange` of the axis, and it is not defined at an index. The formula is an
  elementwise map from `Nat(t)` to the reals. The axis of the result is the axis the
  indices are arranged over, and the formula is broadcast over it. Two formulas that
  read the index of one axis share one `ops.Arrange`, whose result is copied.
  Rejected: `cat.DefinedExpression.template(yarn_ramp() @
  value_at_index(VALUE_OF_EVERY_PAIR, 0, PAIR_INDEX), <an ops.ConstantOp whose value is
  nm.Clamp((PAIR_INDEX - RAMP_START) / (RAMP_END - RAMP_START))>)`, and one
  `ops.Arrange` in front of each formula of the frequencies that reads the index.
  Replacement: `pair_positions() @ clamp_pair_index_to_ramp()` for the ramp, where the
  formula is `nm.Clamp((nm.x - RAMP_START) / (RAMP_END - RAMP_START))`, and
  `pair_positions() @ route((0, 0), (PAIR_POSITIONS,)) @ (raise_base_to_pair_index() *
  (clamp_pair_index_to_ramp() @ interpolate_along_ramp())) @
  ops.Einops.template('t, t -> t')` for the frequencies.

- Integer arithmetic on identifiers is written out over `Natural` datatypes whose bounds
  say which integers each wire can hold. A product of two naturals is an
  `ops.Einops` with nothing contracted, retyped by `ops.with_datatypes` to the product
  of the bounds. A bitwise exclusive or is `ops.BitwiseXor`, a reduction over the axis
  it consumes, and it acts on a `Natural(2^N)` only. A remainder is `ops.Modulo`, and
  carries the divisor's datatype. A fixed table of integers the model never learns, such
  as the multipliers and the primes of a hash, is an `ops.FixedArray`. A prefix of an
  axis, one per order, is a guarded view whose row reads before the first position for
  the orders that do not reach it, so that the reduction reads the unit there.
  Rejected: `generic_operator('\mathrm{hash}', (Nat(v)[L|x],), (Nat(T)[G, K],), degree=(x,))`,
  and an exclusive or over the whole of `L` followed by a modulus per order, which
  hashes the 4-gram under every order.
  Replacement: `multiply_by_multipliers() @ xor_prefixes() @ (hold(...) *
  primes_and_offsets()) @ (reduce_modulo_primes() * hold(OFFSETS)) @ offset_into_table()`
  in `omitted_mechanisms.py`, with the view named prefix reading slot
  `i_L' + i_G + 1 - |G|` for the order `i_G`.

- A bound is chosen so that the arithmetic states its own result, and a cast stands only
  where the declared bound is wider than the values reach. The multipliers of the Engram
  hash are `Natural(2^63 / v)`, where `v` bounds the identifiers, so the product bound
  `v * 2^63 / v` is `2^63` once `nm.cancel_reciprocal_factors` cancels the symbol against
  its reciprocal, and the products are 64-bit integers with nothing cast. A fixed array
  that another fixed array determines is computed from it: the primes are one
  `ops.FixedArray` over an axis of the `|G||K|` pairs, the offsets are the sum of the primes
  before each pair, read through a guarded view whose row `i_GK' + i_GK - |GK|` reads
  before the first prime for the slots the pair does not reach, and both arrays are then
  read over the orders and the heads by one reindexing. The remainder plus the offset
  keeps its cast into the row count, because the addition declares the sum of the two
  bounds and the values stay below the row count for a reason the datatypes do not carry,
  and the inspection box over the cast states the reason.
  Rejected: `MULTIPLIER = cat.Natural(nm.FreeNumeric.named('\bar{\alpha}'))` followed by
  `ops.Cast.template(PRODUCT, to=INT64)`, and `ops.FixedArray.template('o', (G, K),
  TABLE_ROW)` beside `ops.FixedArray.template('p', (G, K), PRIME)`.
  Replacement: `MULTIPLIER = cat.Natural(INT64.max_value / COMPRESSED_IDS.max_value)`,
  `PRODUCT = cat.Natural(nm.cancel_reciprocal_factors(...))`, and `primes_and_offsets()`,
  which copies `FixedArray('p', (GK,))`, sums one copy through the view named before and
  reads both through the view named pair.

### Weights and Linears

- Every weight rides an `ops.Linear`, and after the rules are applied no explicit weight
  array remains in a model's construction.
  Rejected: a `GenericOperator` for an expert projection, and a weight array carried on a
  wire.
  Replacement: `Linear` in both directions.

- A `Linear` with no inputs is a parameter array, because a linear map out of the empty
  product is a constant tensor. The array then draws in the weight-box idiom every other
  learned tensor uses.
  Rejected: a new `Parameter` operator, and a `GenericOperator` standing for a bias table.
  Replacement: a 0-input `ops.Linear`.

- A weight is not lifted over a batch axis. A weight does not vary with the batch, so a
  lifted weight would state that it is drawn once per batch element. State the batch axis
  in the einsum beside the weight instead, per [[Construction Helpers]].
  Rejected: lifting a per-token block that holds parameter arrays over the sequence axis,
  and `morphism_object_lift` applied to a weight table.
  Replacement: `'x m, m f -> x f'`, with the batch axis written into the signature.
  The mechanical half was superseded on 2026-08-21, and the modelling advice stands on its
  own.

- A parameter table that a per-position operation needs is broadcast at the consumer. Give
  the consuming operation a reindexing that drops the degree, and the table is read at its
  own shape whatever position the consumer is at.
  Rejected: lifting a window-by-channel bias table over the compressed entries.
  Replacement: a nullary source and an `ops.AdditionOp` whose reindexing drops the degree.

- A per-index weight produces its axis. An up-projection in a mixture of experts is
  `Linear (m,) -> (e, f)` broadcast over the tokens, so the expert axis rides the implicit
  weight tensor and one weight is not shared between the experts.
  Rejected: explicit per-expert weight arrays, and lifting one `Linear` over the expert
  axis.
  Replacement: `ops.Linear.template(('m',), ('e', 'f'), name='up')`.

- The down-projection of a mixture produces the expert axis a second time and is followed by
  a diagonalisation. The diagonalisation is a copy-`Rearrangement` reindexing that keeps the
  entries where the produced expert and the slot's expert agree. The gates then land on the
  down-projected outputs and the combine is an explicit contraction over the sparse axis.
  The gates may sit on either side, because `Σₑ Wₑ(gₑ⊙hₑ) = Σₑ gₑ(Wₑhₑ)`.
  Rejected, as the form a model notebook draws: `Linear (e, f) -> (m)` consuming the expert
  axis whole with the gates multiplied into the hidden state first.
  Replacement: `Linear (f) -> (e, m)` broadcast over the slots, then the diagonal, then the
  gates, then `Einops.template('k, k m -> m')`.

- The down-projection and its diagonal sit inside the `Experts` block, whose domain and
  codomain do not mention the expert axis, so the merge of the sparse axis with its parent
  stays inside the block. The router outside then scores a dense expert axis and the listing
  shows the selection reading the parent axis into the sparse one.
  Rejected: the down-projection outside the block, which spreads the sparse annotation onto
  the router.
  Replacement: an `Experts` block taking a token in and the sparse pair out.

- The diagonal follows the down-projection's codomain rather than the residual width. Kimi
  K3's down-projection produces the mixture latent, so the diagonal, the gates and the
  combine all happen at bottleneck width and the widening projection runs once, afterwards.

- A scalar head weighting is a `Linear` onto the empty shape, once the score einsum puts the
  weighted axis last.
  Rejected: a custom operator, and an explicit weight vector on a wire.
  Replacement: `ops.Linear.template(('i',), ())`.

- The target of a weight array is what makes a convolution depthwise. A tap table whose
  target carries the window, the heads and the channels is per channel, and a `Linear` over
  the window alone tiled over the channels is one shared filter with the same drawn shape.
  Rejected: `ops.Linear.template((win,), (), 'W^C')` broadcast over the heads and the
  channels.
  Replacement: a parameter array over the window, the heads and the channels, contracted
  over the window against a window view.

- An integer output size on a `Linear` mints that many fresh axes, which the signature that
  consumes them names by position.

- A `Linear` reads two kinds of input. A real datatype is contracted, and a `Natural` index
  selects one of the weights rather than contracting. `ops.selects_weights` is the
  condition, and a `Linear` is linear in the einops-rearrangement sense only when no input
  selects, per [[Operators]].
  Rejected: expanding a selecting `Linear`, which contracts the selection axis away and
  computes the average of every weight where one weight was meant.
  Replacement: `expand_linear_root` returning a selecting `Linear` unchanged, with
  `parameter_arrays` building its weight from every input rather than from the first.

- A `Linear` is a parameter load and a contraction, and an expression may state the weight
  in three forms. The implicit form holds the weight inside the operator, the Para form
  feeds it in from a `Grab`, and the third form is a weight array with no inputs. Once the
  weight is an array the `Linear` is replaced by the `Einops` that contracts it, and the
  `Einops` keeps the `Linear`'s name, so the contraction is read as the projection it
  performs, per [[Linear Expansion]].
  Rejected: naming the expanded operators through `expand_linear_root`'s default name,
  which gives every projection one name.
  Replacement: the Para route, which names the `Einops` after the `Linear`.

- A parameter read by both passes has to be one array with no inputs in the graph, so that
  the forward `Linear` and the backward `Transpose` read one wire.
  Rejected: a shared input wire of the joint morphism, where each pass reads its own copy.
  Replacement: one nullary weight array, merged by `merge_duplicate_roots`.

- A grabbed parameter takes the first operand slots of the seed it belongs to, ahead of the
  operands the seed already had, so the tape arrives above the data in a diagram.
  Rejected: permuting the operands inside `to_para_wrap`, which breaks the round trip back
  to the grabs and the drops the wrap stands for.
  Replacement: `grab_parameters` prepending the parameter weaves and their grabs.

- The gain of an RMS normalisation is a parameter like any weight. It is grabbed out of the
  operator and becomes a weight array, and the operator then stays, because a normalisation
  is not a contraction and has no `Einops` to become.

- A contraction against a parameter array is written as the `Linear` that produces the
  array's axes, broadcast over the axes the two share, followed by the diagonalisation
  that keeps the entries where the axis the weight produced and the axis of the input
  agree. The two forms state the same map. The `Linear` holds the weight inside the
  operator, which is the form every other weight in a model takes, and it leaves the
  reader nothing to check about which positions of the weight are contracted and which
  are shared. The diagonalisation is the copy-`Rearrangement` reindexing the
  down-projection of a mixture of experts already uses. V4.1's grouped output projection
  was the one contraction against a parameter array in the four models, so it is the one
  place the rule applies.
  Rejected: `einops_simplification.einsum(((x, g, j, c), (g, j, c, o)), (x, g, o))`
  against a `weights()` array on `(g, j, c, o)`.
  Replacement: `ops.Linear.template((j, c), (g, o))` lifted over the groups, followed by
  `ops.View.template(reindexing=cat.Rearrangement((0, 1, 1, 2), (x, g, o)))`.

- A linear map whose results are cut into parts is written as one linear map per part. A
  linear map followed by a slice of its results is the linear map whose weight is that
  slice of the rows, so the parts need no axis to be cut from and no slice. The released
  DeepSeek-V4.1-Flash holds the three sets of mHC coefficients as one weight of 24 rows and
  cuts the 24 results into 4, 4 and 16. It also multiplies each part by a scale of its own,
  and each scale is part of one of the three weights.
  Rejected: `ops.Linear.template((n, m), (y,), 'H', bias=True)` onto a 24-wide axis `y`,
  followed by three views named `collapse`, `output` and `combine` that read `y` at the
  shifts 0, 4 and 8.
  Replacement: `ops.Linear.template((n, m), (n,), 'H_0', bias=True)`,
  `ops.Linear.template((n, m), (n,), 'H_1', bias=True)` and
  `ops.Linear.template((n, m), (n, N), 'H_2', bias=True)`, each reading a copy of the
  normalised streams.

- A row read from a table held in a block-scaled format carries the quantisation of the
  table and reaches the projection reading it with no cast. The released Engram table
  is E4M3 with one UE8M0 scale per 32 channels, the lookup multiplies a row by its
  scales into BF16, and the projection rounds it back with the same groups of 32, which
  changes no value, so the model writes the row at the quantisation of the table.
  Rejected: a lookup returning BF16 and a cast to E4M3 in front of the key and value
  projections of Engram, while the inspection box over the table said E4M3.
  Replacement: `embedded` in `quantization/registries/operator_quantisations.py`
  returning the quantisation the policy names for the table, MXFP8, and no cast.
  `quantization/validate_quantization.py` asserts it in
  `check_the_rows_of_the_ngram_table_reach_the_projections_with_no_cast`.

### Selections and sparse axes

- Selection is the pre-existing `ds.TopK`, with a rank-1 target over the scored axis
  broadcast over the queries and a codomain on a `SparseAxis`. Pass `template(axis=)` so
  the caller's axis is consumed, because a freshly minted axis wins canonicality and erases
  the caller's name together with its `assign_values` binding.
  Rejected: a new selection operator, and a fresh axis minted inside the template.
  Replacement: `ds.TopK.template(axis=, k=, name=)`.

- There are two ways to consume a compressed selection, and the one used settles how far the
  sparsity annotation travels. Reach for `ds.Select` where the payload is data on a wire,
  and identify the two axes in a reindexing where the payload is a weight. [[Sparse Axes]]
  holds the mathematics.

- A `Block` confines a merge when the parent axis appears in neither the block's domain nor
  its codomain, so where the box is drawn decides how far a sparse annotation reaches.

- Which axis a model selects over decides which spelling is usable at all. Over an axis the
  model invented, such as compressed entries or an expert axis, the identification costs one
  sub-block. Over the model's own sequence axis it prints on the embedding, on every
  residual, through the mixture and out of the unembedding.
  Rejected: pairing GLM-5.2's selected sequence axis with its parent in a reindexing.
  Replacement: `ds.Select`, with the parent axis on a target.

- A model is written compressed and unpacked by rewriting the graph. Writing the expanded
  form by hand hard-codes a presentation into the model and costs a second wire through
  every consumer of every selection, per [[Sparse Expansion]].
  Rejected: building the mixture on `TopK.complete` as the default, which was implemented,
  read and withdrawn in one session.
  Replacement: `TopK.template` throughout, with `expand_sparse` applied on demand.

- A selection may not be written into a reindexing, because the map from the sparse axis to
  its parent is a runtime value and a reindexing is affine. The dependence on data rides an
  operator or a datatype instead, which is what a `Natural`-typed operand does.

- Applying the gates and summing, where the payload already rides the sparse axis, is one
  einsum dot product.
  Rejected: a multiply followed by a separate sum, and a mask followed by a contraction.
  Replacement: `ops.Einops.template('k, k m -> m')`.

- `ds.TopK` accepts an `nm.Integer` and will hide the model's most interesting constant
  inside the expression, so a selection count is declared as a free symbol like every other
  size.
  Rejected: `ds.TopK(k=nm.Integer(16))`.
  Replacement: `k = nm.FreeNumeric.named('k')`, bound in the one `NumericConfig` call.

- Two selections in one model are labelled apart through `TopK.template(name=)`, or both
  draw as the same sparse axis.

- A selection sits between the router's projection and the softmax, so the softmax
  normalises over the experts that were chosen. The placement decides how far the sparse
  axis reaches in the reverse pass, and it is the order DeepSeek-V3, DeepSeek-V4 and Mixtral
  use.
  Rejected: the selection after the softmax, which normalises over every expert and then
  selects.
  Replacement: the router's projection, then the selection, then the softmax.

- A Para expansion of a selection is written as independent pieces joined by the tape. The
  selection drops the index to a slot, each parametric operation grabs it, and nothing is
  wired between them. A separate pass ties the tape when the wire is wanted, because a tape
  is what makes a feature modular.
  Rejected: writing the routed expansion first and taping it afterwards, and making the pass
  a `ParaWrap`, which is the display form.
  Replacement: `para/algebra/para_sparse_expansion.py`, with `para/algebra/tie_tapes.py` as
  the connector.

- On a model with a sparse axis, expand the selection before grabbing the parameters.
  `parameter_arrays` reads a `Linear`'s target and treats the sparse axis as an axis of the
  activity size, so the expert weights come out as one slab rather than the whole bank.
  Rejected: `grab_parameters` first on a sparse model.
  Replacement: `expand_sparse` first, then `grab_parameters`.

- A complete `TopK` declares its index output as its residual, so the index is dropped and
  grabbed like any other residual and needs no slot convention of its own. The compressed
  form still needs a slot derived from the axis, because there the index is on no wire.

- Do not model a top-k selection with a relaxation. A Gumbel-softmax, a soft top-k and a
  straight-through gate each change the model, none of the three published architectures
  uses one, and the index is a residual rather than a value to be smoothed.
  Rejected: any relaxation of the selection.
  Replacement: the index as a `Natural`-typed residual with no cotangent.

- A selection expands onto the tape. `expand_sparse_onto_tape` is the default expansion,
  and its result, a Para morphism with each index dropped by its `TopK` and grabbed by
  each consumer, is the full form of a model that selects. `expand_sparse` reaches the
  tied form directly, exists to state that the expansion depends on nothing in `para`, and
  has no case for a `Grab` or a `Drop`, so a model already in Para expands on the tape
  alone. The tape crosses a `BlockOperator` without a wire, so a boxed layer plan expands
  on the tape where the wired pass refuses the sparse axis on the box.
  Rejected: extending `expand_sparse`'s classifier to step over a `Drop`, which put a
  Para case into the pass that exists to be free of Para.
  Replacement: `expand_sparse_onto_tape`, with a rule for `Select` and one for a
  `BlockOperator` holding a selection.

- A selection whose values are never read is written in the `ONLY_SELECTION` form of
  `ds.SelectionForm`, and hands out positions alone. The candidate pool of
  DeepSeek-V4.1-Flash keeps 2048 of the blocks and reads none of their scores, so its
  Top-2048 hands out `Nat(P)[x, p]`, the merge is applied to those positions with
  `dst.merge_selected_positions`, and the pool leaves as `Nat(B)[x, C]`. Nothing maps a
  value to one, because the positions state on their own which of the parent's entries
  are active. A consumer reads a payload at them with `ds.IndexSelect`, and a selection
  made among them reports positions of the parent through
  `ds.TopK.template(positions_of=)`.
  Rejected: `KEEP_BLOCKS @ ops.Arithmetic.template(nm.Integer(1)) @ every_offset @ COVER`,
  a `TopK` in the `WEIGHTS` form whose values are replaced by ones, repeated over the
  offsets and merged onto a sparse axis, so that a `Select` could multiply an indicator
  onto the scores it restricts.
  Replacement: `dst.TopK.template(k=npool, axis=P, form=ONLY_SELECTION)`,
  `dst.merge_selected_positions(BLOCK_SPLIT, p_axis, (x,), C)`, a `dst.IndexSelect` on
  the scores, and `dst.TopK.template(axis=C, selected_axis=sB, positions_of=B)`.

- A selection whose values the model never reads is written in the `ONLY_SELECTION` form,
  and `ds.Select` reads the payload at its positions with the selection axis in the target
  of the positions and of the result, `[Nat(n); k], [R; n] -> [R; k]`, broadcast over
  every other axis. Whether the values are read is a fact about the reference's
  computation. A `TopK` followed by `[x > 0]` reads the sign of each surviving value and
  keeps at most `k` entries, those among the top `k` whose value is positive, so it is the
  positions form only where the reference reads the positions alone. DeepSeek-V4.1-Flash's
  indexer returns indices alone, and its attention reads every entry the Top-512 returns
  whatever the entry's score. It keeps fewer than 512 only where a pick is an entry the
  query cannot reach yet, meaning one whose group of tokens ends at or after the query's
  own position. It marks such a pick `-1`, and its kernel gives that slot no weight. The
  `[x > 0]` the notebook drew had no counterpart there. Where the indicator was zero, the
  entry stayed in the softmax, because a gathered entry multiplied by zero scores zero and
  adds `exp(0) = 1` to the denominator.
  Rejected: `over((x,), select_entries(b) @ indicator())` followed by a `ds.Select` on
  the sparse axis, `(s/b), (b) -> (s/b)`, in the Full layers of DeepSeek-V4.1-Flash.
  Replacement: `dst.TopK.template(k=nsel, axis=b, selected_axis=s,
  form=dst.SelectionForm.ONLY_SELECTION)` followed by a `ds.Select` at positions,
  `Nat(b)[s], (b) -> (s)` broadcast over `x` and `c`, which `ds.Select.at_positions`
  builds before it is broadcast.

- A selection over an axis live on a run from its first position hands its slots out on
  an axis live on the same run. `dst.TopK.template(axis=b_reach)` mints `s|x` through
  `AffineSparseAxis.selected_slots`, and the two halves of V4.1 select at different ratios, so their slot
  axes carry different forms and are two axes, read off the selections that mint them.
  Rejected: one declared `s` shared by the encoder's and the decoder's selections.
  Replacement: `s_e` and `s_d`, read off `ENCODER_SELECT` and `DECODER_SELECT`.

- `dst.MergedPositions` computes an array of positions from one block number, and its
  operand and its result are two arrays over two datatypes. The operand is a rank-0
  `Natural` counting the blocks of the split axis, broadcast over the degree and over the
  slots the selection filled. The result is a `Natural` counting the entries of the axis
  the split was taken from, and it carries the offsets as its target, because each
  position it holds is a function of an offset. `deepseek/validate_sparse.py` asserts both
  weaves in `check_merged_positions_weaves`. The reindexing pentagon tsncd drew the
  operator as until 2026-09-15 is the glyph a reindexing is drawn as, so the thick
  datatype wire entering its point and the thick datatype wire leaving its flat edge read
  as one wire carried through a split.
  Rejected: `MergedPositionsBox` drawing `aob.draw_merge_pentagon` pointing left, with the
  stride of the block number placed on the result's datatype row.
  Replacement: a rectangle in the selection family's green named `blk`, with the stride of
  the block number at the left edge on the row the block number arrives on, and the stride
  of each offset at the right edge on the row that offset leaves on.

- A gather reads one entry per slot, so it is elementwise in the slot and the slot axis
  belongs in the degree. Write it as `ds.IndexSelect`, whose targets are a rank-0
  `Natural` index and the payload's parent axis, broadcast over the slots and over
  everything else. A wire passing through an operator says that the operator is broadcast
  over that axis, and a wire that ends at an operator says that the axis is part of a
  target the operator consumes, so the slot axis in a target draws the selected slots as
  a wire the gather consumes and a fresh wire it emits. The two spellings compute the
  same values, and the weaves are the only thing that tells a reader which of the two a
  model means.
  Rejected: `ds.Select.at_positions`, `[Nat(n); k], [R; n] -> [R; k]`, with the slot axis
  in the target of the positions and of the result, broadcast over the queries and the
  channels alone, which the entry gather of DeepSeek-V4.1-Flash used from 2026-09-14.
  Replacement: `ds.IndexSelect` over the degree `(x, s, c)`, with the positions weave
  `Nat(B)[T, T]` read at `(x, s)`, the payload weave `R[T, r|x, T]` read at `(x, c)` and
  the output weave `R[T, T, T]`, which is the form
  `candidate_pool.restrict_to_candidates` already reads the pool's distances with.

- A wire drawn across an operator says that the operator is broadcast over the axis the
  wire carries. A `dst.TopK` reads the axis it selects over, so that axis sits in the
  target of every weave. The operand's axis ends at the left point of the diamond and each
  result's axis starts from the right point. The two are never linked, whatever the
  relation between them. The rule holds in all four values of `dst.SelectionForm`. The
  result rides the sparse axis `k/n` in the `WEIGHTS` form and a dense axis of size `k` in
  the other three, and the parent axis is consumed in every one.
  A consumed axis is never linked across the operator that consumes it. A gather's slot is
  a degree, written as a `ds.IndexSelect` broadcast over it, per the ruling above.
  Rejected: `TopKBox.pass_selected_axis_through` in tsncd, which loosened the first target
  axis of the operand and of each result and linked them, so one wire crossed the diamond
  the way a degree wire crosses an operator.
  Replacement: the operand's axis ending at the left column and each result's axis and
  datatype leaving the right column, with the degree axes alone crossing the box.

### Reindexings, windows and views

- A reindexing acting on some axes is written as the acting `StrideMorphism` times the
  identity on the rest, which is convolution's own form, per [[Stride Category]].
  Rejected: an opaque operator, and a reindexing rebuilt over every axis.
  Replacement: `View.template(reindexing=(stride, id_c))`.

- A reindexing that cuts two axes is the product of one group view per axis and the
  identity on the rest, with a rearrangement before it that brings the grid indices to
  the front. One row over a domain of five positions says the same thing and says it as
  one opaque table, where the product draws one glyph per axis with its own strides and
  the rearrangement draws as the crossing wires. The order the product leaves the target
  in is the order the following `Linear` reads, so the weight is written in that order.
  Rejected: `_dom=(H', W', M, U, V)` with three rows onto `(H, W, M)`, and
  `Linear((M, U, V), (m,), 'W^{A1}')` beside it.
  Replacement: `route((0, 2, 1, 3, 4), (H', W', U, V, M)) @ (grp * grp * id_M)`, whose
  domain is `(H', W', U, V, M)`, and `Linear((U, V, M), (m,), 'W^{A1}')`.

- A stride morphism may name the same axis in its domain and in its codomain, which is the
  statement that a convolution preserves length. A padded convolution has to do so, or the
  gathered array carries a fresh output axis that nothing else names and a residual
  connection has nothing to add to.
  Rejected: a window built with `StrideMorphism.from_matrix`, which mints fresh output axes.
  Replacement: a domain of the grid axes and the offsets, with the codomain naming the grid
  axes again.

- The order of a window's domain decides which axes the following contraction takes, because
  `@` aligns a `Linear` against the last axes of what it is handed.
  Rejected: a domain interleaving each grid axis with its own offset, which silently
  contracts the width and its offset.
  Replacement: a domain putting the grid axes first and the offsets and the channels last.

- An upsample is the transpose of the downsample's gather, built from one shared
  `StrideMorphism`. A nearest-neighbour or sub-pixel upsample cannot be a reindexing,
  because the map from a fine index to a coarse one is not affine.
  Rejected: a nearest-neighbour upsample written as a reindexing.
  Replacement: `transpose.transpose_reindexing` of the downsample's gather, which is a
  transposed convolution and ties the two directions to one gather.

- A position-dependent map consumes the position axis. Positional encoding, rotary
  embeddings and a compressor's positional bias are each a function of the index, and a
  function of the index cannot be tiled over it.

- Draw the smaller form and leave the schedule to the cost tooling. The two rewrites are
  stated in [[Yoneda and Cartesian Tricks]], and applying them in sequence collapses a
  compressor built as a linear, a window view per branch, a parameter array and an addition
  into one window view followed by a linear with its bias flag set.
  Rejected: choosing the presentation so that it looks cheap, which hard-codes a scheduling
  decision the algebra exists to derive.
  Replacement: the two rewrites, applied by hand as a modelling choice.

- A repetition occurs as late as possible. A node read by several morphisms is copied over
  the fan-out of its output and each reader absorbs its own copy, so the broadcast happens
  in each reader's own reindexing rather than once in front of the copy.
  Rejected: leaving a `View` in front of a copy, and offering the fan-out search a
  contraction rule, which would compute the contraction twice.
  Replacement: `reindexing_absorption.absorb_nodes` through
  `merge_producers_into_every_consumer`.

- A per-block selection is put back onto the positions by the merge of the block split,
  read covariantly. The split `(P, u) -> B` is affine and is a `View`. Its inverse rounds
  down and is not affine, and `aops.CovariantView` reads the split the other way, which
  is a function because the split is a bijection. `dst.merge_selected_axis` hands out a
  sparse axis over `B` whose active set is every position of a kept block, a `Select`
  reads the scores there, and a `TopK` over it hands out a selection over `B`.
  Rejected: a nullary `GenericOperator` table over `cat.Natural` giving each position its
  block, consumed by `ds.IndexSelect`, and the mask multiplied onto the scores.
  Replacement: `dst.merge_selected_axis(BLOCK_SPLIT, (pP, u), name='c/B')`, then a
  `dst.Select` on the scores and `dst.TopK.template(axis=cB, selected_axis=sB)`.
  Later the same day the pool moved to the `ONLY_SELECTION` form, per the ruling under
  *Selections and sparse axes*, so the merge is applied to the block numbers and the
  pool carries no values. `dst.merge_selected_axis` remains the merge of a selection in
  the `WEIGHTS` form, which `deepseek/validate_sparse.py` keeps as `merged_selection`.

- A read outside an axis holds the universal unit, and causality is written as the sign of
  an index. A window's negative shift states that the first slots of an early query are
  empty, and `mark_sparse_domains.guarded_view` marks the slot axis `w|x`, where
  `ops.View.template` composes the same rows and marks nothing. A condition no single read
  states, such as the reachability of a compressed entry from a query, is written as a
  relative read followed by a merge, and the relative read is on the array the condition
  restricts. The condition restricts the entries, so the indexer's keys are read at
  `i_b - i_r`, entry `r` back from every entry, which marks the slots `r|b` and makes an
  entry before the first a negative index that reads the unit. One `aops.CovariantView`
  writes each entry's slots onto every query whose newest reachable entry that entry is,
  at `i_x = |a| i_b + i_a + (|a| - 1)`, and its input carries the entries and not the
  offsets, so `aops.CovariantView.broadcast_over_absent_axes_and_merge` broadcasts the
  input over the offsets and the `|a|` queries of a group all read the same entry at the
  same distance. The slots leave the merge as `r|x`, which
  `mark_sparse_codomains.mark_sparse_codomain` derives, the query axis stays dense, and
  every selection holds distances back from the query's newest reachable entry, which the
  gather reads through the same chain. There is no guard field and no mask operator.
  Rejected: `ops.WeightedTriangularLower`, a zero-one mask multiplied onto the scores, a
  `_guard_stride_shift` field on `StrideMorphism` for rows no array position reads, and
  the `reach` view, an `ops.View` through `copy_query @ (hold(x) * distance) @
  delete_distance` that computed the distance from the entry's last token to the query
  onto an axis and deleted it. The stride category is Cartesian, so a copy, a map on the
  copy and a deletion compose to the identity on the position, and the algebra may erase
  the row that carried the causality. The reach was the replacement of 2026-09-14 and was
  rejected on 2026-09-15. Rejected the same day: a second row of the merge writing the
  entry `i_{b_0} - i_r`, so that the output held absolute entries, because the Top-k
  reads the slots as they are.
  Rejected later on 2026-09-15: the `grp` views reading the query low rank and the hidden
  state by query group and offset at `|a| i_{b_0} - i_a + 2 |a| - 2`, with the scores
  merged back to the queries. The group views state the same form as the key read, and
  they read three arrays where the condition restricts one, mark an offset axis `a|b0` no
  computation uses, need the query axis sized `|a| |b|` for that mark, and put every
  operation of the scoring at the group and the offset rather than at the query.
  Replacement: `read_back_from_each_entry` and `write_to_every_query_of_the_group` in
  `notebooks/sota/DeepSeekV41Flash/lightning_indexer.py`, the keys read back from every
  entry and one covariant view onto the queries, each a read or a write no rewrite
  removes.

- Write a causal window as a read back from the current token, `x' = i_x - i_w`, so that
  the guard arises from the negative stride on the window axis. Slot `w` is then the
  token `w` positions back and slot `0` is the current token.
  `mark_sparse_domains.guarded_view` marks `w` as an `AffineGuards.AffineSparseAxis`
  under the form `i_x - i_w`, whose empty end is `LAST`, and the live slots at query `q`
  are `0` to `q`. A positive stride with a negative shift states the same set of live
  slots by the other route, so it leaves two spellings of one read, and the requester
  ruled on 2026-09-15 that the rule "a guard comes from a negative stride" holds
  throughout.
  Rejected: `x' = i_x + i_w + 1 - |w|`, a positive stride on `w` with the shift
  `1 - |w|`, whose guard comes from the shift falling below zero and whose empty end is
  `FIRST`.
  Replacement: `x' = i_x - i_w`, which also reads the taps of a convolution in the order
  a convolution reads them, because slot `w` multiplies the token `w` back.
  Applied to the window of `notebooks/sota/DeepSeekV41Flash.ipynb` and to
  the window fixture of `advanced_axis_dynamics/validate_advanced_axis_dynamics.py`. The
  group windows of a compressor keep their positive stride, at
  `|w| i_b + i_w + \mathrm{shift}`, because a compressed entry holds a group of tokens
  anchored at an absolute position rather than a window ending at the current token.

- Declare a size relation the marking needs on the axis, as a product of the sizes it
  is made of. The query axis of V4.1 is `cat.RawAxis(_size=a.local_size() *
  b.local_size())`, and the decoder's entries take `x.local_size()`, so a view onto the
  query axis is examined for reading past its end and the offsets of the last query
  group are marked. Composition replaces a size symbol by the product it meets, per
  `composition.size_class`, so the template axes named `x` in every einops take the
  product. Nothing derives the relation from the compressor's view, which is the item
  *Axis sizes are declared rather than derived* in [[Open Gaps]].
  Rejected: `cat.RawAxis.named('x')` beside `cat.RawAxis.named('b')`, with the ratio
  relating them only in prose, which leaves the last group's overshoot unprovable.
  Replacement: the product size on `x`, and the concrete sizes bound through `|a|` and
  `|b|` alone.

- Write the part of a reindexing that changes nothing as a `cat.Rearrangement`, a product
  with the identity and a composition, and keep only the rows that compute something in
  the `StrideMorphism`. A rearrangement says that nothing changes, where a unit-stride row
  has to be read to learn the same fact. The one exception is the row that makes an axis
  sparse. A view that produces `b|x` from a dense `b` pairs two different axis objects,
  which a rearrangement cannot do, so that row stays in the stride morphism at unit
  stride, and `mark_sparse_domains.mark_sparse_domain` marks a composite from its last
  factor back to its first so the factored form marks the same domain.
  Rejected: one `StrideMorphism` for the reach with unit-stride rows for the query and
  the entry beside the distance row.
  Replacement, on 2026-09-14: `copy_query @ (hold(x) * distance) @ delete_distance`, with
  `distance` carrying the entry row and the distance row alone. The reach itself was
  rejected on 2026-09-15, per the ruling above, and the rule stands for every
  reindexing.

- Two axes carrying two affine forms are concatenated, and the concatenated axis carries
  no form. The window slots `w|x` and the selected slots `s|x` of a V4.1 attention core
  are live on two different runs, and the union of two runs is no affine form of a
  position, so no one `AffineGuards.AffineSparseAxis` states it.
  `aops.ConcatenateAxes` states the pair as one stride row per part, each writing its own
  axis into a run of a dense axis whose size is the sum, and
  `concatenation_expansion.expand_concatenations` restores each part's own form by
  rewriting every consumer that treats the concatenated positions one at a time. A core
  written with one exponential, one denominator sum and one contraction against the
  concatenated values expands into the two-branch core the notebook writes by hand.
  Rejected: one sparse axis over both kinds of slot, whose live positions would be the
  union of the two runs.
  Replacement: `aops.ConcatenateAxes.template(((h, x, w), (h, x, s)))` and the expansion,
  checked in `advanced_axis_dynamics/validate_advanced_axis_dynamics.py`.

- A concatenated axis references the axes it is made from and is labelled by them. The
  axis `aops.ConcatenateAxes` produces is an `AxisConcatenation.ConcatenatedAxis` whose
  `parts` are the axes laid end to end, whose size is written from theirs on every
  construction, and whose label `agent_display` and tsncd derive from the parts, so the
  window slots `w|x` beside the selected slots `s|x` read `w|x + s|x`. Composition
  replaces a part by the axis it meets and rebuilds the concatenated axis with it, so a
  core written over the raw `w` and `s` is labelled by the guarded axes a mode composes
  into it.
  Rejected: a fresh `cat.RawAxis` named `w+s` at construction, which kept that name after
  composition had replaced both parts.
  Replacement: `AxisConcatenation.ConcatenatedAxis(parts=(w, s))`, built by
  `ConcatenateAxes.template`, checked in
  `advanced_axis_dynamics/validate_advanced_axis_dynamics.py` and in
  `notebooks/sota/DeepSeekV41Flash/validate_deepseek_v41_flash.py`.

- A view that reads a sparse axis names that axis in its codomain, and a `Rearrangement`
  that declares an array names the sparse axis the array carries. Composition identifies
  a `RawAxis` with the sparse axis it meets and the sparse one wins canonicality, so a raw
  declaration spreads the sparse axis onto every array carrying the raw one.
  Rejected: `BLOCK_SPLIT` with the codomain `B`, and `route(..., cat.Array(R, (x, B)))`
  for the indexer's output, which turned the decoder's entries into `[B|x, c]`.
  Replacement: the codomain `B_reach`, and `cat.Array(R, (x, B_reach))`.

- The partition of an axis into the parts that fill it is a deconcatenation. The
  512 channels of a V4.1 latent are the 448 channels the rotary embedding leaves alone
  followed by the 64 it rotates, and `aops.DeconcatenateAxes` cuts the latent into the
  two arrays in one operator. The operator is declared beside `aops.ConcatenateAxes`
  with the same part reindexings, each is the reverse derivative of the other, and its
  standard expansion is a copy followed by one `View` per part. The axis being cut may
  be a declared axis, and the sizes of the parts have to sum to its size.
  Rejected: `ops.View.template(reindexing=(ROPE_TAIL,))`, the affine read of the last
  `|z|` channels at the shift `|c| - |z|`, which leaves the other channels on a second
  read that no operator states.
  Replacement: `aops.DeconcatenateAxes.template(((h, x, zbar), (h, x, z)),
  concatenated=c)`, with `zbar` declared at the size `|c| - |z|`.

### Blocks, repetitions and presentation

- Present compactly with `ops.BlockOperator`, which draws as one bold named box whose body
  renders once as a sub-diagram beside the main figure, and which keeps the definition
  attached so the box can be opened.
  Rejected: an opaque `GenericOperator` used for compactness.
  Replacement: `ops.BlockOperator.template(block, name)`.

- A `Block` with a repetition other than one denotes a loop over its body, and the
  repetition may be a free symbol, so a symbolic layer count draws as a loop over that
  symbol.

- Every block carries a fill colour. An uncoloured block has no drop shadow and cannot be
  seen, which is what makes a mis-nested one hard to debug.
  Rejected: `cat.Block.template` called with no `fill_color`, and a hand-built
  `BlockAesthetics` without one.
  Replacement: a default of white on `Block.template`, with `'white'` written explicitly
  where a block is built by hand.

- A derivative rule never invents a block. The only block in a backward pass is the one a
  forward block produced.
  Rejected: reverse blocks written by the derivative rules around the softmax and around
  each pointwise map.
  Replacement: plain compositions returned by the rules, with `backprop._block` supplying
  the one block a forward block earns.

- A block title is LaTeX as given. A title that is prose is written with an explicit
  `\text{}` from the Python side, and a title that is mathematics carries none.
  Rejected: wrapping every block title in `\text{}` inside the renderer, which made a title
  holding an exponent print its caret literally.
  Replacement: the title passed through unchanged, with the escape written at the one site
  that needs it.

- Only a block's title is drawn, on one line inside the block, and the annotation scales
  with the text. A long title stretches the figure and a title of several lines runs over
  the operations below it, so every title is one short line and a table carries the rest.
  Rejected: a three-line title on each block, and the whole assessment moved into one
  outer title, which stretched the figure to four thousand pixels.
  Replacement: one short line per block, with an assessment table beside the figure.

- A derived expression is drawn as the derivation returned it. Do not assemble a figure from
  separately derived pieces, and do not add a box or a title the derivation did not produce.
  Rejected: a display-side graph that wrapped each derived piece in a titled block inside an
  outer block.
  Replacement: `show_diagram` over the graph the derivation returned, with what a title
  would have said in the caption.

- A diagram is delivered in a notebook's INLINE output. Do not write PNG files into the
  repository to show one, and do not add a script whose purpose is to write them.
  Rejected: rendering a derived expression into a figures folder so it can be viewed
  outside the notebook.
  Replacement: executing the notebook with `DiagramMode.INLINE` and saving its outputs.

- The compressed form is the default for drawing, because it is one wire rather than two
  through every consumer and the smaller picture is the point of drawing one at all.

- A `MultiCategory` draws its rows stacked with a double dashed line between them, and a
  contravariant row draws its body mirrored so the composition order reads backwards with
  every glyph upright.
  Rejected: exchanging the anchors without mirroring the layout, which doubled the terminal
  dots, and mirroring the geometry with a negative scale, which reverses the glyphs and the
  names.
  Replacement: reflecting the built box tree, with the anchors exchanged after it.

- A grab or a drop is drawn onto the morphism it touches, as a `ParaWrap`, so a parameter
  arrives from above and a saved value leaves below. A seed writes its own drop the way it
  absorbs its own grab, so a selection emits its index to a slot rather than onto a wire
  that crosses the figure. [[Para Wrap]] holds the display rules.
  Rejected: a bare drop box at the end of a wire running the width of the picture, and
  laying the tape out along the domain and the codomain, which held every taped wire across
  the whole figure.
  Replacement: `to_para_wrap`, with a merge rule for a drop beside the one for a grab, and a
  drop emitted beside the producer of the wire it saves.

- A wrap does not absorb a seed that reads an operand through an affine index map a
  `cat.Rearrangement` cannot express, so a convolution's `View` keeps its grab beside it
  with the reindexing drawn between the two.

- A subclass of `Elementwise`, and any newly registered `Term`, needs its own box
  registration in the renderer, keyed on the exact constructor. Without one it draws as
  nothing and raises nothing.

- A value one layer publishes and later layers read is a tape slot. The publishing layer
  ends in a `Para.Drop` onto the slot and each reading layer begins with a `Para.Grab`
  from it. Both have the empty product on one side, so neither changes the domain or the
  codomain of the layer it sits in, every sublayer reads and returns the residual and the
  collapse vector alone, and a repeated `cat.Block` carries nothing else. In
  DeepSeek-V4.1-Flash the six slots stand for the fields of the reference's shared
  runtime, `shared_attn`: the encoder's entries and selection, written by each encoder
  Full layer and read by the five Reuse layers of its group, and the decoder's entries,
  indexer keys, candidate mask and selection, written at layer 20, the selection again by
  each Reindex layer, and read by every later decoder layer that uses them. A slot holds
  what was written to it last, as the runtime does. The whole-model figure is sent to the
  browser page with the block bodies left out and the tape drawn absorbed onto the boxes,
  so each slot leaves the box that writes it and enters each box that reads it.
  Rejected: the value on a wire through the repeated block, as a passenger that every
  sublayer the wire passed carried untouched. Each MoE sublayer of the decoder then took
  the entries, the keys, the mask and the selection in and handed them out, and the
  whole-model figure was fourteen thousand pixels tall, most of it wiring.
  Replacement: `Para.Drop` after the publishing box and `Para.Grab` before each reading
  box, with `slots_dropped(model) == slots_grabbed(model)` asserted beside
  `block.dom() == block.cod()`.

- A value a loop overwrites on every iteration is a loop variable, read by a
  `Para.StreamGrab` and written by a `Para.StreamDrop` of one slot. The loop grab reads the
  value the slot held when the iteration started and the loop drop writes the value the
  next iteration starts from, so their order in the body carries no meaning. Loops are a
  property of a `Block`, so the two seeds state the loop variable at the level of the
  loop, and a subclass of a seed is added where a loop variable needs one.
  Rejected: a plain `Drop` of the new maximum inside the attention loop, which the shift
  of the carried maximum, standing after it, read as already overwritten.
  Replacement: `StreamDrop<s>` on the new maximum and `StreamGrab<s>` at every read of the
  old one, written by `tape_loop_variables`, which is the form the loop expansion returns
  since later the same day.

- A body computed once per index of an axis is written as one `ops.BlockOperator`
  broadcast over that axis, and the form is confirmed rather than trusted. The queries and
  the learned sink of V4.1's attention core carry the head axis and the window latents and
  the selected latents do not, so the broadcast form has the reindexings
  `((0,), (0,), (), ())`, the sink arrives as a first operand whose target is `R[]`, and
  the body carries no head axis anywhere. The axis has to lead in every array and in every
  `Einops` result before the body can be written without it, because
  `remove_leading_degree_axes` deletes a prefix of the degree and refuses an operation that
  carries the axis at a later position. `confirm_broadcast_expansion` expands the candidate
  and compares it against the written-out core, so a candidate that states the wrong
  operand varies is refused with the domain position it disagrees on.
  Rejected: the core written out, with `h` in the degree of all eleven of its operations
  and in seven of its arrays, and the `((0,), (0,), (), ())` form asserted only in prose.
  Replacement: `discovering_broadcasts.discover_broadcast_over_axes(core, (h,), 'Core')`,
  or `broadcast_block_over_axes(body, (h,), ((0,), (0,), (), ()), 'Core')` on a body
  written without the head axis, each reported with its `BroadcastConfirmation`.

- A boxed layer that reads and writes a tape is a `ParaBlockOperator`, which lists the
  grabs and the drops of its body on the operator. A grab and a drop have the empty product
  on one side, so a block holding them has the domain and the codomain it would have
  without them, and a plain `ops.BlockOperator` over such a block says nothing about the
  tape. The seeds may stand inside the box, because `tutil.type_search` walks an operator's
  `block` field and finds a grab inside a box exactly as it finds one beside it, so
  `slots_dropped(model) == slots_grabbed(model)` holds either way. Broadcasting the box
  prepends the degree axes to the arrays the seeds carry and leaves the slots alone.
  Rejected: `Para.Drop(tape=SLOT, size=ARRAY)` composed after the boxed mode, with the
  slots visible only in the wiring around the box.
  Replacement: the seeds inside the body, with
  `ParaBlockOperator.template(block, 'Full')` recording them, and
  `broadcast_para_block_over_axes` where the box is broadcast.

- A block carries a long title naming the thing in words, and the `ops.BlockOperator` over
  it carries a short name. The title is read inside the block and in a contents list, and
  the box name is read inside a box a few characters wide in a figure holding forty of
  them, so a reader expands the short name from the title. A title that is prose is written
  with an explicit `\text{}`, because a block title is LaTeX as given.
  Rejected: `title='SWA'` beside `boxed(block, 'SWA')`, which names the box twice and says
  nothing in either place.
  Replacement: `title='\text{Sliding Window Attention}'` on the block, with
  `ops.BlockOperator.template(block, 'SWA')` naming the box.

- A mechanism written out over many operators inside a module is one `ops.BlockOperator`,
  drawn as one box in the module's figure, with the written-out body drawn once beside
  the figure and in the inspection box over the box. The Engram hash is six steps over
  the integer operators, and drawn as a `cat.Block` in place it filled the Engram figure
  with the steps of one operation of the module.
  Rejected: `hash_ngrams` returning `cat.Block.template(...)`, drawn in place inside the
  Engram module.
  Replacement: `boxed(cat.Block.template(...), named_for_layer(HASH_BOX, layer))`, a box
  named $\text{Hash}_1$ whose block keeps the title, the formula and the description.
  A cast between two naturals inside such a body opens its own box, and one whose
  naturals are held under a quantisation, as every index of the quantised model is,
  opens the same box with the format named, per `operator_explanations.explain_cast`.

- A part the reference implements as one module is one `ops.BlockOperator`, and a box
  stands inside a box where the part stands inside a larger part. V4.1's gate runs from
  `W^{R}` to the six normalised gates, which is the whole of the reference's `Gate` module,
  so it is a `Gate` box inside the `MoE` box and the expert projections beside it are read
  as the other piece of the mixture. Three things hold for a nested box with no work. The
  recycling `boxed` performs keeps the sparse axis the box hands out, so the axis the gate
  mints is the axis the experts consume and composition aligns by uid.
  `confirm_broadcast_expansion` confirms the outer box, because
  `lift.morphism_object_lift` lifts a `BlockOperator` over the degree as it lifts any other
  operation and both sides of the comparison hold the nested box.
  `para_sparse_expansion.expand_sparse_onto_tape` drops the index inside the gate box and
  grabs it at each expert projection outside it, because the tape crosses a box without a
  wire. A morphism holding two boxes is read with `node_with_box_named`, since
  `node_with_operator` cannot say which of the two is wanted.
  Rejected: `(picked @ normalise_gates(ke)) * routed_experts(ke) * shared_expert()`, which
  drew the ten operations of the gate flat beside the two expert blocks.
  Replacement: `expert_gate()`, which returns
  `boxed(cat.Block.template(picked @ normalise_gates(ke), title='\text{Expert Gate}',
  fill_color=GATE_COLOUR), 'Gate')` beside the axis it minted.

### Sizes and the configuration stage

- Every size is a free symbol, assigned by name in one `NumericConfig` in the last cell, the
  selection counts and the window widths included, so that no `Integer` literal appears
  anywhere in a construction.
  Rejected: integer literals for the selection counts, the window widths and the compressed
  extents written into the expression.
  Replacement: `nm.FreeNumeric.named(...)` declared beside the axes, bound by
  `config.assign_values(...)`.

- Keep a size name's body to a single letter. `DynamicName.from_str` splits on the
  underscore and `assign_values` matches on the body, so two names such as `k_e` and `k_x`
  collide into one entry.
  Rejected: `k_e` and `k_b` as two distinct sizes.
  Replacement: `k` and `s`, or a `DynamicName(body=...)` built directly.

- A number that is not sourced stays a symbol. The last cell prints what the configuration
  could not bind beside what it did, and the sizes dictionary keeps each bound value's
  provenance beside it.
  Rejected: inventing the undisclosed hidden width, head count, expert hidden and layer
  count of a model so that its configuration would look as complete as another's.
  Replacement: a sizes dictionary carrying provenance, and a printout of the symbols left
  unbound.

- Two letters written in two templates are two axes with two sizes, and `assign_values`
  matches on the name body, so one entry binds both. Name the value width `v` when the two
  must differ.

- An axis minted by an integer size takes the name of whichever letter alignment later puts
  at its position, so the integer in `Linear.template('m', 2)` is assigned through the
  letters the score signature supplies.

- A hand-named tape slot must not take the form the derivation's own slots take.
  `Para.new_slot` names slots from a counter that a derivation resets, so a hand-named `s0`
  and a derived `s0` print alike while being different slots.
  Rejected: naming a UNet's skip-connection slots `s0`, `s1` and `s2`.
  Replacement: `f0`, `f1` and `f2`, named after the feature map they hold.

- Two separate chains of `@` give one axis two size symbols, because `align_axis` merges the
  sizes separately and neither `RawAxis` takes priority. Compare rows by axis uid and print
  the names.

- A label holding more than one symbol keeps every letter and carries each one's assigned
  size as an exponent on that letter, per [[Compound Axis Labels]]. A count of the entries a
  selection keeps is named the way a size is named, between absolute bars, so that the
  count and the extent read alike.
  Rejected: the router's Top-6 over 384 experts drawn `6 of 384`, and a `Natural` over the
  context length drawn `2|b|`.
  Replacement: `|k|^{6} \text{ of } e^{384}`, and `|a|^{2}|b|`, drawn by passing the
  configuration's `assigned_integers_by_name()` to the figure as `assigned_sizes`.

### Composition traps of `@`, `*` and `>>`

- `@` composes in sequence and aligns axes by position. A name plays no part in the
  alignment, so two letters written in two templates are two axes until a composition
  puts them at the same position, per [[Construction Helpers]].

- A composition supplies what either side is missing. Surplus wires on the left pass beside
  the right side as identities, and the left side is lifted over surplus axes on the right.

- A bare tuple on the left of `@` is a `Rearrangement` whose output position copies the
  input its mapping names. Its domain is read from the first codomain position naming each
  index rather than by indexing the right morphism's domain by the mapping value.
  Rejected: reading the domain by indexing `f.dom()` with the mapping value, which worked by
  coincidence for every order-preserving mapping and broke on an interleaving one.
  Replacement: reading the domain from positions.

- `add_excess_lift` compares first objects alone and lifts the whole of the shorter side, so
  a product mixing a rank-deficient operation with held wires lifts the held wires too and
  raises on the axis lengths. Pre-lift the one operation explicitly.
  Rejected: a softmax placed beside a held wire in a product after a rank-3 wire.
  Replacement: `lift.morphism_object_lift(op, cat.ProdObject(axes))` on that operation
  alone, which the model notebooks call `over`.

- `*` composes in parallel, and a wire is carried past an operation by the identity on its
  array, which the model notebooks call `hold`.

- An identity written as an `Einops` signature loses the array's shape inside a product, so
  the product then fails positional alignment.
  Rejected: `Einops.template('feature model -> feature model')` as a carrier for a weight.
  Replacement: `ProdObject((weight_array,)).identity()`.

- `>>` broadcasts a morphism over a batch axis on every input and every output, and the
  batch axis becomes the first axis of every array.

- Import `construction_helpers` for its side effects even where the name is unused, because
  the `@`, `*` and `>>` overloads do not exist without it.

- A contraction built by hand from declared axes uses `einops_simplification.einsum` over
  the axis objects. `Einops.template` mints fresh axes, and a minted axis composed onto a
  declared one can win canonicality and rename the declared one throughout the term.
  Rejected: `ops.Einops.template('x d, x d -> x x')` beside declared axes, and a templated
  `Einops` composed onto a declared latent axis in a model module, which printed the latent
  under the template's own letter.
  Replacement: `einops_simplification.einsum` over `IndexVariable`s, as
  [[DeepSeek-V3 Backward Pass]] records.

- One declared axis may stand at two positions, and read by position that morphism is a
  contraction whose result carries one axis twice. Self-attention written from one copied
  input is the case, and every rule in the package reads a `Broadcasted` that way, per
  [[Invariants]].
  Rejected: deduplicating a shape, testing an axis for membership in a shape, keying a
  dictionary by an axis, and looking a position up with `shape.index(axis)`.
  Replacement: reading through `select_degree`, `select_target`, `target_idx`, a
  reindexing's mapping and a signature's groups.

- Where one axis must be read under a second name, as a key axis beside a query axis, write
  the second reading as an affine identity rather than letting composition merge the two.
  Rejected: three projections composed onto a score signature from one copy, which merges
  the query axis with the key axis and turns the contraction into a contraction of one axis
  with itself.
  Replacement: `diffusion_unet.relabel_axis`, reading the feature map under a second name as
  the affine identity, with each projection lifted over the shape it will be read under.

- Put the operand a composition supplies first in a signature, so the surplus wire is the
  one the composition does not carry.
  Rejected: `'e, e f -> e f'` after an up-projection, which aligns the projection's rank-2
  result against the rank-1 gates operand.
  Replacement: `'e f, e -> e f'`.

- An empty left side of an einops signature parses to one empty input segment, so
  `Einops.template('-> q v')` is a repeat of a scalar. A morphism with an empty domain is
  built by hand.

- `cat.Composed` reads its domain off its first entry and its codomain off its last and
  compares no neighbouring pair, so a composition that does not compose raises nothing. A
  notebook that builds one asserts on `dom()` and `cod()`.
  Rejected: relying on construction to catch a mismatched chain.
  Replacement: an explicit assertion on both ends.

- Composing axes whose sizes are literal integers through `@` raises, because alignment
  builds equality classes for the sizes.
  Rejected: `@` over axes carrying `Integer` sizes.
  Replacement: explicit `Composed` and `ProductOfMorphisms` over already aligned axes.

- Three projections reading one copy of a feature map cannot name different axes, because
  composition aligns by position and lifts each projection over the axes it is missing.

### Activations and numerics

- A function a reader knows is written with its own numeric, and the spelling in the
  primitive numerics lives in `nm.Expandable.expand_to_primitives` and in no formula and
  no inspection box. The sign, the magnitude, the larger of two values and the square
  root are `nm.Sign`, `nm.AbsoluteValue`, `nm.LargerOf` and `nm.SquareRoot`, since
  2026-09-19.
  Rejected: `sign = 2 * nm.IsPositive(d) - 1`, `magnitude = d * sign`,
  `clamped = EPSILON + nm.RectifiedLinear(magnitude - EPSILON)` and
  `nm.Power.template(clamped, 1 / 2)` for Engram's gate, and a box formula
  `y = \sigma(x) = (1 + e^{-x})^{-1}` that appends the expansion.
  Replacement: `nm.Sigmoid(nm.Sign(d) * nm.SquareRoot(nm.LargerOf(nm.AbsoluteValue(d),
  EPSILON)))`, and the box formula `y = \sigma(x)`.

- An elementwise map given by a formula is an `ops.Arithmetic` over `nm.x`. The algebra
  differentiates the formula and `torch_compile` evaluates it, so a derived backward pass is
  checkable against `torch.autograd`.
  Rejected: `ops.Elementwise.template(name='SiLU')`.
  Replacement: `ops.Arithmetic.template(nm.x * nm.Sigmoid(nm.x))`.

- A named `ops.Elementwise` carries no formula. The derivative rule reverses it into the
  primed name, which states nothing, and `torch_compile` evaluates every named `Elementwise`
  as a ReLU, so an activation named `SiLU` differentiates to a box holding nothing and
  computes a ReLU.

- `ops.ReLU.template(name='ReLU')` is the one named map that is what it says, and its
  derivative rule writes the indicator.
  Rejected: the generic `Elementwise` rule's opaque primed name.
  Replacement: a rule of its own emitting the written indicator. Since 2026-09-11 the indicator is `Arithmetic<[x > 0]>`, the derivative of
  `nm.RectifiedLinear`, where it was the named `Elementwise<\mathbb{1}_{>0}>`.

- A clamp is written with `nm.RectifiedLinear`, whose expansion is `x [x > 0]`, so the
  clamp's derivative is a sum of `nm.IsPositive` terms.
  Rejected: `clamp(v, L) = (v + L - sqrt((v - L)^2)) / 2`, which a reader has to work out
  is a clamp, and whose derivative is `0 * inf` at `v = L`.
  Replacement: `ops.Arithmetic.template(nm.x - nm.RectifiedLinear(nm.x - L))` for a clamp
  above `L`, and `nm.x - nm.RectifiedLinear(nm.x - L) + nm.RectifiedLinear(-L - nm.x)` for
  a clamp on both sides.
  Since 2026-09-17 a clamp on both sides is an `nm.Clamp`, per the next ruling, and a
  clamp on one side keeps this form.

- A clamp between two bounds is an `nm.Clamp`, which holds the argument, the lower
  bound and the upper bound. The clamp to the unit interval is the default and prints
  between corner brackets alone, `\ulcorner x \lrcorner`, and any other bounds are
  written on the closing bracket, `\ulcorner x \lrcorner_{-\lambda}^{\lambda}`. It is
  an `nm.Expandable`, and its expansion is
  `lower + (x - lower) [x - lower > 0] - (x - upper) [x - upper > 0]` in `nm.IsPositive`
  terms, so its derivative is the difference of the two indicators.
  Rejected: `nm.RectifiedLinear(y) - nm.RectifiedLinear(y - 1)` for the YaRN ramp, which
  a reader has to work out is a clamp.
  Replacement: `nm.Clamp(y)`, and `nm.Clamp(nm.x, -LIMIT, LIMIT)` for the SwiGLU clamp.

- A function the primitive numerics can also spell is written with its own
  `nm.Expandable` numeric, and a formula keeps that numeric. A pass expands it only to
  compare two formulas spelled differently, and keeps the numeric when the comparison
  fails.
  Rejected: `1 / (1 + nm.E ** -nm.x)` for the logistic function, and
  `nm.x * nm.IsPositive(nm.x)` for the rectified linear function.
  Replacement: `nm.Sigmoid(nm.x)` and `nm.RectifiedLinear(nm.x)`.

- The GELU is written as an `ops.Arithmetic` over `nm.CumulativeGaussian`, which is exact.
  Neither standard approximation is used, because each puts a float into an expression that
  otherwise holds free symbols alone.
  Rejected: `ops.Elementwise.template(name='GELU')`, and the tanh and sigmoid
  approximations.
  Replacement: `ops.Arithmetic.template(nm.CumulativeGaussian(nm.x) * nm.x)`.

- A published name for an activation is not unique, so a name is given only where the
  published name is shorter than the formula and denotes one function. The formula
  `x` times `σ(x)` is the SiLU and also the swish, and the swish is also written with a
  learned scale.
  Rejected: `ops.Arithmetic.template(nm.x * factor, name='|d|^{-1/2}')`, which names the
  factor and leaves the reader to supply the multiplication, and
  `ops.Elementwise.template(name='SiLU')`.
  Replacement: `ops.Arithmetic.template(nm.Sigmoid(nm.x) * nm.x)` with no name, keeping
  `σ` because it is shorter than its formula and denotes one function.

- A numeric's derivative is written through the forward term where that keeps the two
  sharing one node. The logistic density is written as the sigmoid times one minus the
  sigmoid, so a formula and its derivative hold the same `nm.Sigmoid`.
  Rejected: the derivative written in exponentials, whose two terms appear nowhere in the
  formula they came from.
  Replacement: `nm.logistic_density`.

- `ops.Arithmetic` is unary, so a binary subtraction is a negation followed by an addition,
  and the negated value is built once and read by both consumers.
  Rejected: a `GenericOperator` named with a minus sign.
  Replacement: `ops.Arithmetic(formula=-1 * nm.x)` followed by `ops.AdditionOp`.

- A sum prints a negated summand as a subtraction, and a product is written by
  juxtaposition in both repositories, per [[Numerics]].
  Rejected: a sum printed with a plus sign followed by a minus sign.
  Replacement: `negated_part` and `signed_latex`, read by `Addition.to_latex`.to_latex` on 2026-09-14.

- A negative term is written with one minus sign, placed from an `is_negative` check on
  the term, in both repositories, per [[Numerics]].
  Rejected: `negated_part`, which recognised a constant of `-1` alone, so a sum printed
  `a + -2 b`, and a product of two negative factors printed `-2 -x`, which reads as a
  subtraction.
  Replacement: `is_negative` and `without_sign`, read by `signed_latex`,
  `Multiplication.to_latex` and tsncd's `NumericRenderer`, so the two print `a - 2 b`
  and `2 x`.

- A softmax written out at the model level is the definition, with no maximum subtracted.
  Rejected: `ops.SoftMax` left as one box in a model whose other pointwise maps are
  formulas.
  Replacement: an exponential, a sum over the keys and a reciprocal, so that the layer holds
  no operator whose reverse is declared a-priori.

- A softmax has two written forms. `expand_softmax` writes it as an exponential, a row sum
  and a reciprocal, and `expand_shifted_softmax` subtracts the row maximum before the
  exponential. Write the shifted form where the tape needs the maximum, per
  [[Expression Simplification]].
  Rejected: writing every model with the shifted form, which puts a maximum in the
  expression at every softmax where the model states none.
  Replacement: `ops.SoftMax.template()`, expanded on request.

- A softmax normalises one axis, so a mechanism that would need one over two axes is written
  as a published architecture that does not.
  Rejected: full self-attention over height and width at a UNet bottleneck.
  Replacement: axial attention, with one softmax per axis.

- An RMS normalisation reads every position of the normalised axis to write every position
  of it, so it has no partial result and no row in the accumulator registry, per
  [[Design Space]]. `expand_normalize` writes it out as a copy, a square, a sum, an inverse
  root, a scale and a gain, and the sum is the fold that does have one.
  Rejected: reading the operator as a fold, which it is not.
  Replacement: `expand_normalize`, whose contraction is the fold.

- The class of functions the reverse pass assumes is local Lipschitzness together with
  definability in an o-minimal structure. Almost-everywhere differentiability is not closed
  under the composition the reverse functor performs, and the value chosen at a kink is
  harmless because a training loop integrates a conservative field along a curve.
  Rejected: differentiability at all but a finite number of points, and the weaker condition
  of differentiability almost everywhere.
  Replacement: local Lipschitzness and definability, with conservativity as the property
  that composes.

- Adjacent reals that are rotated together are one complex number, and the rotation is
  a product with a complex exponential. `dst.PairsAsComplex` reads `R[z]` as
  `Complex(R)[t]` with `|z| = 2|t|`, the rotation at position `i_x` multiplies pair `i_t`
  by `e^{i i_x theta'[i_t]}`, and `dst.Decomplex` writes the pairs back as reals.
  `nm.ConstantSymbol.IMAGINARY_UNIT` is the numeric for `i`.
  Rejected: a `View` of the split `i_z = 2 i_t + i_J` onto `R[t, J]`, contracted with
  `ops.Einops.template('x t I J, h x t J -> h x t I')` against a table of two-by-two
  rotation matrices `R[x, t, I, J]`, which needs a cosine and a sine.
  Replacement: `over((h, x), pairs_as_complex()) @ over((h,), rotation) @
  over((h, x), complex_as_pairs())`.

### Parameters, tapes and the backward pass

- A model becomes a morphism in Para of Br by grabbing the parameters of its parametric
  seeds. Each seed gains one extra input weave per parameter, a degree-deleting reindexing
  for it, and a `Grab` of a slot named after the parameter, per [[Show Grabbed Parameters]].

- A parameter is shared across the batch by a deletion reindexing rather than by convention,
  so a parameter is a weave with no tiled entry.

- Weight tying falls out of term sharing. The same seed used twice stays one term through
  `@` and `*`, so memoising the slot per seed makes a tied layer grab one slot, and two
  separately built identical layers stay distinct because their axes are.

- The reverse of a parameter grab is a drop onto a gradient slot memoised per parameter, so
  a tied weight's two reverse grabs write one slot and gradient accumulation falls out of
  the comonoid, per [[Training]].

- A grabbed value is its own residual. A rule sees one seed, so it tapes the weight operand
  like any residual, and a pass then deletes the forward drop and points the backward grab
  at the parameter slot the forward pass already wrote.

- A linear map needs no tape. `Transpose` and `ReindexTranspose` are residual-free
  operators, and the reindexing of the second is an argument of the operator, because the
  transpose runs in the direction a `Broadcasted`'s reindexings cannot express.
  Rejected: the generic fallback, which declares the whole domain as the residual, so every
  matrix multiply taped its input for a backward pass that never needed it. Also rejected:
  reusing `ops.Linear` for its own transpose, which draws the two passes alike and would
  make linear expansion mint a second weight.
  Replacement: `para/data_structure/transpose.py`.

- Tape the unexpanded operand and re-apply the node in the backward pass, so the tape holds
  the small array rather than the broadcast copy.

- Declare the reverse derivative of an operator directly rather than declaring a Jacobian
  and transposing it. The transpose of a Jacobian is linear algebra rather than a rewrite,
  and a transposer would need per-operator help at every operator that matters.

- A residual is declared per operator. Taping the whole domain is the memory-worst policy
  and cannot express a softmax's backward pass, which needs the output.

- `Grab` and `Drop` are not each other's reverse in the reversed category. Reversal is a
  categorical construction that exchanges the two ends of a morphism and says nothing about
  a slot, so the swap onto the gradient slot is backpropagation's interpretation of the
  leaf, per [[Derivatives]].
  Rejected: a module that documents itself by naming backpropagation, and the claim that the
  swap is a property of the reversal.
  Replacement: the module documenting the reversed category alone, with the swap in
  `backprop._para_seed`.

- The direction a categorical construction reads is covariant or contravariant, and forward
  and backward name the two training passes. The construction itself is one class holding a
  covariant expression as its body, which is a construction rule in the paper's sense, per
  [[Para Category]].
  Rejected: a pair of recursive functions over two leaf classes that carried nothing to
  interpret.
  Replacement: `Contravariant`, with `ContravariantCategory`.

- A taped pair is a two-row `MultiCategory` whose covariant row is the forward pass and
  whose contravariant row holds the backward pass.

- Simplification may remove only the derivation's own scaffolding. It never reassembles an
  expanded softmax, because the expansion is a modelling choice, per
  [[Expression Simplification]].
  Rejected: any rewrite that folds the exponential and the reciprocal back into a `SoftMax`.
  Replacement: rules that are all linear, with an `Elementwise` as a wall none of them
  crosses, asserted in the notebook.

- A skip connection is a tape slot, so a compression path and an expansion path carry none
  of each other's wires and the contravariant path has the domain and the codomain the
  covariant one has, per [[Para Category]].
  Rejected: concatenating the skip connection, which the package has no operator for, and
  tying the expansion path's weights to the compression path's, which is a different
  architecture and conflates the contravariant reading with the reverse derivative.
  Replacement: a `Grab` and a `Drop` per resolution, with the expansion adding the projected
  skip, since a convolution of a concatenation is the sum of the convolutions of its halves.

- Whether a value is stored or rebuilt is a cost decision rather than a modelling one, per
  [[Recomputing the Exponent in the Backward Pass]]. Every slot the forward pass drops can
  be rebuilt in the backward pass from the taped operands of the operation that produced
  it.

- The backward pass is derived from the shifted softmax, because the tape has to hold the
  maximum the exponent was shifted by. The maximum passes back the zero map, and the
  arithmetic the zero leaves is pruned, per
  [[Recomputing the Exponent in the Backward Pass]].

- A grab inside a block is a grab. Both tape passes walk the graph through its blocks,
  duplicate grabs are removed only within one scope, and grabs are merged across blocks
  whose repetition is one, because a loop block is the one block that changes what a root
  inside it computes.
  Rejected: canonicalising a grab across blocks by renaming alone, which leaves consumers
  reading a wire produced in another block.
  Replacement: retargeting the slot, with `merge_duplicate_roots` keyed on the enclosing
  loop blocks.

- A model has an implicit form, a full form and a tied form, and the full form is the
  one in Para. The implicit form keeps a weight inside its operator and a selection on
  one wire, and is the form a model is written in and can be worked in. The full form
  writes those out, as `grab_parameters` and `expand_sparse_onto_tape` do, with the tape
  as the only coupling between the pieces. The tied form, which `tie_tapes` reaches, has
  the pieces wired and is available without being worked in. A figure shows the
  implicit form or the `ParaWrap` form, and the tied form only when it is asked for.

- A boxed layer states its tape at its own ports, and keeps it inside the body as well.
  Each grab of the body is a leading operand of the box and each drop a trailing result,
  and the box stands inside a `ParaWrap` naming the slot at each of those positions, so
  the wrap has the apparent domain and codomain of the block and the drawn box carries
  one tape per slot. The block on the operator is the block as it was written, seeds
  included, so the body drawn beside the box shows each drop where it saves its value.
  Rejected: the seeds left standing inside the body with the operator listing them, where
  the box has the domain and codomain it would have without them and a reader has to
  search the body. Rejected also: the seeds taken out of the block, which left the body
  drawn beside the box with nothing to say about the tape.
  Replacement: `ParaBlockOperator.template`, which derives the ports through
  `expose_tape_as_ports`, keeps the written block on the operator and returns `wrap_box`
  of the result, per [[Para Block Operator]].

### Anything else

- A correction on how a model is expressed is recorded in this note, in the section it
  belongs to, with the rejected form beside its replacement, and in
  `notebooks/base_features/BuildingAModel.ipynb` where the rule has a code form.

- Check a model's computational path against its reference before changing how the model
  is written, and again after the change. A rewrite that changes which values reach an
  operation changes the model, and a construction that looks redundant can be the one the
  reference computes or the one it does not. A correction from a reviewer is checked the
  same way, because the reviewer can be wrong.
  Rejected: a change to DeepSeek-V4.1-Flash's selection made on the strength of the
  pattern `TopK` followed by `[x > 0]`, and restoring the indicator on the strength of the
  suggestion that it lets the Top-512 select fewer entries.
  Replacement: the reference's `Indexer` and `sparse_attn`, read at the pinned commit,
  which read every reachable pick whatever its score and drop only an unreachable one.

- A notebook demonstrates a feature and is its test, so everything it claims is asserted in
  text and holds with the diagrams turned off, per [[Notebooks]].

- Predeclare every structural `RawAxis` and pass the axis objects rather than strings, since
  exact identity beats guessing at what `@` will align, per [[UIDs and Names]].

- A model built as a module beside its notebook is named after its action, as
  `notebooks/sota/DeepSeekV41Flash/lightning_indexer.py` is. The assertions stay in the
  notebook, so the notebook remains the feature's test, except for a SOTA notebook with a
  package, whose assertions live in the package's validator.

- Two mathematically equivalent presentations of one algorithm are what diagrams are for, so
  the package keeps both the compressed and the expanded selection, and both weight forms,
  rather than choosing one and deleting the other.

- The reverse functor is partial on datatypes rather than on shapes. `Natural(n)` has no
  cotangent, so an index wire is absent from the reverse graph, which answers the
  integer-input question and the top-k question together, per
  [[Selection and the Reverse Pass]].

- A parameter is classified by which pass writes it, and two of the four classes are not
  lenses. The auxiliary-loss-free routing bias is written by the forward pass and has no
  gradient, because it is excluded from the gate value by construction and enters only the
  selection.
  Rejected: giving the routing bias a gradient by pointing at the sequence-wise balance
  loss, which trains the router weights rather than the bias.
  Replacement: a forward-writable tape slot, per [[Training]].

- Where a router's normalisation is taken decides whether an unselected row is in the
  reverse graph at all. Normalising over every expert and masking afterwards leaves every
  logit in the denominator, and normalising over the selected set alone makes the derivative
  to an unselected row exactly zero. [[Training Mixture of Experts Gates]] holds the
  literature.

- A script that lives beside a notebook cannot be run by path when the notebook's folder
  holds a module whose name shadows a top-level package. Run it as a module from the
  repository root.

- A notebook that calls `websocket_transfer` directly rather than `show_diagram` is not
  reached by the diagram override. A new notebook draws through
  `notebooks/display/notebook_diagrams.py`.

- **A rotary embedding is a `dst.Rotary` table of turns multiplied onto the pairs.** The
  user asked on 2026-09-17 for "a Rotary operator class, displayed using the previous
  rotary display type", with V4.1's YaRN rotary "a subset so it's displayed in a similar
  way", and the dynamics visible on hover. `dst.Rotary` has no operands and returns the
  complex array `F[i_x, i_t] = e^{i i_x theta[i_t]}` over the positions and the pairs;
  `dst.YarnRotary(Rotary)` adds the factor and the ramp. The rotation of an array is a
  titled block boxed as an operator: `aops.DeconcatenateAxes` of the channel axis,
  `dst.PairsAsComplex`, an `Einops` against the table over the complex datatype,
  `dst.Decomplex` and `aops.ConcatenateAxes` back onto the declared channel axis
  (`concatenated=c`). The entries of a ratio-2 layer take the position stride `|a|`, and
  the attention output is a `RotationSite` with `conjugated=True`, whose table is followed by `dst.conjugate_complex_values`. The table takes no row in
  the explanation tables, so its inspection box shows its standard expansion. The pattern
  is `notebooks/sota/DeepSeekV41Flash/rotary_embedding.py`, in the integrated package until 2026-09-19.
  Rejected: `generic_operator('\\mathrm{Rot}_{x}', ...)` beside a `cat.DefinedExpression`.
  Replacement: `rotation_box(RotaryKind.YARN, TOKEN_LATENTS)`.

- **An inverse rotation is the conjugate of the table, and its name carries the conjugate
  bar.** The user asked on 2026-09-18 to "make sure the inverse YaRN is notated as the
  inverse YaRN", and asked whether a conjugate over the complex numbers is all it takes.
  A factor of length one is turned back by conjugating it, so a site that turns back
  holds the table of the same base followed by `dst.conjugate_complex_values`, an
  elementwise `nm.Conjugate` named `\overline{x}` outside the table, per
  `rotary_embedding.turns_of_every_position`. The circle of the table keeps its plain
  name, and `deepseek/validate_rotary.check_a_conjugated_rotation_turns_back` checks the
  form. The name under a conjugate bar that the ruling of 2026-09-18 asked for is not
  what the code draws, as recorded on 2026-09-19.
  Rejected: `e^{-\mathrm{i} x}` under the plain name `\mathrm{YaRN}`.
  Replacement: `\overline{e^{\mathrm{i} x}}` under `\overline{\mathrm{YaRN}}`.

### Reversed rulings

- An operator whose value depends on an index stays a `GenericOperator` beside a
  `cat.DefinedExpression`. For the rotary embedding this is superseded: the rotation is
  an `Einops` against a `dst.Rotary` table whose standard expansion writes the positions
  as an `ops.Arrange`. Ruled 2026-09-17 and
  superseded the same day.

- An index map that is not affine is drawn as a constant table over `cat.Natural`, read
  by `ds.IndexSelect`. The table was the block a compressed position belongs to, an
  integer division. Its inverse, the block split, is affine and a bijection, so
  `aops.CovariantView` reads the split covariantly and no table is needed. Ruled
  2026-09-10 and superseded 2026-09-11.

- The expanded form of a selection is available to both kinds of consumer. The earlier
  wording restricted it to a selection whose payload is a weight, which held while the
  expanded form was a constructor. `expand_sparse` turns a `Select` into an `IndexSelect`
  with the index wire routed across the blocks it crosses, so a payload on a wire expands
  as well. Ruled 2026-08-18 and superseded 2026-08-20.

- Lifting a `Broadcasted` with no inputs is well formed. The earlier rule forbade it
  outright, because the degree came back empty and the lifted weave gained tiled slots with
  nothing to imprint. `Broadcasted.backup_degree` now carries the degree of a morphism whose
  domain is empty. The modelling advice stands on its own, which is that a weight does not
  vary with the batch, so the batch axis is written into the einsum beside the weight. Ruled
  2026-08-15 and superseded 2026-08-21.

- The down-projection of a mixture produces the expert axis and is followed by a
  diagonalisation, which is the form the model notebooks draw. The folded form, where the
  down-projection consumes the expert axis whole and the gates are multiplied into the
  hidden state first, states the same map and remains valid arithmetic. Drawing the pieces
  out is what makes the combine a visible reduction and keeps the router's expert axis
  dense. Ruled 2026-08-15 and superseded 2026-08-18.

- The GELU is an `ops.Arithmetic` over `nm.CumulativeGaussian`. The earlier ruling made it a
  named `ops.Elementwise`, because `data_structure/Numeric.py` held no term for the Gaussian
  cumulative distribution function and both standard approximations put a float into the
  expression. The numeric was added the same day, so the reason for the earlier ruling is
  gone. Ruled 2026-08-25 and superseded
  2026-08-25.

- A value that several layers share rides a repetition block as a passenger wire. The
  passenger form was the first construction that let a repeated block hold a layer plan.
  It was replaced by a tape slot the next day, because a wire through the repetition made
  every sublayer the wire passed carry it, and the reference passes nothing between its
  layers but the residual and the collapse vector. GLM-5.2's shared selection still uses
  the passenger form, and Kimi K3's `AttnRes` is a second residual stream, which is a wire
  in its own right. Ruled 2026-09-10 and
  superseded 2026-09-11.

- A selection in the `ONLY_SELECTION` form is read by `ds.Select.at_positions`, with the
  selection axis in the target of the positions and of the result. The form of the
  selection stands. What changed is the operator that reads it, because the slot axis in a
  target draws the selected slots as a wire the gather consumes rather than a wire it is
  broadcast over, and a gather is elementwise in the slot. `ds.IndexSelect` over a degree
  holding the slot axis replaced it. Ruled 2026-09-14 and superseded 2026-09-16.

- The target anchors of a `dst.TopK` are drawn as degree anchors are, so that the axis
  the selection runs over and the axis it hands its outputs out on make one wire crossing
  the diamond. The reading behind the earlier ruling was that `n` before the selection and
  `k/n` after it are one wire read twice. A wire drawn across an operator is the mark of a
  degree axis, meaning an axis the operator is broadcast over, and the axis a selection
  runs over is consumed, so the earlier drawing said that the selection is broadcast over
  the entries it chooses among. The operand's axis now ends at the diamond and each
  result's axis starts from it. Ruled 2026-09-15 and superseded 2026-09-16.

- An operator whose value depends on an index is defined over every position. The
  earlier ruling defined it at one index: the left-hand side of the definition was the
  operator followed by the read of one position, a reindexing with the shift `i_x` and
  no strides, and the right-hand side was an `ops.ConstantOp` or an `ops.Arithmetic`
  whose formula held the free numeric `i_x`. The rotation, the YaRN frequencies and the
  YaRN ramp were its three examples. The user had each rewritten with an `ops.Arrange`,
  the ramp last, so no expression in the repository holds a definition at an index. The
  read of one position stays the way a fixed one-hot is written, and tsncd still draws
  a definition whose left-hand side ends in such a read. Ruled 2026-09-17 and superseded
  2026-09-17.

## Open follow-ups

- The rule that a name is given to an elementwise map only where the published name is
  shorter than the formula and denotes one function has been applied to the modules of
  `notebooks/sota/DeepSeekV41Flash/` and not to every expression in the repository.

## See also

- `notebooks/base_features/BuildingAModel.ipynb` — the rules of construction one at a time,
  asserted, and where a rule with a code form is recorded
- [[SOTA Model Notebooks]] — the published architectures these rules were worked out
  on, and the inventory of what each needed
- [[Notebooks]] — the `notebooks/sota/` section, and how every notebook draws
- [[Operators]] — the standard vocabulary, and the `deepseek/` tip
- [[Construction Helpers]] — what `@`, `*` and `>>` do
- [[Sparse Axes]] — the `SparseAxis` datatype and the two ways to consume a selection
- [[Sparse Expansion]] — the rewrite between the compressed and the expanded forms
- [[Yoneda and Cartesian Tricks]] — the two rewrites of rule 19
- [[Linear Expansion]] — the three forms of a weight
- [[Show Grabbed Parameters]] — the transform into Para of Br
