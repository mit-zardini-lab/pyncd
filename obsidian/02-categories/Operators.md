---
tags: [layer/categories, reference]
code: data_structure/Operators.py, advanced_axis_dynamics/data_structure/Operators.py, deepseek/data_structure.py
status: evolving
---

# Operators

## What it is

An `Operator` states what a [[Broadcasted Category|Broadcasted]] does to its target. It
carries a name and, by convention, a `template(...)` classmethod that returns a whole
`Broadcasted` with its weaves and reindexings already built. In practice an expression is
written as `ops.SoftMax.template()` rather than through a `Broadcasted` constructor.

`Operator.name` has no default value, so every subclass either declares its own default,
as `SoftMax` declares `SoftMax` and `Linear` declares `L`, or is given a name at each
construction, as `Split` and `Join` are. A subclass is therefore free to declare a field
that has no default of its own, which is what `BlockOperator.block`,
`CovariantView.reindexing`, `Transfer.child_kernel` and `Transpose.operator` are. Before
2026-09-13 the base default forced each of those to be declared as optional and filled
with `None`, and the construction that omitted one raised nothing.

## The vocabulary

From `data_structure/Operators.py`:

| operator | what it is |
|---|---|
| `Einops` | a contraction. `Einops.template('q d, x d -> q x')` is the usual way to write a matmul. It carries a `signature: Prod[Prod[int]]`, covered below |
| `Linear` | a contraction whose weight is implicit: N inputs map to M outputs, with an empty reindexing. It is opaque to dependency analysis, per [[Linear Expansion]]. With no inputs at all it is a parameter array, because a linear map out of the empty product is a constant tensor, so the implicit weight is the learned tensor and it draws in the usual weight-box idiom. That case is the whole of what a `Parameter` operator would have been. An input whose target datatype is `Natural` selects rather than contracts, per *A `Linear` reads two kinds of input* below |
| `Elementwise` | a unary pointwise map with no interior the algebra can read, so it is a name. `ReLU` and `Dropout` subclass it, and so do `View` and `Arithmetic` |
| `Arithmetic` | a pointwise map given by a formula in `nm.x`, per [[Numerics]]. `Arithmetic.template(nm.E ** nm.x)` is the exponential and `1 / nm.x` the reciprocal. Its `name` is the formula's latex with `x` at the input's place, so a listing and a diagram print `e^{x}`. Pass `name=` where the published name is shorter than the formula, as `\sigma` is shorter than `(1 + e^{-x})^{-1}`. Nothing reads the name, so a renamed operator behaves identically: every rule that acts on such an operator tests its formula. Its derivative is `solver.algebra.differentiate_numeric.differentiate(formula)`, another `Arithmetic`, and `torch_compile` evaluates the formula directly with no table of names |
| `View` | a reindexing and nothing else, which is the operator of a node, whose whole content is its `reindexings[0]`. It covers a permutation, a broadcast, a diagonal, a convolution window, and a repeat, which is a `View` whose reindexing does not name the repeated axis. It was called `Identity` until 2026-08-21, and was renamed because it is never the identity of anything: it is a view of its input through an index map |
| `CovariantView` | a reindexing read covariantly, so the output at `reindexing(i)` is the input at `i`. It lives in `advanced_axis_dynamics/data_structure/Operators.py` since 2026-09-15 and is written `aops.CovariantView`. It exists for a merge, whose inverse rounds down and is not affine, and it carries the reindexing as a field because a `reindexings` entry points the other way. `deepseek.merge_selected_axis` builds it over a sparse axis and `deepseek.merge_selected_positions` over positions, per [[Sparse Axes]], and its reverse is the `View` of the same reindexing. The reindexing may have several rows and a shift, provided `mark_sparse_codomains.merge_groups` finds an order in which each row writes its own axes as a signed mixed-radix number beside the axes earlier rows determine, and `mark_sparse_codomains.mark_sparse_codomain` replaces every codomain axis the image does not fill by the `AffineGuards.AffineSparseAxis` stating which positions it does, and a `degree` the merge is broadcast over, an axis of which guided by an axis the merge consumes leaves re-guided through a unit-stride degree reindexing, so the indexer's `pos` merge of `(b_0, a\|b_0)` into `x` over the slots `r\|b_0`, leaving `[x, r\|x]`, is one operator, per [[Advanced Axis Dynamics]]. tsncd draws it as the reindexing itself read left to right, a `StrideMorphismBox` built through `CovariantStrideRenderer` in `display/Framework/advanced_axis_dynamics/covariantOperatorBoxes.ts` since 2026-09-16, so it has the height of every other reindexing node, each stride of the first row sits beside the axis it scales, and the point carries the shift, or the name where the shift is zero, as a `View`'s node does. The operator moves each value without changing it, so its operand and its result have the same datatype. The datatype wire therefore runs past the pentagon as a degree axis does |
| `DeconcatenateAxes` | one axis cut into the parts that fill it, with one output per part. It holds the part reindexings a `ConcatenateAxes` of the same parts holds and reads them contravariantly. The reverse derivative of each of the two is the other, and its standard expansion is a copy followed by one `View` per part, per [[Advanced Axis Dynamics]]. The axis being cut may be a declared axis whose size the parts sum to |
| `ConcatenateAxes` | one reindexing per input, each reading one axis covariantly into one shared codomain axis at unit stride and at the offset the parts before it sum to, so the images are disjoint runs that fill it. It is the multi-input `CovariantView`, is written `aops.ConcatenateAxes`, and states the concatenation of two axes carrying two different affine forms, which no one form states. `concatenation_expansion.expand_concatenations` rewrites every consumer that treats the positions of the concatenated axis one at a time into the consumers of the parts, per [[Advanced Axis Dynamics]]. The axis it produces is an `AxisConcatenation.ConcatenatedAxis`, which references its parts, is sized by the sum of their sizes and is labelled by their labels joined, so the window slots `w|x` beside the selected slots `s|x` read `w|x + s|x`. Since 2026-09-17 `template(concatenated=c)` also accepts a declared axis whose size the parts sum to, as `DeconcatenateAxes.template` does, so the cut of a latent axis `c` followed by the concatenation of the same parts returns the array to `c`. The expansion of concatenations reads the parts off the weaves, so it rewrites the consumers of such a concatenation as it rewrites any other |
| `GenericOperator` | an escape hatch that can name anything, built from a signature string |
| `AdditionOp` | addition. tsncd draws it as a bold `+` where the wires meet, and, since 2026-09-19, as a rectangle carrying the same name where every operand is a `Natural`, because an addition of whole numbers is a step of an integer computation and not a junction of a residual stream |
| `Maximum` | a fold under max, which is a contraction, and is registered as one |
| `BitwiseXor` | a fold under the bitwise exclusive or, over the one axis it consumes, whose unit is zero. `template(axis, datatype)` accepts a `Natural` bounded by a power of two and raises `NotAPowerOfTwo` otherwise, because the exclusive or of two values below `2^N` is again below `2^N` and no such bound holds for another datatype. `bit_width` reads the `N`. Added on 2026-09-18 for the Engram n-gram hash. tsncd draws it as a `GlyphBox`: the circle a `Normalize` takes, carrying an upright cross, with `XOR` written under it, since 2026-09-19 |
| `Modulo` | the remainder of the first operand after division by the second, entry by entry. `template('x G, G K -> x G K', dividend, divisor)` builds it from a signature with nothing contracted, and the result carries the divisor's datatype because a remainder is below the divisor. tsncd draws it as a rectangle carrying `\bmod`, by the rule that an arithmetic operation on whole numbers stands in a rectangle |
| `Cast` | the identity on values read into another datatype, an `Elementwise` whose output weave carries the new datatype. A cast from `Natural(a)` to `Natural(b)` with `a <= b` loses nothing. A cast that narrows is exact only where every value the operand can hold is below the new bound, and the block that holds it says why in its description. The Engram hash holds one cast, of a remainder plus an offset into a row number after the addition, since 2026-09-19; its products are 64-bit integers by the bound of the multipliers |
| `FixedArray` | an array the model reads and never learns, such as the multipliers and the primes of a hash. An array another fixed array determines, such as the offsets of the hash, is computed from it rather than declared, since 2026-09-19. It has no operands and the axes of the array stand in the target of its output weave. A `Linear` with no inputs is a learned array and draws as a weight, and an `Embedding` is a learned table read at an index, so neither states a fixed table, and `show_grabbed_parameters` grabs nothing for a `FixedArray` |
| `with_datatypes` | not an operator: `with_datatypes(broadcasted, inputs, outputs)` returns the same `Broadcasted` with the datatype of each weave replaced in order. A signature template gives one datatype to every weave, and an operation on naturals returns a different bound from the ones it reads, so the product of an identifier and a multiplier is an `Einops.template('x L, L -> x L')` retyped to `(Natural(v), Natural(a)) -> Natural(v a)` |
| `SoftMax` | the exponential and the `L1Norm` after it, as one operator |
| `Normalize` | `y = gamma * x * (mean(x^2) + epsilon)^{-1/2} + beta` over the axes it consumes, named `RMSNorm`. Since 2026-09-18 it declares three things. `gain` says that `gamma` multiplies the result, one learned number per position of the normalised axes, and `bias` says that `beta` is added after it. `show_grabbed_parameters.parameter_arrays` grabs whichever the operator declares, the gain first, and `expand_normalize` writes the scaling alone where it declares neither, so a model that normalises with no learned weight writes this operator and not an `L2Norm` beside a scale. `epsilon` is the number added under the root: `ops.SYMBOLIC_EPSILON` where a model does not state it, `ops.NO_EPSILON` for an exact function, and the released number where the model has one. `torch_compile` computes the root mean square itself, where it used to call `nn.LayerNorm`, and a symbolic epsilon compiles to zero |
| `L1Norm` | `y = x / (sum(x) + epsilon)` over the one axis it consumes, which at `ops.NO_EPSILON` is what a `SoftMax` does after its exponential. The divisor is the sum and never the mean. `epsilon` was added on 2026-09-18, because a released kernel adds one so that an axis whose values have all underflowed to zero is not divided by zero, and the Sinkhorn normalisation and the gate of the mixture of experts of DeepSeek-V4.1-Flash each supply the number their kernel adds. `ops.NO_EPSILON` is `nm.Integer(0)`, the unit of addition, so a norm that adds none expands to the bare reciprocal |
| `L2Norm` | `y = x / sqrt(sum(x^2))` over the axes it consumes, added on 2026-09-18. It takes no gain, where a `Normalize` takes one learned weight per position and divides by the root of the mean rather than of the sum, so a `Normalize` is an `L2Norm` times the root of the number of values and times its gain. A model that normalises with no learned weight writes the `L2Norm` and that root, as the four-stream residual of DeepSeek-V4.1-Flash does. `algebra.operator_expansion.expand_l2_norm` writes it as a copy, a square, a contraction, an inverse square root and a product, and `torch_compile` divides by the root of the sum over every dimension of the array. tsncd draws it as the triangle of a `SoftMax` carrying the root tick a `Normalize` carries, as a `GlyphBox`, because the operator may consume more than one axis. Since 2026-09-18 the four-stream residual writes an `ops.Normalize` declaring neither a gain nor a bias instead, because that operator now carries its own epsilon, and the `L2Norm` is the operator for a normalisation stated as a division by the root of a sum. |
| `Embedding` | a gather |
| `ConstantOp` | a nullary source. The `Op` suffix separates it from `nm.Constant`, the numeric naming a mathematical constant |
| `Arrange` | a nullary source over one axis `x` that returns `Nat(x)[x]`, the array holding `i_x` at index `i_x`, so `[0, 1, ..., |x| - 1]`. The axis stands in the target of the output weave, because a tiled operator is one function at every index of its degree. `Arrange.template(x)` names the operator `i_x`. `multiply_by_positions(positions, values)` writes the product of that array with an array of another datatype, which `Einops.template` cannot, because it gives one datatype to every weave. The rotary angle `i_x theta[i_t]` is the case. A formula in a position is an `Arithmetic` applied to the result, and `Arithmetic.template(formula, base=..., output_datatype=...)` names the datatype it returns where it differs from the datatype it reads. `tsncd` draws an `Arrange` as a reversed einops cup joining the axis wire and the `Natural` wire, stroked as the `Natural` wire is and carrying the direction triangle of a datatype wire half way round. No cost, no torch module and no derivative rule is declared for it yet |
| `WeightedTriangularLower` | a causal mask as an operator. Nothing in the algebra reads it, and it is superseded: a causal read is an `ops.View` through a shifted or negatively strided `StrideMorphism`, whose output carries a `cat.AffineSparseAxis`, per [[Padding and Masks as Sparse Axes]] |
| `BlockOperator` | an operator that wraps a whole sub-expression |

> [!tip] There is a second vocabulary in `deepseek/data_structure.py`, and it is easy to miss
> **`TopK`** has a rank-1 target over the scored axis, so it broadcasts over the queries, and
> its codomain is a `SparseAxis`: the same extent, with an `activity` numeric `k`, printing
> `k/n`. Contracting the sparse axis against a raw axis merges them, because `SparseAxis`
> outranks `RawAxis`, so the sparsity annotates the axis itself. Annotating the axis is right for an expert
> axis, where every consumer is sparse. Pass `template(axis=...)` so that the caller's axis is
> consumed directly rather than a fresh `'n'` that would erase its name. The sparse axis is a
> compression of a pair of wires, being the values and the `Natural`-typed indices, which
> the `WEIGHTS_SELECT` form of `TopK.template` builds instead, with the scored axis staying its full size. Compressed is
> the default, and the expanded form is what a rewrite produces, per [[Sparse Axes]].
>
> **`Select`** is beside it, at `(k/n), (n) -> (k/n)`. It is the operator that reads a payload
> at the positions a selection names, and it is the one thing neither an `Einops`, whose
> signature describes a contraction, nor a reindexing, which is affine, can express.
> `Select.at_positions` builds it over positions held as data, `[Nat(n); k], [R; n] -> [R; k]`,
> which is how a `TopK` in the `ONLY_SELECTION` form is read. It draws as the green dot a
> copy of an axis draws as, because a selection read as a one-hot matrix is a contraction
> against the diagonal a copy states, per [[Sparse Axes]].
>
> Also there: `ComplexRotary` and `Decomplex`, which are rotary embeddings over a `Complex`
> datatype, and `RotaryLinear`. `PairsAsComplex`, added on 2026-09-17, is the inverse of
> `Decomplex`: it reads `R[z]` as `Complex(R)[t]` with `|z| = 2|t|`, the first value of a
> pair as the real part, so that a rotation of a pair is a product with a complex
> exponential, per [[Representing Models]]. All have tsncd boxes, in `src/deepseek/display_deepseek.ts`.
> The notebooks under `notebooks/base_features/` that use them, meaning `DeepSeekV3.ipynb` and
> a notebook of the old layout, are legacy, and the module is live, because the notebooks
> under `notebooks/sota/` build on `TopK`. The module is easy to overlook precisely because
> the notebooks beside it are marked dead.
>
> **`Rotary`** and **`YarnRotary`**, added on 2026-09-17, are the table of factors a rotary
> embedding multiplies the channel pairs by. Each has no operands and returns
> `Complex(R)[x, t]` over the positions `x` and the pairs `t`, with both axes in the target
> of the output weave, because a tiled operator is one function at every index of its
> degree and the table differs at every position and pair. A `Rotary` holds
> `F[i_x, i_t] = e^{i stride i_x theta[i_t]}` with `theta[i_t] = base^{-i_t / |t|}`, which is
> the number usually written `base^{-2 i_t / |z|}` over `|z| = 2 |t|` rotated channels. Its
> fields are `name`, `base` and `position_stride`, in that order. `position_stride` is
> the token position one step along `x` stands for. It is one over the tokens and `|a|`
> over the compressed entries, because entry `i_b` is rotated at the position `|a| i_b` of
> the first token of its group. A table turns counterclockwise and holds `e^{+i angle}`.
> An inverse rotary embedding multiplies by the conjugate of that table, which
> `dst.conjugate_complex_values` writes as one elementwise `ops.Arithmetic` named
> `\overline{x}` after it. A factor of length one is turned back by conjugating it, so the
> conjugated table is the table of the same base read in the other direction, and
> DeepSeek-V4.1-Flash multiplies its attention output by it. The `RotationDirection` enum
> and the conjugate bar on a table's name were removed on 2026-09-18, because one table
> and one conjugate say what two tables said.
> `YarnRotary` subclasses `Rotary` and adds `factor`, `ramp_start` and `ramp_end`, in that
> order, with `theta'[i_t] = theta[i_t] (1 - r[i_t] + r[i_t] / factor)` and `r[i_t]` the
> clamp of `(i_t - ramp_start) / (ramp_end - ramp_start)` to the unit interval. The
> rotation of an array of pairs is an `ops.Einops` over the `Complex` datatype against the
> table, between a `PairsAsComplex` and a `Decomplex`, per [[Representing Models]].
> `deepseek/registries/standard_expansions.py` registers the standard expansion of each
> class. The expansion writes the table with `ops.Arrange`, `ops.Arithmetic` and
> `ops.multiply_by_positions`, applies a stride other than one to the positions as an
> `ops.Arithmetic` from `Natural` to `Natural`, and lifts the whole table over the degree
> of an operator that is broadcast. The formula and the description an inspection box
> shows are written from the operator's own axes and fields, per [[Advanced Display]]. The
> module is imported for its side effect by the code that wants the rule, because
> `websocket_transfer/validate_auxiliary_information.py` asserts which operators the
> registry holds when the core alone is loaded. `deepseek/validate_rotary.py` holds the
> checks. tsncd mirrors both classes and draws each with the circle of `ComplexRotary`, per
> [[Terms Mirrored in tsncd]]. Since 2026-09-18 the circle carries the operator's own name
> in a strip above it, outside the glyph, so that `\mathrm{RoPE}` and `\mathrm{YaRN}` are
> told apart by reading the name rather than by a mark inside the cap. `ComplexRotary` names no axis and has no expansion, and it
> stays because the notebooks under `notebooks/base_features/` hold it.

`para/data_structure/` adds the operators the reverse pass introduces: `Transpose` and
`ReindexTranspose`, per [[Derivatives]], `Zero`, and **`Inject`**, the reverse of a
compressed selection, at `(k/n) -> (n)`, per [[Selection and the Reverse Pass]]. `Inject` has
a tsncd box in `src/deepseek/display_deepseek.ts`, drawn as `TopK`'s glyph turned over, so
the two read as one shape and its reverse.

## A `Linear` reads two kinds of input

The datatype of an input's target says how the operator reads that input.

| target datatype | reading | what the weight gains |
|---|---|---|
| a real datatype | contracted, and the operation sums over the target's axes | those axes |
| `Natural(n)` | an index, naming one of `n` weights | one axis of `n` entries |

`Linear : R[m] -> R[f]` holds `R[m, f]` and computes $y_f = \sum_m x_m W_{mf}$.
`Linear : (Nat(n), R[m]) -> R[f]` holds `R[n, m, f]` and computes $y_f = \sum_m x_m W_{imf}$
at the index $i$ the operand carries. The target of an index input is rank 0, because an
index is a single number.

A selection is not a contraction. Writing it as one needs the index expanded into a one-hot
vector of length `n` and contracted away, which materialises `n` values to keep one. Keeping
the selection in the datatype is what lets the index stay runtime data while the rest of the
expression stays affine, per [[Sparse Expansion]].

The expert `Linear` of a mixture of experts is the case the distinction was written for. The
index comes from the router's `TopK`, the weight holds every expert, and
`deepseek.IndexSelect` takes the
same posture for a payload on a wire, so the two consumers of a selection's index read it
identically.

**A `Linear` is a contraction only when no input selects.** `ops.selects_weights` is the
condition. Three passes read a `Linear` as a contraction, and each has to test it:

- [[Linear Expansion]] rewrites a `Linear` into a weight array and an `Einops`. It returns a
  selecting `Linear` unchanged, because the expansion would contract the selection axis away
  and give an average of every weight where one weight was meant.
- [[Show Grabbed Parameters]] builds the weight, through `weight_axes`, and mints the
  selection's axis from the datatype's size.
- [[Derivatives]] reverses a `Linear` to a `Transpose`, which is correct when the map is
  multilinear in $(x, W)$. A selecting `Linear` is multilinear in $(x, W_i)$ at one index, so
  the rule needs the same condition and does not yet test it.
  [[Selection and the Reverse Pass]] carries what follows.

## `Einops.signature`, precisely

`signature: Prod[Prod[int]]` has one outer entry per input segment. Each inner tuple lists,
in positional order and skipping any TILED position, the contraction group that each
non-tiled axis of that segment belongs to.

- Two segments sharing a group number are summed together over that axis.
- A group number appearing in one segment alone is a plain sum-reduction over that axis.

The tuple lines up positionally with `input_weaves[i]._shape`. Each position is either
`WeaveMode.TILED`, carrying no group, or a concrete `RawAxis`, which is exactly the set of
positions `signature[i]` enumerates, in order. `einops_rearrange.segment_group_axes` reads a
group number back to an axis, and section 1 of [[Einops Rearrangement]] is the full account.

**An `Einops` always has exactly one output segment.** `Einops.template` asserts it.

**An axis in the output alone is a repeat, and a repeat is a `View`.**
`Einops.template('q -> q v')` used to raise. Since 2026-08-21 it is a `View` at degree
$(q, v)$ whose reindexing reads $(q)$, because the produced axis is degree that no operand's
reindexing names. `generic_signature`, which `AdditionOp.template` uses, reads a right-only
symbol the same way. `signature_to_broadcast` keeps the older reading for any other operator,
where a right-only symbol is an axis the operator writes, as a `Linear`'s output is, and
`Einops.template` passes `produced_as_degree=True`. There is no repeat operator, and
`algebra.einops_simplification.einsum` has no produced case, per
[[Expression Simplification]].

## The registries an operator can appear in

Adding an operator usually means adding it to some of the registries below. They are the
extension points.

| registry | file | what it states |
|---|---|---|
| accumulator | `algebra/registries/accumulator.py`, through `@register` | how two partial results of a fold are combined, for a concatenation and for a fold run in pieces |
| torch lowering | `torch_compile/torch_compile.py`, through `operation_key=` | how to run it |
| diagram | the `TermDirectory` of the `tsncd` package | how to draw it |

> [!warning] A new operator with no `tsncd` box breaks the diagram and leaves the algebra alone
> "Term type not found in TermDirectory" means the `tsncd` bundle needs rebuilding, per
> [[Diagram Display]].

## Gaps

- Add every operation present in transformers, make the operator constructors consistent,
  and give each operator a formal-name characteristic, per [[Open Gaps]].
- `Maximum` has a row in the accumulator registry and the other monoids, meaning min and
  logsumexp, do not, per [[Open Gaps]].
- **The derivative rule for a `Linear` does not test `ops.selects_weights`.** It reverses
  every `Linear` to a `Transpose`, which is a contraction against the whole weight. For a
  selecting `Linear` the cotangent of the weight is a scatter onto the slabs the index
  named, and a scatter is not in the operator set, per
  [[Selection and the Reverse Pass]]. What the rule writes for the value's cotangent is now
  repaired downstream: `para.algebra.para_sparse_expansion.reads_a_weight` admits a
  `Transpose` of a `Linear` as well as a `Linear`, so expanding a selection onto a derived
  pair gives `Transpose<E>` the index its forward `Linear` reads, and
  `para/validate_backward.py` asserts it. The weight's own cotangent is
  what still needs the scatter.

## See also

- [[Broadcasted Category]] and [[Weaves and Degree]] — what an operator acts on
- [[Construction Helpers]] — the operator overloads that compose these
- [[Einops Rearrangement]] — the algebra on `Einops` signatures
