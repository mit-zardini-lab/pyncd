---
tags: [layer/para, concept]
code: para/processing/show_grabbed_parameters.py
status: working
---

# Show Grabbed Parameters

## What it is

`grab_parameters(m)` is the transform from **Br** to **Para(Br)** that makes a model's
weights operands of the expression. A `Linear` keeps its weight inside the operator, so the
expression has one operand and the weight is a letter drawn on a glyph. A reader of the model
needs no more than that, and a functor that counts, trains or shards the parameters cannot
see them at all.

`grab_parameters` writes each one out. The parametric seed gains one input weave per
parameter, holding the full weight array, and a `Grab` of a fresh [[Para Category|TapeSlot]]
feeds it:

    Linear<Q> : R[q, m] -> R[q, d]   becomes   (Grab<W_Q> * id) ; Linear<Q>

with the `Linear` now reading `R[m, d]` as its FIRST operand. A `Grab` is
composition-neutral, so the domain and codomain of the whole expression are unchanged. The
result is the same morphism in `Para` form, with its parameters named.

**The parameters occupy the first operand slots**, ahead of the operands the seed already
had, in the order `parameter_arrays` returns them. A `Linear` with a bias reads `(W, b, x)`
and a `Normalize` reads `(γ, x)`. The first slot is the top wire in a diagram, so the tape
arrives above the data rather than below it. The order is part of the parametrised form
rather than a display choice, because `derivative._parametrised_linear` and
`torch_compile.parametrised_linear_func` both read the weight off the first slot.

## The rule

`parameter_arrays` states what is parametric, with one `match` case per operator. Everything
else returns `()` and passes through untouched. The arrays are built from the seed's own
weave targets, meaning the size of the target, which is what the operator receives with the
broadcasting excluded, so their axes are literally the operand's and the result's rather than
lookalikes:

| operator | parameters |
|---|---|
| `Linear` | `W` runs over every input and then the out target, through `weight_axes`. `b` is the out target, when `bias` is set |
| `Linear` with no operands | the out target, under the operator's own name |
| `Normalize` | `γ` is the target itself |
| `Embedding` | absent, per *Gaps* below |

`weight_axes` reads one input at a time, and the datatype of its target says what that input
contributes. A real input is contracted, so the weight carries the operand's target axes. A
`Natural(n)` input selects one of `n` weights, so the weight carries an axis of `n` entries
and the operand's own target is empty. [[Operators]] states both readings.

The axis a selection contributes is minted by `selected_axis` from the datatype's size,
because a `Natural` carries a count and not an axis. The size symbol is what ties it to the
axis the selection was made over. The expert `Linear` of a mixture of experts is the case:
`(Nat(n), R[m]) -> R[f]` grabs `R[n, m, f]`, holding every expert, and


A `Linear` with no operands is a learned array, which is how the sink logit of
DeepSeek-V4.1-Flash is written, per [[Representing Models]]. The sum over its inputs is
empty, so its weight is the whole of its result. It becomes the `Grab` alone and no
operator is left behind, and the slot takes the array's own name through
`learned_array_name`, so the sink reads `sink` on the tape where a projection `Q` reads
`W_{Q}`. Until 2026-09-17 such a `Linear` raised, because the reconstruction added an
input weave beside the `backup_degree` that an empty domain carries. A bias on such a
`Linear`, or a degree it is broadcast over, would need an addition or a repeat after the
grab. No model writes either, and `_grabbed_learned_array` raises a `NotImplementedError`
that names the operator.

The parameter's weave has no `TILED` entry and its reindexing deletes the whole degree, as
`Rearrangement((), degree)`, which is the statement that the weight is shared: every batch
index reads the same array.

**A gradient slot has its parameter's shape**, axis for axis. `backprop._para_seed` builds
the drop's array as `tangent.array(grab.size)`, and `derivative._parametrised_linear` writes
the outer product with the weight's own shape as its codomain, so neither can name an axis
the parameter does not carry. A rule is given one seed, so nothing downstream of a parameter
can reach back and change what its gradient is written on.

A selecting mixture of experts is where that shows.
$W^{E}$ is `R[m, k/n, f]`, because the expert bank is indexed by the sparse axis, so
$\mathrm{d}W^{E}$ is `R[m, k/n, f]` too. $W^{g}$ is `R[m, n]`, because the router scores
every expert, so $\mathrm{d}W^{g}$ is `R[m, n]` even though the cotangent reaching it arrived
on `k/n`. The two axes are wired to one node and are different terms, per
[[Selection and the Reverse Pass]]. A router's gradient being sparse is a fact about its
values rather than about its shape, and carrying it needs the scatter that writes zeroes into
the $n - k$ columns.

`ShowGrabbedParameters` is a `functor.Endofunctor`, and it is stateful on purpose. The slot
is memoised per seed, so an expression that reaches the same seed twice, meaning one term
reached along two paths, which is what sharing a layer is in an immutable directed acyclic
graph, grabs one slot read twice, which is weight tying. Two layers built alike but separately
have different axes, so they are different terms, so they get slots of their own.

The slot's name is the parameter's, as `W_Q`, `b_Q` or `γ`, rather than a tape position. The
`s0` counter is for residuals, and [[Backpropagation]] writes those.

## How it prints and draws

- **[[Agent Display]]** prints the standard form as `%2 = Grab<WQ>() : R[m, d]`. Wrapped
  through `to_para_wrap`, the slot appears in the operand's place, as
  `%1 = Linear<Q>(<WQ>[{m}, {d}], %0[q, {m}]) : R[q, {d}]`.
- **[[Diagram Display]]** merges the grab into its seed by the existing rule of
  [[Para Wrap]], and the wrap draws the parametric operation as the body with the parameter
  arrays coming in from the top. Three small extensions made that read. A parametric box,
  meaning `LinearBox` or `NormalizeBox`, sets `raise_rows = false`, so its row stays at the
  glyph's own top edge instead of being lifted to the core's, and the tape visibly runs down
  onto the operation, crossing the degree wires on the way. `NormalizeBox` draws a rectangle
  with a bite out of its bottom-left corner behind its circle whenever the operator has more
  than one operand, which a grabbed γ makes it, so the gain has a box to arrive at. `TopKBox`
  draws the same figure for its second result, and the two read alike. The vertical run crosses no
  `ComposedGap` that could label it, so `BroadcastedBox.annotate_row_links` rests each axis
  name beside it, low on the run by the glyph. `EinopsBox` keeps `raise_rows` true, because
  its row anchors continue into cups and the turn belongs at the edge.

## A weight drawn as a box that reads the tape

`weight_box_fed_by(grab)` writes a grab followed by a `Linear` named after its slot, whose
one operand and one result are the array the grab reads, `Grab<W> ; W : 1 -> [a, b]`. The
box is the weight array of [[Linear Expansion]], a `Linear` with no data, with its weight
written out as an operand the way `ShowGrabbedParameters` writes the weight of every
other `Linear`. `to_para_wrap` absorbs the grab onto it, so a diagram draws a box labelled
with the weight and a tape running down onto it. `box_grabbed_weights(target)` puts the
box after every grab of a term.

The user ruled on 2026-09-17 that the inspection box over a weight draws this form, per
[[Advanced Display]]: a map `W : a -> b` becomes `[ParaWrap(grab(ab), W) : 1 -> ab] *
hold(a)` followed by the `Einops` of `ab` and `a` onto `b`. The form was what
`grab_parameters` wrote for a `Linear` with no operands until earlier the same day, when
such a `Linear` became the `Grab` alone, and tsncd drew it then and draws it now.

A `Linear` with one operand is otherwise read as a map applied to that operand, with its
weight inside the operator, and `linear_expansion.is_parametrised_linear` expects a data
operand after the weight. The weight box is told apart by the grab that feeds it, so it is
written for a diagram alone, after every pass that reads a `Linear` has run.

## A grab written as a weight array

`weight_array_in_place_of(grab)` writes a `Linear` with no operands, named after the slot
of the grab, whose one result is the array the grab reads, `W : 1 -> [a, b]`. The box is
the weight array of [[Linear Expansion]], which holds its weight inside the operator and
reads no tape. `write_grabs_as_weight_arrays(target)` puts it where every grab of a term
stood, so the result is a morphism of **Br** again. The user asked on 2026-09-17 for the
inspection box over a weight to be drawn with no `ParaWrap`, and
`expand_with_parameters.expanded_with_weight_arrays` draws this form, per
[[Advanced Display]]. A `Linear` with no operands is unambiguous, where the weight box of
the section above is told apart from a map by the grab that feeds it.

## Backpropagating the parametrised form

`backprop.forward_backward` runs on the Para form, and the backward pass reads the slots the
forward pass reads. A parametrised linear reverses to a parametrised `Transpose`, giving
$dx$ against the same weight as an operand, with $dW = x \otimes dy$ summed over the batch by
the weight's node and dropped to the gradient slot `dW1`, through `backprop.gradient_slot`,
where $R$ swaps `Grab` and `Drop` as [[Training]] states.

Three mechanisms produce it: the rule, the swap, and `collapse_grabbed_residuals`. The third
rests on the fact that a grabbed value is its own residual. The rule tapes its $W$ operand
like any residual, and the pass deletes that `Drop` and points the backward pass's grab at
the parameter's own slot, on the graph, where the same wire is a node. A tied weight's two
reverse grabs write one gradient slot, which is accumulation stated by the slots. The pass
walks the graph through its blocks, by `graphs.processing.replace_roots`, and it collapses
a grabbed index the same way, so a selecting `Linear`'s transpose grabs the slot the model
dropped the index to, per [[DeepSeek-V3 Backward Pass]].

It is checked numerically. `para/validate_backward.py` compiles the parametrised FFN and
matches $dx$, $dW_1$ and $dW_2$ against `torch.autograd`. `torch_compile` gained the
parametrised `Linear` and `Transpose` as one einsum, read off axis identity, which the
transform guarantees by building the weight from the seed's own weave targets.

A gradient is a chain read by nothing but its `Drop`, which the demand frontier of
[[Hypergraph to Morphism]] never reaches, because the frontier starts at the `cod`. The
conversion builds each such chain into a branch first and injects it into the column that
first consumes a wire the chain reads, so $dW_2$ is drawn beside `Transpose<2>`. That note
has the detail.

## Gaps

- **`Embedding` is absent.** Its table is the vocabulary by the target, and the vocabulary is
  a `Natural` datatype rather than an axis. `selected_axis` now mints exactly that axis for a
  selecting `Linear`, so the objection that an axis invented here is one nothing else names
  has been answered for one operator and not for this one. What an `Embedding`'s table should
  be indexed by has not been settled.
- ~~A selecting `Linear` grabs its weight and does not reverse~~. Closed 2026-09-04.
  `derivative.linear` keeps the index on the transpose and writes the weight gradient through
  `inject.inject_slab`, per [[Selection and the Reverse Pass]], and
  `para/validate_backward.py` asserts the derived pass.
- **`Normalize` reverses to an opaque box.** The plumbing works, with γ grabbed by the
  `R[RMSNorm]` box and dγ summed and dropped to its slot, and the RMS derivative itself is
  undeclared, the way `SoftMax`'s once was.
- **A parametrised `Linear` with a bias does not compile.** `ConstructedLinear` reaches
  `parametrised_linear_func`, and the `View` that carries the bias to the degree raises
  `IndexError` inside `bcast.unsqueeze_guide`. The failure predates the parameter-first
  order and `para/validate_backward.py` exercises no bias, so nothing reported it.
- **Nothing reads the gradient slots.** An optimiser step, meaning grab the parameter, grab
  its gradient and write the parameter back, is the remaining leg of [[Training]], and it is
  the first thing that needs a slot written across derivations rather than within one.
- **A `Grab` has no `degree`**, which is the gap [[Backpropagation]] records, so a
  parametrised pass is detaped before anything reads it as an ordinary morphism.
- **`collapse_grabbed_residuals` and [[Pathway Collapse|pathway collapse]] have not met.**
  The full stack on projected attention is the natural next experiment. Run the collapse
  before `dedup_slots`, because the `s{n}` numbering of `dedup_slots` does not expect a slot
  named after a parameter.

## See also

- [[Notebooks]] — what each notebook of the repository demonstrates
- [[Para Category]] — `Grab`, `Drop`, and why they are composition-neutral
- [[Para Wrap]] — the layering rule and the four-sided box this extends
