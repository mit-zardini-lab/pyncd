---
tags: [layer/para, reference]
code: para/processing/backprop.py, para/registries/derivative.py, para/data_structure/contraction.py, para/algebra/tangent.py, para/algebra/prune_zero_cotangents.py, para/algebra/detape.py
status: working
---

# Backpropagation

## What it is

`backprop.forward_backward(m)` returns a `Taped`, holding two passes. `Taped` is a
[[Para Category|MultiCategory]] of two rows. The covariant row is the forward pass, which
is `m` with its residual saved to a tape. The contravariant row holds the backward pass in
a `Contravariant`, because the pass reads from the cotangent of the codomain back to the
cotangent of the domain. The properties `forward` and `backward` hand each pass back as
the executable covariant morphism, which is what every rewrite in `para/algebra` works
on, and `tsncd` draws the whole `Taped` as the two passes stacked, per
[[Diagram Display]], which draws a taped pair with the slots on the operations. It
implements the method of [[Derivatives]], and it is what [[Training]] listed first under
*Gaps* as "$R$ does not exist".

$$\mathrm{Forward} = (0,0)\,;\,(\mathrm{hold}(X) \otimes \mathrm{drop}(X))\,;\,F
\qquad
\mathrm{Backward} = (\mathrm{grab}(X) \otimes \mathrm{hold}(T^{*}Y))\,;\,R[F]$$

Both are ordinary morphisms in the broadcasted category rather than symbols. The reverse of an
attention is `Einops`, `AdditionOp` and `Elementwise` that the rest of the repository can
read. `para/validate_backward.py` derives the dot product, the matmul, the
softmax and attention, and prints both passes for each.

What the rules write is the expanded form. Step 1 of [[Derivatives]] factors every reindexing
into its own node before differentiating, and the reverse pass keeps them, so the backward
pass of attention arrives as 10 `Einops` and 4 `View` with rank-3 intermediates between them.
[[Expression Simplification]] puts it back together, absorbing the nodes into their readers'
reindexings and merging each broadcast into the sum that follows it, and what comes out is the
five contractions of the FlashAttention backward. The notebook simplifies every pass before
showing it, and exhibits the raw pass separately, labelled *as derived*, in the two sections
about the scaffolding itself, which are the matmul and the expanded softmax. The derived form
is the one to check the derivation against.

## Where it lives

| | |
|---|---|
| `para/processing/backprop.py` | `forward_backward`, the recursion, and the tape wiring |
| `para/registries/derivative.py` | one `Rule` per operator, carrying $R[f]$ and its `Residual` |
| `para/data_structure/contraction.py` | `contract`, an einsum over axis objects, and `Zero` |
| `para/data_structure/transpose.py` | `Transpose` and `ReindexTranspose`, the reverse of what is already linear |
| `para/algebra/tangent.py` | the tangent functor on objects, which is partial |
| `para/data_structure/Para.py` | `TapeSlot`, `Grab`, `Drop`, and now `new_slot` and `reset_slots` |
| `para/data_structure/ParaWrap.py` | `ParaWrap` and `to_para_wrap`, the display form, per [[Para Wrap]] |
| `para/algebra/prune_zero_cotangents.py` | `prune_zero_cotangents`, which removes the zero map the rule for `Maximum` writes, with the chain that fed it and the slots only that chain read |
| `para/algebra/detape.py` | `detape`, which makes each `Drop` an extra output and each `Grab` an extra input, so that a pass can be compiled or read as an ordinary morphism |

## How it recurses

It recurses exactly as the contravariant category is read, and for the same reasons.

| | forward | backward |
|---|---|---|
| `Composed` | each factor | each factor, with the order reversed |
| `ProductOfMorphisms` | each factor | each factor, in place |
| `Block` | a block | a block titled `R[...]`. It is the only way a block enters a backward pass, because a rule never writes one of its own. The reverse of a `SoftMax` is four bare operations, since a block that was not in the expression is one nobody wrote |
| `Rearrangement` | itself | the transpose, described below |
| `Grab` and `Drop` | itself | the other one, on the gradient slot, so `Grab<W1>` reverses to `Drop<dW1>`, which is [[Training]]'s statement that $R$ swaps them. `gradient_slot` is memoised per parameter, so the two reverse grabs of a tied weight write one slot, which is accumulation by the comonoid. A wire with no tangent, such as an index, reverses to nothing |
| `Broadcasted` | the nodes, then the core, plus the `Drop`s | the `Grab`s, then the nodes, then $R[\mathrm{core}]$, then the node transposes |

The difference from `Contravariant.Contravariant` is at the leaves alone. A
`Contravariant` holds the forward expression and leaves every leaf symbolic, and
`backprop` interprets each one.
Everything above the leaves is the same construction written twice, which is worth knowing
before changing either.

### The rearrangement transpose is one rule rather than three

A copy becoming an addition, a permutation becoming a permutation, and a deletion becoming
zero are the same statement: for each domain segment, add up every codomain segment that
`mapping` sent to it. Over zero segments the sum is `contraction.Zero`, over one it is the
identity, and over several it is an `AdditionOp`. That a copy dualises to an addition is the
comonoid identity, and it is why a slot's write is `+=`, per [[Training]].

### The seed case, and why node expansion comes first

[[Derivatives]] argues it. A reindexing is affine so its transpose is determined, an operator
needs a declared rule, and mixing them would make every operator re-derive the broadcasting.
`algebra.node_expansion.expand_to_nodes` splits them, the nodes are transposed generically,
and a rule is only ever given a cleanly broadcast core.

The one thing the note did not state, and the implementation had to settle:

> **The residual is taped unexpanded, and pushed back through the node in the backward pass.**
> The rule needs the core's operands, which are the nodes' outputs, and taping those would
> store the broadcast copy, at $q \times v \times d$ floats where $q \times d$ were written. A
> reindexing is free to redo, so the tape holds the small thing and the backward pass
> re-applies the same node.

### `contract`, and why `Einops.template` is not enough

Every derived contraction is built by `contraction.contract(inputs, output)`, which takes
shapes as tuples of index variables. A rule reads its operator's variables off its weaves,
reindexings and signature through `einops_simplification.index_shapes`, one per degree
position and one per contraction group, so a repeated axis at two positions is two
variables and the transpose of the score contraction of self-attention from one copied
input is written correctly. A shape may also be a tuple of `Axis` objects, in which case
each axis is its own variable. `ops.Einops.template` calls `RawAxis.named` for every
symbol and so mints fresh axes on each call. Fresh axes are right for writing an expression down and
impossible for deriving one, because the reverse pass has to contract against axes that
already exist, per [[UIDs and Names]].

`contract` classifies an axis as absorbed, meaning it is in an operand and not in the result,
or as degree, meaning it is in the result, and picks the operator from that. An axis in the
result that no operand has is a degree axis that no operand's reindexing names, which is a
repeat along it, written as a `View`. The dual of a sum over an axis is therefore a repeat
along it, and a repeat is a node. There is no operator for it, and since 2026-08-21
`ops.Einops.template('q -> q x')` writes the same `View`, per [[Expression Simplification]].

## The rules that exist

| operator | residual | $R[f]$ |
|---|---|---|
| `Einops` with several operands | the operands | contract the cotangent against the others |
| `Einops` with one operand | nothing | a sum reverses to a repeat, and a pointwise map to itself |
| `View`, meaning a node, with a `Rearrangement` reindexing | nothing | a sum over the fibre, written as a contraction |
| `View`, meaning a node, with a strided reindexing | nothing | a `ReindexTranspose` carrying the reindexing whole |
| `ReindexTranspose` | nothing | the node it is the transpose of |
| `Linear` and `Transpose` | nothing | the weaves swapped, so it is the same weight read the other way |
| `Linear` and `Transpose`, parametrised, per [[Show Grabbed Parameters]] | $x$ and $W$ | it is multilinear once the weight is an operand. $dx$ is a `Transpose` keeping $W$ as an operand, $dW$ is $x \otimes dy$ summed over the batch by the weight's node, and $db$ is $dy$ summed likewise. The $W$ residual is a grabbed value, and `show_grabbed_parameters.collapse_grabbed_residuals` turns it back into a grab of the parameter's own slot, so the backward pass reads the slots the forward pass reads |
| `Linear` and `Transpose`, selecting, meaning one operand has a `Natural` datatype | the index as well | the transpose keeps the index, so it reads the slab the forward pass read. $dW$ is the outer product, which is one slab, written by `inject.inject_slab` at the position the index names and zero elsewhere, and the weight's node sums it over the degree. A `Linear` with only an index has no $dx$. The index residual collapses onto the slot the model grabbed it from, per [[DeepSeek-V3 Backward Pass]] |
| `TopK`, complete, per [[Selection and the Reverse Pass]] | the index output | `Inject` of the values' cotangent at the positions the index names. The index has no cotangent, so the reverse has one input fewer than the forward has outputs |
| `AdditionOp` | nothing | the cotangent to every operand, summed over its broadcast axes |
| `SoftMax` | its output | $\mathrm{d}x = y \odot (\mathrm{d}y - \langle \mathrm{d}y, y \rangle)$ |
| `Arithmetic` | its input | $\mathrm{d}y \odot f'(x)$, with $f'$ written out as the formula's derivative, another `Arithmetic`. `e^{x}` reverses through `e^{x}`, and `x^{-1}` through `-x^{-2}` |
| `ReLU` | its input | $\mathrm{d}y \odot [x > 0]$, with the step function written as `Arithmetic<[x > 0]>`. It is the derivative of `nm.RectifiedLinear`, whose expansion is $x\,[x > 0]$, per [[Numerics]] |
| `Elementwise` | its input | $\mathrm{d}y \odot \sigma'(x)$, with $\sigma'$ a primed `Elementwise`, because the map is a name the algebra cannot differentiate |
| `Maximum` | nothing | the zero map, $0 \cdot \mathrm{d}m$, repeated back over the folded axis. A `Maximum` in this package shifts a softmax before its exponent, and a softmax is unchanged by a shift of its scores, so the cotangent that reaches the maximum through the shift is zero in total. The rule is declared a-priori for that use. It is not the reverse derivative of a maximum read for its own value, which would send the cotangent to the position of the largest score, and the package writes no such maximum |
| anything else | its whole domain | a named box, `R[name]` |

A rule declares the reverse derivative directly, rather than a forward `Derivative` that
something else transposes, which collapses steps 3 and 4 of [[Derivatives]] into one. The
justification is that the transpose of $D[f]$ is not derivable from $D[f]$ syntactically,
because transposing a SoftMax Jacobian is linear algebra rather than a rewrite, so it would
have to be declared per operator anyway.

### Pruning the zero cotangents

The rule for `Maximum` writes its zero as a pointwise map whose formula is the constant
$0$, followed by a repeat back over the folded axis, so the cotangent stays a wire and the
derived pass is well formed. Derived as written, the backward pass of a shifted softmax
therefore carries a chain that computes the cotangent of the maximum, multiplies it by
zero and adds the result to the cotangent of the scores, and the forward pass drops the
maximum for that chain alone. `prune_zero_cotangents` removes both.
`pathway_collapse.collapse` runs it before its other rewrites. It deletes every pointwise
map whose formula is the constant zero and every `contraction.Zero`, together with the
`View`s that only reindex a zero wire. Every addition that read a deleted wire is rebuilt
without that operand, and an addition left with one operand of its own shape becomes a
rename of that operand. The roots whose outputs nothing then reads are removed to a
fixed point, and the forward pass loses the `Drop` of any slot no remaining `Grab` reads.
The collapsed pair of a shifted softmax then has the unshifted pair's form, and its
forward pass keeps the `Maximum`, which `para/validate_backward.py` checks.

## The rules

- **What is already linear takes no residual.** A `Linear`, or a node, a repeat included,
  propagates the cotangent on its own, so the rule declares `Residual()` and nothing is taped.
  Before `transpose.py`, `Linear` had no rule and fell to `opaque`, which declares the whole
  domain, so every matrix multiply in a model taped its input for a backward pass that did not
  read it.
- **A node transpose takes no residual, and is asserted not to.** Node transposes are producted
  straight onto the cotangent, so a rule declaring a residual would be wired to the wrong
  wire. `backprop._transpose_node` asserts it.
- **A strided reindexing transposes into a `ReindexTranspose` rather than a contraction.** A
  `Rearrangement` selects, permutes and repeats whole axes, so its fibre is a set of axes and
  the transpose is an ordinary contraction, which is the case [[Expression Simplification]] can
  then absorb and merge, and which is why it is kept. A convolution window sends several domain
  axes to one codomain axis, so its fibre is a diagonal, `contract` cannot name it, and the
  operator carries the whole reindexing instead. It is not `Scatter`, per
  [[Selection and the Reverse Pass]], whose index map arrives on a wire and cannot be written
  down from the term.
- **A transpose is an involution.** `R[Transpose]` is `Linear` and `R[ReindexTranspose]` is the
  node, so `forward_backward` applied to a derived backward pass returns the forward one
  operator for operator. Keeping the involution true is the cheapest check that a rule is a
  transpose rather than merely something with the right domain and codomain.
- **Slots are named rather than identified by UID.** `Para.new_slot` names them `s0`, `s1` and
  so on in creation order, and `forward_backward` resets the counter, because `UID._id` is
  random per process and a listing that prints one stops being comparable with `diff`, per
  [[Invariants]].
- **The tape is the only coupling.** Nothing else crosses between the two morphisms.

## Gaps

- **A `Grab` has no `degree`**, so a pass is detaped before anything reads it as an
  ordinary morphism. `para.algebra.detape` makes each `Drop` an extra output and each
  `Grab` an extra input, in slot order, and raises on a tape operation below the top
  level, so a pass holding a tape operation inside a loop block cannot be detaped.
- ~~The residual is taped once per consumer~~. Closed 2026-08-20.
  `pathway_collapse.dedup_slots` merges same-wire `Drop`s on the hypergraph, where a wire is an
  wire, and canonicalises the backward pass's `Grab` of the removed slot onto the kept
  one, so the consumers follow the node, block interiors included. The derivation still writes
  one slot per consumer, because each seed declares its residual with no global view, and the
  dedup is the post-pass, run before simplification, per [[Pathway Collapse|pathway collapse]].
- ~~A slot is grabbed once per reader~~. Closed 2026-09-04. `pathway_collapse.dedup_roots`
  merges roots wrapping equal morphisms on the same wires, grabs among them, into one root in
  the innermost block enclosing every reader, through `graphs/processing/merge_duplicate_roots.py`
  per [[Functors]]. A loop block is a boundary, so a value inside a loop is grabbed once per
  loop body. The derivation still writes one grab per reader, for the same reason it writes
  one slot per consumer.
- ~~`tangent` is written and never exercised~~. Closed 2026-09-04. A selecting `Linear`'s
  index operand is a `Natural` wire, `_rearrangement` drops it, and `_broadcasted` transposes
  no node for it, because a rule emits no cotangent for an input with no tangent.
- **A loop block reverses with the same repetition**, which is right only if the tape is read
  back in the order it was written. A slot would need an index otherwise.
- **A `Transpose` with an implicit weight is not written as the forward weight read the
  other way.** `expand_linear_root` leaves a `Linear` with several inputs unchanged, so
  the parametrised backward pass of [[Show Grabbed Parameters]] gains no second weight
  under [[Linear Expansion]]. What remains is the unparametrised form. `ExpandLinear`
  leaves a `Transpose` as it finds it, because a `Transpose` is not a `Linear`, and nothing
  reads the forward operator the transpose keeps in its `operator` field to write the
  weight array the forward pass would share. Sharing a weight between the passes goes
  through `show_grabbed_parameters` first, per [[Open Gaps]].
- **The context is not factored out of a `ReindexTranspose`.** It is built with an empty degree
  and the whole of both shapes in the targets of its weaves, so the transpose of a batched
  convolution consumes the batch axis where the forward node was broadcast over it. It is
  correct, and it is harder to read than it needs to be.
- **A `Rearrangement` under a `Contravariant` is still uninterpreted**, and the meaning
  exists a second time beside it, since `backprop._rearrangement` is the addition that
  reversing a copy means. Whether `Contravariant.py` should defer to it is open.

## See also

- [[Derivatives]] — the method this implements, in four steps
- [[Training]] — what a reverse pass is, and the four writers of a parameter slot
- [[Para Category]] — the tape, and the reference construction
- [[Selection and the Reverse Pass]] — the data-dependent case, and `Scatter`
- [[Expression Simplification]] — putting the expanded pass back together afterwards
- [[Pathway Collapse|pathway collapse]] — re-expressing a backward chain by a value the forward computed, deriving the FlashAttention D-trick
- [[Show Grabbed Parameters]] — weights exposed on the domain, and the parametrised passes with their gradient slots
- [[DeepSeek-V3 Backward Pass]] — a whole layer, with a selecting `Linear` and a complete `TopK`
- [[Linear Expansion]] — the other consumer of `expand_to_nodes`
- [[Agent Display]] — how both passes are read
