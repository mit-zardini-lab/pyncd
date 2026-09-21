---
tags: [layer/para, concept]
code: para/registries/derivative.py, deepseek/registries/derivative.py, para/data_structure/inject.py
status: working
---

# DeepSeek-V3 Backward Pass

## What it is

One layer of DeepSeek-V3, written as a compressed sparse expression, expanded onto the tape
so that the `TopK` drops its index to a slot and each expert `Linear` grabs it back, and
differentiated in that form. The derived backward pass grabs the same slot at the injection
that puts the router's cotangent back on the expert axis, at each expert's transpose and at
each expert's weight gradient, and a final stage merges the grabs into one that the three
blocks read. Nothing is tied. The drop and the grabs are connected by the slot's name and by
nothing else, per [[Para Category]].

Every pointwise map in the layer is an `ops.Arithmetic` over a formula, so the backward pass
carries the written derivative of the sigmoid, of $x\,\sigma(x)$, of the softmax and of the
RMSNorm, and no operator falls to the opaque rule of [[Backpropagation]]. The RMSNorm and
the softmax are written out of their operators for the same reason. The model is the one at
[arXiv:2412.19437](https://arxiv.org/abs/2412.19437), with the parts listed under *Gaps*
left out.

## Where it lives

| | |
|---|---|
| `para/registries/derivative.py` | `linear`, with the selecting cases and `_weight_slab_gradient` |
| `deepseek/registries/derivative.py` | `top_k`, with `complete_top_k` for the complete spelling |
| `para/data_structure/inject.py` | `inject`, `inject_onto` and `inject_slab` |
| `graphs/processing/replace_roots.py` | the walk through blocks that `collapse_grabbed_residuals` and `dedup_slots` use |
| `graphs/processing/merge_duplicate_roots.py` | the merge of roots computing one value, which `pathway_collapse.dedup_roots` applies to both passes |

## The three forms

**Compressed.** `deepseek.TopK.template` takes the sigmoid scores on `e` and returns one
wire on the sparse axis `k/e`. Each expert projection is one `Linear` producing the expert
axis, and the down projection produces it a second time under a diagonal, which is the idiom
of [[Sparse Axes]]. One sparse axis, no `Natural` datatype.

**Expanded onto the tape.** `para_sparse_expansion.expand_sparse_onto_tape` gives the
`TopK` a second output, the `Natural(e)` index, dropped to `inject.selection_slot` of the
sparse axis, and gives each producing `Linear` the index as an operand grabbed from that
slot. The listing reads `%36, <idx_{k/e}> = TopK(...)` and
`Linear<G>(<idx_{k/e}>[x, k], ...)` three times. The layer's domain and codomain are
unchanged, per [[Sparse Expansion]].

**Parametrised.** `show_grabbed_parameters.grab_parameters` makes every weight an operand.
A `Linear` whose first operand is a `Natural(e)` selects one of `e` weights, so its weight
is the whole bank, $W^{G} : R[e, m, f]$, $W^{U} : R[e, m, f]$ and $W^{D} : R[e, f, m]$, per
[[Show Grabbed Parameters]]. The four RMSNorm gains are grabs the model declared, because a
norm written out as a formula has no operator for a gain to sit inside.

## The two rules the derivation needed

**A complete `TopK` declares its index as the residual.** `complete_top_k` returns
`Residual(outputs=(1,))` and `inject.inject`, so `backprop` drops the index in the forward
pass and grabs it in the reverse. The forward pass then holds two drops of the index wire,
the model's and the rule's, and `pathway_collapse.dedup_slots` keeps the model's slot,
because a slot outside the `s{n}` numbering ranks ahead of every residual slot, and points
the reverse pass's grab at it.

**A selecting `Linear` declares its index as a residual too.** `derivative.linear` tells the
forms apart by datatype. The transpose keeps the index as an operand, so
`Transpose<G>(<W_G>, <idx_{k/e}>, dh)` reads the slab the forward `Linear` read. The weight
gradient is the outer product $x \otimes \mathrm{d}h$ at the degree $(x, k)$, which is one
slab of the bank per token and slot, and `inject.inject_slab` writes it at the expert the
index names and zero at every other. The weight's own node transpose then sums over
$(x, k)$, and the injection followed by that sum is the scatter-add
[[Selection and the Reverse Pass]] recorded as missing. The index wire came from a grab,
so `collapse_grabbed_residuals` points the reverse pass's grab at the slot the model wrote,
as it does for a weight.

Two smaller changes carry both. `backprop._broadcasted` transposes no node for an input
with no tangent, because a rule emits no cotangent for an index. `replace_roots` walks the
roots of a graph through its blocks, so the two tape passes reach a grab inside a block, and
`dedup_slots` canonicalises a duplicate grab onto its twin only inside one scope, pointing
it at the kept slot across scopes, because a wire cannot be moved between blocks by
renaming it. `pathway_collapse.dedup_roots`, run as the notebook's last stage, merges the
grabs of one slot across blocks. It keeps one grab in the innermost block enclosing every
reader and recomputes the domain of each block on the way, so the index is grabbed once in
each pass, per [[Functors]]. A loop block is a boundary for the merge, and the layer has
none.

## What the backward pass reads as

The listing holds 108 assignments: 61 `Einops`, 15 `Arithmetic`, 13 `AdditionOp`, 13
`Transpose`, 4 `Inject`, one `ReindexTranspose` and one `View`. Read against the model, the sparse axis
ends at `%9 = Inject(<idx_{k/e}>[x, {k}], %8[x, {k}]) : R[x, {e}]`, and every line after it
in the router runs over all `e` experts, so $\mathrm{d}W^{R}$ covers $R[m, e]$ with the
unselected rows the zeroes the injection wrote. The router's injection, the three expert
transposes and the three weight-gradient injections each grab the index, and the router,
the experts and the shared expert grab the normalised tokens five times between them. The
notebook's last stage runs `dedup_roots` on the pair, after which the backward pass holds one
grab of each, placed in `R[Mixture of Experts]` and read by the three blocks, and the
forward pass one grab of the index inside `Experts`. The stage removes two roots from the
forward pass and seven from the backward pass, and the operation counts, the slots, the
domain and the codomain are unchanged. The notebook's `DiagramSettings` sets
`tape=ABSORBED`, so every listing and diagram passes through `to_para_wrap` just before it
is shown and the algebra runs on the `Para` underneath, per [[Diagram Display]]. The wrap
splits a grab read by several siblings into one grab per reader, so the absorbed forms of
the merged passes coincide with the absorbed forms before the merge, and the merge stage is
carried by its assertions, which read the graph. Neither stage ties a drop to the grabs of
its slot.

The notebook's last stage runs `pathway_collapse.collapse` on the merged pair, followed by
`dedup_roots`. Every rewrite in [[Pathway Collapse|pathway collapse]] now reaches through
the blocks, so the softmax three blocks deep collapses as a bare one does. The backward
pass goes from 61 einsums and 15 arithmetics to 60 and 14, and the tape from 45 residual
slots to 39. The softmax's four residuals go, the scores, the exponential, the sum and its
reciprocal, and `R[SoftMax]` holds a negation and one einsum, reading the probabilities $P$
already saved for the value contraction and the row statistic
$D = \langle \mathrm{d}O, O \rangle$, computed one block out from the attention output $O$
already saved for $W^{O}$. The cotangent of the scores is $P \odot \mathrm{d}P - P \odot D$,
which is the FlashAttention backward. The gate normalisation loses its two residuals the
same way. The three `R[RMSNorm]` blocks are unchanged, for the reason under *Gaps*, and
carry
`2 * x` and $-\tfrac{1}{2|m|}(\epsilon + \tfrac{x}{|m|})^{-3/2}$, written by
`solver.registries.numeric_derivative`, and the gain gradients $\mathrm{d}\gamma$ dropped to
their slots. The attention backward is the contractions of [[Backpropagation]]'s
attention with an `R[SoftMax]` block in place of the operator's reverse rule, and
`ReindexTranspose<=>` sums the key path back onto the tokens. The block differentiates the
two paths of the division, holding `e^{x}` for the numerator and `-x^{-2}` for the sum.

The forward pass performs the model's operations and no others, and the backward pass
grabs the weights, the gains, the index and the residuals the forward pass wrote, and drops
one gradient per weight and per gain. The notebook asserts each of these with the diagrams
off.

## The rules

- **Build a contraction from declared axes.** `ops.Einops.template` mints a fresh axis for
  every letter, and a fresh axis composed onto a declared one can win canonicality and take
  the declared axis's name. The first RMSNorm printed `R[x, {m}]` for the query latent `c`
  while its formula read `|c|`. `algebra.einops_simplification.einsum` over the axis objects
  is the construction, per [[Invariants]] under *Axis identity*.
- **Keep the key axis a second axis.** Composition aligns axes by position, so
  self-attention written with one token axis scores a diagonal `R[h, x, x]`.
  `diffusion_unet.relabel_axis` reads the tokens under the key name, and its transpose is a
  `ReindexTranspose`.
- **Write a pointwise map as a formula.** An `ops.Elementwise` carries a name the algebra
  cannot read, and a published name is not unique. `x \sigma(x)` is the SiLU and also the
  swish, and the swish is also written with a learned scale. `deepseek_v3` builds every
  pointwise map from `ops.Arithmetic` and lets the box print the formula, except the
  sigmoid, whose `\sigma` is shorter than its formula and names one function.
- **The softmax is written out.** `ops.SoftMax` reverses through the rule in
  `para.registries.derivative`. Written as an exponential, a sum over the keys and a
  reciprocal, it reverses through the numeric derivatives of `e^{x}` and `x^{-1}` instead,
  and the layer holds no operator whose reverse is declared a-priori. The maximum is not
  subtracted, so the expression is the definition of the softmax.
- **A grab inside a block is a grab.** Both tape passes walk the graph through its blocks.
  `dedup_slots` deletes a duplicate grab only where its twin is in the same scope, and
  `dedup_roots` merges the grabs across blocks whose repetition is 1, because a loop block
  is the one block that changes what a root inside it computes.

## Gaps

- **The decoupled rotary query and key are absent.** A rotary embedding is a linear map
  whose weight depends on the position. A `Linear` broadcast over the tokens shares one
  weight across them, and a `Linear` with the tokens in its target holds a dense matrix over
  positions. A rotation table on a wire would receive a cotangent nothing reads.
- **The causal mask is absent.** `ops.WeightedTriangularLower` has no derivative rule.
- **The auxiliary-loss-free routing bias is absent.** DeepSeek-V3 selects on $s + b$ and
  gates on $s$, which needs the gate read at the index through `deepseek.IndexSelect`, whose
  reverse is the same `Inject` and is not registered, and the `TopK`'s values output left
  unread. [[Training]] classifies $b$ as a parameter the forward pass writes.
- **The embedding, the unembedding and the stack are absent.** An embedding is a `Linear`
  whose only operand is a `Natural` index, which `derivative.linear` reverses into the
  injection of the cotangent onto the slab the token names, so nothing new is needed. A
  repeated block reverses with the same repetition, and [[Backpropagation]] records the
  condition on reading the tape back in order.
- **The weight gradient materialises the bank once per token.** `Inject` at the degree
  $(x, k)$ produces $R[x, k, e, m, f]$, and the sum over $(x, k)$ follows as a separate
  `Einops`, because `merge_into_consumer` merges an einsum into an einsum and `Inject` is
  neither. A scatter-add operator consuming the degree would be the merged form.
- **The RMSNorm keeps four residuals.** Its inverse root's derivative is written as a formula
  in the sum of squares, $-\tfrac{1}{2|m|}(\epsilon + z/|m|)^{-3/2}$, and
  `split_inverse_squares` reads through the power $x^{-2}$ alone. Written as
  $-\tfrac{1}{2|m|} r^{3}$ the chain would collapse onto the saved $r$, and each block would
  keep $r$ and $\hat{x}$ in place of $x$, $z$, $r$ and $\hat{x}$. [[Pathway Collapse]]
  states the general rule.
- **Nothing checks the derivation numerically.** `torch_compile` has no module for a
  complete `TopK`, for `Inject`, or for a `Linear` or `Transpose` with an index operand, so
  `para/validate_backward.py` cannot compile the pair. The three modules are the gather by
  the index followed by the einsum `parametrised_linear_func` already writes.

## See also

- [[Backpropagation]] — the rules table, with the selecting `Linear` and the complete `TopK`
- [[Selection and the Reverse Pass]] — what a selection does to the reverse functor
- [[Sparse Expansion]] — the rewrite between the two forms, and its taped variant
- [[Show Grabbed Parameters]] — the whole expert bank as one weight
- [[Training Mixture of Experts Gates]] — what trains a router, argued and now derived
- [[Notebooks]] — what each notebook of the repository demonstrates
