---
tags: [layer/para, concept]
code: deepseek/data_structure.py, deepseek/registries/derivative.py, deepseek/sparse_expansion.py, para/data_structure/Para.py, para/data_structure/inject.py, para/registries/derivative.py
status: speculative
---

# Selection and the Reverse Pass

## What it is

A top-k is piecewise constant, so its derivative is 0 almost everywhere. Every account of
the problem starts there and then reaches for straight-through estimators.
The structural statement is sharper, and this vault already has the pieces for it:

> A selection is a morphism whose index output has no dual, and whose value path reverses
> into a scatter along that index. It is differentiable, and its index has
> nothing to differentiate.

That reading turns "top-k is not differentiable" into two concrete claims about this
codebase. The reverse pass needs the expanded presentation of a selection, and the operator
set is not closed under the reverse functor $R$ of [[Training]].

## Only the expanded form can be reversed

[[Sparse Axes]] keeps a selection in two presentations. Compressed, `TopK` emits one wire on
a `SparseAxis` and the payload rides the same axis. Expanded, `TopK.complete` emits the pair
that `torch.topk` returns: the surviving values on $k$, and a `Natural(n)` index per slot.

$$[\mathbb{R},\, k/n] \;\cong\; [\mathbb{R},\, k] \;\otimes\; [\mathrm{Nat}(n),\, k]$$

Sending a gradient back to the parent axis requires knowing which of the $n$ each of the $k$
came from. The compressed form does not expose it, because that is precisely what the
compression removed. A correct $R$ is therefore not defined on a compressed selection, and
[[Sparse Expansion|`expand_sparse`]] is a precondition for differentiating as well as for
reading.

Both forms reverse, and `deepseek/registries/derivative.py` registers the rule for each.
*`Inject`, the reverse of a selection* below says what it computes in each, and where the
index comes from.

It is the same rewrite that [[Open Gaps]] item 1 requires before anything analyses the
dependency a gather introduces. The rewrite that makes a selection legible to that
analysis is the rewrite that makes it reversible.

## The dual of `IndexSelect` is a scatter, and it does not exist

After expansion a `Select` is
`hold(values) × IndexSelect(index, payload) ; einops('k, k -> k')`. `IndexSelect` is the pure
gather, $\mathrm{out}[j] = \mathrm{payload}[\mathrm{index}[j]]$, with the index an operand of
datatype `Natural(n)`. Reversing it does two things.

- **The index operand vanishes.** `Natural` has no cotangent, per [[Training]], so the index
  wire is absent from the reverse graph. It did its job by being saved to the tape in the
  forward pass and loaded back.
- **The payload's gradient is a scatter-add**:
  $\mathrm{d\,payload}[i] \mathrel{+}= \sum_{j\,:\,\mathrm{index}[j] = i} \mathrm{d\,out}[j]$.
  The `+=` is the accumulation rule of [[Training]], because a gather may name the same
  source twice.

Read against the one-hot $(n, k)$ matrix, `IndexSelect` is an ordinary contraction and its
dual is the transposed contraction. The transposition is what the adjointness of dispatch
and combine
means in [[Open Gaps]] item 1, and it confirms that a scatter-add is the right dual rather
than a convention.

There is no scatter in [[Operators]], and `Inject` is what stands where one would. Every
argument that forced `Select` and `IndexSelect` into existence runs identically in the
reverse direction. A scatter is not an `Einops`, because an einops signature describes a
contraction and a scatter is not one, and it is not a reindexing, because a reindexing is
affine and $k \to n$ through a runtime index is not. `Inject` takes the index as an operand
and writes each value at the position the index names. The accumulation is the sum over the
degree that follows it, which a weight's node transpose already writes, so a scatter-add is
`Inject` followed by an `Einops`. `IndexSelect` has no rule registered, and the rule would
be that pair at the payload's degree.

The index form is also the one that does not materialise $n \cdot k$, which is the same
reason [[Sparse Axes]] prefers it in the forward direction. A scatter written as a one-hot
contraction would be correct and would defeat the point.

## `Inject`, the reverse of a selection

`para/data_structure/inject.py` declares `Inject`, and
`deepseek/registries/derivative.py` registers it as the reverse derivative of `TopK` in
both spellings by `para.registries.derivative`'s own decorator, so declaring a reverse for
an operator `para` has never heard of needs no edit to the reverse functor.

$$\mathrm{TopK} : [\mathbb{R},\, n] \to [\mathbb{R},\, k/n] \qquad\qquad
\mathrm{Inject} : [\mathrm{Nat}(n),\, k/n] \otimes [\mathbb{R},\, k/n]
\to [\mathbb{R},\, n]$$

The selector is an operand, because an injection has to say which of the `n` positions the
`k` cotangents land on. In the compressed spelling it arrives from a slot
`inject.selection_slot` derives from the sparse axis through `fd.hash_id`, so the drop a
forward expansion writes and the grab this rule writes agree on the slot with nothing
shared between them but the axis they both name. `nm.FreeNumeric.named` derives an id the
same way, and it is the only other place in the package that does. In the complete spelling
the index is the `TopK`'s second output, and `complete_top_k` declares it as the residual,
so `backprop` drops it in the forward pass and grabs it in the reverse. A model that
already drops the index to a slot, which is what `expand_sparse_onto_tape` produces, then
holds two drops of one wire, and `pathway_collapse.dedup_slots` keeps the model's slot and
points the reverse pass's grab at it, since 2026-09-04.

`Inject` writes each surviving cotangent at the position its value came from and zero
everywhere else. The rule is the forward selection's weaves swapped, which is the
construction `transpose.transpose` uses and is this short for the same reason: every
reindexing of a cleanly broadcast operator is the degree identity, so exchanging the input
and output weaves exchanges the domain and the codomain exactly.

It is an operator for the reason `Select` and `IndexSelect` are. A reindexing is affine, and
the map from a slot to the position it selected is runtime data. An `Einops` signature
describes a contraction, and an injection leaves $n - k$ entries untouched rather than
summing over them. The third way to write a map between two axes is an operator, and it is
what is left.

**`Inject` with a rank-0 index is the weight gradient of a selecting `Linear`.**
`inject.inject_slab` writes one slab of the weight, the outer product of the data operand
and the cotangent at the degree, at the position the index names along the selected axis
and zero at every other. The weight's node transpose then sums over the degree, and the two
together accumulate every token's slab into the expert bank. `derivative.linear` writes
this for a `Linear` whose operand has a `Natural` datatype, and keeps the index on the
transpose, so the transpose reads the slab the forward `Linear` read.
[[DeepSeek-V3 Backward Pass]] derives a whole layer through it.

**Where the sparse axis ends.** In a derived backward pass the injection is the last root
carrying `k/n` on the gate path. Everything before it runs over the selected experts, and
everything after it runs over all $n$, so a router's transpose contracts against the whole of
$W^{g}$ and $\mathrm{d}W^{g}$ covers the whole of $R[m, n]$. Both are right, because $W^{g}$
scores every expert and the $n - k$ entries that were not selected are the zeroes the
injection wrote. `para/validate_backward.py` derives it, and asserts that
the sparse backward pass is the dense one with one `Inject` added.

**What the identity cost, before `Inject` existed.** The rule was the identity on the
incoming cotangent. An identity typechecks only where its domain and its codomain are the
same object, and a `SparseAxis` carries the extent of the axis it was carved from, so
composing one made `k/n` and `n` one axis with two names. A listing then declared a wire
`R[k/n]` and read it as `[{n}]` on the next line. Worse, two links of the derived backward
pass had a codomain and a domain that disagreed, and nothing reported them: `cat.Composed`
computes its domain from its first entry and its codomain from its last and compares nothing,
and no validator compares them either. `hypergraph_to_morphism` rebuilt the pass regardless,
because a wire is identified by its uid and a `HypergraphObject`'s `.obj` is allowed
to differ from its neighbours', which is the invariant in [[Invariants]] under *Wire identity
in a hypergraph*. The consequence would have been felt by any pass keying an axis by its UID,
meaning the affine reading of a reindexing and the einsum construction of
[[Torch Compile]], which would
have read `n` and `k/n` as two axes of the same size.

`TopK.complete` raised `SelectionHasNoReverse` until 2026-09-04, because it emits the index
beside the values and a `Natural` datatype has no cotangent, so the rule had one fewer input
than `Inject` needs. Declaring the index output as the residual is what supplies it.

## Where a selector's learning signal comes from instead

If the index has no dual, nothing reaches the thing that produced it, so a selector has to
be trained off the value path. Both live examples are in the notebooks under
`notebooks/sota/`, per [[SOTA Model Notebooks]].

- **A gate multiplier.** An MoE router's scores multiply the expert outputs after the
  selection, so a gradient reaches the router through $g$ even though none reaches through
  $S$. [[Training]] works through the consequences: 6 rows of 384, a softmax Jacobian over
  the chosen set, and a bandit reward with no counterfactual.
- **A separate detached objective.** DeepSeek's lightning indexer gets no gradient from the
  language-modelling loss. It is trained by its own KL loss against the main attention
  distribution, with its input detached from the backbone so that the two objectives cannot
  fight. DSA introduces it at arXiv 2512.02556, and V4 keeps it, at arXiv 2606.19348 section
  4.2.2, after a dense warmup and an indexer-only warmup stage.

In the vocabulary of [[Training]], the first is an ordinary reverse-written parameter that
happens to sit off the index path, and the second is a second loss whose reverse pass is cut
with a stop-gradient, meaning a forward `Drop` with no matching `Grab`.

## The rules

- **Both spellings reverse to `Inject`, and the index comes from the slot.** The compressed
  rule derives the slot from the sparse axis, and the complete rule declares the index
  output as the residual and lets `dedup_slots` find the slot the model wrote.
- **A reverse derivative has the shape its signature asks for.** $R[f]$ goes from
  $T^{*}(\mathrm{cod}\ f)$ to $T^{*}(\mathrm{dom}\ f)$, so the reverse of an
  $n \to k/n$ map is a $k/n \to n$ map and nothing else. Writing an identity where the two
  objects merely have the same extent merges two axes that are not the same axis.
- **An index wire is forward-only.** It is saved, loaded, and never differentiated. A
  `Natural` wire in a reverse graph means something has gone wrong.
  `para/algebra/para_sparse_expansion.py` is what saves it. It applies the same three rules
  `expand_sparse` applies and routes nothing: the `TopK` drops the index to a slot and each
  parametric operation grabs it back, so the index enters no signature and no block's domain
  moves. The pieces are independent, connected by the slot's name and by nothing else.
  `para/algebra/tie_tapes.py` connects them again, by giving every grab and drop of one slot
  the same node and putting that node on a scope's domain or codomain wherever only one half
  of the slot is inside it. Tying the taped expansion reaches the expression `expand_sparse`
  reaches by routing, line for line.
  `deepseek/validate_sparse.py` derives both and asserts it.
  `para/validate_backward.py` runs one `ExpandSparseOntoTape` over a
  derived forward and backward pair, so both name one slot. The forward pass drops the index
  and grabs it at the experts, and the backward pass grabs it at the `Inject` and at
  `Transpose<E>`. `reads_a_weight` is what admits the transpose: a `Transpose` is built from
  the weight its forward `Linear` read, per `para/data_structure/transpose.py`, so a `Linear`
  that selects one of `n` weights reverses to a `Transpose` reading the same slab and the
  index says which slab.

**The order of the passes decides the weight's shape.** Expanding the model before
`grab_parameters` gives $W^{E} : R[n, m, f]$, the whole expert bank, because `weight_axes`
sees a `Linear` whose first input is a `Natural(n)`. Expanding after it gives $R[m, k, f]$,
because the parameter was built while the sparse axis was still in the operator's target.
Expanding first is the order that writes the gradient onto the slabs the index named, and
it is the order a whole layer is derived in.
- **A gather's dual accumulates.** It is never `=`, because a gather may name a source more
  than once, and in a distributed implementation those writes come from different GPUs.
  The ordering is therefore a question about determinism, which DeepSeek-V4 had to answer
  explicitly in its released backward pass, at section 3.3.
- **Do not reach for a relaxation.** Gumbel-softmax, straight-through and soft top-k all
  change the model. The models under `notebooks/sota/` do not use them, and the algebra does
  not need them, because the index is a residual rather than a value to be smoothed. The one
  place a relaxation is used is quantisation, where V4's FP4 quantization-aware training is a
  straight-through estimator. The estimator answers a different problem, and it lives on the
  value path.

## Gaps

- ~~There is no scatter operator~~. Closed 2026-09-04 for `TopK.complete` and for a
  selecting `Linear`, as `Inject` followed by the node transpose's sum. `IndexSelect` still
  has no rule, and its rule is the same pair at the payload's degree.
- **Nothing checks that a composition composes.** `cat.Composed` does not compare its
  neighbours' codomain and domain, and no validator does either, so the two incompatible
  links `Inject` removed were found by walking the morphism by hand. A checker is a small
  piece of work and it would have caught them at the moment the identity was written.
- **`Inject` has no PyTorch module**, so a derived pass holding one is checked
  structurally rather than numerically.
- ~~A selecting `Linear`'s weight gradient is still a contraction~~. Closed 2026-09-04.
  `derivative.linear` tells a selecting `Linear` from a contracting one by the datatype of
  its operands, keeps the index on the transpose and writes $\mathrm{d}W$ through
  `inject_slab`.
- **Nothing checks the two rules numerically.** `torch_compile` has no module for a complete
  `TopK`, for `Inject`, or for a `Linear` with an index operand, so `validate_backward.py`
  cannot compile the derived pair.
- **$R$ itself does not exist**, per [[Training]] under *Gaps*.

## See also

- [[Training]] — the reverse functor, the tape, and the four writers of a parameter
- [[Derivatives]] — how a derivative is built, and where `Scatter` is needed
- [[Sparse Axes]] — the two presentations, and why `Select` and `IndexSelect` exist
- [[Sparse Expansion]] — the rewrite between them
- [[Operators]] — where a scatter would go
- [[Open Gaps]] — item 1, *Data-dependent reindexing with static structure*
- [[DeepSeek-V3 Backward Pass]] — a whole layer derived through the taped expansion
