---
tags: [layer/categories, algorithm]
code: deepseek/sparse_expansion.py, deepseek/validate_sparse.py
status: stable
---

# Sparse Expansion

## What it is

The rewrite between the two presentations of a selection that [[Sparse Axes]] describes. A
model written compressed, with one wire on a `SparseAxis`, is unpacked into the complete
form, with the values beside a `Natural(n)` index wire, which is the pair `torch.topk`
returns, by manipulating the graph.

```python
import deepseek.sparse_expansion as dse
expanded = dse.expand_sparse(morphism_or_hypergraph)
```

This pass reaches the tied form of [[Sparse Axes]] directly. The default expansion of a
selection is `expand_sparse_onto_tape`, in `para/algebra/para_sparse_expansion.py`, which
produces the full form in Para and is what a model already holding a `Grab` or a `Drop`
expands with. This pass has no case for a tape operation, and it exists to state that the
expansion depends on nothing in `para`. The two agree line for line, which
`deepseek/validate_sparse.py` asserts, and the four seed rules below are read by both.

A `SparseAxis` is a compression of two wires:

$$[\mathbb{R},\, k/n] \;\cong\; [\mathbb{R},\, k] \;\otimes\; [\mathrm{Nat}(n),\, k]$$

They are the surviving values on a genuinely $k$-sized axis, and `Natural(n)` indices on that
axis stating which of the $n$ each value came from. The compressed form is the default in
every model notebook, because it is one wire rather than two through every consumer. The
expanded form has a two-output `TopK`, with the index wire entering each expert `Linear`
as an operand.

The two forms were separate constructors until 2026-08-20, being `TopK.template` beside
`TopK.complete` and `Select.template` beside `Select.complete`, so a model had to be written
down once per presentation. The session of 2026-08-18 recorded the missing piece: expansion
should be a graph rewrite, so that a model is written compressed and unpacked on demand.
The rewrite below is that one.

It is not a local splice. Every consumer of the sparse axis changes shape, and the new index
wire has to travel from the `TopK` to consumers that are not its neighbours in the graph,
across blocks. The rewrite therefore works on the hypergraph, where a wire is an
`HypergraphObject`, a new wire is a new uid, and routing it is a matter of putting the same wire
in the right `dom`s and `cod`s.

## The rules

With $S$ the sparse axis, printed `k/n`, $k$ its fresh dense replacement and $n$ its parent
extent:

| compressed | expanded |
|---|---|
| `TopK : [R; e] -> [R; S]` | `TopK : [R; e] -> [R; k], [Nat(e); k]`, where the second output is the index segment, stating which $k$ of the $n$ were selected. The operator's `form` moves from `WEIGHTS` to `WEIGHTS_SELECT`, per `ds.SelectionForm`, and a `TopK` already in a dense form is left alone |
| `Linear W : X -> [Y; S]` | `W : [Nat(e); k], X -> [Y; k]`. $S$ leaves the weight's target and becomes a broadcast axis in the degree, and the index enters as the first operand, as V3's `W_in` takes it, at a rank-0 target read at every $(…, k)$, so the implicit weight tensor is indexed by the chosen expert |
| `Select : [R; S], [R; e] -> [R; S]` | `hold[R; k] × IndexSelect([Nat(e)], [R; e] -> [R]) ; einops('k, k -> k')`, where the held selection values are multiplied back onto the gathered payload. The `IndexSelect` is elementwise in the index: its degree is the whole output array, its targets are the rank-0 `Nat(e)` index and the payload's parent axis, and its index reindexing is the identity prefix, which is the same posture as the expanded Linear's operand |
| `aops.CovariantView : [R; p/P, u] -> [R; c/B]`, the merge of a selection | `aops.CovariantView : [R; p, u] -> [R; c]` for the values, beside `MergedPositions : [Nat(P); p] -> [Nat(B); p, u]` followed by the same `CovariantView` on the positions, `[Nat(B); p, u] -> [Nat(B); c]`, which is the index segment of the merged axis. `MergedPositions` is the split applied to a block number and every offset. The merge consumes the index of the split axis and produces the index of the merged one, per [[Sparse Axes]] |
| `TopK : [R; c/B] -> [R; s/B]`, a selection over a selection | `TopK : [R; c] -> [R; s], [Nat(c); s]` followed by `IndexSelect([Nat(c); s], [Nat(B); c] -> [Nat(B); s])`, which reads the positions of the inner axis at the chosen slots, so the outer index is a segment of the parent. The selection consumes the inner index and produces the outer one |
| `TopK : [R; C], [Nat(B); C] -> [R; s/B]`, a selection over positions | `TopK : [R; C] -> [R; s], [Nat(C); s]` followed by `IndexSelect([Nat(C); s], [Nat(B); C] -> [Nat(B); s])`, which reads the positions operand at the chosen slots. It is the rule above with the inner positions on a wire of the model rather than on the index wire of a sparse axis, per `ds.TopK.template(positions_of=)`, and the operand leaves the expanded selection, which selects over `C` alone |
| anything else touching $S$ | $S$ is substituted by $k$, and the operation is otherwise untouched. A reindexing keeps its mapping and its strides, with only the axis object swapped |

`IndexSelect`, in `deepseek/data_structure.py`, is the pure gather, computing
`out = payload[index]` at every point of the degree. It is the third of the trio: `TopK`
selects, `Select` reads the compressed form, and `IndexSelect` reads the expanded one. It
exists because after expansion the selection is data on a wire, so the reader has to take it
as an operand. It is named after `torch.index_select`. It was briefly called `Grab`, and was
renamed the same day because [[Para Category]] claims that word for its read of a parameter
slot. A log from 2026-08-20 that writes `Grab` in the context of a gather means `IndexSelect`.

## What the rules produce, on the V3 MoE

Compressed, from cell 5 of the notebook, recycled:

```
%2 = Linear<L^g>(%0[{m}]) : R[{n}]
%3 = TopK(%2[{n}]) : R[{k/n}]
%4 = SoftMax(%3[{k/n}]) : R[{k/n}]
%5 = Linear<W0>(%0[{m}]) : R[{k/n}, {f}]
...
%9 = Linear<W2>(%8[k/n, {f}]) : R[k/n, {k/n}, {m}]
%10 = View(%9[k/n, k/n, m]) : R[k/n, m]          <- the diagonal
%1 = Einops(%4[{k/n}], %10[{k/n}, m]) : R[m]
```

Expanded by the rewrite:

```
%2 = Linear<L^g>(%0[{m}]) : R[{n}]
%3, %4 = TopK(%2[{n}]) : R[{k}], Nat[{k}]
%5 = SoftMax(%3[{k}]) : R[{k}]
%6 = Linear<W0>(%0[{m}], %4[k]) : R[k, {f}]
...
%10 = Linear<W2>(%9[k, {f}], %4[k]) : R[k, {m}]
%1 = Einops(%5[{k}], %10[{k}, m]) : R[m]
```

Shape for shape, that is the complete form the V3 notebook hand-builds. The router consumes
a dense `n` that never reappears, the experts are broadcast over a real six-wide `k`, and
the index `%4` is the one wire all three projections share. The diagonal is gone, for the
reason below.

## How it is done

Three phases run over the `Multigraph`, in `_Expansion`.

**Survey.** Every `HypergraphRoot` is classified. A compressed `TopK`, meaning one with the
sparse axis in its output target, is a producer. A `Linear` with the sparse axis in an output
target, or a `Select` whose first operand rides the sparse axis, is a consumer. A `Select`
at positions, per `Select.at_positions`, holds no sparse axis and is generic. An `aops.CovariantView` with a sparse axis on both sides,
and a `TopK` whose input axis is sparse, are chained: each consumes one sparse axis and
produces another, and a chain whose first axis has no producer in scope is downgraded to
generic along with everything it produces. A `BlockOperator` whose body holds a sparse
axis is a nested scope. Everything else is generic.

Each produced $S$ gets a `SparseWire`, holding the fresh `k` axis, one wire for the
index wire, and, once the `TopK` is read, the index array as it will be emitted, being the
`TopK`'s broadcast degree plus $k$ at datatype `Natural(n)`. A consumer whose $S$ has no
producer in scope stays compressed, because there is nothing to expand it with, and a
fragment cut below its `TopK` should survive the pass unchanged.

The substitution from $S$ to $k$ is one `fd.Context` holding an `EqualityClass` per selection
with the `k` axis as canonical. Applying it to a term rewrites the axis everywhere it occurs,
in weave shapes, in `Rearrangement` domains and in `StrideMorphism`s, while leaving every
mapping and stride untouched. Leaving them untouched is what keeping the reindexings intact means
mechanically: the affine structure is never rebuilt, and only the axis object inside it is
swapped.

**Analyze.** A bottom-up pass records, per subtree, which selections are produced inside it
and how many consumers sit inside it. Each decision about a `dom` or a `cod` is then local
arithmetic. The index wire enters a graph's `dom` exactly when the subtree holds a consumer
and not the producer. It exits a graph's `cod` exactly when the subtree holds the producer
and some consumer sits outside.

**Rebuild.** The reconstruction is top-down. A root is rewritten per its class. The `TopK`
root gains the index node in its `ucod`. A consumer `Linear` gains it in its `udom`. A
`Select` root is replaced by the `hold × IndexSelect ; einops` composite, built as a morphism
and converted with `from_morphism` onto the original `dom` nodes plus the index node, with
its fresh `cod` node aliased to the old one.

An auxiliary graph appends the index `HypergraphObject` to its `dom` or `cod` per the
analysis. `HypergraphBlock.template` then recomputes each block's `dom` and `cod` from its
rebuilt body, which is how the `TopK`'s segment crosses a block. In the V4 MoE, `Experts`
comes back as

```
Block[Experts]: dom = R[x,m], Nat[x,k]     cod = R[x,k,m]
```

with the index wire connecting to the router's `TopK` outside purely by node identity. An
index wire that would cross into a block with `repetition > 1` raises, because a loop cannot
import a selection made per iteration from outside.

A `BlockOperator` is a nested scope rather than a crossing. Its body is expanded by a
recursive call to `expand_sparse`, and the box's own weaves are asserted to be free of sparse
axes, so a selection has to be self-contained inside the box. The recursion covers the whole
`v4_flash` model, where `MoE` and `CSA` are boxed with `state -> state` domains and
codomains.

### The diagonal, and why it disappears

The compressed idiom for a per-expert down-projection is to produce and diagonalise. `W^D` at
degree $(x, S)$ produces the target $(S, m)$, which is a block indexed by slot and by expert,
and a copy-`Rearrangement` $(x, S, m) \to (x, S, S, m)$ keeps the entries where the produced
expert and the slot's expert agree. Two facts fall out of the `Linear` rule with no special
case.

1. The produced $S$-target of `W^D` drops, because its $S$ is already in the degree, so the
   clause turning it into a broadcast axis is satisfied by the slot axis it was being
   diagonalised against. The output is $(x, k, m)$.
2. The diagonal's reindexing then copies a $k$ its producer no longer duplicates.
   `_collapse_copies` drops the repeated references to an axis the expansion minted from the
   input reindexings, taking $(0,1,1,2)$ to $(0,1,2)$, at which point the node is a pure
   identity and is spliced out through `UIDRenaming.set_canonical` on its two wires.

The diagonal was the compressed shadow of V3's shared index wire, and the expansion replaces
the shadow with the wire.

### Sizes and names

The `k` axis takes the activity numeric as its size, which is the same symbol the `TopK`
operator's `k` field carries and the same name the notebook's `NumericConfig` assigns, at
`k=6` and `s=512`. A configuration written against the compressed model therefore still
reaches every size in the expanded one, which `check_config_names` in the validator checks.

## Rules of use

- **One compressed `TopK` per `SparseAxis` per scope**, which is asserted. Consumers bind to
  a selection by the axis they carry rather than by a wire, so two `TopK`s minting the same
  `SparseAxis` in one expression would make the index wire ambiguous. Distinct selections
  should be distinct axes, which `TopK.template` guarantees unless an axis is deliberately
  reused.
- **A sparse axis with no `TopK` in scope stays compressed.** A fragment cut below its
  selection survives the pass unchanged.
- **A `Linear` consuming the sparse axis in an input target is not expanded.** That is the
  folded form `(k/e, f) -> (m)`, which hides the combine inside the weight, and the
  substitution would silently turn one column per expert into one column per slot. The
  produce-and-diagonalise idiom the notebooks settled on never reaches the case. Today such a
  `Linear` is substituted like any generic consumer, so failing loudly is the behaviour to
  add, and the notebooks' conventions are what keep it from arising.
- **A sparse axis on a `BlockOperator`'s domain or codomain raises.** The box's weaves would
  have to grow an output, which is a presentation the caller should choose: use a `cat.Block`,
  which the wire can cross, or expand before boxing. The taped pass needs neither, because
  the tape crosses the box without a wire.
- **The `k` axis takes the activity symbol as its size**, so a `NumericConfig` written against
  the compressed model still binds every size.
- **The selection axis is a target only where it means something**, which is at the `TopK`
  that mints it and at the contractions that consume it. Everywhere else, meaning the index
  operands, the `IndexSelect` and the gathered payload, it is broadcast. An index wire a block
  imports is prepended to its `dom`, so `Experts` reads `dom = (Nat[x,k], R[x,m])` and the
  drawn selection rides on top.

## The bug this surfaced in `hypergraph_to_morphism`

The first expanded MoE came back with the token wire and the gate wire swapped, so `W_0` read
the softmax output as its `{m}` input. The graph was right and the conversion back to a
morphism was wrong.

`ComposedBranch.morphism` aligned each stack to `unique_tuple` of its own left nodes,
assuming the previous stack emits nodes in exactly that order. `rolled_subbranches` may
deliver a requested node later than its requested position, because a multi-output subgraph
only becomes exclusive at the roll position of its last output, and the hold identity for its
earlier outputs is rightly skipped. A `Gate` block emitting values and indices consumed by two
different downstream operations is exactly that shape, so the actual column order deviated
from the assumed one and the missing `Rearrangement` wired by position.

The fix, in `graphs/processing/Hypergraph2Morphism.py`, builds each inter-stack rearrangement
from the previous stack's actual `right_nodes()`. Where the old assumption held, the emitted
rearrangement is identical, and usually absent, so the fix is conservative.

### Shaping the drawn form

Three refinements came from reading the drawn expansion, on 2026-08-20. All three shape the
display and none changes the semantics.

- **The index operand is first** on an expanded `Linear`, and an index wire a block imports is
  prepended to its `dom`, so `Experts` reads `dom = (Nat[x,k], R[x,m])` with the selection on
  top.
- **`IndexSelect` is elementwise in the index.** The selection axis is broadcast everywhere
  below the `TopK`, and the only places `k` is a target are the `TopK` that mints it and the
  contractions that consume it, being attention over `s` and the combine over `k`.
- **Rearrangements are hoisted.** The first drawn MoE showed the router's input being copied
  at the router, because `hypergraph_to_morphism` makes each copy at the last possible column.
  `hoist_rearrangements`, in `graphs/processing/Hypergraph2Morphism.py` and now part of every
  reconstruction, pulls the leading rearrangement of each product child out, fuses it forward,
  and fuses adjacent rearrangements, so cascaded fans merge into one at the earliest point and
  the MoE opens with the model's own three-way `(0,0,0)` fan. It is the Cartesian comonoid law
  $\Delta;(f\times f)=f;\Delta$ run as a normalization, per
  [[Yoneda and Cartesian Tricks]].

## Where it is exercised

- `python deepseek/validate_sparse.py` makes structural assertions rather than text
  comparisons. It checks for a two-output `TopK` with `Natural` indices, all three expert
  `Linear`s carrying the rank-0 index operand broadcast over `k`, and `k` out of every weight
  target. It checks that no `View`, and so no diagonal, remains, and that the index wire is on
  the `dom` and `cod` of both the `Gate` block and the perceptron block. It checks that
  `Select` expanded to an `IndexSelect` with an elementwise product, with the payload's parent
  axis still dense, and that no `SparseAxis` remains anywhere in either result. It checks that
  the domains and codomains are preserved and that the configuration names are intact.
  `check_merged_selection` takes a merged selection with a selection over it through both
  passes, and `check_positions_pool` takes a pool written as positions, with a selection
  over the candidates that takes them as an operand, through both.
- **The index on the tape.** `para/algebra/para_sparse_expansion.py` applies the same three
  rules and puts the index on a slot rather than on a wire. It reads the seed rules from this
  module, meaning `classify`, `expand_topk`, `expand_linear` and `collapse_copies`, so the
  routing is the only thing that differs between the two passes. Taping costs no routing at
  all, so the pass is one functor where this one needs three phases, and it leaves an
  expression whose pieces are connected by a slot's name. `para/algebra/tie_tapes.py` then
  reaches this module's own output from that one, line for line.
  [[Selection and the Reverse Pass]] states why the index needs a slot.
- **A GeGLU mixture of experts.** One layer, meaning a sigmoid router, a `TopK`, three
  expert projections and the gate-weighted combine, is the smallest expression in which all
  four rules fire at once. The index enters `W^G`, `W^U` and `W^D`, it crosses the `Experts`
  block's domain to reach them, `W^D`'s degree collapse splices the diagonal out, and the
  domain and codomain of the layer are unchanged. `deepseek/validate_sparse.py` states those
  facts, including that no `SparseAxis` survives and that exactly one `Natural` datatype
  appears.
- **DeepSeek-V4-Flash.** `moe()` expands with the index entering `W^G`, `W^U` and `W^D`, and
  `Experts` importing `Nat[x,k]`. `csa_attention()` expands with
  `IndexSelect(%idx[x,{s}], %cache[{b}, c])` and the cache's `b` dense. The whole `v4_flash`,
  with `MoE` and `CSA` boxed and repeated in the 20-layer group, expands through the
  `BlockOperator` recursion with its `dom` and `cod` unchanged. `IndexSelect` is mirrored
  in tsncd, registered in
  `src/deepseek/data_structure.ts` and drawn by `display_deepseek.ts` as an einops cup
  between the index and the payload, and the index wire is the bold `Natural`-typed second
  output of the `TopK` diamond. A compressed `Select` draws as the green dot a copy of an
  axis draws as, per [[Sparse Axes]].
- `python deepseek/validate_sparse.py` states the shape of each expanded form.

## What it opens

The expanded form makes the MoE dispatch and combine an index wire with a six-wide
contraction, rather than a 256-wide sparse axis. The compressed form gives the combine a
`k/e` contraction, and the expanded one gives it a `Nat`-typed operand. [[Open Gaps]]
records the data-dependent reindexing that the expanded form leaves.

## See also

- [[Sparse Axes]] — the compressed and complete forms this rewrites between
- [[Operators]] — `TopK` and `Select`, which `IndexSelect` joins
- [[Hypergraphs]] and [[Hypergraph to Morphism]] — the form it rewrites in
- [[Selection and the Reverse Pass]] — why this rewrite is also a precondition for differentiating
