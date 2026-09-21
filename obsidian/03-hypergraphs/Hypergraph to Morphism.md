---
tags: [layer/hypergraphs, algorithm]
code: graphs/processing/Hypergraph2Morphism.py
status: stable
---

# Hypergraph to Morphism

## What it is

The hard direction. Going from a morphism to a hypergraph is a walk that discards structure.
Coming back has to invent a nesting of `Composed`, `ProductOfMorphisms` and `Rearrangement`
that reproduces the wiring, which means choosing an order for what the graph left unordered,
and inserting a rearrangement wherever the graph copies, deletes or permutes a wire.

```python
m = h2m.hypergraph_to_morphism(g)
m = h2m.recycle(m)     # normalise a hand-built morphism by round-tripping it
```

`recycle` is how a morphism assembled by hand is put into the canonical shape the rest of
the package expects.

`recycle` is defined on any product category, so it descends through a `cat.Block` and
reads a seed morphism as one node. An `ops.BlockOperator` is a seed morphism of **Br**
that carries a whole expression in its `block`, so `recycle` normalises the level it was
called at and leaves every box as it was written.
`algebra.broadcasted_recycle.BroadcastedRecycle` is the recycling of **Br**, which
recycles that expression too, and the body of every box inside it.

```python
import algebra.broadcasted_recycle as broadcasted_recycle
normalised = broadcasted_recycle.recycle_broadcasted(model)
```

Its descent is a walk over the term rather than over the category. A
`para.data_structure.ParaWrap.ParaWrap` is a morphism that
`graphs.processing.hypergraph_functor` has no case for, so a category functor reads one
as a leaf, and the attention modes of DeepSeek-V4.1-Flash put the attention core inside
exactly that. A term walk also rebuilds from the leaves up, so a box's body already holds
its own recycled boxes by the time the box is reached. The body is recycled and the block
rebuilt around it, because recycling a `cat.Block` whole lifts a wire that passes straight
through it out of the block, which would change the domain and codomain the box's weaves
were built from. [[Diagram Display]] draws through it under
`BlockRecycling.RECYCLED`.

## How it works

The algorithm is a recursive branch construction. Each `Branch` carries the morphism it
denotes, the nodes on its left and right, and which subgraphs it has consumed.

| branch | what it covers |
|---|---|
| `RootBranch` | one leaf morphism |
| `IdentityBranch` | a wire that passes straight through |
| `NestedBranch` | a block |
| `ProductBranch` | independent subgraphs side by side |
| `ComposedBranch` | one branch feeding another |

`graph_to_branch` picks which applies. `make_subbranch`, `rolled_subbranches` and
`make_stacks` do the assembly. `make_rearrangement` builds the permutation that reconciles
the node order a branch produces with the order the next branch requires. `expand_composed`
flattens nested compositions so that the result is no deeper than it needs to be.

`hoist_rearrangements` then normalises the result, added 2026-08-20. The branch construction
makes each copy at the last possible column, so a wire consumed at one point and held for
later is duplicated where the hold begins, which drew the expanded MoE's router input being
copied at the router. Hoisting pulls the leading rearrangement of every product child out of
the product, fuses it into the rearrangement before it, and fuses adjacent rearrangements, so
that cascaded fans merge into one at the earliest point, which is the model's own `(0,0,0)`.
It is the Cartesian comonoid law $\Delta;(f\times f)=f;\Delta$ run as a normalization, per
[[Yoneda and Cartesian Tricks]]. It is neutral on the wiring by construction, and the
Every validator passes with it in place.

## The rules

> [!warning] `HypergraphObject(...) is not in list` means a wiring bug upstream
> The conversion is where a wiring error surfaces rather than where it is. Some node is
> required on the right of a branch and nothing on the left produces it. Check the splice
> that last touched the region, and check whether a container's `dom` or `cod` needed
> rebuilding, through `leaf_splicing.rescope`, per [[Leaf Splicing]]. One systematic cause that was not a bug is fixed,
> in the session recorded as, and the next warning
> describes it. If the error reappears on a valid morphism, suspect that family first.

> [!warning] Pass-through wires and dropped wires used to break the branch walk
> A wire a block passes through untouched keeps its node, so the node sits in that graph's
> `dom` and in its `cod`. The graph becomes its own right-neighbour, and a chain of such
> blocks, meaning a passenger riding several `Residual` blocks, becomes a false cycle. A wire
> a `Rearrangement` drops leaves a dangling `cod` node that no frontier ever requires. In
> either case `exclusive` never emitted the graph, and the reconstruction either degenerated
> silently into identity wires, drawing a loop block as an empty box, or died in
> `make_rearrangement`. It was fixed 2026-08-15. Sequencing now waits only on a graph's
> `boundary_cod`, meaning the `cod` nodes that are neither passed through nor dangling, with
> `gating_right` derived from those, and `ComposedBranch._right_override` projecting the
> tolerated dangles away at the end. A follow-up in the same session deduplicated the
> `IdentityBranch` fallback: a frontier node already delivered by the pass-through of an
> emitted branch must not also be held by an identity, or the copy rides the whole
> composition unused before being deleted, which draws full-length stub wires. The
> Every validator passes across both fixes.

> [!warning] Stacks used to be aligned to what a stack requested rather than to what the previous one emits
> `ComposedBranch.morphism` aligned each stack to `unique_tuple` of its own left nodes,
> assuming the previous stack emits exactly that order. `rolled_subbranches` may deliver a
> requested node later than its requested position, because a multi-output subgraph only
> becomes exclusive at the roll position of its last output, and the identity hold for its
> earlier outputs is rightly skipped. The actual column order can therefore deviate, and the
> missing `Rearrangement` wired the downstream by position rather than by node, which swapped
> two same-rank wires silently. It surfaced 2026-08-20 through the two-output `TopK` blocks of
> [[Sparse Expansion]], whose values and indices go to different consumers. The fix builds
> each inter-stack rearrangement from the previous stack's actual `right_nodes()`. Where the
> old assumption held, the emitted rearrangement is identical, so the fix is
> conservative.

- The conversion is not an inverse in the strict sense. `from_morphism` followed by
  `hypergraph_to_morphism` gives a morphism with the same semantics and possibly a different
  association. `recycle` exists for that: it normalises rather than preserving.
- `ReverseCrawler.crawl` mints a fresh `HypergraphRoot`. Whether the original UID should carry
  over is an open question, also recorded in `PublicCodeTODOs.md`.

**A subgraph with an empty codomain waits for a column that consumes its domain.**
`exclusive` enumerates the graphs to the left of a frontier, which are the producers, so a
sink such as `Para.Drop`, which saves a value and returns no wire, is never a candidate for
it. `boundary_cod` still counts the sink as a consumer that has to be placed before its
producer may be. Left alone, the pair deadlocks: the producer waits for the sink, the wire is
held, the stack reports no progress, `make_stacks` stops, and the wire ends up with nothing on
its left to make it.

`make_root_branch` therefore builds every sink into a branch before the main walk starts, by
calling `make_subbranch` on it. The branch holds the sink and the chain of producers that feed
the sink alone, so for a gradient it holds the outer product and the `Drop` together, and its
domain is the wires that chain shares with the rest of the graph. `SinkPlacement.register`
then files the branch under each node of that domain. The branch is not added to any stack at
that point. It waits until a column is built in which some branch consumes one of those
nodes, and `inject_waiting_sinks` then inserts it into that column, after the consumer. Every
column passes through the injection: the columns `rolled_subbranches` builds, and the
one-branch column `make_subbranch` makes of its origin. The rearrangement in front of the
column delivers the wire once and the column copies it, so nothing is held across the
picture for the tape's sake.

The walk builds columns from right to left, so the column that claims a sink is the rightmost
one consuming a wire of its domain. In the feed-forward backward pass the $dW_2$ chain reads
the incoming cotangent, which `Transpose<2>` also reads, so the chain is drawn beside
`Transpose<2>`, and the $dW_1$ chain reads $dh$ beside `Transpose<1>`. An injected branch's
`newly_processed` joins the column's, so the producer of a wire the sink shares with the live
graph becomes `exclusive` one column to the left of both readers.

`local` in a scope holds that scope's origin, `LEFT`, and what the origin's own injection
placed. A sink injected in another scope is not local, so a producer gated on it is held and
floats up to the scope in which the sink is local. The earlier version marked every sink
`local` in every scope before anything was placed, and that ordering is what the waiting
replaces.

Two kinds of sink never see a column consuming their domain, and `make_root_branch` places
them at an edge. A sink whose domain is read by nothing outside the sink branches is placed
in a final column beside the outputs, and its domain joins the initial frontier so the walk
produces it. A sink whose domain is entirely inputs of the graph, and read by nothing else,
is placed in a leading column beside the inputs, and so is a sink whose chain reads no wire
at all, which is a constant dropped onto a tape slot. The initial value of a loop's carried
value is written that way, and the leading column puts it before the loop that reads the
slot. Until 2026-09-12 a chain reading no wire went to
the final column, and the initial values were drawn after the loop. `edge_column` builds both, as identities
over the carried wires with the sinks beside the identities of the wires they read, and
drops an identity whose wire the sink was the last reader of. A sink that is neither claimed
nor at an edge raises a `ValueError`, because the walk emits every graph reachable from the
codomain and a consumer of the sink's domain is therefore always built. On a graph with no
sink none of this fires.

A source, meaning a `Para.Grab`, needs none of it, because `exclusive` already emits one as
late as possible, one stack to the left of its last-scheduled consumer. A diagram therefore
shows each save beside the first reader of the value and each load at its use. An earlier
version laid the whole tape out along the edge of the picture, with the drops closing, the
grabs leading and the Para types drawn, and attention showed the cost: every taped wire was
held across the full width of the diagram.

> [!warning] A sink is placed once per conversion
> `SinkPlacement.claimed` is one set per conversion, threaded through the branch
> construction, and a sink waiting on two nodes is injected at the first of them to be
> consumed. `ignore` cannot serve, because it restarts at every `make_subbranch`. A wire that
> is a `dom` node of two scheduling scopes once had its `Drop` emitted once per scope. The
> node absorption of [[Expression Simplification]] creates exactly that shape: once the scale
> einsum of the expanded softmax reads the exp output directly, that wire is rolled by the
> scale's scope and by the sum's, and the tape slot was saved twice. It was found 2026-08-20
> by reading the diagram.

## See also

- [[Hypergraphs]] — the form being converted
- [[Hypergraph Analysis]] — what the branch construction consults
- [[Agent Display]] — reads either form, so a conversion is rarely needed just to look
