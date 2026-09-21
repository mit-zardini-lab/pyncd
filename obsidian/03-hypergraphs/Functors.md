---
tags: [layer/hypergraphs, concept]
code: graphs/processing/hypergraph_functor.py, graphs/processing/replace_roots.py, graphs/processing/rewire_blocks.py, graphs/processing/merge_duplicate_roots.py
status: stable
---

# Functors

## What it is

A **functor** transforms a whole expression uniformly: it maps objects to objects and
morphisms to morphisms, preserving composition. Where [[Rewriting]] identifies particular
terms, a functor applies one *rule* everywhere.

`graphs/processing/hypergraph_functor.py` gives the general machinery:

| name | what it is |
|---|---|
| `Functor[L1, M1, L2, M2]` | object map plus morphism map, walked over a morphism or a hypergraph |
| `Endofunctor[L, M]` | source and target the same category |
| `reduce_nodes(context, graph)` | apply a `Context` to a graph, deduplicating the dom and cod by wire |

Subclass it and override the object and morphism handlers. The base provides the traversal.

`graphs/processing/replace_roots.py` is the keyed edit a functor is not. `walk_roots` yields
every root of a graph through its blocks, and `replace_roots` rebuilds the graph with given
roots replaced or removed, keeping every wire. `show_grabbed_parameters.collapse_grabbed_residuals`
and `pathway_collapse.dedup_slots` use the pair, because a grab inside a block is a grab.

## Merging duplicate roots

`graphs/processing/merge_duplicate_roots.py` is the common subexpression elimination of the
graph form. Two roots are duplicates when they wrap equal morphisms and read the same wires.
A morphism carries no uid of its own, so equality compares the operator, the weaves and the
reindexings, with the axes compared by identity, and a wire is a `HypergraphObject` compared
by its uid. Two duplicates therefore compute one value.

`merge_duplicate_roots(graph)` keeps the first duplicate in walk order and renames the wires
the others produced onto its wires with `fd.UIDRenaming`. The kept root is placed in the
innermost `Multigraph` whose enclosing blocks are a prefix of every duplicate's, which is the
lowest common ancestor of the readers. It is inserted after the last child of that scope
producing a wire it reads. Every block's domain and codomain are then recomputed from the
roots inside it, so the kept root's wire enters each block that reads it, and a block whose
every root was merged away is removed. The outermost graph keeps its domain and codomain,
because they are the morphism's.

A loop block, meaning a `BlockTag` whose `repetition` is not 1, is a boundary. The grouping
key includes the tuple of loop blocks enclosing a root, so a root inside a loop is never
merged with a root outside it, and two roots inside the same loop merge within its body. A
root inside a loop computes once per iteration, and a value outside the loop is one value,
so the two are not the same computation even when their morphisms and wires agree.

Merging two roots makes their readers read one wire, so two readers wrapping equal morphisms
become duplicates in turn. The merge is repeated until a round finds none, and every round
removes at least one root. `pathway_collapse.dedup_roots` applies it to both passes of a
`Taped`, after `dedup_slots`, because two grabs of one residual compare equal only once they
read one slot.

The pass assumes a root with an empty domain is deterministic. A `Grab` reads a slot and a
`Broadcasted` with no operand is a constant. No operator in the package samples, and a
sampler would have to be excluded from the grouping when one is added.

## Rewiring blocks after a rewrite

`graphs/processing/rewire_blocks.py` is the machinery the merge and every pass of
[[Pathway Collapse|pathway collapse]] rebuild through. `walk_roots_with_blocks` yields every
root with the blocks enclosing it. `insert_roots` appends roots to the scope a chain of block
uids names. `recompute_block_boundaries` rederives every block's domain and codomain from
where each wire is produced and where it is consumed. A wire produced at one block chain and
consumed at another flows out through the codomain of every block below their common prefix
on the producer's side and in through the domain of every block below it on the consumer's
side. A wire keeps its position in a boundary it already had, a wire new to a boundary is
appended in walk order, and a block emptied by a rewrite is removed.
`replace_roots.replace_roots` accepts a tuple of subgraphs as a replacement and splices them
in place of the root.

A rewrite therefore removes, replaces and adds roots in whatever scope it chooses and calls
`recompute_block_boundaries` once. One placement is ruled out. A root placed outside a block
that reads a wire the block produces and feeds a wire the block reads makes the block and the
root read each other, and no order of the scope carries it. `hypergraph_to_morphism` then
drops the cycle rather than raising. `pathway_collapse` hoists a rewritten chain one block out
only when every wire it reads comes from outside the block, and `merge_duplicate_roots`
hoists a kept root to the enclosing scope only when no wire it reads is produced below that
scope, merging within each scope otherwise.

## See also

- [[Rewriting]] — the pointwise alternative
- [[Crawlers]] — the guided traversal, for when the rule depends on context
- [[Pathway Collapse]] — the pass that rewires blocks after every rewrite
