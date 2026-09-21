---
tags: [layer/categories, algorithm]
code: algebra/einops_rearrange.py
status: stable
---

# Einops Rearrangement

## What it is

Given an expression containing two adjacent `Einops` steps, one feeding an input of the
other, rearrange them into their most efficient form, working entirely from
`Einops.signature`, the weaves and the reindexings. No part of the module manipulates the
`'a b, c b -> a c'` string a template is written with.

It is not a cost search. It is two unconditional structural rules, always applied, in order.

1. **Always unify**, merging the two into one combined `Einops`.
2. **Then always disentangle**, so that when a contraction involves only some of the input
   segments, that subset is pulled out, contracted first, and combined with the rest by an
   outer product.

## What `signature` states

`Einops.signature` is a `Prod[Prod[int]]` with one outer entry per input segment, and one
segment per array consumed. Each inner tuple lists the contraction group of every non-TILED
axis of that segment, in positional order. Two segments sharing a group number are summed
together over that axis. A group number appearing in one segment alone is a plain sum
reduction over that axis.

The entries line up positionally with `Broadcasted.input_weaves[i]._shape`. A position there
is either `cat.WeaveMode.TILED`, meaning a broadcast axis with no group, or a concrete
`RawAxis`, meaning an axis the segment contracts. The `RawAxis` positions are exactly the
ones `signature[i]` enumerates, in the same order, which is what makes the signature readable
without parsing a string:

```python
def segment_group_axes(operator, weaves):
    '''Group number -> the RawAxis it names, from the signature and the weaves.'''
    axes = {}
    for segment_signature, weave in zip(operator.signature, weaves):
        non_tiled = [ax for ax in weave._shape if ax is not cat.WeaveMode.TILED]
        for group, axis in zip(segment_signature, non_tiled):
            axes.setdefault(group, axis)
    return axes
```

`Broadcasted.reindexings` holds one `Rearrangement` per input segment, and an `Einops` always
has exactly one output segment, which `Einops.template` asserts. Every reindexing shares one
`_dom`, the degree. A segment's `.mapping` states which degree positions its own TILED slots
draw from. An absorbed position never appears in `.mapping`, and a broadcast one always does.
The output weave's TILED positions correspond one to one with the shared degree, so the
output's degree is `reindexings[0]._dom`.

## Why these two rules

Counting the elements each form computes explains why these are the right rules to apply
unconditionally. The count is not consulted to choose between candidates.

The cost of an `Einops` is the size of its degree multiplied by the size of its contracted
axes. The degree is `Broadcasted.degree()`. The contracted axes are every axis named
anywhere in `signature`:

```python
def einops_cost(target: cat.Broadcasted) -> nm.Numeric:
    degree_sizes = [axis.local_size() for axis in target.degree()]
    stream_sizes = [
        axis.local_size()
        for axis in segment_group_axes(target.operator, target.input_weaves).values()]
    return nm.Multiplication.template(*degree_sizes, *stream_sizes)
```

Applied to a softmax with an explicit batch axis `y`:

| form | pieces | cost |
|---|---|---|
| incoherent | `y,x->y x` then `y x, x -> y` | `xy + xy = 2xy` |
| unified, after merging | `y,x,x->y` | `xy` |
| coherent, after disentangling | `x,x->` and `y,->y` | `x + y` |

- **Merging never re-materialises an intermediate that does not need to exist**, so it is
  strictly cheaper.
- **Disentangling never re-introduces a cross product between axes that were never related.**
  A single joint contraction over segments from two components that share nothing still
  iterates the full cross product of both components' contracted axes, even though none of
  that cross-relationship is real. Splitting along the connected components can only remove
  that term, and it can never add one.

The partition into connected components is unique, so it is the factoring rather than a
candidate factoring.

## Merging, precisely

`merge_einops(producer, consumer, feed_index)`. The producer's output is an all-TILED array
over the producer's degree, plugged into the consumer's feed slot position for position. The
consumer's feed slot therefore assigns each of the producer's degree axes a role, read
straight off the feed weave:

- a feed position that is `TILED` leaves that axis a degree axis of the merged operation;
- a feed position naming a `RawAxis` means the consumer contracts that axis into the group
  its signature names at that position, and the merged operation absorbs it into that
  group. A group is a position in a signature and never an axis, so a group the producer
  contracts and a group the consumer contracts over one axis stay two groups. Before
  2026-09-05 the merged groups were keyed by `RawAxis`, and self-attention from one copied
  input merged `dQ = dS·K` with `K = X·W_K` reading `X`'s `m` as a degree axis.

Every input segment of the producer is then rewritten through those roles, the consumer's
other segments carry over unchanged, the groups are renumbered from scratch, and the
reindexings are rebuilt as plain `Rearrangement`s against the merged degree.

The old special case for a carrier segment is not needed. It is the case where one segment's
degree axes reproduce the whole of the producer's output.

## Disentangling, precisely

1. Build a graph over the merged operation's input segments, with an edge wherever two share
   a contraction group.
2. Take the connected components. One component leaves nothing to disentangle, and the
   operation is returned unchanged.
3. Build one sub-`Einops` per component. A singleton with no absorbed groups and no axis
   reorder needs no new node at all, because whatever already produces that value becomes a
   join input directly. It still calls `imprint_to_degree` to put the component's degree axes
   back on, or a broadcast axis is dropped and the result is a bare scalar where `s[q]` was
   meant.
4. Build one final join `Einops` combining the components by outer product, with every
   segment sharing the full merged degree as its reindexing domain. A component spanning
   fewer axes than the output degree broadcasts up into the rest. Rearranging `q, q x -> q x`
   feeding `q x, x v -> q v` gives the components `{s[q]}`, spanning `q`, and `{M, N}`,
   spanning `q v`, and the join `q, q v -> q v` broadcasts `s[q]` across `v`.

Worked through on the softmax: the merged `,x,x->` has the segments `[scalar with no groups,
x in group 0, x in group 0]`, so there are two components, `{scalar}` and `{x, x}`. The
component `{x, x}` becomes `Einops('x,x->')`. The component `{scalar}` is a singleton with no
groups, so it needs no new node and is whatever already produces that scalar, which here is
the output of `Elementwise('/z')`. The join is `Einops(',')`. The pair is the coherent softmax
exactly.

## Where it runs

The module is `algebra/einops_rearrange.py`, in the general `algebra/` package because
the merge belongs to no one feature. `algebra/validate_simplification.py` and
`para/validate_backward.py` both call it.

`rearrange_einops(graph)` walks the hypergraph with `flat_subgraphs` and a producer and
consumer lookup by node identity, because [[Hypergraph Analysis]] answers over one level of
direct children and a raw graph is several levels deep. It makes one pass rather than
searching to a fixpoint.

The ordering against [[Linear Expansion]] matters. Expansion has to run after rearrangement,
or the normaliser folds a projection into the score and produces a three-operand contraction
that no tensor core will claim. It was found the hard way, when every matmul fell to the
threads.

## The same rules, driven differently

`merge_rule` is `merge_einops` followed by `disentangle_einops`, offered one pair at a time
to `algebra.merge_into_consumer`, which poses the same producer-and-consumer question and then
splices. `merge_reindexings_and_einops` alternates `reindexing_absorption.absorb`, on the
search that copies a node over its fan-out, with `merge_rule`, on the search that folds a
producer read once, until neither changes the graph, so a reindexing between two
contractions is absorbed and the contractions merge in one call.
`merge_into_consumer.merge_producers_into_consumers(morphism, merge_rule)`, also spelled
`rearrange_local`, differs from `rearrange_einops` in two ways that matter to its caller:

| | `rearrange_einops` | `rearrange_local` |
|---|---|---|
| passes | one sweep | to a fixed point |
| blocks | rebuilt flat | left standing |

A pass over a whole graph needs the first, because it flattens the graph once.
[[Backpropagation]] needs the second, because the reverse of a `SoftMax` is a block, and
flattening it loses the marker saying where that reverse came from. Both call the same
`merge_einops`, so there is one merge algebra and two searches. [[Expression Simplification]]
pairs this rule with the node rule.

## What the disentangled shape gives

Two branches that share no contracted axis are two independent computations, and the
disentangled form says so. The merged form says only that six axes are contracted
somewhere, so anything reading the expression afterwards has to work the independence out
again.

## See also

- [[Operators]] — `Einops.signature`
- [[Weaves and Degree]] — the positional correspondence the whole module rests on
- [[Linear Expansion]] — the ordering constraint
- [[Expression Simplification]] — the other rule this one is driven beside, and the search
