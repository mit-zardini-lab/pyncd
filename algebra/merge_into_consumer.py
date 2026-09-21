'''Merging a morphism into the morphism that consumes it.

Both rewrites in this package perform the same move. Absorbing a node's
reindexing, in `reindexing_absorption`, and composing two contractions, in
`einops_simplification`, each take a `Broadcasted` whose result is read by
another `Broadcasted` and fold it into that reader. Only the algebra of the
merge differs, so the search is written once here and the two rules are passed
in.

Two searches drive the rules. `merge_producers_into_consumers` folds a producer
that is read exactly once. `merge_producers_into_every_consumer` folds a
producer read any number of times into each reader that takes the rule, and
removes the producer once nothing reads it. The second search copies the
producer over the fan-out of its output, which is the naturality of the copy,
and it is offered a rule only where the copied producer costs nothing, meaning
the node rule. A contraction is never copied.

The search runs on the graph rather than on the morphism, because "read once,
and by that one" is a question about wires. A morphism has no wires. It has
positions in a product, one `Rearrangement` away from meaning anything. In the
graph a wire is a `HypergraphObject`, its consumers are the roots that name it in
their `dom`, and counting them is a `Counter`.

`obsidian/02-categories/Expression Simplification.md` is the full account.
'''
from __future__ import annotations
from collections import Counter
from typing import Callable, Iterable

import data_structure.Category as cat
import data_structure.Term as fd
import graphs.data_structure.Hypergraph as hg
import graphs.processing.Hypergraph2Morphism as h2m


'''
A rule is given the producer, the consumer, and which of the consumer's inputs the
producer feeds. The producer is the morphism carrying the shared wire in its
`cod`, and the consumer is the one carrying it in its `dom`. The rule returns
the single morphism the two become, or `None` when it does not apply, which is
the usual answer and not an error.

By default the merged morphism's domain is the consumer's, with `port` replaced
in place by the whole of the producer's domain. Call that the canonical order.
A rule that needs a different order returns
`(morphism, order)`, where `order` indexes into the canonical one;
`algebra.einops_rearrange.merge_rule` needs this, because
`disentangle_einops` re-groups the operands it kept.
'''
type Merged = cat.Morphism | tuple[cat.Morphism, tuple[int, ...]]
type MergeRule = Callable[[cat.Broadcasted, cat.Broadcasted, int],
                          Merged | None]

MERGE_LIMIT = 256


def _count_uses(graph: hg.Hypergraph, counter: Counter, is_outermost: bool = True) -> None:
    '''
    How many times each wire is read.

    Everything that is not a root's own output counts: a root's inputs, a
    block's inputs *and* outputs (a value leaving a block is read outside it),
    a nested scope's outputs, and the whole graph's outputs. Over-counting is
    the safe direction, because it costs a merge that would have been legal, where
    under-counting silently drops a consumer.
    '''
    if is_outermost:
        for obj in graph.cod:
            counter[obj] += 1
    match graph:
        case hg.HypergraphRoot():
            for obj in graph.dom:
                counter[obj] += 1
        case hg.HypergraphBlock(body=body):
            for obj in (*graph.dom, *graph.cod):
                counter[obj] += 1
            _count_uses(body, counter, is_outermost=False)
        case hg.Multigraph():
            if not is_outermost:
                for obj in graph.cod:
                    counter[obj] += 1
            for subgraph in graph.subgraphs():
                _count_uses(subgraph, counter, is_outermost=False)


def _mergeable(graph: hg.Hypergraph) -> bool:
    return (isinstance(graph, hg.HypergraphRoot)
            and isinstance(graph.wraps, cat.Broadcasted))


def canonical_order(
    producer_width: int,
    consumer_width: int,
    port: int,
    order: Iterable[int],
) -> tuple[int, ...]:
    '''
    Translate an order given as "the producer's operands, then the consumer's
    remaining ones" into one indexing the canonical concatenation.

    `einops_rearrange` numbers a merged morphism's operands that way, and this
    is the only place the two conventions meet.
    '''
    translated = []
    for index in order:
        if index < producer_width:
            translated.append(port + index)
        else:
            kept = index - producer_width
            original = kept if kept < port else kept + 1
            translated.append(original if original < port
                              else port + producer_width + original - port - 1)
    return tuple(translated)


def _reading_pairs(
    graph: hg.Multigraph,
) -> Iterable[tuple[int, int, int]]:
    '''Every `(producer, consumer, port)` of sibling `Broadcasted` roots joined by
    one wire, as indices into `graph._subgraphs`.

    Siblings, because a wire that crosses into a block is named on the block's dom or
    cod as well, and a merge would have to rewrite those too. A consumer reading the
    wire at two ports is skipped, because the rules take one port.
    '''
    subgraphs = graph._subgraphs
    for i, producer in enumerate(subgraphs):
        if not _mergeable(producer) or len(producer.cod) != 1:
            continue
        wire = producer.cod[0]
        for j, consumer in enumerate(subgraphs):
            if i == j or not _mergeable(consumer):
                continue
            ports = [p for p, obj in enumerate(consumer.dom) if obj == wire]
            if len(ports) == 1:
                yield i, j, ports[0]


def _merge_pair(
    graph: hg.Multigraph,
    producer_index: int,
    consumer_index: int,
    port: int,
    rules: tuple[MergeRule, ...],
    uses: Counter,
) -> hg.Multigraph | None:
    '''The consumer rewritten by the first rule that takes the pair, or `None`.

    The producer is removed when the consumer was its only reader. Otherwise it stays,
    still read by the others, and the consumer now reads the producer's input directly.
    '''
    subgraphs = graph._subgraphs
    producer = subgraphs[producer_index]
    consumer = subgraphs[consumer_index]
    for rule in rules:
        answer = rule(producer.wraps, consumer.wraps, port)
        if answer is None:
            continue
        merged, order = (answer if isinstance(answer, tuple)
                         else (answer, None))
        udom = (*(obj for obj in consumer.dom[:port]),
                *(obj for obj in producer.dom),
                *(obj for obj in consumer.dom[port + 1:]))
        if order is not None:
            udom = tuple(udom[index] for index in order)
        if len(tuple(merged.dom())) != len(udom):
            raise ValueError(
                f'{getattr(rule, "__name__", rule)}: merged domain is '
                f'{len(tuple(merged.dom()))} wires, expected {len(udom)}')
        root = hg.HypergraphRoot.template(
            merged, udom, tuple(obj for obj in consumer.cod))
        is_last_reader = uses[producer.cod[0]] == 1
        rebuilt = tuple(
            root if k == consumer_index else subgraph
            for k, subgraph in enumerate(subgraphs)
            if k != producer_index or not is_last_reader)
        return hg.Multigraph.template(graph.dom, graph.cod, rebuilt)
    return None


type MergeStep = Callable[
    [hg.Multigraph, tuple[MergeRule, ...], Counter], hg.Multigraph | None]


def _merge_here(
    graph: hg.Multigraph,
    rules: tuple[MergeRule, ...],
    uses: Counter,
) -> hg.Multigraph | None:
    '''One merge between two siblings whose wire is read once, or `None`.'''
    for i, j, port in _reading_pairs(graph):
        if uses[graph._subgraphs[i].cod[0]] != 1:
            continue
        rebuilt = _merge_pair(graph, i, j, port, rules, uses)
        if rebuilt is not None:
            return rebuilt
    return None


def _merge_into_each_here(
    graph: hg.Multigraph,
    rules: tuple[MergeRule, ...],
    uses: Counter,
) -> hg.Multigraph | None:
    '''One merge between two siblings, however many readers the wire has, or
    `None`.'''
    for i, j, port in _reading_pairs(graph):
        rebuilt = _merge_pair(graph, i, j, port, rules, uses)
        if rebuilt is not None:
            return rebuilt
    return None


def _merge_once(
    graph: hg.Hypergraph,
    rules: tuple[MergeRule, ...],
    uses: Counter,
    merge_here: MergeStep,
) -> hg.Hypergraph | None:
    '''The first merge anywhere in `graph`, innermost scopes first.'''
    match graph:
        case hg.HypergraphRoot():
            return None
        case hg.HypergraphBlock(body=body):
            rebuilt = _merge_once(body, rules, uses, merge_here)
            if rebuilt is None:
                return None
            return hg.HypergraphBlock.template(rebuilt, graph.block_tag)
        case hg.Multigraph():
            for i, subgraph in enumerate(graph._subgraphs):
                rebuilt = _merge_once(subgraph, rules, uses, merge_here)
                if rebuilt is not None:
                    return hg.Multigraph.template(
                        graph.dom, graph.cod,
                        tuple(rebuilt if k == i else sub
                              for k, sub in enumerate(graph._subgraphs)))
            return merge_here(graph, rules, uses)
    raise NotImplementedError(f'cannot rewrite {type(graph).__name__}')


def _merge_to_fixed_point[L, M: cat.Morphism](
    graph: hg.Hypergraph[L, M],
    rules: tuple[MergeRule, ...],
    merge_here: MergeStep,
) -> hg.Hypergraph[L, M]:
    '''One merge per pass, with the use counts recomputed each time, until no pair
    takes a rule.

    A merge removes a consumer, so what was read twice before it can be read once
    after, and a chain collapses from the far end inwards. The graph is returned
    as the same object when nothing merged.
    '''
    for _ in range(MERGE_LIMIT):
        uses: Counter = Counter()
        _count_uses(graph, uses)
        rebuilt = _merge_once(graph, rules, uses, merge_here)
        if rebuilt is None:
            return graph
        graph = rebuilt
    raise RuntimeError(
        f'{MERGE_LIMIT} merges without reaching a fixed point - a rule is '
        'probably undoing another one')


def merge_producers_into_consumers_in_graph[L, M: cat.Morphism](
    graph: hg.Hypergraph[L, M],
    *rules: MergeRule,
) -> hg.Hypergraph[L, M]:
    '''Every producer read once by one `Broadcasted`, merged into that consumer by
    the first of `rules` that accepts the pair, until no pair takes one.'''
    return _merge_to_fixed_point(graph, rules, _merge_here)


def merge_producers_into_every_consumer_in_graph[L, M: cat.Morphism](
    graph: hg.Hypergraph[L, M],
    *rules: MergeRule,
) -> hg.Hypergraph[L, M]:
    '''Every producer read by any number of `Broadcasted`s, merged into each of
    them that accepts a rule, until no pair takes one.

    A producer read by several consumers is copied over the fan-out, each copy
    merged into its own reader, and the original removed once nothing reads it.
    A reader that takes no rule, a reader inside a block and an output of the
    graph all keep the original in place. Offer this search a rule only where
    the copies cost nothing, as `reindexing_absorption.absorb_nodes` does.
    '''
    return _merge_to_fixed_point(graph, rules, _merge_into_each_here)


def merge_producers_into_consumers[L, M: cat.Morphism](
    target: cat.ProdCategory[L, M],
    *rules: MergeRule,
) -> cat.ProdCategory[L, M]:
    '''`merge_producers_into_consumers_in_graph` on a morphism, through the graph
    and back.'''
    return h2m.hypergraph_to_morphism(merge_producers_into_consumers_in_graph(
        hg.Multigraph.from_morphism(target), *rules))


def merge_producers_into_every_consumer[L, M: cat.Morphism](
    target: cat.ProdCategory[L, M],
    *rules: MergeRule,
) -> cat.ProdCategory[L, M]:
    '''`merge_producers_into_every_consumer_in_graph` on a morphism, through the
    graph and back.'''
    return h2m.hypergraph_to_morphism(
        merge_producers_into_every_consumer_in_graph(
            hg.Multigraph.from_morphism(target), *rules))
