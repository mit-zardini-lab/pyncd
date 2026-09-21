'''Placing roots in block scopes and rederiving the block boundaries.

A rewrite that removes, replaces or adds roots inside a nested block leaves
the block's domain and codomain stale. `recompute_block_boundaries` rederives
both for every block from where each wire is produced and where it is
consumed, so a wire produced inside one block and read inside another flows
out through the first block's codomain, up to their common ancestor, and in
through the second block's domain. A block emptied by a rewrite is removed.

`insert_roots` appends roots to the scope named by a chain of block uids, and
`walk_roots_with_blocks` yields every root with the blocks enclosing it. A
loop block, meaning one whose `repetition` is not 1, is walked like any other
here. The passes that treat it as a boundary test `is_loop` themselves.
'''
from __future__ import annotations
from collections import defaultdict
from typing import Iterable, Iterator, Mapping

import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Term as fd
import graphs.data_structure.Hypergraph as hg
import utilities.utilities as util

type BlockChain = fd.Prod[fd.UID[hg.HypergraphBlock]]
type Wire = hg.HypergraphObject


def is_loop(block: hg.HypergraphBlock) -> bool:
    return block.block_tag.repetition != nm.Integer(1)


def chain_of(blocks: fd.Prod[hg.HypergraphBlock]) -> BlockChain:
    return tuple(block.uid for block in blocks)


def common_prefix(chains: Iterable[BlockChain]) -> BlockChain:
    prefix: BlockChain = ()
    for position in zip(*chains):
        if len(set(position)) != 1:
            break
        prefix = (*prefix, position[0])
    return prefix


def walk_roots_with_blocks[L, M: cat.Morphism](
    graph: hg.Hypergraph[L, M],
    enclosing_blocks: fd.Prod[hg.HypergraphBlock[L, M]] = (),
) -> Iterator[tuple[hg.HypergraphRoot[L, M], fd.Prod[hg.HypergraphBlock[L, M]]]]:
    '''Every root of `graph`, at any depth, in order, with the blocks around it
    from the outermost inwards.'''
    match graph:
        case hg.HypergraphRoot():
            yield graph, enclosing_blocks
        case hg.HypergraphBlock(body=body):
            yield from walk_roots_with_blocks(body, (*enclosing_blocks, graph))
        case hg.Multigraph():
            for subgraph in graph.subgraphs():
                yield from walk_roots_with_blocks(subgraph, enclosing_blocks)


def insert_roots[L, M: cat.Morphism](
    graph: hg.Hypergraph[L, M],
    placements: Mapping[BlockChain, Iterable[hg.Hypergraph[L, M]]],
    chain: BlockChain = (),
    is_scope_root: bool = True,
) -> hg.Hypergraph[L, M]:
    '''`graph` with each chain's roots appended to the outermost `Multigraph`
    of that chain. The empty chain names the top level. Boundaries are left
    as they were, for `recompute_block_boundaries`.'''
    match graph:
        case hg.HypergraphRoot():
            return graph
        case hg.HypergraphBlock(body=body):
            rebuilt_body = insert_roots(body, placements, (*chain, graph.uid), True)
            return graph if rebuilt_body is body else graph.reconstruct(
                body=rebuilt_body)
        case hg.Multigraph():
            children = tuple(insert_roots(child, placements, chain, False)
                             for child in graph._subgraphs)
            added = tuple(placements.get(chain, ())) if is_scope_root else ()
            unchanged = not added and all(
                child is original
                for child, original in zip(children, graph._subgraphs))
            if unchanged:
                return graph
            return graph.reconstruct(_subgraphs=(*children, *added))
    raise NotImplementedError(f'cannot rewrite {type(graph).__name__}')


def _ordered(existing: fd.Prod[Wire], needed: Iterable[Wire]) -> fd.Prod[Wire]:
    '''`needed`, in the order of `existing` for the wires already there and in
    the order given for the rest.'''
    needed_tuple = util.unique_tuple(needed)
    needed_set = set(needed_tuple)
    kept = tuple(wire for wire in util.unique_tuple(existing) if wire in needed_set)
    kept_set = set(kept)
    return (*kept, *(wire for wire in needed_tuple if wire not in kept_set))


def recompute_block_boundaries[L, M: cat.Morphism](
    graph: hg.Multigraph[L, M],
) -> hg.Multigraph[L, M]:
    '''Every block's domain and codomain rederived from the roots, and every
    emptied block removed. The outermost graph keeps its own domain and
    codomain, because they are the morphism's.

    A wire produced at one block chain and consumed at another flows through
    the codomain of every block below their common prefix on the producer's
    side, and through the domain of every block below it on the consumer's
    side. A wire's position in a block's boundary is kept where it already
    had one, and a wire new to a boundary is appended in the order the roots
    are walked.
    '''
    produced_at: dict[Wire, BlockChain] = {wire: () for wire in graph.dom}
    consumed_at: dict[Wire, list[BlockChain]] = defaultdict(list)
    in_walk_order: dict[Wire, None] = {}
    for root, blocks in walk_roots_with_blocks(graph):
        chain = chain_of(blocks)
        for wire in root.cod:
            produced_at.setdefault(wire, chain)
            in_walk_order.setdefault(wire)
        for wire in root.dom:
            consumed_at[wire].append(chain)
            in_walk_order.setdefault(wire)
    for wire in graph.cod:
        consumed_at[wire].append(())

    needs_dom: dict[BlockChain, list[Wire]] = defaultdict(list)
    needs_cod: dict[BlockChain, list[Wire]] = defaultdict(list)
    for wire in in_walk_order:
        producer = produced_at.get(wire)
        if producer is None:
            continue
        for consumer in consumed_at.get(wire, ()):
            shared = len(common_prefix((producer, consumer)))
            for depth in range(shared + 1, len(producer) + 1):
                needs_cod[producer[:depth]].append(wire)
            for depth in range(shared + 1, len(consumer) + 1):
                needs_dom[consumer[:depth]].append(wire)

    def rebuild(target: hg.Hypergraph[L, M], chain: BlockChain,
                ) -> hg.Hypergraph[L, M] | None:
        match target:
            case hg.HypergraphRoot():
                return target
            case hg.HypergraphBlock(body=body):
                own_chain = (*chain, target.uid)
                rebuilt_body = rebuild(body, own_chain)
                if rebuilt_body is None or not rebuilt_body.subgraphs():
                    return None
                dom = _ordered(target.dom, needs_dom[own_chain])
                cod = _ordered(target.cod, needs_cod[own_chain])
                if rebuilt_body is body and dom == target.dom and cod == target.cod:
                    return target
                return target.reconstruct(
                    body=rebuilt_body.reconstruct(dom=dom, cod=cod),
                    dom=dom, cod=cod)
            case hg.Multigraph():
                children: list[hg.Hypergraph[L, M]] = []
                for child in target._subgraphs:
                    rebuilt = rebuild(child, chain)
                    if rebuilt is None:
                        continue
                    if isinstance(rebuilt, hg.Multigraph):
                        children.extend(rebuilt._subgraphs)
                    else:
                        children.append(rebuilt)
                unchanged = len(children) == len(target._subgraphs) and all(
                    child is original
                    for child, original in zip(children, target._subgraphs))
                return target if unchanged else target.reconstruct(
                    _subgraphs=tuple(children))
        raise NotImplementedError(f'cannot rewrite {type(target).__name__}')

    rebuilt = rebuild(graph, ())
    if rebuilt is None:
        return graph.reconstruct(_subgraphs=())
    return rebuilt
