'''Replacing the leaves of a hypergraph in place, and rebuilding the containers.

Written by Claude Opus 5 (1M context), reasoning effort high.

`all_leaves` walks every operation of a graph, descending through every block, loops
included. `flat_subgraphs` stops at a loop block, so a pass that has to reach the
operations inside one walks with `all_leaves`.

`splice` takes a replacement for each of a set of leaves, keyed by the uid of the leaf,
and puts the new subgraphs where the old leaf stood, however deeply nested that place
was. Every container between the root and the replacement then has a domain and a
codomain that no longer match what its members consume and produce, and `rescope`
rebuilds both from the members. A pass that adds an operation to a graph without
splicing owes the graph the same call, because a block that has gained an operation
reading a wire from outside has gained that wire on its domain and nothing else works
out where.

`obsidian/03-hypergraphs/Leaf Splicing.md` covers the walk, the splice and the order
the rebuilt domain keeps.
'''

from __future__ import annotations
from typing import Iterator
import data_structure.Numeric as nm
import data_structure.Term as fd
import graphs.data_structure.Hypergraph as hg
import utilities.utilities as util


def all_leaves(graph: hg.Hypergraph) -> Iterator[hg.HypergraphRoot]:
    '''Every operation, descending through every block, loops included.'''
    match graph:
        case hg.HypergraphRoot():
            yield graph
        case hg.HypergraphBlock(body=body):
            yield from all_leaves(body)
        case hg.Multigraph():
            for subgraph in graph.subgraphs():
                yield from all_leaves(subgraph)


def _splice_seq(
    graph: hg.Hypergraph,
    replacements: dict[fd.UID, fd.Prod[hg.Hypergraph]],
) -> fd.Prod[hg.Hypergraph]:
    match graph:
        case hg.HypergraphRoot():
            return replacements.get(graph.uid, (graph,))
        case hg.HypergraphBlock(body=hg.Multigraph() as body):
            return (hg.HypergraphBlock.template(
                body=_splice_one(body, replacements),
                block_tag=graph.block_tag),)
        case hg.HypergraphBlock(body=body):
            return (hg.HypergraphBlock.template(
                body=_splice_body(body, replacements),
                block_tag=graph.block_tag),)
        case hg.Multigraph():
            return (_splice_one(graph, replacements),)
    return (graph,)


def _splice_body(
    body: hg.Hypergraph,
    replacements: dict[fd.UID, fd.Prod[hg.Hypergraph]],
) -> hg.Hypergraph:
    '''A block body that is one subgraph rather than a `Multigraph`, as a
    reduction terminal's is, with its replacement in place. One replacement
    stands as the body itself, so a terminal boxed by two kernels is a box
    inside a box, and several are gathered into a `Multigraph` on the body's
    own wires.'''
    spliced = _splice_seq(body, replacements)
    if len(spliced) == 1:
        return spliced[0]
    return hg.Multigraph.template(dom=body.dom, cod=body.cod, subgraphs=spliced)


def _splice_one(
    graph: hg.Hypergraph,
    replacements: dict[fd.UID, fd.Prod[hg.Hypergraph]],
) -> hg.Hypergraph:
    assert isinstance(graph, hg.Multigraph)
    new_subgraphs = util.concat(
        _splice_seq(subgraph, replacements) for subgraph in graph.subgraphs())
    return hg.Multigraph.template(
        dom=graph.dom, cod=graph.cod, subgraphs=new_subgraphs)


def _node_objects(graph: hg.Hypergraph) -> dict[hg.HypergraphObject, hg.HypergraphObject]:
    '''The authoritative object for each wire: what its producer writes,
    falling back to the whole graph's inputs.'''
    authoritative: dict[hg.HypergraphObject, hg.HypergraphObject] = {
        obj: obj for obj in graph.dom}
    for leaf in all_leaves(graph):
        for obj in leaf.cod:
            authoritative[obj] = obj
    return authoritative


def _rescope(
    graph: hg.Hypergraph,
    authoritative: dict[hg.HypergraphObject, hg.HypergraphObject],
    is_outermost: bool,
    carried: int = 0,
) -> hg.Hypergraph:
    '''Rebuild a container's domain and codomain from its members.

    The domain is whatever its subgraphs consume without producing. The
    codomain keeps its nodes, with the objects refreshed from their producers.
    The top container keeps both, because they are its contract with
    everything outside it.

    Which wires appear is recomputed. The order they were already in is kept,
    and only genuinely new wires are appended, in order of first consumption.
    A loop body's last `carried` wires are the seeds of the values the loop
    carries, one per wire of its codomain, per
    `explicit_accumulator.carried_values`, and a new wire is inserted before
    them so that they stay last.

    That order is a deliberate choice made where the container was built.
    `explicit_accumulator.make_block` takes the loop's inputs from
    `stream_graph.sibling_subgraph`, which sorts them by `domain_order`, the
    order in which the surrounding graph produces them.

    Re-deriving the order from what the body happens to consume first discards
    that choice and crosses the wires. Attention's loop would take K before Q,
    because the body splits K before it reads the Q tile, while outside the
    loop Q is produced first.
    '''
    match graph:
        case hg.HypergraphRoot():
            return graph
        case hg.HypergraphBlock(body=body):
            is_loop = graph.block_tag.repetition != nm.Integer(1)
            return hg.HypergraphBlock.template(
                body=_rescope(body, authoritative, is_outermost=False,
                              carried=len(graph.cod) if is_loop else 0),
                block_tag=graph.block_tag)
        case hg.Multigraph():
            new_subgraphs = tuple(
                _rescope(subgraph, authoritative, is_outermost=False)
                for subgraph in graph.subgraphs())
            produced = {
                node for subgraph in new_subgraphs for node in subgraph.cod}
            old_by_node = {obj: obj for obj in (*graph.dom, *graph.cod)}

            def refreshed(node: hg.HypergraphObject) -> hg.HypergraphObject:
                return authoritative.get(node) or old_by_node[node]

            incoming = util.unique_tuple(
                node
                for subgraph in new_subgraphs
                for node in subgraph.dom
                if node not in produced)
            # Stable, so the wires the dom already carried keep their
            # relative order and anything new lands after them.
            established = {node: i for i, node in enumerate(graph.dom)}
            kept = tuple(sorted(
                (node for node in incoming if node in established),
                key=lambda node: established[node]))
            new = tuple(node for node in incoming if node not in established)
            split = len(kept) - carried if carried else len(kept)
            dom_nodes = (
                graph.dom if is_outermost
                else (*kept[:split], *new, *kept[split:]))
            return hg.Multigraph.template(
                dom=tuple(refreshed(node) for node in dom_nodes),
                cod=tuple(refreshed(obj) for obj in graph.cod),
                subgraphs=new_subgraphs)
    return graph


def splice(
    graph: hg.Hypergraph,
    replacements: dict[fd.UID, fd.Prod[hg.Hypergraph]],
) -> hg.Hypergraph:
    '''Replace leaves by uid, then rescope every container.

    Each replacement is a tuple of new subgraphs, spliced flat into the
    container the old leaf occupied.
    '''
    return rescope(_splice_one(graph, replacements))


def rescope(graph: hg.Hypergraph) -> hg.Hypergraph:
    '''
    Rebuild every container's dom and cod from what its members now consume
    and produce, taking each wire's object from its producer.

    Every pass that adds operations to a graph owes the containers around them
    this call. A block that has gained an operation reading a wire from
    outside has gained that wire on its dom, and nothing else works out where.
    `_rescope` states what is preserved, which is the order the dom was
    already in, and why the order matters.
    '''
    return _rescope(graph, _node_objects(graph), is_outermost=True)
