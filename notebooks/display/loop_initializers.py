'''Choosing whether the initializer of a loop variable is drawn.

Written by Claude Opus 5 (1M context), effort 80.

A loop carries each accumulated value from one iteration to the next as a loop
variable, which `para.data_structure.Para.StreamGrab` reads inside the loop and
`Para.StreamDrop` writes for the next iteration. Before the loop, an initializer
computes the starting value of the variable and a `Drop` writes it to the variable's
slot, and the initializers, their drops and the loop stand in one block. The starting
value is the universal unit of the accumulator, and the `StreamGrab` of the slot
inside the loop shows that the variable exists and where the loop first reads it.
`HIDDEN` removes each initializer and the drop that starts its variable just before
the term is drawn, and removes the block that grouped them with the loop once the
loop is all the block holds. `DRAWN` draws the term as it stands. The graph keeps
the initializers, so the choice belongs to the display.
'''
from __future__ import annotations

import enum

import data_structure.Category as cat
import data_structure.Term as fd
import graphs.data_structure.Hypergraph as hg
import graphs.processing.rewire_blocks as rewire_blocks
import para.data_structure.MultiCategory as multi_category
import para.data_structure.Para as Para
import term_utilities.term_utilities as tutil


class LoopInitializers(enum.Enum):
    DRAWN = 'drawn'
    HIDDEN = 'hidden'


def is_loop(subgraph: hg.Hypergraph) -> bool:
    '''Whether `subgraph` is a block whose repetition denotes a loop.'''
    return (isinstance(subgraph, hg.HypergraphBlock)
            and rewire_blocks.is_loop(subgraph))


def loop_variable_slots(term: fd.GeneralTerm) -> frozenset[fd.UID]:
    '''The uid of the slot of every loop variable `term` grabs.'''
    return frozenset(grab.tape.uid for grab in tutil.type_search(Para.StreamGrab, term))


def starts_loop_variable(subgraph: hg.Hypergraph, slots: frozenset[fd.UID]) -> bool:
    '''Whether `subgraph` is the plain `Drop` that writes the starting value of one
    of the loop variables `slots`, as opposed to the `StreamDrop` inside the loop.'''
    return (isinstance(subgraph, hg.HypergraphRoot)
            and isinstance(subgraph.wraps, Para.Drop)
            and not isinstance(subgraph.wraps, Para.StreamDrop)
            and subgraph.wraps.tape.uid in slots)


def without_unread_producers(
    subgraphs: fd.Prod[hg.Hypergraph],
    discarded: frozenset[hg.HypergraphObject],
    codomain: fd.Prod[hg.HypergraphObject],
) -> fd.Prod[hg.Hypergraph]:
    '''`subgraphs` without each subgraph whose every output is a wire in
    `discarded` that no remaining subgraph and no wire of `codomain` reads. The
    inputs of a subgraph removed are discarded in turn, so a chain of operations
    that computed only a removed drop's value is removed with it.'''
    remaining = subgraphs
    discarded_wires = set(discarded)
    while True:
        read = {node for subgraph in remaining for node in subgraph.dom} | set(codomain)
        removed = tuple(
            subgraph for subgraph in remaining
            if subgraph.cod and not is_loop(subgraph)
            and all(node in discarded_wires and node not in read for node in subgraph.cod))
        if not removed:
            return remaining
        removed_uids = {subgraph.uid for subgraph in removed}
        remaining = tuple(subgraph for subgraph in remaining
                          if subgraph.uid not in removed_uids)
        discarded_wires.update(node for subgraph in removed for node in subgraph.dom)


def only_loop_left(body: hg.Hypergraph) -> hg.HypergraphBlock | None:
    '''The loop `body` holds, when it holds that loop alone and has the loop's
    inputs and outputs, and None otherwise.'''
    if not isinstance(body, hg.Multigraph):
        return None
    match body.subgraphs():
        case (loop,) if (is_loop(loop)
                         and set(loop.dom) == set(body.dom)
                         and set(loop.cod) == set(body.cod)):
            return loop
    return None


def hide_initializers_within(
    graph: hg.Hypergraph, slots: frozenset[fd.UID],
) -> hg.Hypergraph:
    '''`graph` without the drops that start the loop variables `slots`, the
    initializers whose values only those drops read, and the blocks left holding
    a loop alone. A graph with nothing removed is returned as it stands.'''
    match graph:
        case hg.HypergraphBlock(body=body):
            hidden = hide_initializers_within(body, slots)
            if hidden is body:
                return graph
            if any(starts_loop_variable(subgraph, slots)
                   for subgraph in body.subgraphs()):
                loop = only_loop_left(hidden)
                if loop is not None:
                    return loop
            return graph.reconstruct(body=hidden)
        case hg.Multigraph():
            subgraphs = tuple(hide_initializers_within(subgraph, slots)
                              for subgraph in graph.subgraphs())
            starts = tuple(subgraph for subgraph in subgraphs
                           if starts_loop_variable(subgraph, slots))
            kept = without_unread_producers(
                tuple(subgraph for subgraph in subgraphs
                      if not starts_loop_variable(subgraph, slots)),
                frozenset(node for start in starts for node in start.dom),
                graph.cod)
            if len(kept) == len(graph.subgraphs()) and all(
                    new is old for new, old in zip(kept, graph.subgraphs())):
                return graph
            return hg.Multigraph.template(dom=graph.dom, cod=graph.cod, subgraphs=kept)
    return graph


def present[L, M: cat.Morphism](
    term: cat.ProdCategory[L, M] | hg.Hypergraph[L, M]
    | multi_category.MultiCategory[L, M],
    initializers: LoopInitializers,
) -> cat.ProdCategory[L, M] | hg.Hypergraph[L, M] | multi_category.MultiCategory[L, M]:
    '''`term` with the initializers of its loop variables drawn as `initializers`
    asks. `DRAWN` returns it as it stands, and so does `HIDDEN` when `term` holds
    no loop variable or is a row of morphisms. A morphism holding a loop variable
    is returned as the hypergraph it converts to.'''
    if initializers is LoopInitializers.DRAWN:
        return term
    slots = loop_variable_slots(term)
    if not slots:
        return term
    match term:
        case multi_category.MultiCategory():
            return term
        case hg.Hypergraph():
            return hide_initializers_within(term, slots)
        case _:
            return hide_initializers_within(hg.Multigraph.from_morphism(term), slots)
