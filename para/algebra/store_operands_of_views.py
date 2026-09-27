# Claude Opus 5.5 (1M context), effort 40.
'''Storing the operand of a chain of views on the tape in place of its result.

A view reads its operand at other positions and computes nothing, so the backward pass
can rebuild the result of a view from its operand with no arithmetic. Where the forward
pass drops the result of a chain of views, and the array the chain starts from has fewer
axes than the result, the drop moves to that array. Every backward grab of the result
becomes a grab of the array followed by the views of the chain, replayed onto the wire
the grab produced. Where the array is already on the tape, the drop of the result is
deleted and its grabs read the slot that holds the array.

The causal read of attention is the case the pass is written for. The mask reads the keys
and the values at `[x, d]` as arrays at `[x, w|x, d]`, one entry for every slot of every
token. After the pass the tape holds the keys and the values at `[x, d]`, as FlashAttention
stores them. `obsidian/07-para/Recomputing the Exponent in the Backward Pass.md` states the
pass beside the recomputation of the exponent.
'''
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

import data_structure.Category as cat
import data_structure.Operators as ops
import data_structure.Term as fd
import graphs.data_structure.Hypergraph as hg
import graphs.processing.Hypergraph2Morphism as h2m
import graphs.processing.replace_roots as replace_roots
import graphs.processing.rewire_blocks as rewire_blocks
import para.data_structure.Para as para
import para.processing.backprop as backprop

VIEW_MOVE_LIMIT = 64


@dataclass(frozen=True)
class ViewChain:
    '''The views a dropped wire is the result of, and the array they start from.

    `views` are in the order the forward pass applies them. `operand_slot` is the slot
    the forward pass already drops `operand` to, or None when it drops it nowhere.'''
    drop: hg.HypergraphRoot
    operand: hg.HypergraphObject
    views: tuple[hg.HypergraphRoot, ...]
    operand_slot: para.TapeSlot | None

    @property
    def slot(self) -> para.TapeSlot:
        return self.drop.wraps.tape

    @property
    def kept_slot(self) -> para.TapeSlot:
        '''The slot that holds the operand once the drop has moved.'''
        return self.slot if self.operand_slot is None else self.operand_slot


def is_view(morphism: cat.Morphism) -> bool:
    return (isinstance(morphism, cat.Broadcasted)
            and isinstance(morphism.operator, ops.View))


def rank_of(wire: hg.HypergraphObject) -> int:
    return len(tuple(wire.obj.shape()))


def producers_of(
    graph: hg.Multigraph,
) -> dict[hg.HypergraphObject, hg.HypergraphRoot]:
    return {wire: root for root in replace_roots.walk_roots(graph) for wire in root.cod}


def drops_of(graph: hg.Multigraph) -> dict[hg.HypergraphObject, hg.HypergraphRoot]:
    return {root.dom[0]: root for root in replace_roots.walk_roots(graph)
            if isinstance(root.wraps, para.Drop)}


def view_chain_of(
    drop: hg.HypergraphRoot,
    producers: Mapping[hg.HypergraphObject, hg.HypergraphRoot],
    drops: Mapping[hg.HypergraphObject, hg.HypergraphRoot],
) -> ViewChain | None:
    '''The chain of views whose result `drop` saves, or None when that result is not a
    view of an array with fewer axes.'''
    views: list[hg.HypergraphRoot] = []
    wire = drop.dom[0]
    while (producer := producers.get(wire)) is not None and (
            is_view(producer.wraps) and len(producer.dom) == 1
            and len(producer.cod) == 1):
        views.append(producer)
        wire = producer.dom[0]
    if not views or rank_of(wire) >= rank_of(drop.dom[0]):
        return None
    operand_drop = drops.get(wire)
    return ViewChain(
        drop=drop, operand=wire, views=tuple(reversed(views)),
        operand_slot=None if operand_drop is None else operand_drop.wraps.tape)


def view_chains(taped: backprop.Taped) -> tuple[ViewChain, ...]:
    '''Every drop of the forward pass that the rewrite would move.'''
    graph = hg.Multigraph.from_morphism(taped.forward)
    producers, drops = producers_of(graph), drops_of(graph)
    return tuple(chain for drop in drops.values()
                 if (chain := view_chain_of(drop, producers, drops)) is not None)


def first_view_chain(graph: hg.Multigraph) -> ViewChain | None:
    producers, drops = producers_of(graph), drops_of(graph)
    return next((chain for drop in drops.values()
                 if (chain := view_chain_of(drop, producers, drops)) is not None),
                None)


def views_left_unread(graph: hg.Multigraph, chain: ViewChain) -> set[hg.HypergraphRoot]:
    '''The views of the chain that nothing reads once the drop of its result is gone,
    taken from the end of the chain back, such as a view computed only to be saved.'''
    readers = Counter(wire for root in replace_roots.walk_roots(graph)
                      if root is not chain.drop for wire in root.dom)
    outputs = set(graph.cod)
    unread: set[hg.HypergraphRoot] = set()
    for view in reversed(chain.views):
        if readers[view.cod[0]] or view.cod[0] in outputs:
            break
        unread.add(view)
        readers.subtract(view.dom)
    return unread


def forward_with_the_drop_moved(
    graph: hg.Multigraph, chain: ViewChain,
) -> hg.Multigraph:
    '''`graph` without the drop of the chain's result or the views nothing reads once
    it is gone, and with a drop of the chain's operand where the operand is on no slot
    yet. The new drop is spliced in where the first view stood, so that it stands in
    the block that reads the operand.'''
    unread = views_left_unread(graph, chain)
    edits: dict[hg.HypergraphRoot, replace_roots.Replacement] = {chain.drop: None}
    edits.update({view: None for view in unread})
    if chain.operand_slot is None:
        first_view = chain.views[0]
        moved = hg.HypergraphRoot.template(
            para.Drop(tape=chain.slot, size=chain.operand.obj), dom=(chain.operand,))
        edits[first_view] = (moved,) if first_view in unread else (first_view, moved)
    return rewire_blocks.recompute_block_boundaries(
        replace_roots.replace_roots(graph, edits))


def grab_rebuilt_through_the_views(
    chain: ViewChain, old_grab: hg.HypergraphRoot,
) -> tuple[fd.Prod[hg.Hypergraph], fd.UIDRenaming]:
    '''The grab of the operand and the replayed views that replace `old_grab`, and the
    renaming that puts the rebuilt result on the wire `old_grab` produced.'''
    grab = hg.HypergraphRoot.template(
        para.Grab(tape=chain.kept_slot, size=chain.operand.obj))
    roots: list[hg.Hypergraph] = [grab]
    wire = grab.cod[0]
    for view in chain.views:
        replayed = hg.HypergraphRoot.template(view.wraps, dom=(wire,))
        roots.append(replayed)
        wire = replayed.cod[0]
    return tuple(roots), fd.UIDRenaming.set_canonical(wire, old_grab.cod[0])


def backward_with_the_views_replayed(
    graph: hg.Multigraph, chain: ViewChain,
) -> hg.Multigraph:
    edits: dict[hg.HypergraphRoot, replace_roots.Replacement] = {}
    renamings: list[fd.UIDRenaming] = []
    for root in replace_roots.walk_roots(graph):
        if isinstance(root.wraps, para.Grab) and root.wraps.tape == chain.slot:
            edits[root], renaming = grab_rebuilt_through_the_views(chain, root)
            renamings.append(renaming)
    return fd.Context(renamings).apply(rewire_blocks.recompute_block_boundaries(
        replace_roots.replace_roots(graph, edits)))


def store_operands_of_views[L, M: cat.Morphism](
    taped: backprop.Taped[L, M],
) -> backprop.Taped[L, M]:
    '''`taped` with every dropped result of a chain of views stored as the array the
    chain starts from, where that array has fewer axes, and rebuilt through the views in
    the backward pass.'''
    forward_graph = hg.Multigraph.from_morphism(taped.forward)
    backward_graph = hg.Multigraph.from_morphism(taped.backward)
    changed = False
    for _ in range(VIEW_MOVE_LIMIT):
        chain = first_view_chain(forward_graph)
        if chain is None:
            break
        changed = True
        forward_graph = forward_with_the_drop_moved(forward_graph, chain)
        backward_graph = backward_with_the_views_replayed(backward_graph, chain)
    else:
        raise RuntimeError(f'{VIEW_MOVE_LIMIT} moved drops without a fixed point')
    if not changed:
        return taped
    return backprop.Taped.from_passes(
        h2m.hypergraph_to_morphism(forward_graph),
        h2m.hypergraph_to_morphism(backward_graph))
