'''Rematerialising a taped value from the taped operands of its contraction.

`pathway_collapse.recompute_elementwise_slots` recomputes a slot across one
pointwise map. This rewrite reaches one contraction further. Where the forward
pass drops `f(einsum(a, b))`, with `f` a chain of pointwise maps and additions and
`a` and `b` already on the tape, the drop is deleted, and every backward grab of it
becomes `Grab(a), Grab(b) -> einsum -> f`, rebuilt onto the wire the grab produced
so that its readers do not move.

An addition in the chain has side operands, such as the negated maximum a shifted
softmax adds to its scores before the exponent. A side operand is rebuilt from a
slot of its own: the wire it is a pointwise image of is grabbed if the forward pass
drops it already, and dropped to a new slot otherwise. The new slot is small,
because a side operand is broadcast over the axis the chain's contraction produced.

On attention the slot is the exponent `e = exp(QK^T)`, at `[q, x]`. Removing it
leaves the five values FlashAttention stores, and the backward pass gains the
recomputed `QK^T` and its exponent. On the shifted expansion the exponent is
`exp(QK^T - m)`, and the backward pass rebuilds it from `Q`, `K` and the maximum
`m`, which joins the tape at `[q]`, so the recomputation is as stable as the
forward pass. Which slots to recompute is the caller's decision.
`notebooks/website/tutorial/derive_training_step.py` recomputes every slot a chain
rebuilds, and `obsidian/07-para/Recomputing the Exponent in the Backward Pass.md` states
the rule and its limits.
'''

from __future__ import annotations

from collections.abc import Iterable
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

RECOMPUTE_LIMIT = 64


@dataclass(frozen=True)
class SideOperand:
    '''An operand of an addition in the chain that is not the chain itself.

    `base` is the forward wire it is rebuilt from, `slot` the slot that wire is
    dropped to, or None when the rewrite has to drop it, and `pointwise` the
    maps applied to it in order.
    '''
    base: hg.HypergraphObject
    slot: para.TapeSlot | None
    pointwise: tuple[hg.HypergraphRoot, ...]

    @property
    def needs_slot(self) -> bool:
        return self.slot is None


@dataclass(frozen=True)
class RecomputeChain:
    '''The forward roots that rebuild one dropped wire from taped operands.

    `contraction` is the einsum, `steps` the roots applied to its result in
    order, each a pointwise map or an addition, `operand_slots` the slot each
    contraction operand is dropped to, and `sides` the side operands of the
    additions among the steps, keyed by the step and the operand position.
    '''
    drop: hg.HypergraphRoot
    contraction: hg.HypergraphRoot
    steps: tuple[hg.HypergraphRoot, ...]
    operand_slots: tuple[para.TapeSlot, ...]
    sides: tuple[tuple[int, int, SideOperand], ...]

    @property
    def slot(self) -> para.TapeSlot:
        return self.drop.wraps.tape

    @property
    def pointwise(self) -> tuple[hg.HypergraphRoot, ...]:
        return tuple(step for step in self.steps if is_pointwise(step.wraps))

    @property
    def added_slots(self) -> tuple[SideOperand, ...]:
        return tuple(side for _, _, side in self.sides if side.needs_slot)


def is_pointwise(morphism: cat.Morphism) -> bool:
    return (isinstance(morphism, cat.Broadcasted)
            and isinstance(morphism.operator, ops.Elementwise)
            and not isinstance(morphism.operator, (ops.View, ops.Dropout)))


def is_addition(morphism: cat.Morphism) -> bool:
    return (isinstance(morphism, cat.Broadcasted)
            and isinstance(morphism.operator, ops.AdditionOp))


def is_contraction(morphism: cat.Morphism) -> bool:
    return (isinstance(morphism, cat.Broadcasted)
            and isinstance(morphism.operator, ops.Einops)
            and len(morphism.input_weaves) >= 2)


def _producers(
    graph: hg.Multigraph,
) -> dict[hg.HypergraphObject, hg.HypergraphRoot]:
    return {wire: root for root in replace_roots.walk_roots(graph)
            for wire in root.cod}


def _drops(graph: hg.Multigraph) -> dict[hg.HypergraphObject, hg.HypergraphRoot]:
    return {root.dom[0]: root for root in replace_roots.walk_roots(graph)
            if isinstance(root.wraps, para.Drop)}


def _has_repeated_axis(wire: hg.HypergraphObject) -> bool:
    shape = tuple(wire.obj.shape())
    return len(shape) != len(set(shape))


def _side_operand(
    wire: hg.HypergraphObject,
    producers: dict[hg.HypergraphObject, hg.HypergraphRoot],
    drops: dict[hg.HypergraphObject, hg.HypergraphRoot],
    rank_limit: int,
) -> SideOperand | None:
    '''The wire as a pointwise image of a taped or tapeable wire, or None.

    A wire is tapeable when its rank is below that of the slot being removed
    and no axis of it is repeated, so that the slot the rewrite adds is smaller
    than the one it removes.
    '''
    pointwise: list[hg.HypergraphRoot] = []
    while True:
        if wire in drops:
            return SideOperand(wire, drops[wire].wraps.tape, tuple(reversed(pointwise)))
        producer = producers.get(wire)
        if (producer is not None and is_pointwise(producer.wraps)
                and len(producer.dom) == 1 and len(producer.cod) == 1):
            pointwise.append(producer)
            wire = producer.dom[0]
            continue
        if len(tuple(wire.obj.shape())) < rank_limit and not _has_repeated_axis(wire):
            return SideOperand(wire, None, tuple(reversed(pointwise)))
        return None


def recompute_chain_of(
    graph: hg.Multigraph, drop: hg.HypergraphRoot,
) -> RecomputeChain | None:
    '''The chain that rebuilds the wire `drop` saves, or None when the wire is
    not a pointwise image of a contraction of taped wires.'''
    producers = _producers(graph)
    drops = _drops(graph)
    rank = len(tuple(drop.dom[0].obj.shape()))
    steps: list[hg.HypergraphRoot] = []
    sides: list[tuple[int, int, SideOperand]] = []
    wire = drop.dom[0]
    while True:
        producer = producers.get(wire)
        if producer is None or len(producer.cod) != 1:
            return None
        if is_pointwise(producer.wraps) and len(producer.dom) == 1:
            steps.append(producer)
            wire = producer.dom[0]
            continue
        if is_addition(producer.wraps) and len(producer.dom) >= 2:
            continuing = _continuing_operand(producer, producers, drops)
            if continuing is None:
                return None
            step = len(steps)
            for position, operand in enumerate(producer.dom):
                if position == continuing:
                    continue
                side = _side_operand(operand, producers, drops, rank)
                if side is None:
                    return None
                sides.append((step, position, side))
            steps.append(producer)
            wire = producer.dom[continuing]
            continue
        if not is_contraction(producer.wraps):
            return None
        if any(operand not in drops for operand in producer.dom):
            return None
        return RecomputeChain(
            drop=drop, contraction=producer, steps=tuple(reversed(steps)),
            operand_slots=tuple(drops[operand].wraps.tape
                                for operand in producer.dom),
            sides=tuple((len(steps) - 1 - step, position, side)
                        for step, position, side in sides))


def _continuing_operand(
    addition: hg.HypergraphRoot,
    producers: dict[hg.HypergraphObject, hg.HypergraphRoot],
    drops: dict[hg.HypergraphObject, hg.HypergraphRoot],
) -> int | None:
    '''The one operand of `addition` behind which a contraction of taped wires
    stands, through pointwise maps and further additions, or None.'''
    found = None
    for position, operand in enumerate(addition.dom):
        wire = operand
        while True:
            producer = producers.get(wire)
            if producer is None or len(producer.cod) != 1:
                break
            if (is_pointwise(producer.wraps) or is_addition(producer.wraps)) and (
                    len(producer.dom) == 1):
                wire = producer.dom[0]
                continue
            if is_addition(producer.wraps):
                wire = None
                break
            if is_contraction(producer.wraps) and all(
                    source in drops for source in producer.dom):
                if found is not None:
                    return None
                found = position
            break
    return found


def recompute_chains(taped: backprop.Taped) -> tuple[RecomputeChain, ...]:
    '''Every slot of the forward pass that the rewrite could remove.'''
    graph = hg.Multigraph.from_morphism(taped.forward)
    chains = []
    for root in replace_roots.walk_roots(graph):
        if isinstance(root.wraps, para.Drop):
            chain = recompute_chain_of(graph, root)
            if chain is not None:
                chains.append(chain)
    return tuple(chains)


def _replay(
    roots: Iterable[hg.HypergraphRoot], wire: hg.HypergraphObject,
) -> tuple[list[hg.Hypergraph], hg.HypergraphObject]:
    rebuilt: list[hg.Hypergraph] = []
    for root in roots:
        copy = hg.HypergraphRoot.template(root.wraps, dom=(wire,))
        rebuilt.append(copy)
        wire = copy.cod[0]
    return rebuilt, wire


def _rebuilt_grab(
    chain: RecomputeChain, old: hg.HypergraphRoot,
    side_slots: dict[hg.HypergraphObject, para.TapeSlot],
) -> tuple[fd.Prod[hg.Hypergraph], fd.UIDRenaming]:
    '''The roots that replace one grab of the removed slot, and the renaming
    that puts the recomputed value on the grab's own wire.'''
    grabs = tuple(
        hg.HypergraphRoot.template(para.Grab(tape=slot, size=array.obj))
        for slot, array in zip(chain.operand_slots, chain.contraction.dom))
    roots: list[hg.Hypergraph] = list(grabs)
    contraction = hg.HypergraphRoot.template(
        chain.contraction.wraps, dom=tuple(grab.cod[0] for grab in grabs))
    roots.append(contraction)
    wire = contraction.cod[0]
    sides = {(step, position): side for step, position, side in chain.sides}
    for index, step in enumerate(chain.steps):
        if is_pointwise(step.wraps):
            replayed, wire = _replay((step,), wire)
            roots.extend(replayed)
            continue
        operands: list[hg.HypergraphObject] = []
        for position in range(len(step.dom)):
            side = sides.get((index, position))
            if side is None:
                operands.append(wire)
                continue
            grab = hg.HypergraphRoot.template(
                para.Grab(tape=side_slots[side.base], size=side.base.obj))
            roots.append(grab)
            replayed, side_wire = _replay(side.pointwise, grab.cod[0])
            roots.extend(replayed)
            operands.append(side_wire)
        addition = hg.HypergraphRoot.template(step.wraps, dom=tuple(operands))
        roots.append(addition)
        wire = addition.cod[0]
    return tuple(roots), fd.UIDRenaming.set_canonical(wire, old.cod[0])


def _place_drops(
    graph: hg.Multigraph, chain: RecomputeChain,
) -> tuple[hg.Multigraph, dict[hg.HypergraphObject, para.TapeSlot]]:
    '''`graph` with a drop added for every side operand the rewrite needs on the
    tape, and the slot of every side operand.'''
    slots: dict[hg.HypergraphObject, para.TapeSlot] = {}
    added: list[hg.Hypergraph] = []
    for _, _, side in chain.sides:
        if side.base in slots:
            continue
        if side.slot is not None:
            slots[side.base] = side.slot
            continue
        slot = para.new_slot()
        slots[side.base] = slot
        added.append(hg.HypergraphRoot.template(
            para.Drop(tape=slot, size=side.base.obj), dom=(side.base,)))
    if not added:
        return graph, slots
    return rewire_blocks.recompute_block_boundaries(
        rewire_blocks.insert_roots(graph, {(): added})), slots


def recompute_contraction_slots[L, M: cat.Morphism](
    taped: backprop.Taped[L, M],
    slots: Iterable[para.TapeSlot] | None = None,
) -> backprop.Taped[L, M]:
    '''`taped` with each slot in `slots` recomputed in the backward pass from the
    taped operands of the contraction that produced it. With `slots` omitted every
    eligible slot is recomputed.

    The forward pass loses the drop and gains a drop for each side operand not
    yet on the tape. Each backward grab of the slot becomes the operand grabs,
    the contraction and the replayed steps, spliced where the grab stood, and
    the recomputed value takes the grab's wire.
    '''
    forward_graph = hg.Multigraph.from_morphism(taped.forward)
    backward_graph = hg.Multigraph.from_morphism(taped.backward)
    chosen = None if slots is None else set(slots)
    changed = False
    for _ in range(RECOMPUTE_LIMIT):
        chain = next(
            (candidate for root in replace_roots.walk_roots(forward_graph)
             if isinstance(root.wraps, para.Drop)
             and (chosen is None or root.wraps.tape in chosen)
             for candidate in (recompute_chain_of(forward_graph, root),)
             if candidate is not None),
            None)
        if chain is None:
            break
        changed = True
        if chosen is not None:
            chosen.discard(chain.slot)
        forward_graph, side_slots = _place_drops(forward_graph, chain)
        forward_graph = rewire_blocks.recompute_block_boundaries(
            replace_roots.replace_roots(forward_graph, {chain.drop: None}))
        edits: dict[hg.HypergraphRoot, replace_roots.Replacement] = {}
        renamings: list[fd.UIDRenaming] = []
        for old in replace_roots.walk_roots(backward_graph):
            if isinstance(old.wraps, para.Grab) and old.wraps.tape == chain.slot:
                roots, renaming = _rebuilt_grab(chain, old, side_slots)
                edits[old] = roots
                renamings.append(renaming)
        backward_graph = fd.Context(renamings).apply(
            rewire_blocks.recompute_block_boundaries(
                replace_roots.replace_roots(backward_graph, edits)))
    else:
        raise RuntimeError(f'{RECOMPUTE_LIMIT} recomputations without a fixed point')
    if chosen:
        names = ', '.join(sorted(slot.uid._name.to_bodies() for slot in chosen))
        raise ValueError(f'No contraction chain rebuilds the slots {names}')
    if not changed:
        return taped
    return backprop.Taped.from_passes(
        h2m.hypergraph_to_morphism(forward_graph),
        h2m.hypergraph_to_morphism(backward_graph))
