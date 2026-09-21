# Claude Opus 5 (1M context), high effort.
'''A boxed block whose tape seeds stand at its domain and its codomain.

A layer plan writes a value one layer publishes onto a tape slot and reads it
back in each layer that needs it, per `obsidian/07-para/Para Category.md`. A
`Para.Grab` and a `Para.Drop` have the empty product on one side, so a block
holding them has the domain and the codomain it would have without them, and
`ops.BlockOperator.template` boxes such a block with no sign that the tape is
touched inside.

`ParaBlockOperator` boxes the same block with the tape at the ports. Each grab
inside the body becomes a leading operand of the box and each drop a trailing
result, so the box reads
`(*grabbed, *apparent_dom) -> (*apparent_cod, *dropped)`, and the box is
returned inside a `ParaWrap` carrying the slots at those positions. From
outside, the wrap has the apparent domain and the apparent codomain of the
block, which is what the model composes with, and a reader sees which slots the
box reads and writes without opening it.

`block` holds the block as the model wrote it, with the seeds standing inside,
which is what a diagram draws beside the box. The user asked for that on
2026-09-15: "The ParaBlockOperator (eg FullAttention) you should still show the
drops internally." `expose_tape_as_ports` derives the ported body from it, and
the weaves, the recorded seeds and the body `expand` lifts all come from that
derivation, so the box and the block state the same morphism.

`obsidian/07-para/Para Block Operator.md` holds the mathematics and the display
rules.
'''
from __future__ import annotations
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import data_structure.Category as cat
import data_structure.Operators as ops
import data_structure.Term as fd
import graphs.data_structure.Hypergraph as hg
import graphs.processing.Hypergraph2Morphism as h2m
import term_utilities.term_utilities as tutil

import algebra.discovering_broadcasts as discovering_broadcasts
import para.data_structure.Para as Para
import para.data_structure.ParaWrap as para_wrap
import para.registries.object_lift  # one lift rule per tape seed

type ParaBBlock[B: cat.Datatype = cat.Reals, A: cat.Axis = cat.RawAxis] = cat.Block[
    cat.Array[B, A],
    Para.Para[cat.Array[B, A], cat.BroadcastedCategory[B, A]]]

type WrappedBox[
    B: cat.Datatype = cat.Reals, A: cat.Axis = cat.RawAxis,
] = para_wrap.ParaWrap[
    cat.Array[B, A], cat.Broadcasted[B, A, ParaBlockOperator[B, A]]]


class TapeBelowTheTopLevel(ValueError):
    '''A block whose body holds a tape seed inside a nested scope, which
    `expose_tape_as_ports` cannot lift to the block's own ports.'''


@dataclass(frozen=True)
class ParaBlockOperator[B: cat.Datatype = cat.Reals, A: cat.Axis = cat.RawAxis](
        ops.BlockOperator[B, A]):
    '''A block as one named operator, reading its grabbed arrays and writing its
    dropped ones through ports of its own.

    `block` is the block as it was written, seeds included. `grabs` holds one
    seed per leading operand of the box and `drops` one per trailing result, in
    the order the ports stand in, so the two say which slot each port carries
    and `wrap_box` reads them to build the `ParaWrap` that states the same thing
    on the box.
    '''
    grabs: fd.Prod[Para.Grab[cat.Array[B, A]]] = ()
    drops: fd.Prod[Para.Drop[cat.Array[B, A]]] = ()

    @classmethod
    def template(
        cls,
        block: ParaBBlock[B, A],
        name: str | None | fd.DynamicName = None,
    ) -> WrappedBox[B, A]:
        '''`block` boxed with its tape at the ports, inside the wrap that names
        the slots those ports carry.

        The wrap is the form a model is written with, because it has the
        apparent domain and codomain of `block`. `bare_box` returns the box
        itself, which is what `expand` and the weaves are read from.
        '''
        return wrap_box(bare_box(block, name))

    def expand(
        self,
        parent: cat.Broadcasted[B, A, ParaBlockOperator[B, A]],
    ) -> cat.ProdCategory[cat.Array[B, A], cat.Morphism[cat.Array[B, A]]]:
        '''The ported body lifted over `parent`'s degree, with a `View` per
        operand carrying that operand's reindexing.

        The body expanded is `expose_tape_as_ports(self.block).block`, whose
        domain and codomain are the ones `parent`'s weaves were built from, so
        the expansion holds no tape seed and the grabbed arrays arrive on the
        operands the wrap tapes.
        '''
        if self != parent.operator:
            raise ValueError(
                f'{parent.operator} is the operator of the morphism handed to '
                f'{self}.expand')
        return discovering_broadcasts.expand_broadcast_of_block(
            parent.reconstruct(operator=ops.BlockOperator(
                name=self.name, block=expose_tape_as_ports(self.block).block)))


@dataclass(frozen=True)
class TapeAsPorts[B: cat.Datatype, A: cat.Axis]:
    '''A block whose tape seeds have been moved to its own ports, beside the
    seeds each port stands for.

    `block` reads one leading operand per entry of `grabs` and writes one
    trailing result per entry of `drops`.
    '''
    block: ParaBBlock[B, A]
    grabs: fd.Prod[Para.Grab[cat.Array[B, A]]]
    drops: fd.Prod[Para.Drop[cat.Array[B, A]]]


def expose_tape_as_ports[B: cat.Datatype, A: cat.Axis](
    block: ParaBBlock[B, A],
) -> TapeAsPorts[B, A]:
    '''`block` with each `Para.Grab` of its body a leading operand and each
    `Para.Drop` a trailing result, beside the seeds those ports stand for.

    The rewrite runs on the body's graph, where a seed is a subgraph of its own
    and the wire it carries becomes the port. `para.algebra.detape` performs the
    same rewrite for a whole pass and does not fit here: it puts a grab at the
    end of the domain rather than the start, it orders the ports by slot name
    rather than by where the seeds stand, and it hands back slot names rather
    than the seeds themselves.

    A seed inside a nested scope of the body raises `TapeBelowTheTopLevel`,
    because lifting it to the block's ports would move a wire across a scope
    this rewrite does not open.
    '''
    graph = hg.Multigraph.from_morphism(block.body)
    subgraphs = graph._subgraphs
    grabs = tuple(i for i, sub in enumerate(subgraphs)
                  if _is_seed_of(sub, Para.Grab))
    drops = tuple(i for i, sub in enumerate(subgraphs)
                  if _is_seed_of(sub, Para.Drop))
    taped = frozenset(grabs) | frozenset(drops)
    body = h2m.hypergraph_to_morphism(hg.Multigraph.template(
        (*(subgraphs[i].cod[0] for i in grabs), *graph.dom),
        (*graph.cod, *(subgraphs[i].dom[0] for i in drops)),
        tuple(sub for i, sub in enumerate(subgraphs) if i not in taped)))
    deeper = tuple(tape_seeds_of(body))
    if deeper:
        raise TapeBelowTheTopLevel(
            f'{len(deeper)} tape seeds stand inside a nested scope of the '
            f'body, on the slots '
            f'{[Para.slot_of(Para.entry_of(seed)).uid.to_latex() for seed in deeper]}')
    return TapeAsPorts(
        block=block.reconstruct(body=body),
        grabs=tuple(subgraphs[i].wraps for i in grabs),
        drops=tuple(subgraphs[i].wraps for i in drops))


def tape_seeds_of(target: cat.Morphism) -> Iterator[Para.ParaMorphism]:
    '''Every tape seed standing in the structure of `target`, a nested block
    included.

    The walk follows the construction rules and never enters an operator, so the
    seeds a `ParaBlockOperator` records are not among them, and never enters the
    body of a `ParaWrap`, whose own entries already state its tape.
    '''
    match target:
        case Para.ParaMorphism():
            yield target
        case cat.Block(body=body):
            yield from tape_seeds_of(body)
        case cat.Composed(content=content) | cat.ProductOfMorphisms(content=content):
            for part in content:
                yield from tape_seeds_of(part)


def _is_seed_of(subgraph: hg.Hypergraph, kind: type) -> bool:
    return (isinstance(subgraph, hg.HypergraphRoot)
            and isinstance(subgraph.wraps, kind))


def bare_box[B: cat.Datatype, A: cat.Axis](
    block: ParaBBlock[B, A],
    name: str | None | fd.DynamicName = None,
) -> cat.Broadcasted[B, A, ParaBlockOperator[B, A]]:
    '''`block` boxed with its tape at the ports, and no wrap over it.

    The weaves and the reindexings are the ones `ops.BlockOperator.template`
    writes for the ported body, so the box has one operand per grab in front of
    the block's own operands and one result per drop after the block's own
    results. The operator carries `block` itself, seeds included.
    '''
    exposed = expose_tape_as_ports(block)
    return _record_tape_seeds(
        ops.BlockOperator.template(exposed.block, name), block, exposed)


def bare_box_of[B: cat.Datatype, A: cat.Axis](
    target: WrappedBox[B, A] | cat.Broadcasted[B, A, ParaBlockOperator[B, A]],
) -> cat.Broadcasted[B, A, ParaBlockOperator[B, A]]:
    '''The box itself, given either a wrapped box or a box.'''
    if isinstance(target, para_wrap.ParaWrap):
        return target.body
    return target


def wrap_box[B: cat.Datatype, A: cat.Axis](
    box: cat.Broadcasted[B, A, ParaBlockOperator[B, A]],
) -> WrappedBox[B, A]:
    '''`box` with the slot each of its taped ports carries written onto it.

    The grabbed slots stand at the leading operands and the dropped ones at the
    trailing results, which is where `bare_box` put the ports, so the wrap's own
    domain and codomain are the apparent ones of the block.
    '''
    grabs = tuple(Para.entry_of(grab) for grab in box.operator.grabs)
    drops = tuple(Para.entry_of(drop) for drop in box.operator.drops)
    return para_wrap.ParaWrap(
        body=box,
        grabs=(*grabs, *(None,) * (len(box.dom()) - len(grabs))),
        drops=(*(None,) * (len(box.cod()) - len(drops)), *drops))


class WrapDisagreesWithBlock(ValueError):
    '''A wrap over a boxed block whose entries do not match the block's operands and
    results one for one, so its tapes cannot be written into the block as seeds.'''


def box_with_wrapped_tapes_as_seeds[B: cat.Datatype, A: cat.Axis](
    wrap: para_wrap.ParaWrap[
        cat.Array[B, A], cat.Broadcasted[B, A, ops.BlockOperator[B, A]]],
) -> WrappedBox[B, A]:
    '''The box of `wrap` as a `ParaBlockOperator` whose block holds the tapes of
    `wrap` as seeds, inside the wrap that names the same slots at its ports.

    `para_wrap.to_para_wrap` writes a grab that feeds a boxed block onto the port of
    the box, and the block inside still reads that operand on a wire, so the body
    drawn for the box shows no tape. The box returned here reads its tape twice over,
    as a box `template` builds does: at the ports of the box, and on the seeds inside
    the block. The grabbed operands lead and the dropped results trail, which is the
    order of ports every `ParaBlockOperator` has.

    A seed inside the block carries the array the block reads or writes at that port.
    Where the box is broadcast over a degree, the array at the port carries the degree
    as well, as `broadcast_para_block_over_axes` states.

    The block takes a tag of its own, derived from the tag it had and the slots it now
    holds, because the block standing elsewhere under the old tag holds no seed.
    '''
    box = wrap.body
    block = box.operator.block
    if (len(wrap.grabs), len(wrap.drops)) != (len(block.dom()), len(block.cod())):
        raise WrapDisagreesWithBlock(
            f'{len(wrap.grabs)} grab entries and {len(wrap.drops)} drop entries over '
            f'a block with {len(block.dom())} operands and {len(block.cod())} results')
    grabbed = tuple(i for i, entry in enumerate(wrap.grabs) if entry is not None)
    kept_inputs = tuple(i for i, entry in enumerate(wrap.grabs) if entry is None)
    dropped = tuple(j for j, entry in enumerate(wrap.drops) if entry is not None)
    kept_outputs = tuple(j for j, entry in enumerate(wrap.drops) if entry is None)
    tag = block.block_tag
    seeded = block.reconstruct(
        body=para_wrap.ParaWrap(
            body=block.body, grabs=wrap.grabs, drops=wrap.drops).to_base(),
        block_tag=tag.reconstruct(uid=tag.uid.reconstruct(_id=fd.hash_id(
            (tag.uid._id, repr(wrap.grabs), repr(wrap.drops))))))
    return wrap_box(box.reconstruct(
        operator=ParaBlockOperator(
            name=box.operator.name,
            block=seeded,
            grabs=tuple(Para.grab_of(wrap.grabs[i], block.dom()[i]) for i in grabbed),
            drops=tuple(Para.drop_of(wrap.drops[j], block.cod()[j]) for j in dropped)),
        input_weaves=tuple(box.input_weaves[i] for i in (*grabbed, *kept_inputs)),
        reindexings=tuple(box.reindexings[i] for i in (*grabbed, *kept_inputs)),
        output_weaves=tuple(box.output_weaves[j] for j in (*kept_outputs, *dropped))))


def broadcast_para_block_over_axes[B: cat.Datatype, A: cat.Axis](
    body: ParaBBlock[B, A],
    degree_axes: fd.Prod[A],
    degree_readings: fd.Prod[fd.Prod[int]],
    name: str | None | fd.DynamicName = None,
) -> WrappedBox[B, A]:
    '''`discovering_broadcasts.broadcast_block_over_axes` with the tape of
    `body` at the ports and the slots on a wrap over the box it builds.

    `degree_readings` holds one reading per operand of `body`, as it does there.
    A grabbed operand is read at every position of the degree, because the
    expansion lifts the body over the whole degree and the array a grab inside
    the lifted body reads carries every degree axis. A dropped result carries
    the whole degree for the reason every result does.
    '''
    exposed = expose_tape_as_ports(body)
    whole_degree = tuple(range(len(degree_axes)))
    return wrap_box(_record_tape_seeds(
        discovering_broadcasts.broadcast_block_over_axes(
            exposed.block, degree_axes,
            (*(whole_degree,) * len(exposed.grabs), *degree_readings), name),
        body, exposed))


def _record_tape_seeds[B: cat.Datatype, A: cat.Axis](
    broadcast: cat.Broadcasted[B, A, ops.BlockOperator[B, A]],
    block: ParaBBlock[B, A],
    exposed: TapeAsPorts[B, A],
) -> cat.Broadcasted[B, A, ParaBlockOperator[B, A]]:
    '''`broadcast`, built from the ported body, with the written `block` and the
    seeds each port carries on its operator.'''
    return broadcast.reconstruct(operator=ParaBlockOperator(
        name=broadcast.operator.name,
        block=block,
        grabs=exposed.grabs,
        drops=exposed.drops))


def broadcast_grabs[B: cat.Datatype, A: cat.Axis](
    parent: WrappedBox[B, A] | cat.Broadcasted[B, A, ParaBlockOperator[B, A]],
) -> fd.Prod[Para.Grab[cat.Array[B, A]]]:
    '''Each grab of the box, reading the array the degree gives it, which is the
    array standing at the leading operand the grab was moved to.

    The slot is unchanged, because a slot names the value and the degree names
    which member of it one index of the broadcast reads.
    '''
    box = bare_box_of(parent)
    return tuple(
        Para.grab_of(Para.entry_of(grab), array)
        for grab, array in zip(box.operator.grabs, box.dom()))


def broadcast_drops[B: cat.Datatype, A: cat.Axis](
    parent: WrappedBox[B, A] | cat.Broadcasted[B, A, ParaBlockOperator[B, A]],
) -> fd.Prod[Para.Drop[cat.Array[B, A]]]:
    '''Each drop of the box, writing the array standing at the trailing result
    the drop was moved to.'''
    box = bare_box_of(parent)
    results = tuple(box.cod())
    return tuple(
        Para.drop_of(Para.entry_of(drop), array)
        for drop, array in zip(box.operator.drops,
                               results[len(results) - len(box.operator.drops):]))


def slots_grabbed(target: fd.GeneralTerm) -> frozenset[Para.TapeSlot]:
    '''Every slot read anywhere in `target`: by a bare `Para.Grab`, and by the
    grab side of a `ParaWrap`, which is where a boxed block and an operation
    fed from the tape both carry theirs.

    A term states its tape in one of two forms. A grab stands as a seed of its
    own while the algebra works on it, and it stands on the morphism it feeds
    once the block has been boxed or once `para_wrap.to_para_wrap` has been
    applied. Both are read here, so a notebook asks the same question of either.
    '''
    return frozenset(
        {grab.tape for grab in tutil.type_search(Para.Grab, target)}
        | _slots_of(entry for wrap in tutil.type_search(para_wrap.ParaWrap, target)
                    for entry in wrap.grabs))


def slots_dropped(target: fd.GeneralTerm) -> frozenset[Para.TapeSlot]:
    '''Every slot written anywhere in `target`, in either of the two forms
    `slots_grabbed` reads.'''
    return frozenset(
        {drop.tape for drop in tutil.type_search(Para.Drop, target)}
        | _slots_of(entry for wrap in tutil.type_search(para_wrap.ParaWrap, target)
                    for entry in wrap.drops))


def _slots_of(entries: Iterable[Para.SlotEntry]) -> frozenset[Para.TapeSlot]:
    return frozenset(Para.slot_of(entry) for entry in entries
                     if entry is not None)
