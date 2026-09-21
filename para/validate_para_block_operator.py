# Claude Opus 5 (1M context), high effort.
'''Checking that a boxed block carries its tape at its ports and broadcasts it.

The block below reads a gain from one slot, adds it to its own operand and to an
operand it shares with every index of the degree, and saves the total to a second
slot. One operand carries the head axis and the other does not, so the reindexings
of the broadcast form are `((0,), ())` and the shared operand is read through a
repeat.

    python para/validate_para_block_operator.py

Five things are checked. The grab becomes the box's leading operand and the drop
its trailing result, the block on the operator keeps the two seeds so that the
sub-diagram beside the box can draw them, the wrap over the box carries the two
slots at those positions and has the apparent domain and codomain of the block,
the slots are found through the wrapped box as through the block, and the arrays
the two ports carry gain the degree axis once the box is broadcast and
expanded.

A sixth concerns a plain box that a `ParaWrap` tapes, which is what
`para_wrap.to_para_wrap` leaves where a grab standing beside a box feeds that box
alone. `box_with_wrapped_tapes_as_seeds` rebuilds it as a `ParaBlockOperator` whose
block holds the tape as seeds, with the grabbed operand leading, the dropped result
trailing, the domain and codomain of the wrap unchanged, a tag of its own on the
block, and the ports `expose_tape_as_ports` derives from that block agreeing with
the ports of the box. The same holds where the box is broadcast over a degree, and
the seeds inside then carry the arrays of the block while the ports carry the
degree.

`obsidian/07-para/Para Block Operator.md` holds the mathematics.
'''
from __future__ import annotations
import sys

import construction_helpers as ch  # noqa: F401 - operator overloads
import construction_helpers.lift as lift
import data_structure.Category as cat
import data_structure.Operators as ops
import term_utilities.term_utilities as tutil

import algebra.reindexing_absorption as reindexing_absorption
import algebra.discovering_broadcasts as discovering_broadcasts
import para.data_structure.Para as Para
import para.data_structure.ParaBlockOperator as ParaBlockOperator
import para.data_structure.ParaWrap as para_wrap

R = cat.Reals()
HEAD = cat.RawAxis.named('h')
TOKEN = cat.RawAxis.named('x')
CHANNEL = cat.RawAxis.named('d')
PER_HEAD = cat.Array(R, (TOKEN, CHANNEL))
GAIN_SLOT = Para.new_slot()
TOTAL_SLOT = Para.new_slot()


def hold(array: cat.Array) -> cat.Rearrangement:
    return cat.ProdObject((array,)).identity()


def route(mapping: tuple[int, ...],
          arrays: tuple[cat.Array, ...]) -> cat.Rearrangement:
    return cat.Rearrangement(mapping, arrays)


def over(axes: tuple[cat.RawAxis, ...],
         morphism: cat.BroadcastedCategory) -> cat.BroadcastedCategory:
    return lift.morphism_object_lift(morphism, cat.ProdObject(axes))


def layer() -> ParaBlockOperator.ParaBBlock:
    '''A body that grabs a gain, adds it to its operand and to the operand every
    index of the degree shares, and drops the total.'''
    return cat.Block.template(
        (hold(PER_HEAD) * hold(PER_HEAD)
         * Para.Grab(tape=GAIN_SLOT, size=PER_HEAD))
        @ route((0, 2, 1), (PER_HEAD, PER_HEAD, PER_HEAD))
        @ (over((TOKEN, CHANNEL), ops.AdditionOp.template()) * hold(PER_HEAD))
        @ over((TOKEN, CHANNEL), ops.AdditionOp.template())
        @ route((0, 0), (PER_HEAD,))
        @ (hold(PER_HEAD) * Para.Drop(tape=TOTAL_SLOT, size=PER_HEAD)),
        title='Layer', fill_color='white')


def plain_layer() -> cat.Block:
    '''The body of `layer` with the gain arriving on its second operand and the
    total leaving on its second result, so it holds no seed.'''
    return cat.Block.template(
        (hold(PER_HEAD) * hold(PER_HEAD) * hold(PER_HEAD))
        @ route((0, 1, 2), (PER_HEAD, PER_HEAD, PER_HEAD))
        @ (over((TOKEN, CHANNEL), ops.AdditionOp.template()) * hold(PER_HEAD))
        @ over((TOKEN, CHANNEL), ops.AdditionOp.template())
        @ route((0, 0), (PER_HEAD,)),
        title='Plain layer', fill_color='white')


def taped_plain_box(
    box: cat.Broadcasted,
) -> para_wrap.ParaWrap:
    '''`box` with its second operand grabbed and its second result dropped, as
    `para_wrap.to_para_wrap` writes a grab and a drop standing beside a box.'''
    return para_wrap.ParaWrap(
        body=box, grabs=(None, GAIN_SLOT, None), drops=(None, TOTAL_SLOT))


def seeded_cases() -> dict[str, bool]:
    plain = plain_layer()
    per_head_and_head = cat.Array(R, (HEAD, TOKEN, CHANNEL))
    taped = taped_plain_box(ops.BlockOperator.template(plain, 'P'))
    seeded = ParaBlockOperator.box_with_wrapped_tapes_as_seeds(taped)
    seeded_box = ParaBlockOperator.bare_box_of(seeded)
    exposed = ParaBlockOperator.expose_tape_as_ports(seeded_box.operator.block)
    taped_broadcast = taped_plain_box(
        discovering_broadcasts.broadcast_block_over_axes(
            plain, (HEAD,), ((0,), (0,), ()), 'P'))
    seeded_broadcast = ParaBlockOperator.box_with_wrapped_tapes_as_seeds(
        taped_broadcast)
    broadcast_box = ParaBlockOperator.bare_box_of(seeded_broadcast)
    return {
        'a taped plain box becomes a ParaBlockOperator':
            isinstance(seeded_box.operator, ParaBlockOperator.ParaBlockOperator),
        'the rebuilt wrap keeps the domain and codomain of the wrap it replaces':
            tuple(seeded.dom()) == tuple(taped.dom())
            and tuple(seeded.cod()) == tuple(taped.cod()),
        'the rebuilt wrap names the slots at the leading operand and trailing result':
            seeded.grabs == (GAIN_SLOT, None, None)
            and seeded.drops == (None, TOTAL_SLOT),
        'the block of the rebuilt box holds the tape as seeds':
            tuple(ParaBlockOperator.tape_seeds_of(seeded_box.operator.block))
            == (Para.Grab(tape=GAIN_SLOT, size=PER_HEAD),
                Para.Drop(tape=TOTAL_SLOT, size=PER_HEAD)),
        'the ports derived from the seeded block are the ports of the rebuilt box':
            exposed.grabs == seeded_box.operator.grabs
            and exposed.drops == seeded_box.operator.drops
            and tuple(exposed.block.dom()) == tuple(seeded_box.dom())
            and tuple(exposed.block.cod()) == tuple(seeded_box.cod()),
        'the seeded block takes a tag of its own and keeps its aesthetics':
            seeded_box.operator.block.block_tag.uid != plain.block_tag.uid
            and seeded_box.operator.block.block_tag.aesthetics
            == plain.block_tag.aesthetics,
        'rebuilding the same wrap twice gives the same tag':
            ParaBlockOperator.box_with_wrapped_tapes_as_seeds(taped) == seeded,
        'a broadcast box keeps the degree at its ports and not on its seeds':
            tuple(broadcast_box.dom())
            == (per_head_and_head, per_head_and_head, PER_HEAD)
            and tuple(grab.size for grab in broadcast_box.operator.grabs)
            == (PER_HEAD,)
            and tuple(seeded_broadcast.dom()) == tuple(taped_broadcast.dom()),
    }


def cases() -> dict[str, bool]:
    block = layer()
    wrapped = ParaBlockOperator.ParaBlockOperator.template(block, 'L')
    box = ParaBlockOperator.bare_box_of(wrapped)
    broadcast = ParaBlockOperator.broadcast_para_block_over_axes(
        block, (HEAD,), ((0,), ()), 'L')
    broadcast_box = ParaBlockOperator.bare_box_of(broadcast)
    expansion = broadcast_box.operator.expand(broadcast_box)
    per_head_and_head = cat.Array(R, (HEAD, TOKEN, CHANNEL))
    repeats = [operation
               for operation in discovering_broadcasts.operations_of(expansion)
               if reindexing_absorption.is_node(operation)]
    return {
        'the wrapped box has the apparent domain and codomain of the block':
            tuple(wrapped.dom()) == tuple(block.dom())
            and tuple(wrapped.cod()) == tuple(block.cod()),
        'the box reads the grabbed array first and writes the dropped one last':
            tuple(box.dom()) == (PER_HEAD, PER_HEAD, PER_HEAD)
            and tuple(box.cod()) == (PER_HEAD, PER_HEAD),
        'the wrap names the two slots at those positions':
            wrapped.grabs == (GAIN_SLOT, None, None)
            and wrapped.drops == (None, TOTAL_SLOT),
        'the operator records the one grab and the one drop of its body':
            tuple(grab.tape for grab in box.operator.grabs) == (GAIN_SLOT,)
            and tuple(drop.tape for drop in box.operator.drops) == (TOTAL_SLOT,),
        'the block on the operator keeps its seeds, for the sub-diagram':
            tuple(ParaBlockOperator.tape_seeds_of(box.operator.block))
            == (Para.Grab(tape=GAIN_SLOT, size=PER_HEAD),
                Para.Drop(tape=TOTAL_SLOT, size=PER_HEAD)),
        'the ported body the box expands holds no tape seed':
            not tuple(ParaBlockOperator.tape_seeds_of(
                ParaBlockOperator.expose_tape_as_ports(box.operator.block).block))
            and not tuple(ParaBlockOperator.tape_seeds_of(expansion)),
        'the wrap writes back out as the grab, the box and the drop':
            tuple(ParaBlockOperator.tape_seeds_of(wrapped.to_base()))
            == (Para.Grab(tape=GAIN_SLOT, size=PER_HEAD),
                Para.Drop(tape=TOTAL_SLOT, size=PER_HEAD)),
        'the slots found through the wrapped box are those of the block':
            ParaBlockOperator.slots_grabbed(wrapped)
            == ParaBlockOperator.slots_grabbed(block) == frozenset({GAIN_SLOT})
            and ParaBlockOperator.slots_dropped(wrapped)
            == ParaBlockOperator.slots_dropped(block)
            == frozenset({TOTAL_SLOT}),
        'the broadcast reads the first operand per head and shares the second':
            tuple(broadcast.dom()) == (per_head_and_head, PER_HEAD)
            and tuple(broadcast.cod()) == (per_head_and_head,),
        'the broadcast reads the grabbed array at every index of the degree':
            tuple(broadcast_box.dom())
            == (per_head_and_head, per_head_and_head, PER_HEAD),
        'the broadcast grab and drop carry the head axis':
            tuple(grab.size for grab
                  in ParaBlockOperator.broadcast_grabs(broadcast))
            == (per_head_and_head,)
            and tuple(drop.size for drop
                      in ParaBlockOperator.broadcast_drops(broadcast))
            == (per_head_and_head,),
        'the broadcast grab and drop keep their slots':
            tuple(grab.tape for grab
                  in ParaBlockOperator.broadcast_grabs(broadcast)) == (GAIN_SLOT,)
            and tuple(drop.tape for drop
                      in ParaBlockOperator.broadcast_drops(broadcast))
            == (TOTAL_SLOT,),
        'the expansion keeps the domain and codomain of the box':
            tuple(expansion.dom()) == tuple(broadcast_box.dom())
            and tuple(expansion.cod()) == tuple(broadcast_box.cod()),
        'the expansion repeats the shared operand along the head':
            len(repeats) == 1,
        'lifting a bare grab prepends the axes to its array':
            lift.morphism_object_lift(
                Para.Grab(tape=GAIN_SLOT, size=PER_HEAD),
                cat.ProdObject((HEAD,))).size == per_head_and_head,
    }


if __name__ == '__main__':
    results = {**cases(), **seeded_cases()}
    for name, passed in results.items():
        print(f'  {"ok  " if passed else "FAIL"}  {name}')
    sys.exit(0 if all(results.values()) else 1)
