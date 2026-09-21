'''Checking the seeds of a tape slot indexed by the iteration of a repeated block.

Written by Claude Opus 5 (1M context), effort high.

    python para/validate_loop_seeds.py

`Para.LoopDrop` writes the member of a slot that one iteration of a repeated
block holds and `Para.LoopGrab` reads the member the index it carries names, per
`obsidian/07-para/Para Category.md`. Two fixtures stand here, and both are a
block repeated four times whose body drops a computed value onto the member the
block's counter names. In the first the grab carries that counter, so each
iteration reads what it wrote, which is a layer stack whose first layer drops the
compressed entries and whose later layers grab them. In the second the grab
carries `N - 1 - i`, so iteration `i` reads what iteration `N - 1 - i` wrote,
which is the expansion path of a UNet reading the skip the contraction path
dropped.

The checks are structural, because a UID is random per process and the text of a
listing is not stable across runs while the shapes are. `type_search` finds both
seeds, `Para.entry_of` and `Para.grab_of` round-trip a seed through the entry
that stands for it, `to_para_wrap` absorbs each seed onto the operation it
touches and holds a `Para.LoopSlot` carrying the index in its place, `tie_tapes`
refuses both seeds, `tape_members` reports one member per iteration of the block,
and `agent_display.listing` prints the index beside the slot's name.
'''
from __future__ import annotations

import construction_helpers as ch  # noqa: F401 - the operator overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd
import term_utilities.term_utilities as tutil

import agent_display as ad
import para.algebra.tie_tapes as tie_tapes
import para.data_structure.Para as Para
import para.data_structure.ParaWrap as para_wrap
import para.processing.tape_members as tape_members

REPETITION = 4
COUNTER = nm.FreeNumeric.named('i')
REVERSED_INDEX = nm.Integer(REPETITION - 1) - COUNTER
AXIS = cat.RawAxis.named('x')
ARRAY = cat.Array(cat.Reals(), (AXIS,))


def entries_slot() -> Para.TapeSlot:
    '''A slot named `e`, so that a listing of a fixture reads the same however
    many fixtures were built before it. `Para.new_slot` numbers a slot in
    creation order, which would number these two differently.'''
    return fd.DynamicName('e', code_form='entries').capture(Para.TapeSlot())


def indexed_layer_stack(grabbed_index: nm.Numeric) -> cat.Block:
    '''A block repeated `REPETITION` times whose body computes a value from its
    input, drops the value onto the member of its slot that `COUNTER` names,
    grabs the member `grabbed_index` names, and contracts the grabbed member
    with its input.'''
    identity = cat.ProdObject((ARRAY,)).identity()
    slot = entries_slot()
    body = (cat.Rearrangement(mapping=(0, 0), _dom=(ARRAY,))
            @ ((AXIS >> ops.Arithmetic.template(nm.E ** nm.x)) * identity)
            @ (Para.LoopDrop(tape=slot, size=ARRAY, index=COUNTER) * identity)
            @ (identity
               * Para.LoopGrab(tape=slot, size=ARRAY, index=grabbed_index))
            @ ops.Einops.template('x, x -> x'))
    return cat.Block.template(
        body, title='layer', repetition=REPETITION, index_name='i')


def check_seeds_are_found() -> None:
    '''`type_search` finds the indexed drop and the indexed grab of a fixture,
    each carrying the index it was written with, and finds no stream seed, which
    is what keeps the two families of subclasses apart.'''
    for grabbed_index in (COUNTER, REVERSED_INDEX):
        stack = indexed_layer_stack(grabbed_index)
        drops = tuple(tutil.type_search(Para.LoopDrop, stack))
        grabs = tuple(tutil.type_search(Para.LoopGrab, stack))
        assert len(drops) == 1, f'{len(drops)} indexed drops in the fixture'
        assert len(grabs) == 1, f'{len(grabs)} indexed grabs in the fixture'
        assert drops[0].index == COUNTER, f'the drop is at {drops[0].index}'
        assert grabs[0].index == grabbed_index, f'the grab is at {grabs[0].index}'
        assert drops[0].tape == grabs[0].tape, 'the two seeds name one slot'
        assert not tuple(tutil.type_search(Para.StreamGrab, stack)), \
            'an indexed grab is not a stream loop variable'
        assert not tuple(tutil.type_search(Para.StreamDrop, stack)), \
            'an indexed drop is not a stream loop variable'


def check_reversed_index_reads_another_iteration() -> None:
    '''The grab of the expansion path names the iteration the contraction path
    dropped at, which is `N - 1 - i` and is not the counter.'''
    grab, = tuple(tutil.type_search(
        Para.LoopGrab, indexed_layer_stack(REVERSED_INDEX)))
    assert grab.index == REVERSED_INDEX, f'the grab is at {grab.index}'
    assert grab.index != COUNTER, 'the reversed index is not the counter'
    assert grab.index.to_latex() == '3 - i', grab.index.to_latex()


def check_entry_rebuilds_its_seed() -> None:
    '''`Para.entry_of` returns a `Para.LoopSlot` carrying the slot and the index,
    and `Para.grab_of` and `Para.drop_of` rebuild the seed the entry stands
    for.'''
    slot = entries_slot()
    grab = Para.LoopGrab(tape=slot, size=ARRAY, index=REVERSED_INDEX)
    drop = Para.LoopDrop(tape=slot, size=ARRAY, index=COUNTER)
    for seed in (grab, drop):
        entry = Para.entry_of(seed)
        assert isinstance(entry, Para.LoopSlot), f'{type(entry).__name__} entry'
        assert Para.slot_of(entry) == slot, 'the entry names the seed\'s slot'
        assert Para.index_of(entry) == seed.index, 'the entry names the index'
    assert Para.grab_of(Para.entry_of(grab), ARRAY) == grab
    assert Para.drop_of(Para.entry_of(drop), ARRAY) == drop
    assert Para.index_of(slot) is None, 'a plain slot names no iteration'
    renamed = fd.DynamicName('f').capture(Para.TapeSlot())
    moved = Para.entry_on_slot(Para.entry_of(grab), renamed)
    assert Para.slot_of(moved) == renamed, 'a renamed entry names the new slot'
    assert Para.index_of(moved) == REVERSED_INDEX, 'a rename keeps the index'


def check_para_wrap_holds_the_index() -> None:
    '''`to_para_wrap` leaves no bare seed and writes a `Para.LoopSlot` carrying
    the index onto the operation each seed touches, and `to_base` writes the
    seeds back out.'''
    for grabbed_index in (COUNTER, REVERSED_INDEX):
        wrapped = para_wrap.to_para_wrap(indexed_layer_stack(grabbed_index))
        wraps = tuple(tutil.type_search(para_wrap.ParaWrap, wrapped))
        entries = tuple(entry for wrap in wraps
                        for entry in (*wrap.grabs, *wrap.drops)
                        if entry is not None)
        assert len(entries) == 2, f'{len(entries)} entries on the wraps'
        assert all(isinstance(entry, Para.LoopSlot) for entry in entries), \
            f'{[type(entry).__name__ for entry in entries]}'
        assert {Para.index_of(entry).to_latex() for entry in entries} \
            == {COUNTER.to_latex(), grabbed_index.to_latex()}, \
            'the wraps carry the indices the seeds carried'
        assert not tuple(tutil.type_search(Para.ParaMorphism, wrapped)), \
            'every seed was absorbed onto an operation'
        rewritten = tuple(
            seed for wrap in wraps
            for seed in tutil.type_search(Para.ParaMorphism, wrap.to_base()))
        assert len(rewritten) == 2, f'{len(rewritten)} seeds written back out'
        assert all(seed.index is not None for seed in rewritten), \
            'each seed written back out carries an index'


def check_tie_tapes_refuses_an_indexed_seed() -> None:
    '''`tie_tapes` refuses an indexed seed, because a repeated block is one
    morphism and no wire inside it carries the member of another iteration.'''
    for grabbed_index in (COUNTER, REVERSED_INDEX):
        try:
            tie_tapes.tie_tapes(indexed_layer_stack(grabbed_index))
        except ValueError:
            continue
        raise AssertionError(
            f'tie_tapes tied a slot grabbed at {grabbed_index.to_latex()}')


def check_tape_members_indexes_by_the_loop() -> None:
    '''`tape_members` reports one member of the slot per iteration of the block,
    indexed by the block's counter, and reports it as a plain member rather than
    a loop variable a stream carries across its iterations.'''
    members = tape_members.tape_members(indexed_layer_stack(COUNTER))
    assert len(members) == 1, f'{len(members)} members of one slot'
    member, = members
    assert member.indexed_code_form() == 'entries[i]', member.indexed_code_form()
    assert member.carried_by is None, 'an indexed member is not carried'
    assert member.grabs() == 1 and member.drops() == 1, \
        f'{member.grabs()} grabs and {member.drops()} drops'


def check_listing_prints_the_index() -> None:
    '''`agent_display.listing` prints the index beside the slot's name, on the
    bare seeds and on the wraps that absorbed them.'''
    stack = indexed_layer_stack(REVERSED_INDEX)
    bare = ad.listing(stack)
    assert 'LoopDrop<e[i]>' in bare, bare
    assert 'LoopGrab<e[3 - i]>' in bare, bare
    absorbed = ad.listing(para_wrap.to_para_wrap(stack))
    assert '<e[i]>' in absorbed, absorbed
    assert '<e[3 - i]>' in absorbed, absorbed


CHECKS = (
    check_seeds_are_found,
    check_reversed_index_reads_another_iteration,
    check_entry_rebuilds_its_seed,
    check_para_wrap_holds_the_index,
    check_tie_tapes_refuses_an_indexed_seed,
    check_tape_members_indexes_by_the_loop,
    check_listing_prints_the_index,
)


if __name__ == '__main__':
    print('==== the counter at the grab, each iteration reading what it wrote ====')
    print(ad.listing(indexed_layer_stack(COUNTER)))
    print('==== absorbed onto the operations they touch ====')
    print(ad.listing(para_wrap.to_para_wrap(indexed_layer_stack(COUNTER))))
    print('==== the reversed index at the grab, as a UNet skip ====')
    print(ad.listing(indexed_layer_stack(REVERSED_INDEX)))
    print('==== absorbed onto the operations they touch ====')
    print(ad.listing(para_wrap.to_para_wrap(indexed_layer_stack(REVERSED_INDEX))))
    for check in CHECKS:
        check()
        print(f'ok {check.__name__}')
    print('all indexed tape seed checks passed')
