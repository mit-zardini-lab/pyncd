'''Reading the members of a tape, indexed by the loops around the operations that
touch them.

Written by Claude Fable 5.1, effort 80.

A tape slot has one name, and a loop repeats every operation inside it. A plain
`Grab` or `Drop` inside a loop reads or writes a different member of the tape on
every iteration, because the residual, the weight or the index of one iteration
is an array of its own, so the member it touches on iteration `l` of the loop
`layer` is the slot indexed by that loop, written `weight_G[layer]` in code and
`W_{G}[l]` in a diagram. A `StreamGrab` or a `StreamDrop` reads or writes the loop
variable the loop carries from one iteration to the next, per
`para.data_structure.Para`, which is one member for the whole loop, so it is
indexed by the loops outside that loop alone. A `LoopGrab` or a `LoopDrop` names
the iteration whose member it touches in an index of its own, and is indexed here
by the loops around it, as a plain grab and a plain drop are.

`tape_touches` walks a morphism and returns every grab and drop with the loops
and the blocks around it. `tape_members` groups the touches by slot and by index
path, so a weight grabbed three times inside one loop is one member.
`rename_tape_members` rebuilds every grab and drop with a name a caller derives
from its touch, which `notebooks/display/tape_naming.py` uses to draw each
member's indices onto its label. A loop's index is the name of its
`cat.BlockTag`, which `cat.Block.template` sets from `index_name`, and a loop
whose tag has no name is indexed by a letter chosen by its nesting depth.

`obsidian/01-foundations/Code Forms.md` states the code form of a member and how
a loop is named.
'''
from __future__ import annotations

import enum
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd
import graphs.data_structure.Hypergraph as hg
import graphs.processing.Hypergraph2Morphism as h2m
import para.data_structure.Contravariant as contravariant
import para.data_structure.MultiCategory as multi_category
import para.data_structure.Para as Para
import para.data_structure.ParaWrap as para_wrap

UNNAMED_LOOP_SYMBOLS = ('i', 'j', 'k')


@dataclass(frozen=True)
class LoopIndex:
    '''The index of one loop: the symbol it is drawn with, the identifier it is
    written with in code, and the number of iterations.'''
    symbol: str
    code_form: str
    repetition: nm.Numeric


def is_loop(block_tag: cat.BlockTag) -> bool:
    return block_tag.repetition != nm.Integer(1)


def loop_index(block_tag: cat.BlockTag, depth: int) -> LoopIndex:
    '''The index of the loop `block_tag` denotes, at nesting depth `depth`: its
    tag's name where the tag has one, and otherwise `i`, `j`, `k`, `i2` and so on
    by depth.'''
    name = block_tag.uid._name
    if name is None or not name.to_bodies():
        letter = UNNAMED_LOOP_SYMBOLS[depth % len(UNNAMED_LOOP_SYMBOLS)]
        round_number = depth // len(UNNAMED_LOOP_SYMBOLS)
        symbol = letter if round_number == 0 else f'{letter}{round_number + 1}'
        return LoopIndex(symbol=symbol, code_form=symbol,
                         repetition=block_tag.repetition)
    return LoopIndex(symbol=name.to_latex(), code_form=name.code_form_or_identifier(),
                     repetition=block_tag.repetition)


class TapeSide(enum.Enum):
    GRAB = 'grab'
    DROP = 'drop'


@dataclass(frozen=True)
class TapeTouch:
    '''One grab or drop, with the loops and the blocks around it, outermost
    first. `is_loop_variable` reports a `StreamGrab` or a `StreamDrop`, whose slot is
    carried across the innermost loop rather than indexed by it.'''
    slot: Para.TapeSlot
    side: TapeSide
    is_loop_variable: bool
    loops: fd.Prod[LoopIndex]
    blocks: fd.Prod[str]
    array: object

    def indices(self) -> fd.Prod[LoopIndex]:
        '''The loops whose iteration selects a different member of the tape.'''
        if self.is_loop_variable and self.loops:
            return self.loops[:-1]
        return self.loops

    def carried_by(self) -> LoopIndex | None:
        '''The loop a loop variable is carried across, and `None` for a plain
        grab or drop.'''
        if self.is_loop_variable and self.loops:
            return self.loops[-1]
        return None

    def slot_code_form(self) -> str:
        name = self.slot.uid._name
        if name is None:
            return f'slot_{self.slot.uid._id % 997}'
        return name.code_form_or_identifier()

    def indexed_code_form(self) -> str:
        '''The member in code: the slot's code form followed by one bracketed
        index per loop that selects a member, as `weight_G[layer]`.'''
        return self.slot_code_form() + ''.join(
            f'[{index.code_form}]' for index in self.indices())

    def indexed_latex(self) -> str:
        '''The member as drawn: the slot's name followed by one bracketed index
        symbol per loop that selects a member, as `W_{G}[l]`.'''
        name = self.slot.uid._name
        latex = name.to_latex() if name is not None else self.slot.uid.to_latex()
        return latex + ''.join(f'[{index.symbol}]' for index in self.indices())


@dataclass(frozen=True)
class TapeMember:
    '''One member of the tape: a slot at one index path, with every touch of it.'''
    slot: Para.TapeSlot
    indices: fd.Prod[LoopIndex]
    carried_by: LoopIndex | None
    touches: fd.Prod[TapeTouch]

    def name(self) -> str:
        name = self.slot.uid._name
        return name.to_bodies() if name is not None else f'?{self.slot.uid._id % 997}'

    def code_form(self) -> str | None:
        name = self.slot.uid._name
        return name.code_form if name is not None else None

    def indexed_code_form(self) -> str:
        return self.touches[0].indexed_code_form()

    def indexed_latex(self) -> str:
        return self.touches[0].indexed_latex()

    def grabs(self) -> int:
        return sum(touch.side is TapeSide.GRAB for touch in self.touches)

    def drops(self) -> int:
        return sum(touch.side is TapeSide.DROP for touch in self.touches)

    def block_paths(self) -> fd.Prod[fd.Prod[str]]:
        '''The distinct paths of block titles the member is touched under.'''
        return tuple(dict.fromkeys(touch.blocks for touch in self.touches))

    def array(self) -> object:
        return self.touches[0].array


type Naming = Callable[[TapeTouch], fd.DynamicName | None]


# ==========================================================================
# The walk.
# ==========================================================================
def _block_title(block_tag: cat.BlockTag) -> str | None:
    aesthetics = block_tag.aesthetics
    return aesthetics.title if aesthetics is not None else None


def _inside_block(
    block_tag: cat.BlockTag, loops: fd.Prod[LoopIndex], blocks: fd.Prod[str],
) -> tuple[fd.Prod[LoopIndex], fd.Prod[str]]:
    title = _block_title(block_tag)
    inner_blocks = (*blocks, title) if title else blocks
    inner_loops = (*loops, loop_index(block_tag, len(loops))) if is_loop(block_tag) else loops
    return inner_loops, inner_blocks


def _entry_touch(
    entry: Para.NamedEntry, side: TapeSide, array: object,
    loops: fd.Prod[LoopIndex], blocks: fd.Prod[str],
) -> TapeTouch:
    return TapeTouch(
        slot=Para.slot_of(entry), side=side,
        is_loop_variable=isinstance(entry, Para.StreamSlot),
        loops=loops, blocks=blocks, array=array)


def _touches(
    target: object, loops: fd.Prod[LoopIndex], blocks: fd.Prod[str],
) -> Iterator[TapeTouch]:
    match target:
        case cat.Composed(content=parts) | cat.ProductOfMorphisms(content=parts):
            for part in parts:
                yield from _touches(part, loops, blocks)
        case cat.Block(body=body, block_tag=block_tag):
            yield from _touches(body, *_inside_block(block_tag, loops, blocks))
        case Para.Grab(tape=slot, size=array):
            yield TapeTouch(slot, TapeSide.GRAB,
                            isinstance(target, (Para.StreamGrab, Para.ReductionGrab)),
                            loops, blocks, array)
        case Para.Drop(tape=slot, size=array):
            yield TapeTouch(slot, TapeSide.DROP,
                            isinstance(target, (Para.StreamDrop, Para.ReductionDrop)),
                            loops, blocks, array)
        case para_wrap.ParaWrap(body=body, grabs=grabs, drops=drops):
            for entry, array in zip(grabs, body.dom()):
                if entry is not None:
                    yield _entry_touch(entry, TapeSide.GRAB, array, loops, blocks)
            for entry, array in zip(drops, body.cod()):
                if entry is not None:
                    yield _entry_touch(entry, TapeSide.DROP, array, loops, blocks)
            yield from _touches(body, loops, blocks)
        case cat.Broadcasted(operator=ops.BlockOperator(block=block)):
            yield from _touches(block, loops, blocks)


def as_rows(term: object) -> tuple[cat.Morphism, ...]:
    '''The morphisms `term` holds: one per row of a `MultiCategory`, the
    morphism of a hypergraph, and a morphism itself.'''
    match term:
        case multi_category.MultiCategory():
            return tuple(contravariant.covariant_body(row) for row in term)
        case hg.Hypergraph():
            return (h2m.hypergraph_to_morphism(term),)
        case _:
            return (term,)  # type: ignore[return-value]


def tape_touches(term: object) -> fd.Prod[TapeTouch]:
    '''Every grab and drop of `term`, in the order they stand, each with the
    loops and the blocks around it.'''
    return tuple(touch for row in as_rows(term) for touch in _touches(row, (), ()))


def tape_members(term: object) -> fd.Prod[TapeMember]:
    '''The members of the tape `term` touches, one per slot and index path, in
    the order of their first touch.'''
    grouped: dict[tuple[int, tuple[str, ...], str | None], list[TapeTouch]] = {}
    for touch in tape_touches(term):
        carried = touch.carried_by()
        key = (touch.slot.uid._id,
               tuple(index.code_form for index in touch.indices()),
               carried.code_form if carried is not None else None)
        grouped.setdefault(key, []).append(touch)
    return tuple(
        TapeMember(slot=touches[0].slot, indices=touches[0].indices(),
                   carried_by=touches[0].carried_by(), touches=tuple(touches))
        for touches in grouped.values())


def members_text(members: fd.Prod[TapeMember]) -> str:
    '''One line per member: its code form with its indices, the loop it is
    carried across where it is a loop variable, how many grabs and drops touch
    it, and the blocks it is touched under.'''
    lines = []
    for member in members:
        carried = (f' carried across {member.carried_by.code_form}'
                   if member.carried_by is not None else '')
        paths = '; '.join(' > '.join(path) for path in member.block_paths())
        lines.append(
            f'{member.indexed_code_form()}{carried}: '
            f'{member.grabs()} grabs, {member.drops()} drops'
            + (f', under {paths}' if paths else ''))
    return '\n'.join(lines)


# ==========================================================================
# Renaming.
# ==========================================================================
def _renamed_entry(
    entry: Para.SlotEntry, side: TapeSide, array: object,
    loops: fd.Prod[LoopIndex], blocks: fd.Prod[str], naming: Naming,
) -> Para.SlotEntry:
    if entry is None:
        return None
    new_name = naming(_entry_touch(entry, side, array, loops, blocks))
    if new_name is None:
        return entry
    return Para.entry_on_slot(entry, new_name.capture(Para.slot_of(entry)))


def _renamed(
    target: object, loops: fd.Prod[LoopIndex], blocks: fd.Prod[str], naming: Naming,
) -> object:
    match target:
        case cat.Composed(content=parts) | cat.ProductOfMorphisms(content=parts):
            rebuilt = tuple(_renamed(part, loops, blocks, naming) for part in parts)
            if all(new is old for new, old in zip(rebuilt, parts)):
                return target
            return target.reconstruct(content=rebuilt)
        case cat.Block(body=body, block_tag=block_tag):
            inner = _renamed(body, *_inside_block(block_tag, loops, blocks), naming)
            return target if inner is body else target.reconstruct(body=inner)
        case Para.Grab(tape=slot, size=array) | Para.Drop(tape=slot, size=array):
            side = TapeSide.GRAB if isinstance(target, Para.Grab) else TapeSide.DROP
            is_loop_variable = isinstance(
                target, (Para.StreamGrab, Para.StreamDrop, Para.ReductionGrab, Para.ReductionDrop))
            new_name = naming(TapeTouch(slot, side, is_loop_variable, loops, blocks, array))
            if new_name is None:
                return target
            return target.reconstruct(tape=new_name.capture(slot))
        case para_wrap.ParaWrap(body=body, grabs=grabs, drops=drops):
            new_grabs = tuple(
                _renamed_entry(entry, TapeSide.GRAB, array, loops, blocks, naming)
                for entry, array in zip(grabs, body.dom()))
            new_drops = tuple(
                _renamed_entry(entry, TapeSide.DROP, array, loops, blocks, naming)
                for entry, array in zip(drops, body.cod()))
            new_body = _renamed(body, loops, blocks, naming)
            if new_grabs == grabs and new_drops == drops and new_body is body:
                return target
            return target.reconstruct(body=new_body, grabs=new_grabs, drops=new_drops)
        case cat.Broadcasted(operator=ops.BlockOperator(block=block) as operator):
            new_block = _renamed(block, loops, blocks, naming)
            if new_block is block:
                return target
            return target.reconstruct(operator=operator.reconstruct(block=new_block))
        case _:
            return target


def rename_tape_members[T](term: T, naming: Naming) -> T:
    '''`term` with every grab and drop rebuilt on a slot named by `naming` from
    its touch, and left as it stands where `naming` returns `None`. A slot's uid
    keeps its id, so the renamed slot is the slot with a new label. A
    `MultiCategory` is renamed row by row and a hypergraph is converted to a
    morphism first.'''
    match term:
        case multi_category.MultiCategory():
            def rename_row(row):
                renamed = _renamed(contravariant.covariant_body(row), (), (), naming)
                if isinstance(row, contravariant.Contravariant):
                    return contravariant.Contravariant(renamed)
                return renamed
            return type(term).from_iter(rename_row(row) for row in term)  # type: ignore[return-value]
        case hg.Hypergraph():
            return _renamed(h2m.hypergraph_to_morphism(term), (), (), naming)  # type: ignore[return-value]
        case _:
            return _renamed(term, (), (), naming)  # type: ignore[return-value]
