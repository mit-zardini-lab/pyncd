'''The Para construction, as two seed morphisms writing and reading a tape.

A tape is a set of named slots. `Drop<s>` writes the slot `s` and `Grab<s>`
reads it, and the pair is the only coupling between a forward pass and the
backward pass derived from it.

The two seeds are the identities of the usual triple presentation of a
parametrised morphism,

    Grab<s> : I -> A          written <A, id, 1>
    Drop<s> : A -> I          written <1, id, A>

so a parametrised `f : A x X -> Y`, written `<A, f, 1> : X -> Y`, is
`(Grab<s> * id) ; f`. Two seeds are therefore enough for the algebra, and every
pass over a `Para` has two cases rather than one per parametrised operator.

A loop variable is a value a repeated block overwrites on every iteration. It is
read and written by two subclasses of the seeds. `StreamGrab<s>` reads the value the
slot held when the iteration started, and `StreamDrop<s>` writes the value the next
iteration starts from. Within one iteration every operation reads wires, which is
the monoidal expression of the body, and the loop drop is the one operation whose
effect reaches past the body's end, so where it stands in the body carries no
meaning. The repetition is a property of the block, per `cat.BlockTag`, and the
two seeds state which of the block's values the repetition carries.

A repeated block also reads and writes one member of the tape per iteration, and
two further subclasses name the member. `LoopGrab<s>` reads the member of `s` that
the iteration `index` wrote, and `LoopDrop<s>` writes the member of `s` that the
iteration `index` holds. The index is a numeric over the counter of a repeated
block around the seed, which `cat.Block.template` names with `index_name` and
`nm.FreeNumeric.named` reads as a numeric. A drop and a grab that carry the same
counter therefore name the member one iteration writes and that same iteration
reads. An index that is an expression of the counter names the member of another
iteration, and the expansion path of a UNet reads the skip of the contraction path
at `N - 1 - i`.

A reduction across processors is exchanged in rounds, and two further subclasses
pass a value between processors, where the stream seeds load from and save to a tape
that persists across one processor's iterations. `ReductionDrop<s>` sends this
processor's partial to the processors that read it in the round, and
`ReductionGrab<s>` receives a partner's partial, so the slot names the value
exchanged rather than a place it is kept. The partial itself is carried round the
loop on its wire, and which partner a round reads is stated by the loop's repetition.

Both seeds are composition-neutral. Their domain or codomain is the empty
product, so a taped morphism has the same domain and codomain as the untaped one
and a tape threads through a `Block` without changing its type.

`ParaWrap.py` presents the general case, a body morphism carrying its grabs and
drops, and adds no mathematical value. It exists so that a tape arriving beside
an operator and a tape arriving at a box of its own are drawn differently.

`obsidian/07-para/Para Category.md` gives the construction and its reference,
and `obsidian/07-para/Training.md` says what the four writers of a slot are.
'''
from __future__ import annotations
from typing import Self, Iterable, Any
from dataclasses import dataclass, field
from enum import Enum
import itertools
import data_structure.Numeric as nm
import data_structure.Term as fd
import data_structure.Category as cat

@dataclass(frozen=True)
class TapeSlot(fd.UTerm): ...


# Slot names are `s0, s1, ...` in creation order. A UID is random per process,
# so a listing that printed one would stop being diffable between runs. A
# derivation resets the counter before it starts.
_slots = itertools.count()


def reset_slots() -> None:
    global _slots
    _slots = itertools.count()


def new_slot() -> TapeSlot:
    return slot_named(next(_slots)).capture(TapeSlot())


def slot_named(index: int) -> fd.DynamicName:
    '''The name of the slot numbered `index`: `s7` drawn, `slot_7` in code.'''
    return fd.DynamicName(f's{index}', code_form=f'slot_{index}')

@dataclass(frozen=True)
class ParaMorphism[L](cat.Morphism[L]):
    '''A tape operation on the slot `tape`, carrying an array of size `size`.'''
    tape: TapeSlot
    size: L

    # Declared without an annotation, so `@dataclass` leaves it alone and it
    # stays out of `__dataclass_fields__`, which `reconstruct` iterates.
    @property
    def name(self) -> fd.DynamicName | None:
        return self.tape.uid._name

@dataclass(frozen=True)
class Grab[L](ParaMorphism[L]):
    def dom(self) -> cat.ProdObject[L]: return cat.ProdObject()
    def cod(self) -> cat.ProdObject[L]: return cat.ProdObject((self.size,))

@dataclass(frozen=True)
class Drop[L](ParaMorphism[L]):
    def dom(self) -> cat.ProdObject[L]: return cat.ProdObject((self.size,))
    def cod(self) -> cat.ProdObject[L]: return cat.ProdObject()


@dataclass(frozen=True)
class StreamGrab[L](Grab[L]):
    '''A grab of a loop variable: the value the slot held when the iteration
    started, which the initializer dropped before the first iteration and the
    `StreamDrop` of the iteration before dropped after that.'''


@dataclass(frozen=True)
class StreamDrop[L](Drop[L]):
    '''A drop of a loop variable, read by the next iteration.

    Every grab of the slot in the same iteration reads the value the iteration
    started with, wherever the drop stands in the body.
    '''


@dataclass(frozen=True)
class LoopGrab[L](Grab[L]):
    '''A grab of the member of `tape` that the iteration `index` wrote.

    The index is a numeric over the counter of a repeated block around the grab.
    '''
    index: nm.Numeric


@dataclass(frozen=True)
class LoopDrop[L](Drop[L]):
    '''A drop onto the member of `tape` that the iteration `index` holds.

    Every `LoopGrab` of `tape` carrying the same index reads the member this
    drop wrote.
    '''
    index: nm.Numeric


@dataclass(frozen=True)
class ReductionGrab[L](Grab[L]):
    '''A grab receiving a partner processor's partial in one round of an exchange,
    the value the partner's `ReductionDrop` of the same slot sent in that round.
    Which partner is read is a property of the interconnect, which the loop's
    repetition, a `Topology`, names.'''


@dataclass(frozen=True)
class ReductionDrop[L](Drop[L]):
    '''A drop sending this processor's partial to the processors that read it in
    one round of an exchange, through their `ReductionGrab` of the same slot. The
    partial is also carried on its wire to the accumulator that folds a received
    partial into it.'''


@dataclass(frozen=True)
class ReductionSlot(fd.Term):
    '''A tape slot naming a value exchanged between processors, which is the entry
    a `ParaWrap` holds for a `ReductionGrab` on its grab side and for a
    `ReductionDrop` on its drop side.'''
    slot: TapeSlot


@dataclass(frozen=True)
class StreamSlot(fd.Term):
    '''A tape slot as a loop variable, which is the entry a `ParaWrap` holds for
    a `StreamGrab` on its grab side and for a `StreamDrop` on its drop side.'''
    slot: TapeSlot


@dataclass(frozen=True)
class LoopSlot(fd.Term):
    '''A tape slot at the iteration `index` of a repeated block, which is the
    entry a `ParaWrap` holds for a `LoopGrab` on its grab side and for a
    `LoopDrop` on its drop side.'''
    slot: TapeSlot
    index: nm.Numeric


type SlotEntry = None | TapeSlot | StreamSlot | LoopSlot | ReductionSlot
type NamedEntry = TapeSlot | StreamSlot | LoopSlot | ReductionSlot


def slot_of(entry: NamedEntry) -> TapeSlot:
    return entry if isinstance(entry, TapeSlot) else entry.slot


def index_of(entry: NamedEntry) -> nm.Numeric | None:
    '''The iteration `entry` names, and `None` for an entry naming no
    iteration.'''
    return entry.index if isinstance(entry, LoopSlot) else None


def entry_on_slot(entry: NamedEntry, slot: TapeSlot) -> NamedEntry:
    '''`entry` naming `slot` in place of the slot it named, which is what a pass
    that gives a slot a new name writes in the entry's place.'''
    return slot if isinstance(entry, TapeSlot) else entry.reconstruct(slot=slot)


def entry_of(operation: ParaMorphism) -> NamedEntry:
    '''The `ParaWrap` entry that stands for `operation`.'''
    if isinstance(operation, (StreamGrab, StreamDrop)):
        return StreamSlot(operation.tape)
    if isinstance(operation, (LoopGrab, LoopDrop)):
        return LoopSlot(operation.tape, operation.index)
    if isinstance(operation, (ReductionGrab, ReductionDrop)):
        return ReductionSlot(operation.tape)
    return operation.tape


def grab_of(entry: NamedEntry, size: L) -> Grab[L]:
    if isinstance(entry, StreamSlot):
        return StreamGrab(tape=entry.slot, size=size)
    if isinstance(entry, LoopSlot):
        return LoopGrab(tape=entry.slot, size=size, index=entry.index)
    if isinstance(entry, ReductionSlot):
        return ReductionGrab(tape=entry.slot, size=size)
    return Grab(tape=entry, size=size)


def drop_of(entry: NamedEntry, size: L) -> Drop[L]:
    if isinstance(entry, StreamSlot):
        return StreamDrop(tape=entry.slot, size=size)
    if isinstance(entry, LoopSlot):
        return LoopDrop(tape=entry.slot, size=size, index=entry.index)
    if isinstance(entry, ReductionSlot):
        return ReductionDrop(tape=entry.slot, size=size)
    return Drop(tape=entry, size=size)


type Para[L, M: cat.Morphism] = cat.ProdCategory[L, M | Grab[L] | Drop[L]]