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

A model that generates text runs one pass per step, and a cache keeps arrays from
one pass for the passes after it. Two further subclasses read and write such a slot.
`CacheGrab<s>` reads the entries the slot holds for the tokens of the earlier
passes, and `CacheDrop<s>` appends the entries of the tokens of this pass. The grab
reads only the earlier tokens and the drop writes only the new ones, so the pair
states the memory traffic of a cache: `|K|` entries loaded and `|n|` stored in a pass
over `n` new tokens, where `K` is every earlier token or the last `|K|` of them that a
sliding window keeps. A later grab reads the last `|K|` entries alone, so a drop may
append only the last `|K|` entries of its pass. The state of a scan and the running
sums of linear attention are caches kept over one token, and their drop appends one
entry per pass. The loop over passes is outside the
expression, which is why the stream seeds cannot state a cache, and
`caching.registries.standard_expansions` writes `caching.data_structure.Caching` out
with these two.

A slot is outer or inner, according to whether the Para construction is applied
outside the lifting of an expression over a product of axes or inside it. An
`OuterTapeSlot` holds a value the expression reads from outside, such as the weight
of a learned linear map. Broadcasting the expression over an axis does not change
the array such a slot holds, so a grab of it lifted over `x` is the grab followed by
a repeat along `x`, and a drop of it lifted over `x` is a sum over `x` followed by
the drop. A plain `TapeSlot` is inner. It holds a value one index of the broadcast
reads or writes on its own, such as the uniform sample a dropout compares against,
a residual taped for the reverse pass, or the entries of a cache, and a seed of it
lifted over `x` reads or writes an array carrying `x`. Every seed class above reads
and writes a slot of either kind, and `para.registries.object_lift` states the two
lifts.

Both seeds are composition-neutral. Their domain or codomain is the empty
product, so a taped morphism has the same domain and codomain as the untaped one
and a tape threads through a `Block` without changing its type.

`ParaWrap.py` presents the general case, a body morphism carrying its grabs and
drops, and adds no mathematical value. It exists so that a tape arriving beside
an operator and a tape arriving at a box of its own are drawn differently.

`obsidian/07-para/Para Category.md` gives the construction and its reference,
`obsidian/07-para/Outer and Inner Tape Slots.md` the two kinds of slot, and
`obsidian/07-para/Training.md` says what the four writers of a slot are.
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
class TapeSlot(fd.UTerm):
    '''A slot of the tape. A slot of this class and of no subclass is an inner tape
    slot, which holds one array for every index of every axis an expression reading
    it is broadcast over.'''


@dataclass(frozen=True)
class OuterTapeSlot(TapeSlot):
    '''An outer tape slot, which holds one array however many axes an expression
    reading it is broadcast over. The weight of a learned linear map and the gain of
    a normalisation are held in outer slots, and so is the cotangent the reverse pass
    writes for either.'''


def new_slot_like(slot: TapeSlot) -> TapeSlot:
    '''A new unnamed slot, outer where `slot` is outer and inner where it is
    inner.'''
    return OuterTapeSlot() if isinstance(slot, OuterTapeSlot) else TapeSlot()


# Slot names are `s0, s1, ...` in creation order. A UID is random per process,
# so a listing that printed one would stop being diffable between runs. A
# derivation resets the counter before it starts.
_slots = itertools.count()


def reset_slots() -> None:
    global _slots
    _slots = itertools.count()


def new_slot() -> TapeSlot:
    '''An inner slot named `s<n>`, which is the name a residual taped for the
    reverse pass carries.'''
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
class CacheGrab[L](Grab[L]):
    '''A grab of the entries a cache holds for the tokens of the earlier passes, which
    the `CacheDrop` of the same slot appended in those passes. Its array holds an axis
    `K` in place of the tokens, and it loads the last `|K|` entries appended, with the
    universal unit at a position before the first append. `K` is every earlier token
    for a cache of the whole past, and a fixed count for a cache kept for a sliding
    window.'''


@dataclass(frozen=True)
class CacheDrop[L](Drop[L]):
    '''A drop appending the entries of the tokens of this pass to a cache, which the
    `CacheGrab` of the same slot reads in every later pass. The grab reads the last
    `|K|` entries appended, so a drop may append the last `|K|` entries of the pass
    alone, as the drop of a state kept over one token appends one entry.'''


@dataclass(frozen=True)
class CacheTapeSlot(fd.Term):
    '''A tape slot kept between the passes of generation, which is the entry a
    `ParaWrap` holds for a `CacheGrab` on its grab side and for a `CacheDrop` on its
    drop side.'''
    slot: TapeSlot


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


type NamedEntry = TapeSlot | StreamSlot | LoopSlot | ReductionSlot | CacheTapeSlot


@dataclass(frozen=True)
class KeptAndDropped(fd.Term):
    '''An entry of a `ParaWrap` for an operand or a result that stays on its wire
    and is also written onto the slot `dropped` names. It stands where a value is
    both passed on and saved: the tokens of a pass, which a concatenation reads
    and a cache appends, or a residual that the next operation reads and the
    reverse pass loads. `dropped` is the entry of the drop, so a
    `Para.CacheTapeSlot` stands for a `CacheDrop`.'''
    dropped: NamedEntry


type SlotEntry = (None | TapeSlot | StreamSlot | LoopSlot | ReductionSlot
                  | CacheTapeSlot | KeptAndDropped)


def slot_of(entry: NamedEntry) -> TapeSlot:
    return entry if isinstance(entry, TapeSlot) else entry.slot


def is_kept(entry: SlotEntry) -> bool:
    '''Whether the operand or result an entry of a `ParaWrap` stands for stays on
    its wire, which it does where the entry is `None` or a `KeptAndDropped`.'''
    return entry is None or isinstance(entry, KeptAndDropped)


def grabbed_entry(entry: SlotEntry) -> NamedEntry | None:
    '''The entry of the slot an operand comes off, and `None` for an operand
    that arrives on its wire.'''
    return None if is_kept(entry) else entry


def operand_dropped_entry(entry: SlotEntry) -> NamedEntry | None:
    '''The entry of the slot an operand is also written to, which a
    `KeptAndDropped` names, and `None` for every other operand.'''
    return entry.dropped if isinstance(entry, KeptAndDropped) else None


def result_dropped_entry(entry: SlotEntry) -> NamedEntry | None:
    '''The entry of the slot a result is written to: the entry a `KeptAndDropped`
    names, the entry itself where the result goes onto the tape alone, and `None`
    where the result leaves on its wire alone.'''
    return entry.dropped if isinstance(entry, KeptAndDropped) else entry


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
    if isinstance(operation, (CacheGrab, CacheDrop)):
        return CacheTapeSlot(operation.tape)
    return operation.tape


def grab_of(entry: NamedEntry, size: L) -> Grab[L]:
    if isinstance(entry, StreamSlot):
        return StreamGrab(tape=entry.slot, size=size)
    if isinstance(entry, LoopSlot):
        return LoopGrab(tape=entry.slot, size=size, index=entry.index)
    if isinstance(entry, ReductionSlot):
        return ReductionGrab(tape=entry.slot, size=size)
    if isinstance(entry, CacheTapeSlot):
        return CacheGrab(tape=entry.slot, size=size)
    return Grab(tape=entry, size=size)


def drop_of(entry: NamedEntry, size: L) -> Drop[L]:
    if isinstance(entry, StreamSlot):
        return StreamDrop(tape=entry.slot, size=size)
    if isinstance(entry, LoopSlot):
        return LoopDrop(tape=entry.slot, size=size, index=entry.index)
    if isinstance(entry, ReductionSlot):
        return ReductionDrop(tape=entry.slot, size=size)
    if isinstance(entry, CacheTapeSlot):
        return CacheDrop(tape=entry.slot, size=size)
    return Drop(tape=entry, size=size)


type Para[L, M: cat.Morphism] = cat.ProdCategory[L, M | Grab[L] | Drop[L]]