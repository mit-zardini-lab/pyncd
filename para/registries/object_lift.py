# Claude Opus 5 (1M context), high effort.
# Outer slots added by Claude Opus 5.5 (1M context), effort 40, on 2026-09-26.
'''Broadcasting a tape seed over a product of axes.

`construction_helpers.lift.morphism_object_lift` descends to the leaves of a
morphism and has a case for each of the four constructions and for
`cat.Broadcasted`. A `Para.Grab` and a `Para.Drop` are none of those. Each
carries the array it reads or writes in its own `size` field, and the slot it
reads or writes says how that array is lifted.

A seed of an inner slot reads or writes one array for every index of the lifted
axes, so lifting it prepends the lifted axes to its array and leaves the slot
alone. The slot is what says which value is read, and the value a loop's
iteration or a broadcast operation's index reads is one member of the slot, so
broadcasting a grab does not mint a slot per index.
`para.processing.tape_members` indexes the members a grab inside a repeated
block touches, and `obsidian/07-para/Para Block Operator.md` says what a grab
inside a broadcast block reads.

A seed of an outer slot reads or writes one array whatever the lifted axes are.
A grab lifted over `x` is therefore the grab of the same array followed by a
repeat along `x`, which `algebra.reindexing_absorption` folds into the
operations that read it. A drop lifted over `x` sums over `x` and then drops the
sum. The sum is the transpose of the repeat, and the reverse pass writes the
cotangent of a grabbed parameter into an outer slot, so the lift of the reverse
of a grab is the reverse of the lifted grab.
`obsidian/07-para/Outer and Inner Tape Slots.md` states the two kinds.
'''
from __future__ import annotations

import algebra.einops_simplification as einops_simplification
import construction_helpers.lift as lift
import construction_helpers.simple_helper as chsh
import data_structure.Category as cat

import para.data_structure.Para as Para


@lift.register_object_lift(Para.Grab, Para.Drop)
def lift_tape_seed_over_axes[B: cat.Datatype, A: cat.Axis](
    seed: Para.ParaMorphism[cat.Array[B, A]],
    lift_by: cat.ProdObject[A],
) -> cat.ProdCategory[cat.Array[B, A], cat.Morphism]:
    '''`seed` read at every index of `lift_by`: the seed of an inner slot reading
    or writing the lifted array, and the seed of an outer slot reading or writing
    its own array beside a repeat or a sum along `lift_by`.'''
    if not isinstance(seed.tape, Para.OuterTapeSlot):
        return seed.reconstruct(
            size=lift.object_object_lift(seed.size, lift_by)[0])
    if isinstance(seed, Para.Grab):
        return chsh.make_composed(seed, repeat_along(seed.size, lift_by))
    return chsh.make_composed(transposed_repeat_along(seed.size, lift_by), seed)


def repeat_along[B: cat.Datatype, A: cat.Axis](
    array: cat.Array[B, A],
    lift_by: cat.ProdObject[A],
) -> cat.Broadcasted[B, A]:
    '''The `View` from `array` to `array` lifted over `lift_by`, reading `array` at
    every index of `lift_by`.'''
    kept, lifted = _positional_variables(array, lift_by)
    return einops_simplification.einsum(
        (kept,), (*lifted, *kept), array.datatype)


def transposed_repeat_along[B: cat.Datatype, A: cat.Axis](
    array: cat.Array[B, A],
    lift_by: cat.ProdObject[A],
) -> cat.Broadcasted[B, A]:
    '''The `Einops` from `array` lifted over `lift_by` to `array`, summing over
    every index of `lift_by`.'''
    kept, lifted = _positional_variables(array, lift_by)
    return einops_simplification.einsum(
        ((*lifted, *kept),), kept, array.datatype)


def _positional_variables[A: cat.Axis](
    array: cat.Array,
    lift_by: cat.ProdObject[A],
) -> tuple[tuple[einops_simplification.IndexVariable, ...],
           tuple[einops_simplification.IndexVariable, ...]]:
    '''One index variable per position of `array` and one per position of
    `lift_by`, labelled by position, so a lifted axis that `array` also carries is
    read as a second index.'''
    return (
        einops_simplification.positional_shape(array),
        tuple(einops_simplification.IndexVariable(('lifted', k), axis)
              for k, axis in enumerate(lift_by)))
