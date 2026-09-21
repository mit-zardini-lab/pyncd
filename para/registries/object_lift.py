# Claude Opus 5 (1M context), high effort.
'''Broadcasting a tape seed over a product of axes.

`construction_helpers.lift.morphism_object_lift` descends to the leaves of a
morphism and has a case for each of the four constructions and for
`cat.Broadcasted`. A `Para.Grab` and a `Para.Drop` are none of those. Each
carries the array it reads or writes in its own `size` field, so lifting one
prepends the lifted axes to that array and leaves the slot alone.

The slot is what says which value is read, and the value a loop's iteration or a
broadcast operation's index reads is one member of the slot, so broadcasting a
grab does not mint a slot per index. `para.processing.tape_members` indexes the
members a grab inside a repeated block touches, and
`obsidian/07-para/Para Block Operator.md` says what a grab inside a broadcast
block reads.
'''
from __future__ import annotations

import construction_helpers.lift as lift
import data_structure.Category as cat

import para.data_structure.Para as Para


@lift.register_object_lift(Para.Grab, Para.Drop)
def lift_tape_seed_over_axes[L, A: cat.Axis](
    seed: Para.ParaMorphism[L],
    lift_by: cat.ProdObject[A],
) -> Para.ParaMorphism[L]:
    '''`seed` reading or writing its slot at the array `lift_by` was prepended
    to, which is the array the operations around it carry once they are lifted
    over the same axes.'''
    return seed.reconstruct(
        size=lift.object_object_lift(seed.size, lift_by)[0])
