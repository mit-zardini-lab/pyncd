'''The tangent functor, on objects.

The functor is defined on datatypes rather than on shapes, and it is partial.
`Reals` has a tangent. `Natural(n)`, which is an index, has none. A factor
whose datatype has no tangent is absent from the reverse pass. No zero
cotangent is written for it, and no cotangent passes through it.

The partiality settles the top-k case. `ArgTopK`'s codomain is all `Natural`,
so its reverse derivative has an empty domain and the operator does not appear
in the reverse graph. `obsidian/07-para/Selection and the Reverse Pass.md` is
the account.

A `ProdObject` therefore maps to a shorter one, and any `Rearrangement` over it
has to be reindexed. `positions` is that reindexing, from an old factor
position to a new one.
'''
from __future__ import annotations
from typing import Iterable

import data_structure.Category as cat


def datatype(target: cat.Datatype) -> cat.Datatype | None:
    '''The tangent datatype, or `None` where the datatype has no tangent.

    A `Natural` is an index and has no tangent. Every other datatype is taken
    to be a real-valued payload and is returned unchanged, which is exact while
    `Reals` and `Natural` are the only two datatypes.
    '''
    match target:
        case cat.Natural():
            return None
        case _:
            return target


def array[B: cat.Datatype, A: cat.Axis](
    target: cat.Array[B, A]
) -> cat.Array[B, A] | None:
    '''The same shape over the tangent datatype, or `None` where there is
    no tangent datatype.'''
    found = datatype(target.datatype)
    return None if found is None else target.reconstruct(datatype=found)


def obj[B: cat.Datatype, A: cat.Axis](
    target: Iterable[cat.Array[B, A]]
) -> cat.ProdObject[cat.Array[B, A]]:
    '''The tangent of a product object. A factor with no tangent is dropped, so
    the result may be shorter than the argument.'''
    return cat.ProdObject.from_iter(
        found for found in (array(item) for item in target)
        if found is not None)


def positions[B: cat.Datatype, A: cat.Axis](
    target: Iterable[cat.Array[B, A]]
) -> tuple[int | None, ...]:
    '''Old factor position to new factor position, `None` where the factor has
    no tangent and is dropped.'''
    found: list[int | None] = []
    kept = 0
    for item in target:
        if array(item) is None:
            found.append(None)
        else:
            found.append(kept)
            kept += 1
    return tuple(found)
