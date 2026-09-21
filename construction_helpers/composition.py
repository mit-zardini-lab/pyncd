'''Sequential composition, `@`, with the axes of the two sides identified.

`f @ g` requires that the codomain of `f` and the domain of `g` be the same
objects. Two axes written with the same letter are two axes, with different UIDs
and different size symbols, so `align_axes` pairs them by position and records
each pair as an `fd.EqualityClass` in an `fd.Context`. Applying the context to
the `Composed` rewrites both sides into the canonical axis.

The account of alignment, and of the case `add_excess_lift` handles badly, is in
`obsidian/02-categories/Construction Helpers.md`.
'''
from __future__ import annotations
from typing import Callable, Iterable, Iterator, overload
import data_structure.Term as fd
import data_structure.Category as cat
import data_structure.Numeric as nm
import construction_helpers.product as chp
import construction_helpers.lift as chl
import utilities.utilities as util
from enum import Enum

type AxialObject = cat.ProdObject[cat.RawAxis] | cat.ProdObject[cat.Array[cat.Datatype, cat.RawAxis]]

def print_axis(target: cat.Axis):
    name = target.uid._name
    if name is None:
        return 'noname'
    return name.to_bodies()

def print_axial_object(target: AxialObject):
    return ' '.join(
        f'[{' '.join(print_axis(b) for b in a.shape())}]'
        if isinstance(a, cat.Array)
        else print_axis(a)
        for a in target
    )

def get_axes(
    objs: AxialObject
) -> Iterator[cat.Axis]:
    return (
        axis
        for obj in objs
        for axis in
        (obj.shape()
         if isinstance(obj, cat.Array)
         else (obj,))
    )

def align_axes(
        left: AxialObject,
        right: AxialObject,
        ctx: fd.Context,
) -> fd.Context:
    left_axes = tuple(get_axes(left))
    right_axes = tuple(get_axes(right))
    if len(left_axes) != len(right_axes):
        raise ValueError(f"Cannot align axes of different lengths. Tried to align: \n{print_axial_object(left)}\n{print_axial_object(right)}")
    for left_axis, right_axis in zip(left_axes, right_axes):
        axis_merge, size_merge = align_axis(left_axis, right_axis)
        ctx.append_bucket(axis_merge)
        if size_merge is not None:
            ctx.append_bucket(size_merge)
    return ctx

def align_axis(
    left: cat.Axis,
    right: cat.Axis,
) -> tuple[fd.EqualityClass, fd.EqualityClass | None]:
    '''Identify two axes, and separately their two sizes.

    A `RawAxis` carries nothing but a size, and any other axis carries the
    structure a later pass reads, so a `RawAxis` is given the lower priority and
    the merged class takes the other axis as its canonical member. The sizes merge
    per `size_class`.
    '''
    left_priority = 1 - isinstance(left, cat.RawAxis)
    right_priority = 1 - isinstance(right, cat.RawAxis)
    axis_class = fd.EqualityClass.template(left, priority=left_priority).merge(
        fd.EqualityClass.template(right, priority=right_priority))
    return axis_class, size_class(
        left.local_size(), right.local_size(), left_priority, right_priority)


def size_class(
    left: nm.Numeric,
    right: nm.Numeric,
    left_priority: int,
    right_priority: int,
) -> fd.EqualityClass | None:
    '''The class identifying the sizes of two aligned axes.

    Two size symbols merge as the axes do. A symbol beside a size written in other
    symbols, as `|a| |b|` is for a query axis sized by its entries and its compression
    ratio, is replaced by that size, which composition then carries to every array
    naming the symbol. Two sizes both written in other symbols are left as they are,
    because neither is a degree of freedom the class could rename.
    '''
    left_symbol = isinstance(left, fd.UTerm)
    right_symbol = isinstance(right, fd.UTerm)
    if left_symbol and right_symbol:
        return fd.EqualityClass.template(left, priority=left_priority).merge(
            fd.EqualityClass.template(right, priority=right_priority))
    if left_symbol:
        return fd.EqualityClass(_type=type(left), bucket={left.uid}, canonical=right,
                                priority=right_priority)
    if right_symbol:
        return fd.EqualityClass(_type=type(right), bucket={right.uid}, canonical=left,
                                priority=left_priority)
    return None

def align_composed(
    *targets: cat.BroadcastedCategory | cat.StrideCategory,
    ctx: fd.Context | None = None
) -> cat.Composed:
    '''Compose a chain, aligning each morphism's codomain with the next domain.

    A `Composed` among the targets is flattened into the chain first, so that
    each alignment runs between two adjacent leaves.
    '''
    targets_expanded = tuple(
        member for m in targets for member in (
            m.content if isinstance(m, cat.Composed) else (m,)
        )
    )
    ctx = ctx or fd.Context()
    for m in zip(targets_expanded[:-1], targets_expanded[1:]):
        ctx = align_axes(
            m[0].cod(),
            m[1].dom(),
            ctx
        )
    return ctx.apply(cat.Composed(
        targets_expanded
    ))

class ExcessProductSide(Enum):
    TOP = 'TOP'
    BOTTOM = 'BOTTOM'

def slice_side(
    target: tuple | cat.ProdObject,
    amount: int,
    side: ExcessProductSide,
) -> tuple | cat.ProdObject:
    if side == ExcessProductSide.TOP:
        return target[:amount]
    return target[-amount:]

def excess_product[L](
    left: cat.ProdObject[L],
    right: cat.ProdObject[L],
    side: ExcessProductSide = ExcessProductSide.TOP
) -> tuple[cat.ProdObject[L] | None, cat.ProdObject[L] | None]:
    '''The entries by which one of the two products is longer than the other.

    The excess is returned in the slot of the side that lacks it, so a longer
    `left` gives `(None, excess)` and a longer `right` gives `(excess, None)`.
    The caller pads that side with what it receives. `side` selects whether the
    excess is taken from the front or the back of the longer product.
    '''
    excess_left = len(left) - len(right)
    if excess_left > 0:
        return (None, cat.ProdObject.from_iter(slice_side(left, excess_left, side)))
    elif excess_left < 0:
        return (cat.ProdObject.from_iter(slice_side(right, -excess_left, side)), None)
    return (None, None)

def add_excess_lift[B: cat.Datatype](
    left: cat.BroadcastedCategory[B, cat.RawAxis],
    right: cat.BroadcastedCategory[B, cat.RawAxis],
) -> tuple[cat.BroadcastedCategory[B, cat.RawAxis],
           cat.BroadcastedCategory[B, cat.RawAxis]]:
    '''Lift whichever side has the fewer axes over the leading axes of the other.

    The comparison reads the first object of each side alone, and the lift is
    then applied to the whole of the shorter morphism. A product that mixes an
    operation of one rank with a wire of another therefore lifts the wire too,
    and `align_axes` raises. Lift the one operation before composing, with
    `chl.morphism_object_lift`.
    '''
    added_lift = excess_product(
        left.cod()[0].shape(), right.dom()[0].shape())
    if added_lift[0] is not None:
        left = chl.morphism_object_lift(left, added_lift[0])
    elif added_lift[1] is not None:
        right = chl.morphism_object_lift(right, added_lift[1])
    return (left, right)

@overload
def composition[B:cat.Datatype](
    left: cat.BroadcastedCategory[B, cat.RawAxis],
    right: cat.BroadcastedCategory[B, cat.RawAxis]
    ) -> cat.BroadcastedCategory[B, cat.RawAxis]: ...
@overload
def composition(
    left: cat.StrideCategory[cat.RawAxis],
    right: cat.StrideCategory[cat.RawAxis]
) -> cat.StrideCategory[cat.RawAxis]: ...

def composition( # type: ignore
        left,
        right):
    '''Compose two morphisms, or a bare mapping tuple against a morphism.

    A tuple on either side is read as a `Rearrangement`'s mapping and given the
    domain the other side implies. Excess axes are lifted and excess wires are
    carried through as identities, so the two boundaries meet.
    '''
    if isinstance(left, tuple):
        # The mapping sends each codomain position to a domain index, so the
        # domain is read from `right.dom()` at the first codomain position
        # naming each index. An index named twice needs no cross-check here,
        # because `align_composed` identifies every copy of it with the same
        # object of `right.dom()`.
        dom_length = max(left) + 1
        dom = tuple(
            right.dom()[left.index(idx)]
            for idx in range(dom_length)
        )
        left = cat.Rearrangement(
            mapping=left,
            _dom=dom
        )
    if isinstance(right, tuple):
        right = cat.Rearrangement(
            mapping=right,
            _dom = tuple(left.cod())
        )
    if isinstance(left.cod()[0], cat.Array):
        left, right = add_excess_lift(left, right)
    excess_left, excess_right = excess_product(left.cod(), right.dom(), ExcessProductSide.BOTTOM)
    if excess_left is not None:
        left = chp.morphism_product((left, excess_left))
    elif excess_right is not None:
        right = chp.morphism_product((right, excess_right))
    return align_composed(left, right)

cat.Morphism.__matmul__ = composition  # type: ignore
cat.Morphism.__rmatmul__ = lambda x, y: composition(y, x)  # type: ignore
