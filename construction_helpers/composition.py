from __future__ import annotations
from typing import Callable, Iterable, Iterator, overload
import data_structure.Term as fd
import data_structure.Category as cat
import construction_helpers.product as chp
import construction_helpers.lift as chl
import utilities.utilities as util
from enum import Enum

type AxialObject = cat.ProdObject[cat.RawAxis] | cat.ProdObject[cat.Array[cat.Datatype, cat.RawAxis]]

def get_axes(
    objs: AxialObject
):
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
):
    for lo, ro in zip(left, right):
        if isinstance(lo, cat.Array) and isinstance(ro, cat.Array):
            if lo.datatype != ro.datatype:
                raise TypeError(
                    f"Datatype mismatch in composition: "
                    f"codomain has {lo.datatype!r}, "
                    f"domain has {ro.datatype!r}"
                )
    left_axes = tuple(get_axes(left))
    right_axes = tuple(get_axes(right))
    if len(left_axes) != len(right_axes):
        raise ValueError("Cannot align axes of different lengths")
    for left_axis, right_axis in zip(left_axes, right_axes):
        ctx.append_iter((left_axis, right_axis))
    return ctx

def align_composed(
    *targets: cat.BroadcastedCategory | cat.StrideCategory,
    ctx: fd.Context | None = None
):
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
):
    if side == ExcessProductSide.TOP:
        return target[:amount]
    return target[-amount:]

def excess_product[L](
    left: cat.ProdObject[L],
    right: cat.ProdObject[L],
    side: ExcessProductSide = ExcessProductSide.TOP
) -> tuple[cat.ProdObject[L] | None, cat.ProdObject[L] | None]:
    excess_left = len(left) - len(right)
    if excess_left > 0:
        return (None, cat.ProdObject.from_iter(slice_side(left, excess_left, side)))
    elif excess_left < 0:
        return (cat.ProdObject.from_iter(slice_side(right, -excess_left, side)), None)
    return (None, None)

def _array_signature(arr) -> tuple | None:
    """Shape signature of an Array as (datatype_type, ((name, size),...)).
    Returns None if arr is not an Array."""
    if not isinstance(arr, cat.Array):
        return None
    return (
        type(arr.datatype),
        tuple(
            (
                ax.uid._name.body if ax.uid._name else None,
                getattr(ax._size, '_value', None),
            )
            for ax in arr.shape()
        )
    )


def _find_composition_reordering(
    left_cod: cat.ProdObject,
    right_dom: cat.ProdObject,
) -> tuple[int, ...] | None:
    """Find a permutation of right_dom indices putting matched elements FIRST,
    excess elements LAST — the layout excess_product with BOTTOM expects.

    Returns new_order tuple where new_order[new_pos] = orig_pos, or None
    if reordering is unnecessary or impossible.
    """
    if len(left_cod) >= len(right_dom):
        return None

    left_sigs = [_array_signature(obj) for obj in left_cod]
    if any(s is None for s in left_sigs):
        return None

    right_sigs = [_array_signature(obj) for obj in right_dom]

    matched: list[int] = []
    used: set[int] = set()
    for left_sig in left_sigs:
        for i, right_sig in enumerate(right_sigs):
            if i not in used and right_sig == left_sig:
                matched.append(i)
                used.add(i)
                break

    if len(matched) != len(left_cod):
        return None

    excess = [i for i in range(len(right_dom)) if i not in used]
    new_order = tuple(matched + excess)

    if new_order == tuple(range(len(right_dom))):
        return None  # Already correct order

    return new_order


def add_excess_lift[B: cat.Datatype](
    left: cat.BroadcastedCategory[B, cat.RawAxis],
    right: cat.BroadcastedCategory[B, cat.RawAxis],
):
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
    if isinstance(left, tuple):
        dom_length = max(left) + 1
        dom = tuple(
            util.iallequals(
                right.dom()[cod_idx]
                for cod_idx in left
                if cod_idx == idx
            )
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
    if len(left.cod()) < len(right.dom()):
        new_order = _find_composition_reordering(left.cod(), right.dom())
        if new_order is not None:
            # inv_order[orig_pos] = new_pos: maps from reordered inputs back to
            # right's original slot order, needed by right's internal einsums.
            inv_order = [0] * len(new_order)
            for new_pos, orig_pos in enumerate(new_order):
                inv_order[orig_pos] = new_pos
            reordered_dom = cat.ProdObject.from_iter(
                right.dom()[orig_pos] for orig_pos in new_order
            )
            rearrangement = cat.Rearrangement(
                mapping=tuple(inv_order),
                _dom=tuple(reordered_dom),
            )
            right = align_composed(rearrangement, right)
    excess_left, excess_right = excess_product(left.cod(), right.dom(), ExcessProductSide.BOTTOM)
    if excess_left is not None:
        left = chp.morphism_product((left, excess_left))
    elif excess_right is not None:
        right = chp.morphism_product((right, excess_right))
    return align_composed(left, right)

cat.Morphism.__matmul__ = composition  # type: ignore
cat.Morphism.__rmatmul__ = lambda x, y: composition(y, x)  # type: ignore