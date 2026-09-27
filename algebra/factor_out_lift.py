# Claude Opus 5.5 (1M context), effort 40.
'''Factoring the axes a `Broadcasted` is lifted over out of it.

`construction_helpers.lift.broadcasted_stride_lift` lifts a `Broadcasted` over a
product of axes. It puts one `WeaveMode.TILED` entry per axis in front of every
weave, and it puts the identity on those axes in front of every reindexing.
`factor_out_lift` undoes the lift. It counts the leading positions of the degree
that every weave tiles at its own leading positions and that every reindexing
carries straight through, which means each such position reads the same position
of the operand and no other position of the operand reads it. It returns the axes
at those positions beside the `Broadcasted` with those positions removed. Lifting
the second over the first gives back the original.

An operator of **Br** never reads the index of its degree, so the operator
computed at one index of the factored axes is the whole of what the original
computes at every index. An inspection box therefore writes out the factored
operation, and the figure around it states the broadcast.
`notebooks/display/expand_with_parameters.py` does so. A parameter grabbed from a
`Para.OuterTapeSlot` is read once whatever the operation is lifted over, so the
expansion the box draws, lifted over the factored axes, is the expansion of the
operation the figure holds. `obsidian/07-para/Outer and Inner Tape Slots.md` states
the argument.

A reindexing is read by its structure. A `cat.Rearrangement` carries a leading
position straight through when its mapping sends the position to itself and no
later entry of the mapping reads it. A product carries the positions its leading
factors carry, and a `StrideMorphism` those whose row is the unit row of the
position with no shift and whose column holds no other stride. A reindexing of any
other form carries none, so the count is at most the true count and never more.
'''
from __future__ import annotations

from dataclasses import dataclass

import construction_helpers.product as chp
import data_structure.Category as cat
import data_structure.Numeric as nm


@dataclass(frozen=True)
class FactoredLift[B: cat.Datatype, A: cat.Axis]:
    '''`base` lifted over `lifted_over` is the `Broadcasted` the two were factored
    from.'''
    lifted_over: cat.ProdObject[A]
    base: cat.Broadcasted[B, A]


def factor_out_lift[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A],
) -> FactoredLift[B, A]:
    '''`target` as the lift of a `Broadcasted` over the leading positions of its
    degree, taking every leading position that can be factored out.'''
    count = lifted_position_count(target)
    if count == 0:
        return FactoredLift(lifted_over=cat.ProdObject(), base=target)
    degree = tuple(target.degree())
    return FactoredLift(
        lifted_over=cat.ProdObject(degree[:count]),
        base=target.reconstruct(
            input_weaves=tuple(
                without_leading_entries(weave, count) for weave in target.input_weaves),
            output_weaves=tuple(
                without_leading_entries(weave, count) for weave in target.output_weaves),
            reindexings=tuple(
                without_leading_positions(reindexing, count)
                for reindexing in target.reindexings),
            backup_degree=(cat.ProdObject(degree[count:])
                           if target.has_empty_domain() else None)))


def lifted_position_count(target: cat.Broadcasted) -> int:
    '''The number of leading positions of the degree of `target` that every weave
    tiles at its leading positions and every reindexing carries straight through.'''
    weaves = (*target.input_weaves, *target.output_weaves)
    return min((len(tuple(target.degree())),
                *(leading_tiled_count(weave) for weave in weaves),
                *(leading_identity_width(reindexing)
                  for reindexing in target.reindexings)))


def leading_tiled_count(weave: cat.Weave) -> int:
    '''The number of `WeaveMode.TILED` entries at the front of `weave`.'''
    count = 0
    for entry in weave._shape:
        if entry is not cat.WeaveMode.TILED:
            break
        count += 1
    return count


def without_leading_entries[B: cat.Datatype, A: cat.Axis](
    weave: cat.Weave[B, A], count: int,
) -> cat.Weave[B, A]:
    return weave.reconstruct(_shape=tuple(weave._shape[count:]))


def leading_identity_width(reindexing: cat.StrideCategory) -> int:
    '''The number of leading positions of the domain of `reindexing` that it
    carries straight through to the same positions of its codomain.'''
    match reindexing:
        case cat.Rearrangement(mapping=mapping):
            return rearrangement_identity_width(mapping)
        case cat.ProductOfMorphisms(content=factors):
            width = 0
            for factor in factors:
                factor_width = leading_identity_width(factor)
                width += factor_width
                if not is_identity_of_width(factor, factor_width):
                    break
            return width
        case cat.StrideMorphism():
            return stride_identity_width(reindexing)
    return 0


def is_identity_of_width(reindexing: cat.StrideCategory, width: int) -> bool:
    return len(reindexing.dom()) == width and len(reindexing.cod()) == width


def rearrangement_identity_width(mapping: tuple[int, ...]) -> int:
    width = 0
    while (width < len(mapping) and mapping[width] == width
           and all(later > width for later in mapping[width + 1:])):
        width += 1
    return width


def stride_identity_width(reindexing: cat.StrideMorphism) -> int:
    width = 0
    rows = reindexing._cod_stride_shift
    while width < min(len(rows), len(reindexing._dom)):
        axis, strides, shift = rows[width]
        is_unit_row = (
            axis == reindexing._dom[width] and nm.is_zero(shift)
            and all(stride == nm.Integer(1) if column == width else nm.is_zero(stride)
                    for column, stride in enumerate(strides)))
        is_read_by_no_later_row = all(
            nm.is_zero(later_strides[width]) for _, later_strides, _ in rows[width + 1:])
        if not (is_unit_row and is_read_by_no_later_row):
            break
        width += 1
    return width


def without_leading_positions[A: cat.Axis](
    reindexing: cat.StrideCategory[A], count: int,
) -> cat.StrideCategory[A]:
    '''`reindexing` with its first `count` domain and codomain positions removed,
    where `leading_identity_width` has found that it carries them straight
    through.'''
    if count == 0:
        return reindexing
    match reindexing:
        case cat.Rearrangement(mapping=mapping, _dom=dom):
            return cat.Rearrangement(
                tuple(position - count for position in mapping[count:]),
                tuple(dom[count:]))
        case cat.ProductOfMorphisms(content=(first, *rest)):
            first_width = len(first.dom())
            if first_width > count:
                return chp.morphism_product(
                    (without_leading_positions(first, count), *rest))
            if not rest:
                return cat.ProdObject().identity()
            return without_leading_positions(
                chp.morphism_product(tuple(rest)), count - first_width)
        case cat.StrideMorphism(_dom=dom, _cod_stride_shift=rows):
            return reindexing.reconstruct(
                _dom=tuple(dom[count:]),
                _cod_stride_shift=tuple(
                    (axis, tuple(strides[count:]), shift)
                    for axis, strides, shift in rows[count:]))
    raise LiftNotFactorable(
        f'a {type(reindexing).__name__} carries no leading position straight through, '
        f'and {count} were asked to be removed')


class LiftNotFactorable(ValueError):
    '''A reindexing asked to lose leading positions it does not carry straight
    through.'''
