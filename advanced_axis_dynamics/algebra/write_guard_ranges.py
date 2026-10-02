# Claude Opus 5.5 (1M context), effort 40.
'''Writing the positions of a guarded axis that hold a value, for the line of indices an
inspection box draws under a formula.

An `AffineSparseAxis` holds a value at position `j`, taken at positions `i` of its guides,
where `0 <= sum(g i) + s j + c < extent`, and the universal unit elsewhere. The line
under a formula names the range of each index the formula holds for every position of
its axis, and the user asked on 2026-09-28 that the range of a guarded index name the
positions that hold a value.

`guarded_interval` solves the form for `j` where the stride `s` is 1 or -1 and writes the
interval `[lower, upper]` of those positions. Each end is the larger or the smaller of a
bound of the axis, `0` or `|axis| - 1`, and a bound of the form, and a bound is left out
where the other one is at least as tight at every position of the guides, per
`lowest_value` and `highest_value` of `AffineGuards`. A causal read of `x` over `|x|`
slots reads `[0, i_{x}]`, and a window of `|w|` slots reads
`[0, \\min(i_{x}, |w| - 1)]`. A stride of any other size makes each bound the floor of a
quotient, and the user chose on 2026-09-28 to state the condition in that case rather
than the floors.

`guarded_condition` writes the condition for every guarded axis, with the term of the
axis alone between the bounds of the form: `j_{w|x} \\le i_{x}`, or
`|u|\\, j_{P|x} \\le i_{x}` for a stride of `-|u|`. The hover over the index shows it.

The caller gives the LaTeX of the index of the axis and of the index of each guide, in
the order of `guides`. A bound is written with its positive terms first, the terms of
the indices before the constant in each sign, and a coefficient in front of its index, so
the window reads `\\min(i_{x}, |w| - 1)` and a compressed read reads
`|a|\\, j_{r|x} \\le i_{x} + 1 - |a|`.
'''
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import data_structure.Numeric as nm

import advanced_axis_dynamics.data_structure.AffineGuards as AffineGuards


class GuidesAndIndicesDiffer(ValueError):
    '''A guarded axis given a number of guide indices other than its number of guides.'''


@dataclass(frozen=True)
class Bound:
    '''One bound on the positions of a guarded axis holding a value, as LaTeX written
    at the indices, and the lowest and the highest value it takes over the positions
    of the guides.'''
    latex: str
    lowest: nm.Numeric
    highest: nm.Numeric


@dataclass(frozen=True)
class SignedTerm:
    is_negative: bool
    latex: str


def minus(value: nm.Numeric) -> nm.Numeric:
    return nm.Multiplication.template(nm.Integer(-1), value)


def index_term(coefficient: nm.Numeric, index: str) -> SignedTerm:
    magnitude = nm.without_sign(coefficient)
    written = (index if magnitude == nm.Integer(1)
               else rf'{nm.product_part_latex(magnitude)}\, {index}')
    return SignedTerm(is_negative=nm.is_negative(coefficient), latex=written)


def constant_terms(constant: nm.Numeric) -> list[SignedTerm]:
    '''The terms of `constant` with their signs, the products of symbols before the
    integers.'''
    collected = nm.collect_like_terms(constant)
    parts = collected.content if isinstance(collected, nm.Addition) else (collected,)
    terms = [SignedTerm(is_negative=nm.is_negative(part),
                        latex=nm.juxtaposed_latex(nm.without_sign(part)))
             for part in parts if not nm.is_zero(part)]
    return sorted(terms, key=lambda term: term.latex[:1].isdigit())


def affine_latex(coefficients: Sequence[nm.Numeric], indices: Sequence[str],
                 constant: nm.Numeric) -> str:
    '''`sum(coefficient * index) + constant` as LaTeX, positive terms first.'''
    terms = [index_term(coefficient, index)
             for coefficient, index in zip(coefficients, indices)
             if not nm.is_zero(coefficient)]
    terms += constant_terms(constant)
    ordered = ([term for term in terms if not term.is_negative]
               + [term for term in terms if term.is_negative])
    if not ordered:
        return '0'
    written = ('-' if ordered[0].is_negative else '') + ordered[0].latex
    for term in ordered[1:]:
        written += f' {"-" if term.is_negative else "+"} {term.latex}'
    return written


def checked_guide_indices(axis: AffineGuards.AffineSparseAxis,
                          guide_indices: Sequence[str]) -> Sequence[str]:
    if len(guide_indices) != len(axis.guides):
        raise GuidesAndIndicesDiffer(
            f'{len(guide_indices)} guide indices for {len(axis.guides)} guides')
    return guide_indices


def form_bound(axis: AffineGuards.AffineSparseAxis, guide_indices: Sequence[str],
               sign: int, shift: nm.Numeric) -> Bound:
    '''The bound `sign * sum(g i) + shift`, with its lowest and highest values over
    the guides.'''
    strides = tuple(nm.Multiplication.template(nm.Integer(sign), stride)
                    for stride in axis.guide_strides)
    return Bound(
        latex=affine_latex(strides, checked_guide_indices(axis, guide_indices), shift),
        lowest=AffineGuards.lowest_value(strides, shift, axis.guides),
        highest=AffineGuards.highest_value(strides, shift, axis.guides))


def constant_bound(value: nm.Numeric) -> Bound:
    return Bound(latex=affine_latex((), (), value), lowest=value, highest=value)


def is_at_least(tighter: Bound, looser: Bound) -> bool:
    return nm.is_nonnegative_for_positive_symbols(
        nm.Addition.template(tighter.lowest, minus(looser.highest)))


def is_at_most(tighter: Bound, looser: Bound) -> bool:
    return nm.is_nonpositive_for_positive_symbols(
        nm.Addition.template(tighter.highest, minus(looser.lowest)))


def kept_lower_bounds(axis_bound: Bound, form: Bound | None) -> list[Bound]:
    if form is None or is_at_least(axis_bound, form):
        return [axis_bound]
    if is_at_least(form, axis_bound):
        return [form]
    return [axis_bound, form]


def kept_upper_bounds(axis_bound: Bound, form: Bound | None) -> list[Bound]:
    if form is None or is_at_most(axis_bound, form):
        return [axis_bound]
    if is_at_most(form, axis_bound):
        return [form]
    return [form, axis_bound]


def written_end(bounds: Sequence[Bound], extremum: str) -> str:
    if len(bounds) == 1:
        return bounds[0].latex
    return rf'\{extremum}({", ".join(bound.latex for bound in bounds)})'


def guarded_interval(axis: AffineGuards.AffineSparseAxis,
                     guide_indices: Sequence[str]) -> str | None:
    '''The interval of the positions of `axis` that hold a value at the guide indices,
    or `None` where the stride of the axis is not 1 or -1.'''
    last = constant_bound(AffineGuards.last_position(axis))
    extent = axis.extent if axis.crosses_end() else None
    if axis.stride == nm.Integer(-1):
        upper = form_bound(axis, guide_indices, 1, axis.shift)
        lower = (None if extent is None else form_bound(
            axis, guide_indices, 1,
            nm.Addition.template(axis.shift, minus(extent), nm.Integer(1))))
    elif axis.stride == nm.Integer(1):
        lower = form_bound(axis, guide_indices, -1, minus(axis.shift))
        upper = (None if extent is None else form_bound(
            axis, guide_indices, -1,
            nm.Addition.template(extent, nm.Integer(-1), minus(axis.shift))))
    else:
        return None
    lowers = kept_lower_bounds(constant_bound(nm.Integer(0)), lower)
    uppers = kept_upper_bounds(last, upper)
    return f'[{written_end(lowers, "max")}, {written_end(uppers, "min")}]'


def guarded_condition(axis: AffineGuards.AffineSparseAxis, index: str,
                      guide_indices: Sequence[str]) -> str | None:
    '''The condition under which position `index` of `axis` holds a value at the guide
    indices, as `lower \\le s' j \\le upper` with `s'` the size of the stride and each
    side left out where the form does not bound it. `None` where the sign of the
    stride is not settled by the sizes.'''
    indices = checked_guide_indices(axis, guide_indices)
    strides = axis.guide_strides
    negated = tuple(map(minus, strides))
    extent = axis.extent if axis.crosses_end() else None
    if nm.is_negative_for_positive_symbols(axis.stride):
        term = index_term(minus(axis.stride), index).latex
        upper = affine_latex(strides, indices, axis.shift)
        lower = (None if extent is None else affine_latex(
            strides, indices,
            nm.Addition.template(axis.shift, minus(extent), nm.Integer(1))))
    elif nm.is_positive_for_positive_symbols(axis.stride):
        term = index_term(axis.stride, index).latex
        lower = affine_latex(negated, indices, minus(axis.shift))
        upper = (None if extent is None else affine_latex(
            negated, indices,
            nm.Addition.template(extent, nm.Integer(-1), minus(axis.shift))))
    else:
        return None
    sides = [side for side in (lower, term, upper) if side is not None]
    return r' \le '.join(sides)
