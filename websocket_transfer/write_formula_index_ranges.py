# Claude Opus 5.5 (1M context), effort 40.
'''Giving each index on the line under a formula the positions of its axis that hold a
value.

`letter_formula_indices` lists the indices a formula holds for every position of their
axes, reading the formula alone. Where the axis of an index is an
`AffineGuards.AffineSparseAxis` with guides, `index_records_with_ranges` adds the
condition under which a position of the axis holds a value, for the hover over the
index, and the interval of those positions where the stride of the axis is 1 or -1, for
the line, per `advanced_axis_dynamics/algebra/write_guard_ranges.py`. The user asked for
both on 2026-09-28.

An index is matched to its axis by the LaTeX of the axis name among `axes`, the axes of
the operator or the block the formula is shown over. Two guarded axes of one name with
different forms, as the compressed and the uncompressed `r|x` of DeepSeek-V4.1-Flash
have, leave the index without a range, because the formula does not say which one it
means. A guide the formula names is written with its letter, and a guide the formula does
not name takes the next spare letter and stands on the line before the index it guides,
so every range names only indices already introduced.
'''
from __future__ import annotations

from collections.abc import Sequence

import data_structure.Category as cat
import websocket_transfer.letter_formula_indices as letter_formula_indices
import websocket_transfer.websockets_transfer as wst

import advanced_axis_dynamics.algebra.write_guard_ranges as write_guard_ranges
import advanced_axis_dynamics.data_structure.AffineGuards as AffineGuards


def axis_latex(axis: cat.Axis) -> str | None:
    '''The LaTeX a formula writes for `axis`: its name without the size a display pass
    wrote as an exponent, with its subscripts braced.'''
    name = axis.uid._name
    if name is None:
        return None
    return letter_formula_indices.with_braced_subscripts(
        name.with_exponent(None).to_latex())


def guard_form_key(axis: AffineGuards.AffineSparseAxis) -> tuple[object, ...]:
    return (tuple(axis_latex(guide) for guide in axis.guides), axis.guide_strides,
            axis.stride, axis.shift, axis.extent, axis.local_size())


def guarded_axes_by_latex(
    axes: Sequence[cat.Axis],
) -> dict[str, AffineGuards.AffineSparseAxis]:
    '''Each guarded axis with guides among `axes`, by the LaTeX of its name, leaving
    out a name two different guards share.'''
    by_latex: dict[str, AffineGuards.AffineSparseAxis] = {}
    ambiguous: set[str] = set()
    for axis in axes:
        if not isinstance(axis, AffineGuards.AffineSparseAxis) or not axis.guides:
            continue
        latex = axis_latex(axis)
        if latex is None:
            continue
        held = by_latex.setdefault(latex, axis)
        if guard_form_key(held) != guard_form_key(axis):
            ambiguous.add(latex)
    return {latex: axis for latex, axis in by_latex.items() if latex not in ambiguous}


def index_records_with_ranges(
    lettered: letter_formula_indices.LetteredFormula, axes: Sequence[cat.Axis],
) -> list[wst.FormulaIndexRecord]:
    '''The indices `lettered` holds for every position of their axes, each guarded one
    with its condition and its interval, and each guide it needs placed before it.'''
    guarded = guarded_axes_by_latex(axes)
    index_of = {axis: f'{letter}_{{{axis}}}' for axis, letter in lettered.letters}
    spare_letters = iter(lettered.spare_letters)
    placed: list[wst.FormulaIndexRecord] = []
    placed_axes: set[str] = set()

    def guide_indices_of(axis: AffineGuards.AffineSparseAxis) -> list[str] | None:
        indices = []
        for guide in axis.guides:
            guide_latex = axis_latex(guide)
            if guide_latex is None:
                return None
            if guide_latex not in index_of:
                letter = next(spare_letters, None)
                if letter is None:
                    return None
                index_of[guide_latex] = f'{letter}_{{{guide_latex}}}'
            place(guide_latex)
            indices.append(index_of[guide_latex])
        return indices

    def place(latex: str) -> None:
        if latex in placed_axes:
            return
        placed_axes.add(latex)
        record: wst.FormulaIndexRecord = {'index': index_of[latex], 'axis': latex}
        axis = guarded.get(latex)
        guide_indices = None if axis is None else guide_indices_of(axis)
        if axis is not None and guide_indices is not None:
            interval = write_guard_ranges.guarded_interval(axis, guide_indices)
            condition = write_guard_ranges.guarded_condition(
                axis, record['index'], guide_indices)
            if interval is not None:
                record['range'] = letter_formula_indices.with_braced_subscripts(interval)
            if condition is not None:
                record['condition'] = letter_formula_indices.with_braced_subscripts(
                    condition)
        placed.append(record)

    for latex in lettered.free_axes:
        place(latex)
    return placed
