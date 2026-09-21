# Claude Opus 5 (1M context), high effort.
'''The operator that folds the partial results over the parts of a concatenation.

A part is one input of an `aops.ConcatenateAxes`, meaning one of the axes whose
positions fill the concatenated axis. `F(Concat(x, y)) = B_F(F(x), F(y))` is the
rewrite of `concatenation_expansion.expand_concatenations`, and `B_F` depends on what
`F` did with the concatenated axis. Where `F` is broadcast over the axis, its result
carries the axis and `B_F` is another `aops.ConcatenateAxes`, which the rewrite builds
without asking here. Where `F` consumes the axis and hands out a result without it,
`F` folded the axis, and `B_F` folds the two partial results into one.

The fold of a concatenation is the accumulator of the fold, so the rule is read from
`algebra.registries.accumulator` rather than written again here. A run over part of an
axis leaves a partial result whatever divided the axis, so a contraction over a
concatenated axis combines by addition for the reason a streamed contraction does:
`+` is the accumulator of a sum. The same table answers the question for the partial
result one iteration of a loop holds.

An operator with no row there refuses the rewrite, and two refuse it for a reason.
A `ops.SoftMax` over the concatenated axis reads every position to normalise, so its
result over a part is not a partial of its result over the whole. A `ops.Linear` holds
its weight rather than naming it, and it does declare an accumulator, so the refusal
is stated here: cutting the concatenated axis into parts would cut the weight with it
and give the two parts one name.

`obsidian/02-categories/Advanced Axis Dynamics.md` states the rewrite.
'''
from __future__ import annotations

import data_structure.Category as cat
import data_structure.Operators as ops

import algebra.registries.accumulator as accumulator


class FoldHasNoCombination(Exception):
    '''An operator that consumes a concatenated axis and for which no rule says how the
    results over the parts combine.'''


OPERATORS_HOLDING_THEIR_WEIGHT: tuple[type[cat.Operator], ...] = (ops.Linear,)


def combination_for(operator: cat.Operator) -> cat.Operator:
    '''The operator folding two partial results of `operator` into one, which is the
    accumulator `algebra.registries.accumulator` declares for it.'''
    if isinstance(operator, OPERATORS_HOLDING_THEIR_WEIGHT):
        raise FoldHasNoCombination(
            f'{type(operator).__name__} holds its weight rather than naming it, so '
            'the parts of a concatenated axis it contracts would carry one weight '
            'under one name')
    combination = accumulator.accumulator_for(operator)
    if combination is None:
        raise FoldHasNoCombination(
            f'{type(operator).__name__} consumes a concatenated axis and no rule says '
            'how its results over the parts combine')
    return combination
