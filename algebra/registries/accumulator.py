# Claude Opus 5 (1M context), high effort.
'''The operator that folds two partial results of a fold into one.

An operator that consumes an axis produces its result by running over the positions of
that axis, so a run over part of the axis leaves a partial result. The accumulator is
the operator combining two such partials, and it answers one question however the axis
came to be divided.

A stream loop asks it of the partial result one iteration holds.
`advanced_axis_dynamics.registries.part_combination` asks it of a concatenation, whose
partials are the results over the parts, which is why a contraction over a concatenated
axis combines by addition: `+` is the accumulator of a sum.

A `ops.SoftMax`, a `ops.L1Norm`, a `ops.L2Norm` and a `ops.Normalize` have no row, because each reads
every position of the axis to write every position and has no partial result at all.

`obsidian/02-categories/Advanced Axis Dynamics.md` states the rewrite that splits a
fold over a concatenated axis into the folds over its parts.
'''
from __future__ import annotations
from typing import Callable

import data_structure.Category as cat
import data_structure.Operators as ops


type AccumulatorRule = Callable[[], cat.Operator]

RULES: dict[type[cat.Operator], AccumulatorRule] = {}


def register(
    *kinds: type[cat.Operator]) -> Callable[[AccumulatorRule], AccumulatorRule]:
    def decorate(rule: AccumulatorRule) -> AccumulatorRule:
        for kind in kinds:
            RULES[kind] = rule
        return rule
    return decorate


def accumulator_for(operator: cat.Operator) -> cat.Operator | None:
    '''The operator folding two partial results of `operator` into one, or `None`
    where no rule declares one.

    The lookup walks the type's MRO, as `para.registries.derivative.rule_for` does, so
    a subclass takes its parent's rule unless it declares one.
    '''
    for kind in type(operator).__mro__:
        if kind in RULES:
            return RULES[kind]()
    return None


@register(ops.Einops, ops.Linear)
def sum_the_partial_contractions() -> cat.Operator:
    '''A contraction over an axis is the sum of the contractions over any division of
    that axis, because a sum over a disjoint union is the sum of the sums.

    A `ops.Linear` contracts its input axes against the weight it holds. Expanded,
    with `expand_linear=True`, the terminal is the `ops.Einops` of that contraction
    and this rule is reached through it. Unexpanded, the `ops.Linear` is the terminal
    itself and accumulates the same way its expansion would.
    '''
    return ops.AdditionOp()


@register(ops.Maximum)
def take_the_greater_partial_maximum() -> cat.Operator:
    '''A maximum over an axis is the maximum of the maxima over any division of that
    axis.

    The accumulator and the operator are both `ops.Maximum`, which is pointwise,
    consumes no axis and costs one tick either way. A row maximum can therefore be a
    reduction with its own accumulator and `Shuffle` over the grid the row sum uses,
    rather than only a stream-carried fold, and FlashAttention-2 uses the reduction
    form with one lane partition carrying two exchanges.
    '''
    return ops.Maximum()
