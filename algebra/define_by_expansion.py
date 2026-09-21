# Claude Fable 5.1, effort 80.
'''An operator stated as equal to its standard expansion.

`algebra.registries.standard_expansions` holds the rule that writes an operator out in
its primitives. `define_by_standard_expansion` pairs the operator with what the rule
returns in a `cat.DefinedExpression`, which a diagram draws as the operator, `:=` and
the expansion.
'''
from __future__ import annotations

import data_structure.Category as cat
import algebra.registries.standard_expansions as standard_expansions


class OperatorHasNoStandardExpansion(Exception):
    '''An operator handed to `define_by_standard_expansion` that no rule of
    `algebra.registries.standard_expansions` writes out.'''


def define_by_standard_expansion[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A],
) -> cat.DefinedExpression[cat.Array[B, A], cat.Broadcasted[B, A]]:
    expansion = standard_expansions.expand_standard(target)
    if expansion is None:
        raise OperatorHasNoStandardExpansion(
            f'{type(target.operator).__name__} has no rule in '
            'algebra.registries.standard_expansions')
    return cat.DefinedExpression.template(target, expansion)
