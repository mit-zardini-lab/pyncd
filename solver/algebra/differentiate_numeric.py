'''Differentiating a `Numeric` with respect to its free input.

A `nm.FreeInput` is the one unnamed variable a formula may mention, `x` in
`e^{x}` or in `x^{-1}`, and an `ops.Arithmetic` is the elementwise map that
formula denotes. Its derivative is a formula again, and `differentiate` is the
pass that writes it.

`differentiate` decides only whether a subterm is constant, through
`nm.contains_free_input`. Every other case is a rule in
`solver/registries/numeric_derivative.py`, keyed by `Numeric` class through
`solver.data_structure.numeric_derivative_rule`. A caller imports the registry
for its side effect, as `data_structure.Operators` does.

The module imports nothing but `Numeric` and the rule table, so
`data_structure.Operators` can reach it without a cycle.

`obsidian/01-foundations/Numerics.md` states the rules and why the results come
out readable.
'''
from __future__ import annotations

import data_structure.Numeric as nm
import solver.data_structure.numeric_derivative_rule as numeric_derivative_rule


def natural_logarithm(target: nm.Numeric) -> nm.Numeric:
    return nm.Logarithm.template(nm.Constant(nm.ConstantSymbol.EULER), target)


def differentiate(target: nm.Numeric) -> nm.Numeric:
    '''d(target)/dx, for `x` the `nm.FreeInput`.'''
    if not nm.contains_free_input(target):
        return nm.Integer(0)
    return numeric_derivative_rule.rule_for(target)(target)
