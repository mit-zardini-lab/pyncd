'''The type a derivative rule has, and the table the rules register into.

A rule differentiates one `nm.Numeric` class with respect to the one
`nm.FreeInput` a formula may mention. The rules are in
`solver/registries/numeric_derivative.py`, one per class, and the pass that
applies them is `solver.algebra.differentiate_numeric.differentiate`.

The table sits in `data_structure/` because both of those modules read it and
neither may import the other. A rule calls `differentiate` on its subterms, so
the registry imports the algebra. The algebra therefore cannot import the
registry, and looks the rule up through this module instead.

`obsidian/01-foundations/Numerics.md` describes what a `Numeric` is and what
differentiating one means.
'''
from __future__ import annotations
from typing import Callable

import data_structure.Numeric as nm


type Rule = Callable[[nm.Numeric], nm.Numeric]

RULES: dict[type[nm.Numeric], Rule] = {}


class NoNumericDerivativeRule(Exception):
    '''A `Numeric` class that no registered rule differentiates.'''


def register[T: nm.Numeric](
    *kinds: type[T],
) -> Callable[[Callable[[T], nm.Numeric]], Callable[[T], nm.Numeric]]:
    '''Add one rule to the table under each of the classes it differentiates.

    The generic ties the rule's argument to the classes it is registered
    against, so a rule for `nm.Power` reads `target.exponent` without narrowing
    the type first.
    '''
    def decorate(
        rule: Callable[[T], nm.Numeric],
    ) -> Callable[[T], nm.Numeric]:
        for kind in kinds:
            RULES[kind] = rule  # type: ignore[assignment]
        return rule
    return decorate


def rule_for(target: nm.Numeric) -> Rule:
    '''Most specific first. `Addition` and `Multiplication` are both
    `Associative`, so the walk up the method resolution order lets a rule
    registered on `Associative` stand in for whichever of them has no rule of
    its own.
    '''
    for kind in type(target).__mro__:
        if kind in RULES:
            return RULES[kind]
    raise NoNumericDerivativeRule(
        f'no derivative rule for {type(target).__qualname__}. The rules are in '
        'solver/registries/numeric_derivative.py, and a caller has to import '
        'that module for its side effect before it differentiates a formula.')
