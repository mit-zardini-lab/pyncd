# Claude Fable 5.1, effort 80.
'''The standard expansion of an operator, collated in one registry.

An operator such as a `ops.SoftMax`, an `ops.L1Norm` or an `ops.Normalize` is one
morphism to the algebra and is written out in its primitives by a rule in
`algebra.operator_expansion`. Each rule is a function of its own, and until this
registry existed nothing listed them, so a reader who wanted to know which operators
could be written out had to read that module. The registry maps each operator class to
its rule, the formula the rule writes out, and a sentence saying what the operator
computes, and the display reads it to offer the expansion of every such operator in a
figure.

`algebra.operator_expansion` registers its rules with the decorator here, so the rule
bodies stay beside their documentation and the registry holds the table. A rule reads
every shape off the operator's own domain and codomain, so the expansion is at the
operator's own degree, and a rule is given the `cat.Broadcasted` that carries the
operator rather than the operator alone.

`obsidian/05-backends/Advanced Display.md` states how the display uses the table.
'''
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import data_structure.Category as cat


type ExpansionRule = Callable[[cat.Broadcasted], cat.BroadcastedCategory]
type TextOfOperator = str | Callable[[cat.Broadcasted], str]


@dataclass(frozen=True)
class StandardExpansion:
    '''One row of the registry. `formula` is LaTeX and `description` is prose. Either
    is one text for every operator of the class, or a function that writes the text
    from the `cat.Broadcasted` that carries the operator, so a formula names the axes
    of the operator it is shown over and mentions a bias only where there is one.'''
    rule: ExpansionRule
    formula: TextOfOperator
    description: TextOfOperator

    def formula_of(self, target: cat.Broadcasted) -> str:
        return self.formula if isinstance(self.formula, str) else self.formula(target)

    def description_of(self, target: cat.Broadcasted) -> str:
        return (self.description if isinstance(self.description, str)
                else self.description(target))


RULES: dict[type[cat.Operator], StandardExpansion] = {}


def register(
    *kinds: type[cat.Operator], formula: TextOfOperator, description: TextOfOperator,
) -> Callable[[ExpansionRule], ExpansionRule]:
    '''Register `rule` as the standard expansion of every class in `kinds`.'''
    def decorate(rule: ExpansionRule) -> ExpansionRule:
        for kind in kinds:
            RULES[kind] = StandardExpansion(
                rule=rule, formula=formula, description=description)
        return rule
    return decorate


def expansion_for(operator: cat.Operator) -> StandardExpansion | None:
    '''The registry row for `operator`, or `None` where no rule declares one.

    The lookup walks the type's MRO, as `algebra.registries.accumulator` does, so a
    subclass takes its parent's rule unless it declares one.
    '''
    for kind in type(operator).__mro__:
        if kind in RULES:
            return RULES[kind]
    return None


def has_standard_expansion(target: object) -> bool:
    '''Whether `target` is a `cat.Broadcasted` whose operator the registry writes out.'''
    return (isinstance(target, cat.Broadcasted)
            and expansion_for(target.operator) is not None)


def expand_standard[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A],
) -> cat.BroadcastedCategory[B, A] | None:
    '''`target` written out by its registered rule, or `None` where it has none.'''
    row = expansion_for(target.operator)
    return None if row is None else row.rule(target)


def registered_operators() -> tuple[type[cat.Operator], ...]:
    '''The operator classes the registry writes out, in registration order.'''
    return tuple(RULES)
