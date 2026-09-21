'''The derivative of a formula, with one rule per `Numeric` class.

`solver.algebra.differentiate_numeric.differentiate` decides only whether a
subterm is constant. Every case below is a row in the table declared by
`solver.data_structure.numeric_derivative_rule`, so a `Numeric` class added
outside `solver` is differentiated by adding a row rather than by editing the
pass. Import this module for its side effect, as
`import solver.registries.numeric_derivative`, wherever a formula has to be
differentiated.

    Integer, FreeNumeric, Constant,  0     constants. A FreeNumeric is a size
    UnitOfMeasure                          symbol rather than a variable, and a
                                           UnitOfMeasure is a scale
    FreeInput                        1
    Addition                         the sum of the parts' derivatives
    Multiplication                   the product rule, one term per part
    Power                            b^e (e' ln b + e b'/b), which collapses to
                                     e^{x} -> e^{x}, x^n -> n x^{n-1}
    Logarithm                        a'/(a ln b) - b'/(b ln b) * log_b(a)
    CumulativeGaussian               the standard normal density at the
                                     argument, times the argument's derivative
    IsPositive                       0
    Conjugate                        the conjugate of the argument's derivative
    Expandable                       the derivative of the expansion, for a
                                     class with no row of its own.
                                     RectifiedLinear takes it, and gives
                                     1_{u > 0} u'
    Sigmoid                          the logistic density at the argument,
                                     times the argument's derivative

Results go through the `template` constructors, so the canonicalisation in
`Numeric.py` keeps them readable. Identities are dropped and integer constants are
folded, and the derivative of `e^{x}` is literally `e^{x}` rather than
`e^{x} * 1 * (1 + 0)`.

`obsidian/01-foundations/Numerics.md` covers the mathematics.
'''
from __future__ import annotations

import data_structure.Numeric as nm
import solver.data_structure.numeric_derivative_rule as numeric_derivative_rule
import solver.algebra.differentiate_numeric as differentiate_numeric

register = numeric_derivative_rule.register


# ==========================================================================
# The rules.
# ==========================================================================
@register(nm.Integer, nm.FreeNumeric, nm.Constant, nm.UnitOfMeasure)
def constant(target: nm.Numeric) -> nm.Numeric:
    '''A `FreeNumeric` is a size symbol rather than a variable, so it is a
    constant here alongside the literals, the named mathematical constants and
    the units of measure.
    `differentiate` returns zero for any subterm holding no free input, so the
    rule is reached only through a direct call.
    '''
    return nm.Integer(0)


@register(nm.FreeInput)
def free_input(target: nm.Numeric) -> nm.Numeric:
    return nm.Integer(1)


@register(nm.Addition)
def addition(target: nm.Addition) -> nm.Numeric:
    return nm.Addition.template(
        *(differentiate_numeric.differentiate(part) for part in target.content))


@register(nm.Multiplication)
def multiplication(target: nm.Multiplication) -> nm.Numeric:
    '''The product rule: one term per part, that part differentiated and the
    others left alone. A part that is constant contributes a term holding a
    zero, and `Multiplication.template` has `Integer(0)` as its absorbing
    element, so `Addition.template` then drops the term.
    '''
    parts = target.content
    return nm.Addition.template(*(
        nm.Multiplication.template(
            differentiate_numeric.differentiate(part),
            *(other for j, other in enumerate(parts) if j != i))
        for i, part in enumerate(parts)))


@register(nm.Power)
def power(target: nm.Power) -> nm.Numeric:
    '''Two spellings of the same number. A constant exponent takes the school
    rule, and everything else takes the general one.
    '''
    if not nm.contains_free_input(target.exponent):
        return constant_exponent_power(target)
    return general_power(target)


def constant_exponent_power(target: nm.Power) -> nm.Numeric:
    '''n b^{n-1} b', written directly. The general rule gives
    `n * b^{-1} * b^{n}`, which is the same number and not the same term,
    because collecting powers is algebra that `template` does not do.
    '''
    return nm.Multiplication.template(
        target.exponent,
        nm.Power.template(target.base, target.exponent + nm.Integer(-1)),
        differentiate_numeric.differentiate(target.base))


def general_power(target: nm.Power) -> nm.Numeric:
    '''b^e (e' ln b + e b'/b). An Euler base makes ln b equal to 1 and the
    canonicalisation drops it, so `e^{x} -> e^{x}` falls out.
    '''
    terms = []
    if nm.contains_free_input(target.exponent):
        terms.append(nm.Multiplication.template(
            differentiate_numeric.differentiate(target.exponent),
            differentiate_numeric.natural_logarithm(target.base)))
    if nm.contains_free_input(target.base):
        terms.append(nm.Multiplication.template(
            target.exponent,
            differentiate_numeric.differentiate(target.base),
            nm.Power.template(target.base, nm.Integer(-1))))
    return nm.Multiplication.template(target, nm.Addition.template(*terms))


@register(nm.Logarithm)
def logarithm(target: nm.Logarithm) -> nm.Numeric:
    '''log_b(a) is ln a / ln b, differentiated as a quotient.'''
    terms = []
    if nm.contains_free_input(target.argument):
        terms.append(nm.division(
            differentiate_numeric.differentiate(target.argument),
            target.argument))
    if nm.contains_free_input(target.base):
        terms.append(nm.Multiplication.template(
            nm.Integer(-1),
            target,
            nm.division(
                differentiate_numeric.differentiate(target.base), target.base)))
    return nm.division(
        nm.Addition.template(*terms),
        differentiate_numeric.natural_logarithm(target.base))


@register(nm.CumulativeGaussian)
def cumulative_gaussian(target: nm.CumulativeGaussian) -> nm.Numeric:
    '''The chain rule over the standard normal density.

    `nm.standard_normal_density` writes the density as
    `(2 pi)^{-1/2} e^{-u^{2}/2}`, which is the derivative of the distribution
    function at `u`.
    '''
    return nm.Multiplication.template(
        nm.standard_normal_density(target.argument),
        differentiate_numeric.differentiate(target.argument))


@register(nm.Conjugate)
def conjugate(target: nm.Conjugate) -> nm.Numeric:
    '''The conjugate of the argument's derivative. Conjugation is additive and
    commutes with multiplication by a real number, so the derivative of the
    conjugate with respect to a real variable is the conjugate of the
    derivative.
    '''
    return nm.Conjugate(differentiate_numeric.differentiate(target.argument))


@register(nm.IsPositive)
def is_positive(target: nm.IsPositive) -> nm.Numeric:
    '''Zero. The indicator is constant on each side of zero, and the step at
    zero has no derivative. Writing zero at the step is the convention that
    PyTorch, TensorFlow and JAX take for a ReLU, and
    `obsidian/07-para/Derivatives.md` states why the reverse pass may take it.
    '''
    return nm.Integer(0)


@register(nm.Expandable)
def expandable(target: nm.Expandable) -> nm.Numeric:
    '''The derivative of the expansion, for an `Expandable` with no rule of its
    own. A class whose expanded derivative shares no subterm with the class, as
    `Sigmoid`'s does not, registers a rule of its own, which the walk up the
    method resolution order reaches first.
    '''
    return differentiate_numeric.differentiate(target.expand_to_primitives())


@register(nm.Sigmoid)
def sigmoid(target: nm.Sigmoid) -> nm.Numeric:
    '''The chain rule over the logistic density.

    `nm.logistic_density` writes the density as `\\sigma(u) (1 - \\sigma(u))`
    rather than as `e^{-u} (1 + e^{-u})^{-2}`. The derivative then holds the
    same `nm.Sigmoid` term the formula it came from holds, where the
    exponential spelling holds two terms that appear nowhere in the formula.
    `pathway_collapse.split_off_forward_formulas` compares a backward formula
    against a forward one structurally, so the shared spelling is what lets it
    read the sigmoid off the tape.
    '''
    return nm.Multiplication.template(
        nm.logistic_density(target.argument),
        differentiate_numeric.differentiate(target.argument))
