"""The operations each operator performs per element and the role of each index it
reads, registered per operator."""

from __future__ import annotations

from collections.abc import Callable

import algebra.einops_simplification as einops_simplification
import data_structure.Category as cat
import data_structure.Operators as ops
import performance_modeling.morphism_work as morphism_work


type Variable = einops_simplification.IndexVariable
type Shape = tuple[Variable, ...]
type WorkRule = Callable[
    [cat.Broadcasted, tuple[Shape, ...], Shape, Shape],
    tuple[dict[Variable, morphism_work.IndexRole], Shape, float, str],
]

RULES: dict[type[cat.Operator], WorkRule] = {}


def register_work(operator_type: type[cat.Operator]) -> Callable[[WorkRule], WorkRule]:
    def register(rule: WorkRule) -> WorkRule:
        RULES[operator_type] = rule
        return rule
    return register


def rule_for(operator: cat.Operator) -> WorkRule:
    for operator_type in type(operator).__mro__:
        if operator_type in RULES:
            return RULES[operator_type]
    raise morphism_work.UnsupportedWork(
        f"No arithmetic estimate is registered for {type(operator).__name__}")


@register_work(ops.Einops)
def contraction_work(
    morphism: cat.Broadcasted, inputs: tuple[Shape, ...], output: Shape, degree: Shape,
) -> tuple[dict[Variable, morphism_work.IndexRole], Shape, float, str]:
    read = set(variable for shape in inputs for variable in shape)
    reduced = read.difference(output)
    roles = {variable: morphism_work.IndexRole.REDUCE if variable in reduced
             else morphism_work.IndexRole.REPEAT if variable not in read
             else morphism_work.IndexRole.TILE
             for variable in (*output, *read)}
    operations = max(0, len(inputs) - 1) + bool(reduced)
    resource = "matrix" if reduced and len(inputs) == 2 else "scalar"
    return roles, (), float(operations), resource


@register_work(ops.Maximum)
@register_work(ops.Product)
def maximum_work(
    morphism: cat.Broadcasted, inputs: tuple[Shape, ...], output: Shape, degree: Shape,
) -> tuple[dict[Variable, morphism_work.IndexRole], Shape, float, str]:
    '''A `Maximum` folds every input dimension absent from its output, as a
    contraction does under a different monoid, at one compare per element on
    the scalar unit. A `Product` folds them at one multiply per element.'''
    read = set(variable for shape in inputs for variable in shape)
    reduced = read.difference(output)
    roles = {variable: morphism_work.IndexRole.REDUCE if variable in reduced
             else morphism_work.IndexRole.REPEAT if variable not in read
             else morphism_work.IndexRole.TILE
             for variable in (*output, *read)}
    return roles, (), 1.0, "scalar"


@register_work(ops.Linear)
def linear_work(
    morphism: cat.Broadcasted, inputs: tuple[Shape, ...], output: Shape, degree: Shape,
) -> tuple[dict[Variable, morphism_work.IndexRole], Shape, float, str]:
    '''A `Linear` with its weight inside the operator, or the parametrised
    form with the weight as its first operand and the data as its last. The
    weight's dimensions are reported as weights in the first form and are an
    ordinary operand in the second.'''
    if not inputs or any(
            not isinstance(weave.datatype, cat.Reals)
            for weave in morphism.input_weaves):
        raise morphism_work.UnsupportedWork("Cost dense Linear operations with real operands")
    data = inputs[-1]
    reduced = tuple(variable for variable in data if variable not in degree)
    features = tuple(variable for variable in output if variable not in degree)
    if set(reduced).intersection(features):
        raise morphism_work.UnsupportedWork(
            "Linear target dimensions need distinct axes")
    roles = {variable: morphism_work.IndexRole.TILE for variable in (*degree,
        *features)}
    roles.update({variable: morphism_work.IndexRole.REDUCE for variable in reduced})
    for operand in inputs[:-1]:
        unknown = tuple(variable for variable in operand if variable not in roles)
        if unknown:
            raise morphism_work.UnsupportedWork(
                f"A Linear operand carries {len(unknown)} dimensions the contraction does not")
    weights = (*reduced, *features) if len(inputs) == 1 else ()
    return roles, weights, 2.0, "matrix"


@register_work(ops.Elementwise)
@register_work(ops.AdditionOp)
@register_work(ops.View)
def elementwise_work(
    morphism: cat.Broadcasted, inputs: tuple[Shape, ...], output: Shape, degree: Shape,
) -> tuple[dict[Variable, morphism_work.IndexRole], Shape, float, str]:
    read = set(variable for shape in inputs for variable in shape)
    if read.difference(output):
        raise morphism_work.UnsupportedWork("An elementwise operation cannot reduce")
    roles = {variable: morphism_work.IndexRole.TILE if variable in read
             else morphism_work.IndexRole.REPEAT for variable in output}
    operator = morphism.operator
    if isinstance(operator, ops.View):
        operations = 0.0
    elif isinstance(operator, ops.AdditionOp):
        operations = float(max(0, len(inputs) - 1))
    elif isinstance(operator, ops.Arithmetic):
        operations = 1.0
    else:
        estimates = {"gelu": 8.0, "relu": 1.0, "sigmoid": 4.0,
                     "exp": 4.0, "sqrt": 4.0, "tanh": 4.0}
        if operator.operator not in estimates:
            raise morphism_work.UnsupportedWork(
                f"No scalar estimate for {operator.operator!r}")
        operations = estimates[operator.operator]
    return roles, (), operations, "scalar"


@register_work(ops.SoftMax)
@register_work(ops.Normalize)
@register_work(ops.LayerNorm)
def normalization_work(
    morphism: cat.Broadcasted, inputs: tuple[Shape, ...], output: Shape, degree: Shape,
) -> tuple[dict[Variable, morphism_work.IndexRole], Shape, float, str]:
    roles = {variable: morphism_work.IndexRole.TILE if variable in degree
             else morphism_work.IndexRole.LOCAL for variable in output}
    normalized = [shape for shape in inputs if shape == output]
    gains = [shape for shape in inputs if shape != output]
    if len(normalized) != 1 or any(set(shape).difference(output) for shape in gains):
        raise morphism_work.UnsupportedWork(
            f"A normalization reads one operand of its own shape {output} and gains "
            f"indexed within that shape, received {inputs}")
    if isinstance(morphism.operator, ops.SoftMax):
        return roles, (), 8.0, "scalar"
    return roles, (), 7.0 + len(gains), "scalar"
