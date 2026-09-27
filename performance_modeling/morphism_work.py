"""Read operation dimensions and arithmetic from broadcast morphisms.

`caching.algebra.count_pass_operations` counts the operations of a pass with
`read_symbolic_work`, and `advanced_axis_dynamics.algebra.absorb_linear_maps` compares
the orders of a chain of contractions by the same count.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from collections.abc import Mapping

import algebra.einops_simplification as einops_simplification
import data_structure.Category as cat
import data_structure.Numeric as nm


class UnsupportedWork(ValueError):
    pass


class IndexRole(enum.Enum):
    TILE = "tile"
    REDUCE = "reduce"
    LOCAL = "local"
    REPEAT = "repeat"


@dataclasses.dataclass(frozen=True)
class WorkIndex:
    variable: einops_simplification.IndexVariable
    size: int
    role: IndexRole


@dataclasses.dataclass(frozen=True)
class MorphismWork:
    morphism: cat.Broadcasted
    indices: tuple[WorkIndex, ...]
    inputs: tuple[tuple[int, ...], ...]
    output: tuple[int, ...]
    weights: tuple[int, ...]
    operations_per_element: float
    compute_resource: str

    def shape(self, positions: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(self.indices[position].size for position in positions)


@dataclasses.dataclass(frozen=True)
class SymbolicWorkIndex:
    variable: einops_simplification.IndexVariable
    size: nm.Numeric
    role: IndexRole


@dataclasses.dataclass(frozen=True)
class SymbolicWork:
    """The work of a morphism with every index size left as the numeric its axis
    carries, so the arithmetic is written before any count is bound. `inputs`,
    `output` and `weights` give positions into `indices`, as `MorphismWork` does."""
    morphism: cat.Broadcasted
    indices: tuple[SymbolicWorkIndex, ...]
    inputs: tuple[tuple[int, ...], ...]
    output: tuple[int, ...]
    weights: tuple[int, ...]
    operations_per_element: float
    compute_resource: str

    @property
    def elements(self) -> nm.Numeric:
        return nm.Multiplication.template(*(index.size for index in self.indices))

    @property
    def operations(self) -> nm.Numeric:
        """The arithmetic of one evaluation, the elements at the operations per
        element, which every work rule states as a whole number."""
        return self.elements * integer_numeric(self.operations_per_element)

    def bound(self, bindings: Mapping[nm.Numeric, int]) -> MorphismWork:
        return MorphismWork(
            self.morphism,
            tuple(WorkIndex(index.variable, positive_size(index.size, bindings), index.role)
                  for index in self.indices),
            self.inputs, self.output, self.weights,
            self.operations_per_element, self.compute_resource)


def integer_numeric(value: float) -> nm.Integer:
    if not float(value).is_integer():
        raise UnsupportedWork(f"The operations per element {value} is not a whole number")
    return nm.Integer(int(value))


def numeric_value(
    expression: nm.Numeric, bindings: Mapping[nm.Numeric, int | float],
) -> float:
    if expression in bindings:
        return float(bindings[expression])
    match expression:
        case nm.Integer(_value=value):
            return float(value)
        case nm.Constant() | nm.UnitOfMeasure():
            return expression.to_float()
        case nm.Addition(content=terms):
            return sum(numeric_value(term, bindings) for term in terms)
        case nm.Multiplication(content=terms):
            return math.prod(numeric_value(term, bindings) for term in terms)
        case nm.Power(base=base, exponent=exponent):
            return numeric_value(base, bindings) ** numeric_value(exponent, bindings)
        case nm.Logarithm(base=base, argument=argument):
            return math.log(numeric_value(argument, bindings), numeric_value(base, bindings))
        case _:
            raise UnsupportedWork(f"Bind the unresolved size {expression!r}")


def positive_size(expression: nm.Numeric, bindings: Mapping[nm.Numeric, int]) -> int:
    return positive_value(numeric_value(expression, bindings))


def positive_value(value: float) -> int:
    if not math.isfinite(value) or value < 1 or not value.is_integer():
        raise ValueError(f"An axis size must be a positive integer, received {value}")
    return int(value)


def read_symbolic_work(morphism: cat.Broadcasted) -> SymbolicWork:
    """The work of `morphism` with every index size as the numeric its axis
    carries, read through the rule registered for its operator."""
    import performance_modeling.registries.operation_work as operation_work

    if len(morphism.output_weaves) != 1:
        raise UnsupportedWork("The work reader requires one output per morphism")
    inputs, output, degree = einops_simplification.index_shapes(morphism)
    variables = tuple(dict.fromkeys((*degree, *output, *(
        variable for shape in inputs for variable in shape))))
    rule = operation_work.rule_for(morphism.operator)
    roles, weights, operations, resource = rule(morphism, inputs, output, degree)
    positions = {variable: index for index, variable in enumerate(variables)}
    return SymbolicWork(
        morphism,
        tuple(SymbolicWorkIndex(variable, variable.axis.local_size(), roles[variable])
              for variable in variables),
        tuple(tuple(positions[variable] for variable in shape) for shape in inputs),
        tuple(positions[variable] for variable in output),
        tuple(positions[variable] for variable in weights), operations, resource,
    )


def read_morphism_work(
    morphism: cat.Broadcasted,
    bindings: Mapping[nm.Numeric, int] | None = None,
) -> MorphismWork:
    """`read_symbolic_work` with every index size bound to an integer."""
    return read_symbolic_work(morphism).bound(bindings or {})
