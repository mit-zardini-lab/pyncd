# Claude Opus 5.5 (1M context), effort 40.
'''Evaluating parts of Kimi K3 on numbers, at small sizes.

`notebooks/caching/mamba/evaluate_numerically.py` evaluates the operators of a scan over
the tokens: the views, the covariant writes, the contractions, the linear maps, the
elementwise maps and the loop with its stream seeds. This module adds the three more
operators the delta attention and the attention residuals hold:

    a box                   evaluated as its body lifted over its degree, per
                            `discovering_broadcasts.expand_broadcast_of_block`
    an RMS normalisation    `v / sqrt(mean(v^2) + epsilon)` over its target, times
                            the gain handed to the evaluation under the name of the
                            normalisation
    a softmax               over its target

and replaces every named constant of `released_constants` and every size by its value
before the evaluation, because a formula evaluated in PyTorch reads no free symbol.

The universal unit is read as zero, as the Mamba evaluator reads it. Zero is the unit
of the sum over the taps of a convolution, so the convolution before the first token is
evaluated as the reference pads it. Zero is not the unit of a softmax, so a softmax is
evaluated here only over an axis every position of which holds a value.
'''
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

import algebra.discovering_broadcasts as discovering_broadcasts
import caching.algebra.derive_cached_pass as derive_cached_pass
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import torch_compile.torch_compile as torch_compile

import notebooks.caching.mamba.evaluate_numerically as mamba_evaluation
from notebooks.caching.mamba.evaluate_numerically import (
    DTYPE, CachesBetweenPasses, LinearValues, Tensors, name_key, sizes_by_symbol)
from notebooks.sota.KimiK3.released_constants import (
    GATE_LOWER_BOUND, L2_NORM_EPSILON, LATENT_NORM_EPSILON, NORM_EPSILON,
    ROUTER_EPSILON, SITU_GATE_BOUND, SITU_UP_BOUND)

CONSTANT_VALUES: dict[nm.FreeNumeric, nm.Numeric] = {
    NORM_EPSILON: nm.Integer(1) / nm.Integer(10 ** 5),
    LATENT_NORM_EPSILON: nm.Integer(1) / nm.Integer(10 ** 6),
    L2_NORM_EPSILON: nm.Integer(1) / nm.Integer(10 ** 6),
    ROUTER_EPSILON: nm.Integer(1) / nm.Integer(10 ** 20),
    GATE_LOWER_BOUND: nm.Integer(-5),
    SITU_GATE_BOUND: nm.Integer(4),
    SITU_UP_BOUND: nm.Integer(25),
}
'''Every named constant of the model with the value the reference gives it, written as
an integer or a ratio of integers so that the formulas stay exact numerics.'''


def with_the_released_constants[T: cat.Morphism](term: T) -> T:
    '''`term` with every named constant replaced by its value in the reference.'''
    for symbol, value in CONSTANT_VALUES.items():
        term = derive_cached_pass.with_numeric_replaced(term, symbol, value, {})
    return term


def number_of(numeric: nm.Numeric) -> float:
    '''The value of a numeric that holds no free symbol.'''
    return float(torch_compile.numeric_torch(numeric, torch.zeros((), dtype=DTYPE)))


@dataclass
class Evaluation(mamba_evaluation.Evaluation):
    '''One pass of an expression of Kimi K3 evaluated on numbers.'''
    gains: Mapping[str, torch.Tensor] | None = None

    def operator(self, target: cat.Broadcasted, read: Tensors,
                 degree_sizes: tuple[int, ...]) -> torch.Tensor:
        match target.operator:
            case ops.Normalize():
                return self.normalised(target, read[0])
            case ops.SoftMax():
                return self.softmax(target, read[0])
        return super().operator(target, read, degree_sizes)

    def broadcasted(self, target: cat.Broadcasted, inputs: Tensors) -> Tensors:
        if isinstance(target.operator, ops.BlockOperator):
            if not tuple(target.degree()):
                return self.morphism(target.operator.block, inputs)
            return self.morphism(
                discovering_broadcasts.expand_broadcast_of_block(target), inputs)
        return super().broadcasted(target, inputs)

    def normalised(self, target: cat.Broadcasted, operand: torch.Tensor) -> torch.Tensor:
        '''The RMS normalisation of `operand` over the axes of its target, which the read
        puts last, times the gain named after the operator.'''
        normalised_axes = len(target.input_weaves[0].target().shape())
        dims = tuple(range(operand.dim() - normalised_axes, operand.dim()))
        epsilon = number_of(target.operator.epsilon)
        result = operand / torch.sqrt(operand.pow(2).mean(dim=dims, keepdim=True) + epsilon)
        if target.operator.gain:
            result = result * (self.gains or {})[name_key(target.operator.name)]
        return result

    def softmax(self, target: cat.Broadcasted, operand: torch.Tensor) -> torch.Tensor:
        '''The softmax of `operand` over the axes of its target, which the read puts
        last.'''
        normalised_axes = len(target.input_weaves[0].target().shape())
        flat = operand.flatten(start_dim=operand.dim() - normalised_axes)
        return flat.softmax(dim=-1).reshape(operand.shape)


def evaluate(expression: cat.Morphism, inputs: Tensors,
             weights: Mapping[str, LinearValues], gains: Mapping[str, torch.Tensor],
             sizes: Mapping[str, int]) -> Tensors:
    '''`expression` evaluated on `inputs`, with every size named in `sizes` and every
    named constant at its value in the reference.'''
    expression = with_the_released_constants(expression)
    bindings = sizes_by_symbol(expression, sizes)
    for symbol, value in bindings.items():
        expression = derive_cached_pass.with_numeric_replaced(
            expression, symbol, nm.Integer(value), {})
    evaluation = Evaluation(
        weights={name_key(name): values for name, values in weights.items()},
        bindings={}, caches=CachesBetweenPasses(),
        gains={name_key(name): gain for name, gain in gains.items()})
    return evaluation.morphism(expression, inputs)
