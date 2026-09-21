'''
Writing an operator out in its primitives.

A `SoftMax` is one operator to the algebra: one rule for its derivative, one
kernelization guide, one glyph. Written out it is

    softmax_x(s) = exp(s) / sum_x exp(s)

which is an `Arithmetic` (`e^{x}`), a copy, a contraction over `x`, another
`Arithmetic` for `x^{-1}`, and a pointwise product: five morphisms the rest of
the algebra already differentiates, simplifies and fuses. The expansion is
the rewrite from the one to the five.

The four morphisms after the exponential are the L1 norm,

    l1norm_x(v) = v / sum_x v

which is an `ops.L1Norm` of its own, so a softmax is also one step from
`exp ; L1Norm`. `rewrite_softmax_as_l1_norm` is that step and
`expand_l1_norm` writes the norm out. `expand_softmax` composes the two, so it
returns the same five morphisms it always has. The Sinkhorn iteration of
DeepSeek-V4.1-Flash normalises a row and then a column, and the mixture of
experts normalises its gates, and each of those is the same operator.

`expand_softmax` states what the operator means, and is prior to any rewrite that
reads the expanded form.

Every shape is read off the operator's own domain and codomain, so a softmax already
broadcast over other axes expands at that degree: the contraction is over the
axis in the operator's target and everything else is carried along.

`expand_l2_norm` writes out an `ops.L2Norm`,

    l2norm_m(v) = v / (sum_m v^2)^{1/2}

which is a copy, a square, a contraction, an inverse square root and a product. The
same five morphisms make up most of the expansion of a `Normalize`, which divides the
sum by the number of values and multiplies by a learned gain besides. A model that
normalises with no gain, as the four-stream residual of DeepSeek-V4.1-Flash does,
writes the `L2Norm` and the root of the number of values rather than a `Normalize`
whose gain is not there.

`expand_normalize` does the same for a `Normalize`, which is

    rmsnorm_m(x) = gamma * x * ((sum_m x^2) / |m| + epsilon)^{-1/2}

and comes out as a copy, a square, a contraction of the normalized axis, the inverse
square root of the mean with the epsilon added, and two products. The gain reaches the
operator as an operand, put there by
`para.processing.show_grabbed_parameters`, and the expansion keeps it on its wire.

`expand_shifted_softmax` writes the same softmax with its scores shifted by their
maximum before the exponent,

    softmax_x(s) = exp(s - m) / sum_x exp(s - m),    m = max_x s

which is the form a floating-point implementation evaluates, because `exp` of an
unshifted score overflows. The shift adds a `Maximum`, a negation and an
`AdditionOp` in front of the exponent and changes nothing else. A softmax is
invariant to a shift of its argument, so the maximum contributes nothing to the
derivative, and `para.registries.derivative` declares its cotangent zero. The
tape of a pair derived from this form carries the maximum beside the reciprocal
row sum, which are the two numbers FlashAttention folds into its log-sum-exp.
'''
from __future__ import annotations
import functools
import operator
from dataclasses import dataclass
from algebra.registries.expansion_wording import TEXT as text

import data_structure.Term as fd
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import graphs.processing.hypergraph_functor as functor
from construction_helpers import simple_helper as chsh

import algebra.einops_simplification as es
import algebra.registries.standard_expansions as standard_expansions
import algebra.write_index_notation as write_index_notation


def consumed_shape_and_degree[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A],
) -> tuple[tuple[es.IndexVariable, ...], tuple[es.IndexVariable, ...]]:
    '''The shape an operator reads, and the positions of it the operator does not
    consume, which are its degree.

    Raises where the operator writes a shape other than the one it reads, which
    every operator written out here does.
    '''
    inputs, output, _ = es.index_shapes(target)
    shape = inputs[0]
    if shape != output:
        raise ValueError(
            f'a {type(target.operator).__name__} reads and writes one shape, and '
            f'this one reads {shape} and writes {output}')
    return shape, tuple(target.input_weaves[0].select_degree(shape))


def consumed_letters[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A],
) -> fd.Prod[str]:
    '''The letter of each axis the operator of `target` consumes, which are the target
    axes of its data operand, the last one. A formula names them, so the box over
    the RMSNorm of the query latent reads `RMSNorm_{q}` and sums over `i_{q}`.'''
    return write_index_notation.axis_letters(
        tuple(target.input_weaves[-1].target().shape()))


def added_epsilon_latex(epsilon: nm.Numeric) -> str:
    '''` + epsilon` where a norm adds one, and the empty string where it adds none.
    `ops.NO_EPSILON` is the unit of addition, and a formula that named it would read
    `+ 0`.'''
    return '' if epsilon == ops.NO_EPSILON else f' + {epsilon.to_latex()}'


def l1_norm_formula[B: cat.Datatype, A: cat.Axis](target: cat.Broadcasted[B, A]) -> str:
    letters = consumed_letters(target)
    return (rf'L^{{1}}_{{{write_index_notation.subscript_of(letters)}}}(v) = '
            rf'\frac{{v}}{{{write_index_notation.sum_over(letters)}'
            rf'{write_index_notation.read_at("v", letters)}'
            rf'{added_epsilon_latex(target.operator.epsilon)}}}')


def softmax_formula[B: cat.Datatype, A: cat.Axis](target: cat.Broadcasted[B, A]) -> str:
    letters = consumed_letters(target)
    return (rf'\mathrm{{softmax}}_{{{write_index_notation.subscript_of(letters)}}}(s) = '
            rf'\frac{{e^{{s}}}}{{{write_index_notation.sum_over(letters)}'
            rf'e^{{{write_index_notation.read_at("s", letters)}}}}}')


def l2_norm_formula[B: cat.Datatype, A: cat.Axis](target: cat.Broadcasted[B, A]) -> str:
    letters = consumed_letters(target)
    return (rf'L^{{2}}_{{{write_index_notation.subscript_of(letters)}}}(v) = '
            rf'\frac{{v}}{{\sqrt{{{write_index_notation.sum_over(letters)}'
            rf'{write_index_notation.read_at("v", letters)}^{{2}}}}}}')


GAIN_SYMBOL = r'\gamma'
BIAS_SYMBOL = r'\beta'


def normalize_formula[B: cat.Datatype, A: cat.Axis](target: cat.Broadcasted[B, A]) -> str:
    letters = consumed_letters(target)
    operator = target.operator
    scaled = (rf'x \left(\frac{{1}}{{{write_index_notation.element_count(letters)}}}'
              rf'{write_index_notation.sum_over(letters)}'
              rf'{write_index_notation.read_at("x", letters)}^{{2}} + '
              rf'{operator.epsilon.to_latex()}\right)^{{-1/2}}')
    if operator.gain:
        scaled = rf'{scaled} \odot {GAIN_SYMBOL}'
    if operator.bias:
        scaled = rf'{scaled} + {BIAS_SYMBOL}'
    return (rf'\mathrm{{RMSNorm}}_{{{write_index_notation.subscript_of(letters)}}}(x) = '
            rf'{scaled}')


def inverse_of_sum(epsilon: nm.Numeric) -> ops.Arithmetic:
    '''The map from a sum to its reciprocal, with `epsilon` added to the sum first:
    `s -> (s + epsilon)^{-1}`. At `ops.NO_EPSILON` the addition folds away and the map
    is the bare reciprocal.'''
    return ops.Arithmetic(
        formula=nm.Integer(1) / (nm.x + epsilon),
        name=fd.DynamicName(f'(x{added_epsilon_latex(epsilon)})^{{-1}}'
                            if epsilon != ops.NO_EPSILON else 'x^{-1}'))


@standard_expansions.register(
    ops.L1Norm,
    formula=l1_norm_formula,
    description=(text.L1_NORM_EXPANSION_DESCRIPTION))
def expand_l1_norm[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A],
    normaliser_first: bool = True,
) -> cat.BroadcastedCategory[B, A]:
    '''
    `copy ; ((sum ; 1/(z + epsilon)) * id) ; scale`, at the norm's own degree.
    Anything that is not an `L1Norm` comes back unchanged. The epsilon is the one the
    operator carries, and at `ops.NO_EPSILON` the reciprocal is the bare one.

    `normaliser_first` states which operand of the scale carries the reciprocal. The two
    orders are the same morphism up to a `Rearrangement`.
    '''
    if (not isinstance(target, cat.Broadcasted)
            or not isinstance(target.operator, ops.L1Norm)):
        return target
    array = tuple(target.dom())[0]
    datatype = array.datatype
    shape, kept = consumed_shape_and_degree(target)

    copy = cat.Rearrangement((0, 0), (array,))
    normaliser = chsh.make_composed(
        es.einsum((shape,), kept, datatype),
        es.einsum((kept,), kept, datatype,
                  operator=inverse_of_sum(target.operator.epsilon)))
    identity = cat.ProdObject((array,)).identity()
    if normaliser_first:
        scale = es.einsum((kept, shape), shape, datatype)
        product = chsh.make_product(normaliser, identity)
    else:
        scale = es.einsum((shape, kept), shape, datatype)
        product = chsh.make_product(identity, normaliser)
    return chsh.make_composed(copy, product, scale)


@dataclass
class ExpandL1Norm[B: cat.Datatype](functor.Endofunctor[
    cat.Array[B, cat.RawAxis], cat.Broadcasted[B, cat.RawAxis],
]):
    '''Every `L1Norm` in a morphism or hypergraph, written out.'''
    normaliser_first: bool = True
    def apply_root(self, target: cat.Broadcasted[B, cat.RawAxis]):
        return expand_l1_norm(target, normaliser_first=self.normaliser_first)


def expand_l1_norms(target):
    return ExpandL1Norm()(target)


INVERSE_ROOT = ops.Arithmetic(
    formula=nm.x ** (nm.Integer(-1) / nm.Integer(2)),
    name=fd.DynamicName('x^{-1/2}'))


@standard_expansions.register(
    ops.L2Norm,
    formula=l2_norm_formula,
    description=(text.L2_NORM_EXPANSION_DESCRIPTION))
def expand_l2_norm[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A],
) -> cat.BroadcastedCategory[B, A]:
    '''`copy ; ((square ; sum ; x^{-1/2}) * id) ; scale`, at the norm's own degree.
    Anything that is not an `L2Norm` comes back unchanged.

    The rule is `expand_normalize` without the division by the number of values,
    without the epsilon under the root and without the gain, which are the three
    things a `Normalize` has and this operator does not.
    '''
    if (not isinstance(target, cat.Broadcasted)
            or not isinstance(target.operator, ops.L2Norm)):
        return target
    array = tuple(target.dom())[0]
    datatype = array.datatype
    shape, kept = consumed_shape_and_degree(target)

    squared = es.einsum((shape,), shape, datatype,
                        operator=ops.Arithmetic(formula=nm.x ** nm.Integer(2)))
    normaliser = chsh.make_composed(
        squared,
        es.einsum((shape,), kept, datatype),
        es.einsum((kept,), kept, datatype, operator=INVERSE_ROOT))
    return chsh.make_composed(
        cat.Rearrangement((0, 0), (array,)),
        chsh.make_product(normaliser, cat.ProdObject((array,)).identity()),
        es.einsum((kept, shape), shape, datatype))


@dataclass
class ExpandL2Norm[B: cat.Datatype](functor.Endofunctor[
    cat.Array[B, cat.RawAxis], cat.Broadcasted[B, cat.RawAxis],
]):
    '''Every `L2Norm` in a morphism or hypergraph, written out.'''
    def apply_root(self, target: cat.Broadcasted[B, cat.RawAxis]):
        return expand_l2_norm(target)


def expand_l2_norms(target):
    return ExpandL2Norm()(target)


def exponential_of_softmax[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A],
) -> cat.BroadcastedCategory[B, A]:
    '''The `e^{x}` a softmax applies to its scores, at the softmax's degree.'''
    datatype = tuple(target.dom())[0].datatype
    shape, _ = consumed_shape_and_degree(target)
    return es.einsum((shape,), shape, datatype,
                     operator=ops.Arithmetic(formula=nm.E ** nm.x))


def l1_norm_of_softmax[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A],
) -> cat.Broadcasted[B, A]:
    '''The `ops.L1Norm` a softmax applies to its exponentiated scores.

    It reads the weaves and the reindexings the softmax reads, so it consumes
    the axis the softmax normalised over and carries the same degree.
    '''
    return target.reconstruct(operator=ops.L1Norm(
        name=fd.DynamicName('L^{1}'), contracted=target.operator.contracted))


def rewrite_softmax_as_l1_norm[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A],
) -> cat.BroadcastedCategory[B, A]:
    '''
    `exp ; L1Norm`, at the softmax's own degree, which is the softmax in one
    step rather than in five. Anything that is not a `SoftMax` comes back
    unchanged.
    '''
    if (not isinstance(target, cat.Broadcasted)
            or not isinstance(target.operator, ops.SoftMax)):
        return target
    return chsh.make_composed(exponential_of_softmax(target),
                              l1_norm_of_softmax(target))


@dataclass
class RewriteSoftMaxAsL1Norm[B: cat.Datatype](functor.Endofunctor[
    cat.Array[B, cat.RawAxis], cat.Broadcasted[B, cat.RawAxis],
]):
    '''Every `SoftMax` in a morphism or hypergraph, as `exp ; L1Norm`.'''
    def apply_root(self, target: cat.Broadcasted[B, cat.RawAxis]):
        return rewrite_softmax_as_l1_norm(target)


def rewrite_softmaxes_as_l1_norms(target):
    return RewriteSoftMaxAsL1Norm()(target)


@standard_expansions.register(
    ops.SoftMax,
    formula=softmax_formula,
    description=(text.SOFTMAX_EXPANSION_DESCRIPTION))
def expand_softmax[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A],
    normaliser_first: bool = True,
) -> cat.BroadcastedCategory[B, A]:
    '''
    `exp ; copy ; ((sum ; 1/z) * id) ; scale`, at the softmax's own degree, which
    is the exponential followed by the L1 norm written out. Anything that is not
    a `SoftMax` comes back unchanged.

    `normaliser_first` states which operand of the scale carries `1/z`. The two
    orders are the same morphism up to a `Rearrangement`.
    '''
    if (not isinstance(target, cat.Broadcasted)
            or not isinstance(target.operator, ops.SoftMax)):
        return target
    return chsh.make_composed(
        exponential_of_softmax(target),
        expand_l1_norm(l1_norm_of_softmax(target),
                       normaliser_first=normaliser_first))


def expand_shifted_softmax[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A],
) -> cat.BroadcastedCategory[B, A]:
    '''
    `copy ; (id * (max ; negate)) ; add ; exp ; copy ; ((sum ; 1/z) * id) ; scale`,
    at the softmax's own degree. Anything that is not a `SoftMax` comes back
    unchanged.
    '''
    if (not isinstance(target, cat.Broadcasted)
            or not isinstance(target.operator, ops.SoftMax)):
        return target
    array = tuple(target.dom())[0]
    datatype = array.datatype
    shape, kept = consumed_shape_and_degree(target)
    identity = cat.ProdObject((array,)).identity()
    maximum = es.einsum((shape,), kept, datatype, operator=ops.Maximum())
    negated = es.einsum((kept,), kept, datatype,
                        operator=ops.Arithmetic(formula=-1 * nm.x))
    shift = es.einsum((shape, kept), shape, datatype, operator=ops.AdditionOp())
    shifted = chsh.make_composed(
        cat.Rearrangement((0, 0), (array,)),
        chsh.make_product(identity, chsh.make_composed(maximum, negated)),
        shift)
    return chsh.make_composed(shifted, expand_softmax(target))


@dataclass
class ExpandShiftedSoftMax[B: cat.Datatype](functor.Endofunctor[
    cat.Array[B, cat.RawAxis], cat.Broadcasted[B, cat.RawAxis],
]):
    '''Every `SoftMax` in a morphism or hypergraph, written out with the shift.'''
    def apply_root(self, target: cat.Broadcasted[B, cat.RawAxis]):
        return expand_shifted_softmax(target)


def expand_shifted_softmaxes(target):
    return ExpandShiftedSoftMax()(target)


@dataclass
class ExpandSoftMax[B: cat.Datatype](functor.Endofunctor[
    cat.Array[B, cat.RawAxis], cat.Broadcasted[B, cat.RawAxis],
]):
    '''Every `SoftMax` in a morphism or hypergraph, written out.'''
    normaliser_first: bool = True
    def apply_root(self, target: cat.Broadcasted[B, cat.RawAxis]):
        return expand_softmax(target, normaliser_first=self.normaliser_first)


def expand_softmaxes(target):
    return ExpandSoftMax()(target)


def inverse_root_of_mean(
    element_count: nm.Numeric,
    epsilon: nm.Numeric,
) -> ops.Arithmetic:
    '''The map from a sum of `element_count` squares to the reciprocal of their root
    mean square, with `epsilon` added under the root:
    `s -> (s / element_count + epsilon)^{-1/2}`.'''
    return ops.Arithmetic(
        formula=((nm.x / element_count + epsilon)
                 ** (nm.Integer(-1) / nm.Integer(2))),
        name=fd.DynamicName(
            f'(x / {element_count.to_latex()}'
            f'{added_epsilon_latex(epsilon)})^{{-1/2}}'))


@standard_expansions.register(
    ops.Normalize,
    formula=normalize_formula,
    description=(text.NORMALIZE_EXPANSION_DESCRIPTION))
def expand_normalize[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A],
) -> cat.BroadcastedCategory[B, A]:
    '''
    `copy ; ((square ; sum ; (x/|m| + epsilon)^{-1/2}) * id) ; scale`, at the
    normalization's own degree, with a gain operand carried past the scaling and
    multiplied into the result and a bias operand added after it. Anything that is
    not a `Normalize` comes back unchanged.

    The sum is of the squares over every axis the operator consumes. The
    `Arithmetic` after it divides the sum by the number of elements summed, adds
    the epsilon the operator carries under the root, and takes the inverse square
    root, so the result scales each value by the reciprocal of the root mean square.
    The reviewer asked on 2026-09-16 for the expansion an inspection box draws to be
    complete, and this rule is the one the box draws.

    The gain and the bias are operands when `para.processing.show_grabbed_parameters`
    has grabbed them and turned each grab into a weight array. The expansion keeps
    those operands on their wires, as
    `algebra.linear_expansion.expand_parametrised_linear_root` keeps a weight on its
    own, so each array stays one operation. A `Normalize` that has been given no
    operand beyond its data expands to the scaling alone, whichever parameters it
    declares.
    '''
    if (not isinstance(target, cat.Broadcasted)
            or not isinstance(target.operator, ops.Normalize)):
        return target
    arrays = tuple(target.dom())
    data = arrays[-1]
    datatype = data.datatype
    inputs, output, _ = es.index_shapes(target)
    shape = inputs[-1]
    if shape != output:
        raise ValueError(
            f'a Normalize reads and writes one shape, and this one reads {shape} '
            f'and writes {output}')
    kept = tuple(target.input_weaves[-1].select_degree(shape))

    squared = es.einsum((shape,), shape, datatype,
                        operator=ops.Arithmetic(formula=nm.x ** nm.Integer(2)))
    element_count = functools.reduce(
        operator.mul,
        (axis.local_size() for axis in target.input_weaves[-1].target().shape()),
        nm.Integer(1))
    inverse_root = chsh.make_composed(
        es.einsum((shape,), kept, datatype),
        es.einsum((kept,), kept, datatype,
                  operator=inverse_root_of_mean(
                      element_count, target.operator.epsilon)))
    identity = cat.ProdObject((data,)).identity()
    normalized = chsh.make_composed(
        cat.Rearrangement((0, 0), (data,)),
        chsh.make_product(chsh.make_composed(squared, inverse_root), identity),
        es.einsum((kept, shape), shape, datatype))
    return combine_normalize_parameters(target, arrays[:-1], inputs[:-1], shape,
                                        datatype, normalized)


class NormalizeParametersDoNotMatchTheOperator(ValueError):
    '''A `Normalize` given a number of operands ahead of its data that is neither zero
    nor the number of parameters it declares.'''


def multiply_by_gain[B: cat.Datatype, A: cat.Axis](
    gain: cat.Array[B, A],
    gain_shape: fd.Prod[es.IndexVariable],
    shape: fd.Prod[es.IndexVariable],
    datatype: B,
    scaled: cat.BroadcastedCategory[B, A],
) -> cat.BroadcastedCategory[B, A]:
    '''`scaled` multiplied position by position by the gain, which stays on its own
    wire past the scaling so that the fusion can place the array as one operation.'''
    return chsh.make_composed(
        chsh.make_product(cat.ProdObject((gain,)).identity(), scaled),
        es.einsum((gain_shape, shape), shape, datatype))


def add_bias[B: cat.Datatype, A: cat.Axis](
    bias: cat.Array[B, A],
    bias_shape: fd.Prod[es.IndexVariable],
    shape: fd.Prod[es.IndexVariable],
    datatype: B,
    scaled: cat.BroadcastedCategory[B, A],
) -> cat.BroadcastedCategory[B, A]:
    '''`scaled` with the bias added position by position, which is the last step of a
    normalisation that declares one.'''
    return chsh.make_composed(
        chsh.make_product(cat.ProdObject((bias,)).identity(), scaled),
        es.einsum((bias_shape, shape), shape, datatype, operator=ops.AdditionOp()))


def multiply_by_gain_and_add_bias[B: cat.Datatype, A: cat.Axis](
    gain: cat.Array[B, A],
    bias: cat.Array[B, A],
    data: cat.Array[B, A],
    gain_shape: fd.Prod[es.IndexVariable],
    bias_shape: fd.Prod[es.IndexVariable],
    shape: fd.Prod[es.IndexVariable],
    datatype: B,
    scaled: cat.BroadcastedCategory[B, A],
) -> cat.BroadcastedCategory[B, A]:
    '''`scaled` multiplied by the gain and then offset by the bias, reading the three
    operands in the order `(gain, bias, data)` the operator supplies them in. The
    `cat.Rearrangement` carries the bias past the gain, because the product that holds
    the gain beside the data has to stand next to the data.'''
    return chsh.make_composed(
        cat.Rearrangement((1, 0, 2), (gain, bias, data)),
        chsh.make_product(
            cat.ProdObject((bias,)).identity(),
            multiply_by_gain(gain, gain_shape, shape, datatype, scaled)),
        es.einsum((bias_shape, shape), shape, datatype, operator=ops.AdditionOp()))


def combine_normalize_parameters[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A],
    parameters: fd.Prod[cat.Array[B, A]],
    parameter_shapes: fd.Prod[fd.Prod[es.IndexVariable]],
    shape: fd.Prod[es.IndexVariable],
    datatype: B,
    scaled: cat.BroadcastedCategory[B, A],
) -> cat.BroadcastedCategory[B, A]:
    '''`scaled` multiplied by the gain and then offset by the bias, for whichever of
    the two the operator declares. `parameters` are the operands ahead of the data, in
    the order `show_grabbed_parameters.parameter_arrays` grabs them, and an operator
    whose parameters have not been grabbed has none of them.'''
    operator = target.operator
    declared = sum((operator.gain, operator.bias))
    if len(parameters) == 0:
        return scaled
    if len(parameters) != declared:
        raise NormalizeParametersDoNotMatchTheOperator(
            f'a Normalize declaring gain={operator.gain} and bias={operator.bias} was '
            f'given {len(parameters)} operands ahead of its data')
    if operator.gain and operator.bias:
        gain, bias = parameters
        gain_shape, bias_shape = parameter_shapes
        return multiply_by_gain_and_add_bias(
            gain, bias, tuple(target.dom())[-1], gain_shape, bias_shape, shape,
            datatype, scaled)
    if operator.gain:
        return multiply_by_gain(parameters[0], parameter_shapes[0], shape, datatype,
                                scaled)
    return add_bias(parameters[0], parameter_shapes[0], shape, datatype, scaled)


@dataclass
class ExpandNormalize[B: cat.Datatype](functor.Endofunctor[
    cat.Array[B, cat.RawAxis], cat.Broadcasted[B, cat.RawAxis],
]):
    '''Every `Normalize` in a morphism or hypergraph, written out.'''
    def apply_root(self, target: cat.Broadcasted[B, cat.RawAxis]):
        return expand_normalize(target)


def expand_normalizes(target):
    return ExpandNormalize()(target)


@dataclass
class ExpandStandardOperators[B: cat.Datatype](functor.Endofunctor[
    cat.Array[B, cat.RawAxis], cat.Broadcasted[B, cat.RawAxis],
]):
    '''Every operator in a morphism or hypergraph that
    `algebra.registries.standard_expansions` holds a rule for, written out by that
    rule. A rule declines a form of its operator it does not write out by
    returning it, so a `Linear` whose weight is still inside the operator stays.'''
    def apply_root(self, target: cat.Broadcasted[B, cat.RawAxis]):
        if not isinstance(target, cat.Broadcasted):
            return target
        expanded = standard_expansions.expand_standard(target)
        return target if expanded is None else expanded


def expand_standard_operators(target):
    return ExpandStandardOperators()(target)
