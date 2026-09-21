# Claude Fable 5.1, effort 80.
'''Writing the formula and the description of one `Linear` from its own axes.

The reviewer asked on 2026-09-17 for the equation an inspection box shows over a weight
to be generated from the weight, to include the bias only where the map has one, and to
use the index notation of `algebra.write_index_notation`. `linear_formula` reads the
operands and the result of the `cat.Broadcasted` that carries the `Linear`, so the
query's down projection reads

    y[i_{q}] = \\sum_{i_{m} \\in m} x[i_{m}]\\, W_{W^{Qa}}[i_{m}, i_{q}]

and the sink logit, which has no operand, reads `y[i_{h}] = sink[i_{h}]`.

The parameters carry the names `show_grabbed_parameters.parameter_arrays` gives them,
which are the names of the slots an expansion draws on the tape, so the formula and the
figure under it name the same arrays. The axes of the degree are left out, because the
map is broadcast over them and the arrays are shared by every position of them. An
operand that selects among weights, per `ops.Linear`, is an index `j` the weight is read
at, and it is not summed over.

`algebra.linear_expansion` registers both functions with the rule that
writes a `Linear` out.
'''
from __future__ import annotations

import algebra.write_index_notation as write_index_notation
import data_structure.Category as cat
import data_structure.Operators as ops
import para.processing.show_grabbed_parameters as show_grabbed_parameters
from algebra.registries.expansion_wording import TEXT as text


def numbered(symbol: str, count: int) -> tuple[str, ...]:
    '''`symbol` once where `count` is one, and `symbol_{1}`, `symbol_{2}` otherwise.'''
    if count == 1:
        return (symbol,)
    return tuple(f'{symbol}_{{{number + 1}}}' for number in range(count))


def linear_formula[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A, ops.Linear],
) -> str:
    '''The LaTeX of what `target` computes at one index of its degree.'''
    parameters = show_grabbed_parameters.parameter_arrays(target)
    weight = parameters[0][0].to_latex()
    targets = tuple(weave.target() for weave in target.input_weaves)
    selecting = tuple(isinstance(operand.datatype, cat.Natural) for operand in targets)
    contracted_axes = tuple(
        axis for operand, selects in zip(targets, selecting) if not selects
        for axis in operand.shape())
    produced_axes = tuple(target.output_weaves[0].target().shape())
    letters = write_index_notation.axis_letters((*contracted_axes, *produced_axes))
    contracted, produced = letters[:len(contracted_axes)], letters[len(contracted_axes):]

    selections = iter(numbered('j', sum(selecting)))
    data = iter(numbered('x', len(selecting) - sum(selecting)))
    remaining = iter(contracted)
    factors: list[str] = []
    weight_indices: list[str] = []
    for operand, selects in zip(targets, selecting):
        if selects:
            weight_indices.append(next(selections))
            continue
        own = tuple(next(remaining) for _ in operand.shape())
        factors.append(write_index_notation.read_at(next(data), own))
        weight_indices.extend(write_index_notation.index_of(letter) for letter in own)
    weight_indices.extend(write_index_notation.index_of(letter) for letter in produced)
    read_weight = f'{weight}[{", ".join(weight_indices)}]' if weight_indices else weight
    product = r'\, '.join((*factors, read_weight))
    formula = (f'{write_index_notation.read_at("y", produced)} = '
               f'{write_index_notation.sum_over(contracted)}{product}')
    if target.operator.bias:
        bias = parameters[1][0].to_latex()
        formula += f' + {write_index_notation.read_at(bias, produced)}'
    return formula


def linear_description[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A, ops.Linear],
) -> str:
    '''What a `Linear` computes, in sentences, saying only what `target` has: no
    contraction for a learned array, and a bias only where there is one.'''
    if target.has_empty_domain():
        return (text.LEARNED_ARRAY_DESCRIPTION)
    sentences = [
        text.LINEAR_MAP_SENTENCE]
    if ops.selects_weights(target):
        sentences.append(
            text.LINEAR_INDEX_OPERAND_SENTENCE)
    if target.operator.bias:
        sentences.append(text.LINEAR_BIAS_SENTENCE)
    sentences.append(
        text.LINEAR_ARRAYS_FROM_THE_TAPE_SENTENCE)
    return ' '.join(sentences)
