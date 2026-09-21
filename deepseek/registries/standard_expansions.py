# Claude Fable 5.1, effort 80.
'''The standard expansions of the two rotary tables, registered into the table of
`algebra.registries.standard_expansions`.

Import this module for its side effect, as
`import deepseek.registries.standard_expansions`, wherever a `dst.Rotary` or a
`dst.YarnRotary` has to be written out or shown beside its expansion.
`websocket_transfer/validate_auxiliary_information.py` asserts which operators the
registry holds when nothing but the core is loaded, so `deepseek/data_structure.py`
does not import this module.

A rule writes the table the operator's docstring states. The positions are an
`ops.Arrange` over the position axis. Where the position stride is not one they pass
through an `ops.Arithmetic` from `Natural` to `Natural` that multiplies by the stride.
The stride could instead be a factor of the angle. It is applied to the position
because the figure then carries the token position `|a| i_b` on a wire of its own
before the wire meets the frequencies, which is the quantity the released code reads
with `freqs_cis[::ratio]`. The frequencies are an `ops.Arithmetic` applied to an
`ops.Arrange` over the pair axis. For YaRN the power of the base and the ramp read one
copied `ops.Arrange`, as `yarn_frequencies_defined` of
`notebooks/sota/DeepSeekV41Flash/omitted_mechanisms.py` writes them.
`ops.multiply_by_positions` forms the angle of every position and pair, and the last
`ops.Arithmetic` is `e^{i x}`. A table turns counterclockwise, and an inverse rotary
embedding conjugates it with `dst.conjugate_complex_values` after the table rather
than holding a table of its own.

Every `ops.Arithmetic` is named by hand, in LaTeX written from the operator's own
fields, because `to_latex` prints a quotient as a product with a power of minus one.
A figure drawn with assigned sizes writes each size onto the name of its symbol, and
the names, the formula and the description leave those sizes out.

`obsidian/02-categories/Operators.md` lists the two operators, and
`obsidian/06-practice/Representing Models.md` holds the rulings on how a rotation is
written.
'''
from __future__ import annotations

from dataclasses import dataclass

import algebra.einops_simplification as einops_simplification
import algebra.registries.standard_expansions as standard_expansions
import algebra.write_index_notation as write_index_notation
import construction_helpers.lift as chl
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd  # for 'foundations'
import deepseek.data_structure as dst
from construction_helpers import simple_helper as chsh
from algebra.registries.expansion_wording import TEXT as text

IMAGINARY_UNIT = nm.Constant(nm.ConstantSymbol.IMAGINARY_UNIT)
UNIT_STRIDE = nm.Integer(1)
FREQUENCY_SYMBOL = '\\theta'
YARN_FREQUENCY_SYMBOL = "\\theta'"
RAMP_SYMBOL = 'r'
TABLE_SYMBOL = 'F'
TRANSPOSE_NAME = 'tr'

TURN_FACTOR_LATEX = 'e^{\\mathrm{i} x}'


class RotaryTableHasOtherTarget(Exception):
    '''A `dst.Rotary` whose output weave does not hold exactly the position axis and
    the pair axis in its target.'''


@dataclass(frozen=True)
class RotaryTable[B: cat.Datatype, A: cat.Axis, O: dst.Rotary]:
    '''What a rule reads off the `cat.Broadcasted` that carries a rotary operator:
    the two axes in the target of its output weave, the datatype of the real and the
    imaginary part, and the LaTeX letter of each axis.'''
    operator: O
    positions: A
    pairs: A
    real: B
    position_letter: str
    pair_letter: str

    def pair_count_latex(self) -> str:
        return write_index_notation.element_count((self.pair_letter,))

    def pair_index_latex(self) -> str:
        return write_index_notation.index_of(self.pair_letter)

    def read_at_pair(self, array: str) -> str:
        return write_index_notation.read_at(array, (self.pair_letter,))

    def has_unit_stride(self) -> bool:
        return self.operator.position_stride == UNIT_STRIDE


def rotary_table_of[B: cat.Datatype, A: cat.Axis, O: dst.Rotary](
    target: cat.Broadcasted[dst.Complex[B], A, O],
) -> RotaryTable[B, A, O]:
    table_axes = tuple(target.output_weaves[0].target().shape())
    if len(table_axes) != 2:
        raise RotaryTableHasOtherTarget(
            f'{len(table_axes)} axes in the target of a {type(target.operator).__name__}'
            f': {table_axes}')
    positions, pairs = table_axes
    position_letter, pair_letter = write_index_notation.axis_letters(table_axes)
    return RotaryTable(
        operator=target.operator, positions=positions, pairs=pairs,
        real=target.output_weaves[0].datatype.base,
        position_letter=position_letter, pair_letter=pair_letter)


def without_written_sizes[T: fd.GeneralTerm](value: T) -> T:
    '''`value` with the exponent taken off every name in it. A figure drawn with
    assigned sizes writes each size onto the name of its symbol as an exponent, so
    the size of `a` prints `|a|_{2}`. A formula names the symbol alone, as
    `write_index_notation.axis_letters` names an axis by its letter alone.'''
    if isinstance(value, fd.DynamicName):
        return value if value.exponent is None else value.with_exponent(None)
    return fd.deep_reconstruct(value, without_written_sizes)


def symbol_latex(value: nm.Numeric) -> str:
    return without_written_sizes(value).to_latex()


def operand_latex(value: nm.Numeric) -> str:
    '''The LaTeX of `value` as one operand of a product, a difference, a quotient or
    a power, which is in parentheses where `value` is a sum or a product.'''
    latex = symbol_latex(value)
    return f'({latex})' if isinstance(value, (nm.Addition, nm.Multiplication)) else latex


def named_by_formula(latex: str) -> fd.DynamicName:
    '''The name of an `ops.Arithmetic` whose whole LaTeX is one body.
    `fd.DynamicName.from_str` splits a text at its first underscore and reads the
    rest as a subscript, and a base written with a subscript of its own holds one.'''
    return fd.DynamicName(body=latex)


def token_positions[B: cat.Datatype, A: cat.Axis, O: dst.Rotary](
    table: RotaryTable[B, A, O],
) -> cat.BroadcastedCategory[cat.Natural, A]:
    '''The token position every index of the position axis stands for, which is the
    index itself at the stride one and the stride times the index otherwise.'''
    arranged = ops.Arrange.template(table.positions)
    if table.has_unit_stride():
        return arranged
    stride = table.operator.position_stride
    multiply_by_stride = ops.Arithmetic.template(
        stride * nm.x, base=arranged.cod()[0],
        output_datatype=cat.Natural(stride * table.positions.local_size()),
        name=named_by_formula(f'{operand_latex(stride)} x'))
    return chsh.make_composed(arranged, multiply_by_stride)


def raise_base_to_pair_index[B: cat.Datatype, A: cat.Axis, O: dst.Rotary](
    table: RotaryTable[B, A, O],
    pair_indices: cat.Array[cat.Natural, A],
) -> cat.Broadcasted[B | cat.Natural, A]:
    '''The frequency `base^{-i_t / |t|}` of every pair, read off the index of the
    pair. An index is a natural number and a frequency is a real.'''
    base = table.operator.base
    return ops.Arithmetic.template(
        nm.Power.template(base, nm.Integer(-1) * nm.x / table.pairs.local_size()),
        base=pair_indices, output_datatype=table.real,
        name=named_by_formula(
            f'{operand_latex(base)}^{{-x / {table.pair_count_latex()}}}'))


def rotary_frequencies[B: cat.Datatype, A: cat.Axis, O: dst.Rotary](
    table: RotaryTable[B, A, O],
) -> cat.BroadcastedCategory[B | cat.Natural, A]:
    pair_indices = ops.Arrange.template(table.pairs)
    return chsh.make_composed(
        pair_indices, raise_base_to_pair_index(table, pair_indices.cod()[0]))


def ramp_latex(index: str, start: nm.Numeric, end: nm.Numeric) -> str:
    '''The YaRN ramp at `index`, `(index - start) / (end - start)` clamped to the
    unit interval, which corner brackets denote as `nm.Clamp` prints them.'''
    return (f'\\ulcorner ({index} - {operand_latex(start)}) / '
            f'({symbol_latex(end)} - {operand_latex(start)}) \\lrcorner')


def interpolation_latex(ramp: str, factor: nm.Numeric) -> str:
    return f'1 - {ramp} + {ramp} / {operand_latex(factor)}'


def clamp_pair_index_to_ramp[B: cat.Datatype, A: cat.Axis](
    table: RotaryTable[B, A, dst.YarnRotary],
    pair_indices: cat.Array[cat.Natural, A],
) -> cat.Broadcasted[B | cat.Natural, A]:
    '''The YaRN ramp of every pair, which is zero up to the pair `ramp_start`, one
    from the pair `ramp_end`, and linear between them.'''
    start, end = table.operator.ramp_start, table.operator.ramp_end
    return ops.Arithmetic.template(
        nm.Clamp((nm.x - start) / (end - start)),
        base=pair_indices, output_datatype=table.real,
        name=named_by_formula(ramp_latex('x', start, end)))


def interpolate_along_ramp[B: cat.Datatype, A: cat.Axis](
    table: RotaryTable[B, A, dst.YarnRotary],
) -> cat.Broadcasted[B, A]:
    '''The factor `1 - r + r / factor` of a ramp value `r`, which is one where the
    ramp is zero and the reciprocal of the YaRN factor where the ramp is one.'''
    factor = table.operator.factor
    return ops.Arithmetic.template(
        nm.Integer(1) - nm.x + nm.x / factor,
        base=cat.Array(table.real, (table.pairs,)),
        name=named_by_formula(interpolation_latex('x', factor)))


def yarn_frequencies[B: cat.Datatype, A: cat.Axis](
    table: RotaryTable[B, A, dst.YarnRotary],
) -> cat.BroadcastedCategory[B | cat.Natural, A]:
    '''The power of the base times the factor that interpolates along the ramp. Both
    read the index of the pair, so the indices are arranged once and copied.'''
    pair_indices = ops.Arrange.template(table.pairs)
    indices = pair_indices.cod()[0]
    return chsh.make_composed(
        pair_indices,
        cat.Rearrangement((0, 0), (indices,)),
        chsh.make_product(
            raise_base_to_pair_index(table, indices),
            chsh.make_composed(clamp_pair_index_to_ramp(table, indices),
                               interpolate_along_ramp(table))),
        einops_simplification.einsum(
            ((table.pairs,), (table.pairs,)), (table.pairs,), table.real))


def rotation_factors[B: cat.Datatype, A: cat.Axis, O: dst.Rotary](
    table: RotaryTable[B, A, O],
) -> cat.Broadcasted[B | dst.Complex[B], A]:
    '''The factor `e^{\\mathrm{i} y}` of every angle `y`. An angle is a real and a
    factor is a complex number.'''
    return ops.Arithmetic.template(
        nm.Power.template(nm.Constant(), IMAGINARY_UNIT * nm.x),
        base=cat.Array(table.real, (table.positions, table.pairs)),
        output_datatype=dst.Complex(table.real),
        name=named_by_formula(TURN_FACTOR_LATEX))


def table_from_frequencies[B: cat.Datatype, A: cat.Axis, O: dst.Rotary](
    target: cat.Broadcasted[dst.Complex[B], A, O],
    table: RotaryTable[B, A, O],
    frequencies: cat.BroadcastedCategory[B | cat.Natural, A],
) -> cat.BroadcastedCategory[B | cat.Natural | dst.Complex[B], A]:
    '''The factor of every position and pair, from the frequency of every pair, with
    the domain and the codomain of `target`.

    The table is written over the position axis and the pair axis alone and then
    lifted over the degree of `target`, so an operator broadcast over a degree
    expands at that degree.
    '''
    positions = token_positions(table)
    over_positions_and_pairs = chsh.make_composed(
        chsh.make_product(positions, frequencies),
        ops.multiply_by_positions(positions.cod()[0], frequencies.cod()[0]),
        rotation_factors(table))
    return in_the_layout_of(
        target, chl.morphism_object_lift(over_positions_and_pairs, target.degree()))


def in_the_layout_of[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A],
    degree_first: cat.BroadcastedCategory[B, A],
) -> cat.BroadcastedCategory[B, A]:
    '''`degree_first`, whose one result carries the degree of `target` ahead of the
    axes of its target, followed by the `ops.View` that moves every axis to the
    position the output weave of `target` gives it. A weave that already leads with
    its tiled positions needs no view.'''
    weave = target.output_weaves[0]
    is_tiled = tuple(isinstance(entry, cat.WeaveMode) for entry in weave._shape)
    tiled_count = sum(is_tiled)
    produced_positions = tuple(
        sum(is_tiled[:wanted]) if tiled
        else tiled_count + wanted - sum(is_tiled[:wanted])
        for wanted, tiled in enumerate(is_tiled))
    if produced_positions == tuple(range(len(produced_positions))):
        return degree_first
    wanted_position_of = {produced: wanted
                          for wanted, produced in enumerate(produced_positions)}
    return chsh.make_composed(degree_first, ops.View.template(
        base=weave.datatype,
        reindexing=cat.Rearrangement(
            mapping=tuple(wanted_position_of[produced]
                          for produced in range(len(produced_positions))),
            _dom=tuple(target.cod()[0].shape())),
        name=TRANSPOSE_NAME))


def factor_formula[B: cat.Datatype, A: cat.Axis, O: dst.Rotary](
    table: RotaryTable[B, A, O], frequency_symbol: str) -> str:
    '''`F[i_x, i_t] = e^{i stride i_x frequency[i_t]}`, with no stride where the
    stride is one.'''
    stride = ('' if table.has_unit_stride()
              else f'{operand_latex(table.operator.position_stride)}\\, ')
    entry = write_index_notation.read_at(
        TABLE_SYMBOL, (table.position_letter, table.pair_letter))
    factor = (f'e^{{\\mathrm{{i}}\\, {stride}'
              f'{write_index_notation.index_of(table.position_letter)}\\, '
              f'{table.read_at_pair(frequency_symbol)}}}')
    return f'{entry} = {factor}'


def base_power_latex[B: cat.Datatype, A: cat.Axis, O: dst.Rotary](
    table: RotaryTable[B, A, O]) -> str:
    return (f'{operand_latex(table.operator.base)}^{{-{table.pair_index_latex()} / '
            f'{table.pair_count_latex()}}}')


def rotary_formula[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[dst.Complex[B], A, dst.Rotary],
) -> str:
    table = rotary_table_of(target)
    return (f'{factor_formula(table, FREQUENCY_SYMBOL)},\\quad '
            f'{table.read_at_pair(FREQUENCY_SYMBOL)} = {base_power_latex(table)}')


def yarn_rotary_formula[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[dst.Complex[B], A, dst.YarnRotary],
) -> str:
    table = rotary_table_of(target)
    operator = table.operator
    ramp = table.read_at_pair(RAMP_SYMBOL)
    return (f'{factor_formula(table, YARN_FREQUENCY_SYMBOL)},\\quad '
            f'{table.read_at_pair(YARN_FREQUENCY_SYMBOL)} = {base_power_latex(table)}'
            f'\\, ({interpolation_latex(ramp, operator.factor)}),\\quad {ramp} = '
            f'{ramp_latex(table.pair_index_latex(), operator.ramp_start, operator.ramp_end)}')


TURN_SENTENCE = (
    text.ROTARY_TURN_SENTENCE)


def table_sentence[B: cat.Datatype, A: cat.Axis, O: dst.Rotary](
    table: RotaryTable[B, A, O]) -> str:
    if table.has_unit_stride():
        return text.ROTARY_TABLE_UNIT_STRIDE_SENTENCE.format(
            position_letter=table.position_letter, pair_letter=table.pair_letter)
    stride = symbol_latex(table.operator.position_stride)
    return text.ROTARY_TABLE_STRIDE_SENTENCE.format(
        position_letter=table.position_letter, pair_letter=table.pair_letter,
        stride=stride)


FASTEST_PAIR_SENTENCE = (
    text.ROTARY_FASTEST_PAIR_SENTENCE)

FASTEST_PAIR_BEFORE_YARN_SENTENCE = (
    text.ROTARY_FASTEST_PAIR_BEFORE_YARN_SENTENCE)

YARN_SENTENCE = (
    text.YARN_SENTENCE)


def rotary_description[B: cat.Datatype, A: cat.Axis, O: dst.Rotary](
    target: cat.Broadcasted[dst.Complex[B], A, O],
) -> str:
    table = rotary_table_of(target)
    return ' '.join((TURN_SENTENCE, table_sentence(table),
                     FASTEST_PAIR_SENTENCE))


def yarn_rotary_description[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[dst.Complex[B], A, dst.YarnRotary],
) -> str:
    table = rotary_table_of(target)
    return ' '.join((TURN_SENTENCE, table_sentence(table),
                     FASTEST_PAIR_BEFORE_YARN_SENTENCE, YARN_SENTENCE))


@standard_expansions.register(
    dst.Rotary, formula=rotary_formula, description=rotary_description)
def write_rotary_table_out[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[dst.Complex[B], A, dst.Rotary],
) -> cat.BroadcastedCategory[B | cat.Natural | dst.Complex[B], A]:
    table = rotary_table_of(target)
    return table_from_frequencies(target, table, rotary_frequencies(table))


@standard_expansions.register(
    dst.YarnRotary, formula=yarn_rotary_formula, description=yarn_rotary_description)
def write_yarn_rotary_table_out[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[dst.Complex[B], A, dst.YarnRotary],
) -> cat.BroadcastedCategory[B | cat.Natural | dst.Complex[B], A]:
    table = rotary_table_of(target)
    return table_from_frequencies(target, table, yarn_frequencies(table))
