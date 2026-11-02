# An einsum as a function of its shapes.
#
# An `Einops` `Broadcasted` is an einsum: multiply the operands, sum over every
# axis that is in an operand and not in the result. Written that way it is a
# function of shapes alone, meaning which axes each operand has and which the
# result has, and the weaves, the reindexings and the contraction groups are
# all recoverable from them. `einsum` is that reconstruction and `shapes` is
# the direction back.
#
# Two callers need it, which is why it is here rather than in either of them:
#
#   - `para.data_structure.contraction.contract` builds a contraction from
#     axis objects, because a reverse pass has to contract against axes that
#     already exist. It is this function under another name.
#   - anything rewriting a contraction, which is easier to state on shapes
#     than on weaves: "these operands, that result".
#
# ## Merging two of them lives elsewhere
#
# `algebra/einops_rearrange.py` owns that, and has since before this
# module: `merge_einops` inlines one `Einops` into the next and
# `disentangle_einops` then splits the result back into its independent
# components: always unify, then always disentangle. It carries each operand's own
# datatype through, which a rebuild from shapes cannot.
#
# So there is no merge here. `einops_rearrange.merge_rule` is that pass
# phrased as an `algebra.merge_into_consumer` rule, in the module beside this one.
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

import data_structure.Term as fd
import data_structure.Category as cat
import data_structure.Operators as ops
import para.data_structure.transpose as transpose
import term_utilities.term_utilities as tutil


type Shape[A: cat.Axis] = tuple[A, ...]


@dataclass(frozen=True, eq=False)
class IndexVariable:
    '''An index an operation runs over, with the axis that sizes it.

    A `Broadcasted` runs over one variable per degree position and one per
    contraction group. Two positions carrying one axis are two variables when
    they are two positions, and one variable when one reindexing reads one
    degree position at both. `index_shapes` reads the variables off a
    morphism and `einsum` builds a morphism from shapes written in them, so a
    repeated axis never stands for a repeated index.

    Two variables are equal when their labels are. The axis is the one at the
    position the variable was read from, so one variable read at two
    positions may carry two axis objects of one size, and a morphism built
    from it keeps each wire's own axes.
    '''
    label: object
    axis: cat.Axis

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IndexVariable) and self.label == other.label

    def __hash__(self) -> int:
        return hash(self.label)


type Entry = cat.Axis | IndexVariable


def _axis(entry: Entry) -> cat.Axis:
    return entry.axis if isinstance(entry, IndexVariable) else entry


def positional_shape(array: cat.Array) -> tuple[IndexVariable, ...]:
    '''One variable per position of `array`, for a map applied to it pointwise.'''
    return tuple(IndexVariable(('position', k), axis)
                 for k, axis in enumerate(array.shape()))


def index_shapes(
    target: cat.Broadcasted,
) -> tuple[tuple[tuple[IndexVariable, ...], ...], tuple[IndexVariable, ...],
           tuple[IndexVariable, ...]]:
    '''The operand shapes, the result shape and the degree of `target`, as
    index variables.

    A tiled position of an operand reads the degree position its reindexing
    names, so it carries that position's variable. A target position of an
    `Einops` carries its contraction group's variable, read off the signature
    the way `einops_rearrange.segment_group_axes` reads it. A target position
    of any other operator carries a variable labelled by its axis, because the
    operator states no grouping, so a `SoftMax` shares its target axis between
    its input and its output by that identity. `linear_indices_by_position`
    reads the targets of a `Linear` or a `Transpose` by position and pairs
    the weight's positions with the operand and result positions they index.

    Every reindexing must select, permute or repeat degree positions, so that
    `term_utilities.get_mapping` reads it as a mapping. A strided reindexing
    sends several degree positions to one, and its transpose is
    `transpose.ReindexTranspose` rather than a contraction.
    '''
    degree = tuple(IndexVariable(('degree', j), axis)
                   for j, axis in enumerate(target.degree()))
    signature = (target.operator.signature
                 if isinstance(target.operator, ops.Einops) else None)

    def variables(weave: cat.Weave, tiled: Iterable[IndexVariable],
                  groups: Iterable[int] | None) -> tuple[IndexVariable, ...]:
        tiled_variables = iter(tiled)
        group_numbers = iter(groups) if groups is not None else None
        return tuple(
            next(tiled_variables) if isinstance(entry, cat.WeaveMode)
            else IndexVariable(('group', next(group_numbers)), entry)
            if group_numbers is not None
            else IndexVariable(('target', entry), entry)
            for entry in weave._shape)

    inputs = []
    for i, (weave, reindexing) in enumerate(
            zip(target.input_weaves, target.reindexings)):
        try:
            mapping = tuple(tutil.get_mapping(reindexing))
        except (ValueError, KeyError) as error:
            raise ValueError(
                f'operand {i} of {target.operator} is read through a strided '
                'reindexing, which index_shapes cannot write as positions') from error
        read = tuple(degree[j] for j in mapping)
        inputs.append(variables(
            weave, read, signature[i] if signature is not None else None))
    output = variables(target.output_weaves[0], degree, None)
    if isinstance(target.operator, (ops.Linear, transpose.Transpose)):
        inputs, output = linear_indices_by_position(target, inputs, output)
    return tuple(inputs), output, degree


def linear_indices_by_position(
    target: cat.Broadcasted,
    inputs: list[tuple[IndexVariable, ...]],
    output: tuple[IndexVariable, ...],
) -> tuple[list[tuple[IndexVariable, ...]], tuple[IndexVariable, ...]]:
    '''The index variables of a linear map, pairing weight positions with data
    and result positions. Tiled positions keep their reindexing's variables.

    A parametrised map reads its weight first, an optional bias next, selection
    indices next, and the data last. The weight holds selection axes followed
    by contracted and produced target positions. A transpose exchanges those
    two target groups while keeping the selection axes and weight order.
    '''
    def label_target_positions(weave: cat.Weave, variables: tuple[IndexVariable, ...],
                   label: object) -> tuple[IndexVariable, ...]:
        return tuple(
            variable if isinstance(entry, cat.WeaveMode)
            else IndexVariable((label, position), variable.axis)
            for position, (entry, variable) in enumerate(zip(weave._shape, variables)))

    def replace_targets(weave: cat.Weave, variables: tuple[IndexVariable, ...],
                        labels: Iterable[IndexVariable]) -> tuple[IndexVariable, ...]:
        replacements = iter(labels)
        return tuple(
            variable if isinstance(entry, cat.WeaveMode)
            else IndexVariable(next(replacements).label, variable.axis)
            for entry, variable in zip(weave._shape, variables))

    result = label_target_positions(target.output_weaves[0], output, 'result')
    operands = [label_target_positions(weave, variables, ('operand', position))
                for position, (weave, variables) in enumerate(
                    zip(target.input_weaves, inputs))]
    if len(operands) < 2 or isinstance(target.input_weaves[0].datatype, cat.Natural):
        return operands, result
    produced = tuple(target.output_weaves[0].select_target(result))
    biased = isinstance(target.operator, ops.Linear) and target.operator.bias
    data_positions = range(2 if biased else 1, len(operands))
    contracted = tuple(variable for position in data_positions
                       for variable in target.input_weaves[position].select_target(
                           operands[position]))
    weight_weave = target.input_weaves[0]
    weight_variables = tuple(weight_weave.select_target(operands[0]))
    selected_count = len(weight_variables) - len(contracted) - len(produced)
    if selected_count < 0:
        raise ValueError(
            f'the weight of {target.operator} has {len(weight_variables)} target '
            f'positions for {len(contracted)} contracted and {len(produced)} '
            'produced positions')
    selected = weight_variables[:selected_count]
    weight_labels = (selected + produced + contracted
                     if isinstance(target.operator, transpose.Transpose)
                     else selected + contracted + produced)
    operands[0] = replace_targets(weight_weave, operands[0], weight_labels)
    if biased:
        operands[1] = replace_targets(target.input_weaves[1], operands[1], produced)
    return operands, result


def unique[A: cat.Axis](axes: Iterable[A]) -> Shape[A]:
    '''Order-preserving deduplication. `dict.fromkeys` keys on the axis Term.'''
    return tuple(dict.fromkeys(axes))


def einsum[B: cat.Datatype, A: cat.Axis](
    inputs: Iterable[Iterable[Entry]],
    output: Iterable[Entry],
    datatype: B = cat.Reals(),
    operator: cat.Operator | None = None,
) -> cat.Broadcasted[B, A]:
    '''
    Build the `Broadcasted` that reads operands of shapes `inputs` and writes
    one of shape `output`. A shape is a tuple of index variables, or of axes,
    in which case each axis is its own variable and two positions carrying one
    axis are read as one index. `index_shapes` and `positional_shape` write
    the variable form, which is the one a rewrite of an existing morphism
    should use.

    An axis is classified by where it appears:

        in the inputs only              absorbed, meaning it is contracted over
        in the output                   DEGREE   - it is broadcast over

    There is no third case. An axis in the output only is *repeated along*,
    and a repeat is not an operator: it is a degree axis that an operand's
    reindexing does not name. A reindexing maps the output's degree to the
    operand's, so an axis missing from it is one the operand is broadcast
    along, which is what repeating is. `sum over x` and `repeat along x` are
    the two readings of the same missing column, one in each direction.

    The operator is inferred unless given:

        anything absorbed              `Einops`, with the contraction groups
        nothing absorbed, one operand  `View` - a pure reindexing, which
                                       includes every repeat
        nothing absorbed, several      `Einops` with an empty signature, which
                                       is the pointwise product

    Passing `operator` overrides the inference, which is how an `AdditionOp` or
    an elementwise map gets built at a degree that is not written anywhere as a
    string.
    '''
    inputs = tuple(tuple(shape) for shape in inputs)
    output = tuple(output)
    if operator is None and inputs == (output,):
        # Nothing absorbed, nothing repeated, nothing permuted. Returning a real
        # identity rather than a `View` operator matters: `make_composed`
        # tests with `term_utilities.is_identity`, which does not look inside a
        # `Broadcasted`, so a no-op written as one rides the whole composition.
        return cat.ProdObject(
            (cat.Array(datatype, tuple(map(_axis, output))),)).identity()
    if len(unique(output)) != len(output):
        raise ValueError(
            'the result shape reads one index at two positions, which is a '
            f'written diagonal and has no Broadcasted form: {output}')

    in_entries = unique(entry for shape in inputs for entry in shape)
    # The degree is the whole output, in the output's order. A variable no
    # operand has is still degree, and it is one that no reindexing names.
    degree = tuple(map(_axis, output))
    absorbed = tuple(entry for entry in in_entries if entry not in set(output))
    absorbed_index = {entry: index for index, entry in enumerate(absorbed)}
    degree_index = {entry: index for index, entry in enumerate(output)}

    input_weaves = tuple(
        cat.Weave(datatype, tuple(
            _axis(entry) if entry in absorbed_index else cat.WeaveMode.TILED
            for entry in shape))
        for shape in inputs)
    output_weaves = (
        cat.Weave(datatype, (cat.WeaveMode.TILED,) * len(output)),)
    # A reindexing takes the output's degree to this operand's degree, so a
    # missing variable is a broadcast and a repeated one is a diagonal read.
    reindexings = tuple(
        cat.Rearrangement(
            tuple(degree_index[entry] for entry in shape
                  if entry in degree_index),
            degree)
        for shape in inputs)

    if operator is None:
        if absorbed:
            operator = ops.Einops(
                name=fd.DynamicName('einops'),
                signature=tuple(
                    tuple(absorbed_index[entry] for entry in shape
                          if entry in absorbed_index)
                    for shape in inputs))
        elif len(inputs) == 1:
            operator = ops.View()
        else:
            operator = ops.Einops(
                name=fd.DynamicName('einops'),
                signature=tuple(() for _ in inputs))

    return cat.Broadcasted[B, A](
        operator=operator,
        input_weaves=input_weaves,
        output_weaves=output_weaves,
        reindexings=reindexings,
        backup_degree=cat.ProdObject(degree) if not inputs else None,
    )


def is_einsum(target: cat.Morphism) -> bool:
    '''A `Broadcasted` whose operator multiplies its operands and sums.'''
    return (isinstance(target, cat.Broadcasted)
            and isinstance(target.operator, ops.Einops)
            and len(target.output_weaves) == 1)
