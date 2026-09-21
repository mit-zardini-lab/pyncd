# Claude Opus 5, effort high.
'''The write of an array of values into the residual at token positions that arrive as
data, which is the scatter of an image span in DeepSeek-V4.1-Flash.

Written by Claude Fable 5.1, reasoning effort 80, in
`notebooks.sota.DeepSeekV41Flash.vision_pathway`. Split into this module by
Claude Opus 5, effort high, on 2026-09-19, when the user asked for every mechanism to be
stated in its newest form.

A reindexing maps every position of a result to a position of an operand by an affine
map, so it cannot state a write whose target position is read from an array. The
operator that states it is `para.data_structure.inject.Inject`, which the package added
for the reverse pass of a selection: it reads an array of indices and an array of
values, and returns an array holding, at each position, the sum of the values whose
index names that position. Every position no index names holds zero.
`obsidian/07-para/Selection and the Reverse Pass.md` records that a forward scatter is
the same operator.

    ones_over               an array of ones of the shape of the positions
    inject_at_positions     one `inject.Inject` onto the token axis, broadcast over the
                            trailing axes of the values
    write_at_positions      the values injected, the ones injected, and the residual
                            kept where no index names the position

`write_at_positions` returns `h (1 - Inject(pos, 1)) + Inject(pos, v)`, which holds the
value at every token an index names and the residual's own row at every other token. The
image pathway writes the cell features with it and the delimiter vectors with it again.

`INJECT_EXPLANATION` and `ONES_EXPLANATION` say what the two operators are, in the form
`operator_explanations` holds its own, and `explain_ones` is the row of the ones
of a write and the positive infinity of the pinned block by the value of the constant.
The explanation tables of the omissions notebook and of the integrated model both take
those rows from here.
'''
from __future__ import annotations

import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import para.data_structure.inject as inject

from notebooks.display.explain_operators import OperatorExplanation
from notebooks.sota.DeepSeekV41Flash.construction_idioms import hold, over, route
from notebooks.sota.DeepSeekV41Flash.declared_axes import R, m, x
from notebooks.sota.DeepSeekV41Flash.omitted_mechanisms import STATE
from notebooks.sota.DeepSeekV41Flash.reference_links import model_lines
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

TOKEN_MASK = cat.Array(R, (x,))


def ones_over(shape: tuple[cat.Axis, ...]) -> cat.Broadcasted:
    '''An array of `shape` that holds one at every index.'''
    return cat.Broadcasted(
        operator=ops.ConstantOp(value=nm.Integer(1)),
        input_weaves=(),
        output_weaves=(cat.Weave(R, (cat.WeaveMode.TILED,) * len(shape)),),
        reindexings=(),
        backup_degree=cat.ProdObject(tuple(shape)))


def inject_at_positions(
    positions: cat.Array,
    degree: tuple[cat.Axis, ...],
) -> cat.Broadcasted:
    '''An `inject.Inject` onto the token axis that adds each value at the token its
    index names and leaves zero at every other token. The index consumes the whole
    shape of `positions`. The values carry that shape and `degree` after it, and the
    write is the same at every index of `degree`, which the index ignores.
    `inject.inject_onto` gives the index the whole degree, so it cannot build this.'''
    tiled = (cat.WeaveMode.TILED,) * len(degree)
    index_shape = tuple(positions.shape())
    return cat.Broadcasted(
        operator=inject.Inject(),
        input_weaves=(cat.Weave(positions.datatype, index_shape),
                      cat.Weave(R, index_shape + tiled)),
        output_weaves=(cat.Weave(R, (x,) + tiled),),
        reindexings=(route((), tuple(degree)),
                     cat.ProdObject(tuple(degree)).identity()))


def write_at_positions(
    positions: cat.Array,
    values: cat.Array,
) -> cat.BroadcastedCategory:
    '''From the residual, `positions` and `values` to the residual with each value in
    place of the row its position names. The values are injected, a one is injected at
    the same positions, and the residual is multiplied by one minus the injected ones
    before the injected values are added.'''
    written_tokens = ((hold(positions) * ones_over(tuple(positions.shape())))
                      @ inject_at_positions(positions, ()))
    unwritten_tokens = written_tokens @ ops.Arithmetic.template(
        nm.Integer(1) - nm.x, base=TOKEN_MASK)
    return (route((0, 1, 1, 2), (STATE, positions, values))
            @ (hold(STATE) * unwritten_tokens * inject_at_positions(positions, (m,)))
            @ (ops.Einops.template('x m, x -> x m') * hold(STATE))
            @ over((x, m), ops.AdditionOp.template()))


INJECT_EXPLANATION = OperatorExplanation(
    title=r'\text{Inject}',
    formula=r'\mathrm{Inject}(p, v)[i_{x}] = \sum_{i \,:\, p[i] = i_{x}} v[i]',
    description=text.INJECT_DESCRIPTION,
    references=(model_lines(1238, 1239), model_lines(1235, 1237)))

ONES_EXPLANATION = OperatorExplanation(
    title=r'\text{Ones}',
    formula=r'1[i] = 1',
    description=text.ONES_DESCRIPTION,
    references=(model_lines(1239), model_lines(1235, 1237)))


def explain_ones(target: cat.Broadcasted) -> OperatorExplanation | None:
    '''The row for an `ops.ConstantOp` holding one, which is the array a write injects
    beside its values. Any other constant is left for the table that joins this row
    with the rows of the other constants.'''
    if target.operator.value == nm.Integer(1):
        return ONES_EXPLANATION
    return None
