'''The reverse derivative of a covariant read, registered into `para`'s rule table.

Written by Claude Opus 5 (1M context), reasoning effort medium.

`para.registries.derivative.RULES` keys by operator type and `register` is its
decorator, so an operator declared outside `para` gets a reverse derivative by adding a
row rather than by editing the reverse functor. Import this module for its side effect,
as `import advanced_axis_dynamics.registries.derivative`, wherever a merge or a
concatenation has to be differentiated. `deepseek.registries.derivative` imports it,
because a merged selection holds an `aops.CovariantView`.

The three operators here are linear, so none declares a residual. A merge reverses
into the contravariant read of the same reindexing, an `ops.View`, which reads each
output position from the input position its reindexing names. A concatenation reverses
into the deconcatenation of the same part reindexings, and a deconcatenation reverses
into the concatenation.

`obsidian/02-categories/Advanced Axis Dynamics.md` states the feature, and
`obsidian/07-para/Derivatives.md` the reverse derivative the rules are written in.
'''
from __future__ import annotations

import data_structure.Category as cat
import data_structure.Operators as ops
import construction_helpers.lift as chl
import para.registries.derivative as derivative
import advanced_axis_dynamics.data_structure.Operators as aops


class CovariantViewNotExpanded(Exception):
    '''A `CovariantView` whose reindexing names axes other than its weaves carry.'''


@derivative.register(aops.CovariantView)
def covariant_view(
    target: cat.Broadcasted,
) -> tuple[derivative.Residual, cat.BroadcastedCategory]:
    '''A merge is a bijection, so its transpose is its inverse: the `View` of the
    same reindexing, with no residual. The reindexing has to name the axes the
    weaves carry, which is so once a merged selection has been expanded.'''
    operator = target.operator
    dom_axes = tuple(target.input_weaves[0].target().shape())
    cod_axes = tuple(target.output_weaves[0].target().shape())
    if (tuple(operator.reindexing.dom()) != dom_axes
            or tuple(operator.reindexing.cod()) != cod_axes):
        raise CovariantViewNotExpanded(
            f'{operator.reindexing} does not run between {dom_axes} and {cod_axes}')
    view = ops.View.template(
        base=target.input_weaves[0].datatype, reindexing=operator.reindexing)
    return derivative.Residual(), chl.morphism_object_lift(view, target.degree())


@derivative.register(aops.ConcatenateAxes)
def concatenate_axes(
    target: cat.Broadcasted,
) -> tuple[derivative.Residual, cat.BroadcastedCategory]:
    '''The cotangent on the concatenated axis cut back into the parts, which is the
    `aops.DeconcatenateAxes` of the same part reindexings.

    A concatenation is linear and has no residual. Its transpose reads each part's run
    of positions out of the one incoming cotangent, so part `k` receives the positions
    `[offset_k, offset_k + |part_k|)` and no others.
    `advanced_axis_dynamics/registries/standard_expansions.py` writes the
    deconcatenation out as a copy followed by one `ops.View` per part.
    '''
    return derivative.Residual(), cat.Broadcasted(
        operator=aops.DeconcatenateAxes(
            name=target.operator.name,
            part_reindexings=target.operator.part_reindexings),
        input_weaves=target.output_weaves,
        output_weaves=target.input_weaves,
        reindexings=(target.reindexings[0],))


@derivative.register(aops.DeconcatenateAxes)
def deconcatenate_axes(
    target: cat.Broadcasted,
) -> tuple[derivative.Residual, cat.BroadcastedCategory]:
    '''The cotangents on the parts laid end to end along the axis that was cut, which
    is the `aops.ConcatenateAxes` of the same part reindexings, with no residual.'''
    return derivative.Residual(), cat.Broadcasted(
        operator=aops.ConcatenateAxes(
            name=target.operator.name,
            part_reindexings=target.operator.part_reindexings),
        input_weaves=target.output_weaves,
        output_weaves=target.input_weaves,
        reindexings=(target.reindexings[0],) * len(target.output_weaves))
