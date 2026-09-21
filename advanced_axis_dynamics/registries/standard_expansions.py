# Claude Fable 5.1, effort 80.
'''The standard expansion of a deconcatenation, registered into the table of
`algebra.registries.standard_expansions`.

Import this module for its side effect, as
`import advanced_axis_dynamics.registries.standard_expansions`, wherever a
deconcatenation has to be written out or shown beside its expansion.

`obsidian/02-categories/Advanced Axis Dynamics.md` states the feature.
'''
from __future__ import annotations

import data_structure.Term as fd # for 'foundations'
import data_structure.Category as cat
import data_structure.Operators as ops
import construction_helpers.product as chp
from construction_helpers import simple_helper as chsh
import algebra.registries.standard_expansions as standard_expansions
import advanced_axis_dynamics.data_structure.Operators as aops
from algebra.registries.expansion_wording import TEXT as text


@standard_expansions.register(
    aops.DeconcatenateAxes,
    formula='y_k[j] = x\\Big[j + \\sum_{l < k} \\lvert p_l \\rvert\\Big]',
    description=(
        text.DECONCATENATION_EXPANSION_DESCRIPTION))
def copy_then_view_each_part(
    target: cat.Broadcasted,
) -> cat.BroadcastedCategory:
    '''One copy of the input per part, each read through the `ops.View` of that part's
    reindexing.'''
    whole = target.dom()[0]
    views = tuple(part_view(target, reindexing)
                  for reindexing in target.operator.part_reindexings)
    copy = cat.Rearrangement(mapping=(0,) * len(views), _dom=(whole,))
    return chsh.make_composed(copy, chsh.make_product(*views))


def part_view(target: cat.Broadcasted,
              reindexing: cat.StrideMorphism) -> cat.BroadcastedCategory:
    '''The `ops.View` reading the whole array at the positions of one part, which is
    that part's reindexing beside the identity on the degree.

    The axis being cut stands at one position of the input weave and the degree fills
    the rest, so the reindexing is the product of the identity on the degree axes
    before it, the part's own row, and the identity on the degree axes after it.
    '''
    degree = tuple(target.degree())
    shape = target.input_weaves[0]._shape
    position = next(index for index, entry in enumerate(shape)
                    if not isinstance(entry, cat.WeaveMode))
    return ops.View.template(
        base=target.input_weaves[0].datatype,
        reindexing=chp.morphism_product((
            cat.ProdObject(degree[:position]).identity(),
            reindexing,
            cat.ProdObject(degree[position:]).identity())),
        name=part_name(reindexing))


def part_name(reindexing: cat.StrideMorphism) -> fd.DynamicName | None:
    '''The name the `View` of one part draws under, which is the part axis's own name
    where the reindexing has none.'''
    if reindexing.name is not None:
        return reindexing.name
    return getattr(reindexing._dom[0].uid, '_name', None)
