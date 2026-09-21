'''The grouped low-rank output projection of DeepSeek-V4.1-Flash.

Written by Claude Opus 5, effort high.

The 64 heads are cut into 8 groups of 8, each group's 4096 values are mapped to its own
1024-wide rank by a weight of its own, and one final weight reads all eight ranks
together and returns the 5120-wide residual.

The first map is a `Linear` that produces the group and the rank, followed by a
diagonalisation keeping the entries where the group the weight produced and the group of
the input agree. An `Einops` against a parameter array states the same map, and the
`Linear` states it with the weight inside the operator, which is the form every other
weight in the model takes.

Every operation here is the same map at every query, so the projection is one box
computed once per query. The body is written for one query and
`project_over_all_queries` writes the same map out over every query, so that
`discovering_broadcasts.confirm_broadcast_expansion` can check the box against it.
`discover_broadcast_over_axes` cannot derive the box here, because the head split is a
reindexing that computes and the pass refuses to delete a degree position from one.
'''
from __future__ import annotations

import algebra.discovering_broadcasts as discovering_broadcasts
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import data_structure.Term as fd

from notebooks.sota.DeepSeekV41Flash.construction_idioms import over
from notebooks.sota.DeepSeekV41Flash.declared_axes import c, g, h, j, m, o, x
from notebooks.sota.DeepSeekV41Flash.reference_links import model_lines
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

OUTPUT_COLOUR = '#D5C8E8'


def split_heads_into_groups() -> cat.Broadcasted:
    '''One query's 64 heads cut into 8 groups of 8: h = |j| g + j on the head axis, times
    the identity on the channels.'''
    return ops.View.template(
        reindexing=(sc.StrideMorphism(
            _dom=(g, j),
            _cod_stride_shift=((h, (j.local_size(), nm.Integer(1)), nm.Integer(0)),),
            name=fd.DynamicName('grp')),
            cat.ProdObject((c,)).identity()),
        name='grp')


def contract_each_group_against_its_weight() -> cat.BroadcastedCategory:
    '''Each group's eight heads of 512 against that group's own weight: a `Linear` from
    the heads within a group and the channels onto the group and the rank, broadcast over
    the groups, then the diagonal keeping the entries where the group the weight produced
    and the group of the input agree.'''
    return (over((g,), ops.Linear.template((j, c), (g, o), 'W^{Oa}'))
            @ ops.View.template(
                reindexing=cat.Rearrangement((0, 0, 1), (g, o)),
                name='diag'))


def project_one_query() -> cat.Block:
    '''One query's heads into the residual width: the group split, the block-diagonal map
    into eight ranks of 1024, and one weight over all of them.'''
    return cat.Block.template(
        split_heads_into_groups()
        @ contract_each_group_against_its_weight()
        @ ops.Linear.template((g, o), (m,), 'W^{Ob}'),
        title=text.OUTPUT_TITLE, fill_color=OUTPUT_COLOUR,
        description=text.PROJECT_ONE_QUERY_DESCRIPTION,
        references=(model_lines(787),))


PROJECTION_BODY = project_one_query()


def project_over_all_queries() -> cat.BroadcastedCategory:
    '''The same map written out over every query, which the box is confirmed against. It
    lifts the body the box holds rather than building a second one, because
    `cat.Block.template` mints a fresh tag per call.'''
    return over((x,), PROJECTION_BODY)


PROJECTION = discovering_broadcasts.broadcast_block_over_axes(
    PROJECTION_BODY, (x,), ((0,),), 'Out')
PROJECTION_CONFIRMATION = discovering_broadcasts.confirm_broadcast_expansion(
    project_over_all_queries(), PROJECTION)


def output_projection() -> cat.Broadcasted:
    '''The projection as one box computed once per query.'''
    return PROJECTION
