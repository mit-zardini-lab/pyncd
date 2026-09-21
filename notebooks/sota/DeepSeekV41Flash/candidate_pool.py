'''The hierarchical sparse indexer of DeepSeek-V4.1-Flash and the pool it publishes.

Written by Claude Opus 5, effort high.

The decoder's Full layer scores every distance a query has reached, takes the best score
in each block of eight consecutive distances, keeps the 2048 best blocks and publishes
the 16,384 distances those blocks cover. Each Reindex layer below it scores inside that
pool alone, which costs the same however long the context is.

Splitting the distances into blocks of eight is affine and is a `View`. Putting a
per-block selection back onto the distances applies the same split to each kept block
number at every offset, which `dst.merge_selected_positions` does, and the covariant
view inside it lays the 2048 by 8 results out along the candidate axis.

Every operation of the pool is the same map at every token, so the pool is one box
computed once per token, `CANDIDATE_POOL`, which is what the decoder's Full mode
composes. The body reads one token's scores at `[r|x]`, and the slot axis keeps the
query axis as its guide although no array of the body carries it, because the guide says
which of its positions hold a value and the box's degree says which token they belong
to. `select_candidate_blocks_of_every_token` writes the same selection out over every
token, so that `discovering_broadcasts.confirm_broadcast_expansion` can check the box
against it.
'''
from __future__ import annotations

import advanced_axis_dynamics.algebra.mark_sparse_domains as mark_sparse_domains
import algebra.discovering_broadcasts as discovering_broadcasts
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import data_structure.Term as fd
import deepseek.data_structure as dst

from notebooks.sota.DeepSeekV41Flash.construction_idioms import over, route
from notebooks.sota.DeepSeekV41Flash.declared_axes import B, C, P, R, npool, nsel, u, x
from notebooks.sota.DeepSeekV41Flash.lightning_indexer import (
    entries_back_axis, reach_d, s_d)
from notebooks.sota.DeepSeekV41Flash.reference_links import model_lines
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

POOL_COLOUR = '#F5E6D9'
POOL_BOX = 'Pool'

BLOCK_SPLIT = sc.StrideMorphism(
    _dom=(P, u),
    _cod_stride_shift=((reach_d, (u.local_size(), nm.Integer(1)), nm.Integer(0)),),
    name=fd.DynamicName('blk'))
BLOCK_VIEW = mark_sparse_domains.guarded_view(reindexing=(BLOCK_SPLIT,), name='blk')
u_reach = BLOCK_VIEW.cod()[0].shape()[1]
P_reach = mark_sparse_domains.sparse_axis_after_fold(u_reach)
KEEP_BLOCKS = dst.TopK.template(k=npool, axis=P_reach, name='p/P',
                                form=dst.SelectionForm.ONLY_SELECTION)
p_axis, = KEEP_BLOCKS.cod()[0].shape()
COVER = dst.merge_selected_positions(BLOCK_SPLIT, p_axis, cat.ProdObject(()), C,
                                     name='cand')
POOL = cat.Array(cat.Natural(B.local_size()), (x, C))
back_d = entries_back_axis(B)
REINDEX_SELECT = dst.TopK.template(k=nsel, axis=C, selected_axis=s_d, positions_of=back_d,
                                   form=dst.SelectionForm.ONLY_SELECTION)


def select_candidate_blocks_of_one_token() -> cat.Block:
    '''One token's scores turned into the distances each Reindex layer scores inside:
    the kept block numbers, laid out at every offset along the candidate axis.'''
    block_max = over((P,), cat.Broadcasted(
        operator=ops.Maximum(),
        input_weaves=(cat.Weave(R, (u,)),),
        output_weaves=(cat.Weave(R, ()),),
        reindexings=(cat.ProdObject().identity(),)))
    return cat.Block.template(
        BLOCK_VIEW @ block_max @ KEEP_BLOCKS @ COVER,
        title=text.POOL_TITLE, fill_color=POOL_COLOUR,
        description=text.SELECT_CANDIDATE_BLOCKS_OF_ONE_TOKEN_DESCRIPTION,
        references=(model_lines(583, 609), model_lines(597, 598), model_lines(603, 604),
            model_lines(606), model_lines(608), model_lines(609)))


POOL_BODY = select_candidate_blocks_of_one_token()


def select_candidate_blocks_of_every_token() -> cat.BroadcastedCategory:
    '''The same selection written out over every token, which the box is confirmed
    against. It lifts the body the box holds rather than building a second one, because
    `cat.Block.template` mints a fresh tag per call.'''
    return over((x,), POOL_BODY)


CANDIDATE_POOL = discovering_broadcasts.broadcast_block_over_axes(
    POOL_BODY, (x,), ((0,),), POOL_BOX)
POOL_CONFIRMATION = discovering_broadcasts.confirm_broadcast_expansion(
    select_candidate_blocks_of_every_token(), CANDIDATE_POOL)


def restrict_to_candidates() -> cat.BroadcastedCategory:
    '''A Reindex layer's own scores read at the pool's distances, before its Top-512.'''
    T = cat.WeaveMode.TILED
    scores = cat.Array(R, (x, reach_d))
    return route((1, 0), (scores, POOL)) @ cat.Broadcasted(
        operator=dst.IndexSelect(),
        input_weaves=(cat.Weave(cat.Natural(B.local_size()), (T, T)),
                      cat.Weave(R, (T, reach_d))),
        output_weaves=(cat.Weave(R, (T, T)),),
        reindexings=(cat.ProdObject((x, C)).identity(),
                     cat.Rearrangement((0,), (x, C))))
