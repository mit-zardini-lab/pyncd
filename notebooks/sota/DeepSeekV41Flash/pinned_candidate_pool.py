'''The candidate pool of DeepSeek-V4.1-Flash with the newest block of entries pinned.

Written by Claude Fable 5.1, reasoning effort 80, and moved into this package by Claude
Fable 5.1, reasoning effort 25, on 2026-09-19, when the user asked for every mechanism to
be stated in its newest form.

`notebooks.sota.DeepSeekV41Flash.candidate_pool` scores every block of eight distances
by its best entry and keeps the 2048 best blocks. The released code gives one block a
score of positive infinity before it chooses, so that block is always among the kept.
This module writes that step into the pool, between the block maximum and the
Top-2048, and keeps the pool one box computed once per token.

The model numbers a query's entries by their distance back from the newest entry the
query reaches. Block 0 therefore holds the newest entries of every query, eight of them
once the query reaches eight, and the pin is the same map at every token. A constant
holding `nm.ConstantSymbol.INFINITY` is written at block 0 by an `aops.CovariantView`
of the row that names position 0, which is how the package writes a one-hot array.
Every other block of the written array holds the universal unit, which an addition
reads as zero, so the addition leaves those blocks at their scores.

The released code numbers its blocks by absolute entry and pins the block that holds
the query's newest entry, `last = (compress_lens - 1) // block_size`. That block holds
between one and eight entries, and all of them are among the eight newest, so every
entry the released code pins is in block 0 here. Block 0 holds up to seven older
entries beside them. The division of the entries into blocks already differs between
the two numberings wherever the count a query reaches is not a multiple of eight.

The view into blocks, the Top-2048 and the cover are imported from `candidate_pool`
unchanged, and a mode still takes `POOL`, `REINDEX_SELECT` and
`restrict_to_candidates` from that module. `PINNED_CANDIDATE_POOL` has the domain and
the codomain of `candidate_pool.CANDIDATE_POOL` and stands in its place.
'''
from __future__ import annotations

import advanced_axis_dynamics.data_structure.Operators as aops
import algebra.discovering_broadcasts as discovering_broadcasts
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import data_structure.Term as fd

import notebooks.sota.DeepSeekV41Flash.candidate_pool as candidate_pool
import notebooks.sota.DeepSeekV41Flash.omitted_mechanisms as omitted_mechanisms
from notebooks.sota.DeepSeekV41Flash.construction_idioms import hold, over
from notebooks.sota.DeepSeekV41Flash.declared_axes import R, x
from notebooks.sota.DeepSeekV41Flash.reference_links import model_lines
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

PIN = 'pin'
PIN_COLOUR = '#FBF3D5'
POSITIVE_INFINITY = nm.Constant(nm.ConstantSymbol.INFINITY)
BLOCK_SCORES_OF_ONE_TOKEN = cat.Array(R, (candidate_pool.P_reach,))


def positive_infinity() -> cat.Broadcasted:
    return cat.Broadcasted(
        operator=ops.ConstantOp(value=POSITIVE_INFINITY),
        input_weaves=(),
        output_weaves=(cat.Weave(R, ()),),
        reindexings=())


def write_at_block_zero() -> cat.Broadcasted:
    '''One number written at block 0 of an array over the blocks a query reaches. The
    row has an empty domain and names position 0, so its covariant reading writes that
    one position, and every other block holds the universal unit.
    `aops.CovariantView.template` asks for a bijection between two index boxes, which
    a row onto one position of many is not, so the node is written by hand.'''
    block_zero = sc.StrideMorphism(
        _dom=(), _cod_stride_shift=((candidate_pool.P_reach, (), nm.Integer(0)),),
        name=fd.DynamicName(PIN))
    return cat.Broadcasted(
        operator=aops.CovariantView(name=fd.DynamicName(PIN), reindexing=block_zero),
        input_weaves=(cat.Weave(R, ()),),
        output_weaves=(cat.Weave(R, (candidate_pool.P_reach,)),),
        reindexings=(cat.ProdObject().identity(),))


def pin_newest_block() -> cat.Block:
    '''One token's block scores with positive infinity added at block 0. Under the
    model's numbering the newest entry is at distance 0 for every query, so the block
    the reference pins is block 0 at every token and the pin is the same map at every
    token.'''
    infinity_at_block_zero = positive_infinity() @ write_at_block_zero()
    return cat.Block.template(
        (hold(BLOCK_SCORES_OF_ONE_TOKEN) * infinity_at_block_zero)
        @ over((candidate_pool.P_reach,), ops.AdditionOp.template()),
        title=text.PIN_TITLE, fill_color=PIN_COLOUR,
        formula=(r'\mathrm{pin}(b)[i_{P}] = b[i_{P}] + J[i_{P}], \qquad '
                 r'J[0] = \infty, \qquad J[i_{P}] = 0 \text{ for } i_{P} \neq 0'),
        description=text.PIN_NEWEST_BLOCK_DESCRIPTION,
        references=(model_lines(602, 605), model_lines(597, 599), model_lines(607)))


def select_candidate_blocks_of_one_token() -> cat.Block:
    '''`candidate_pool.select_candidate_blocks_of_one_token` with the pin between the
    block maximum and the Top-2048.'''
    return cat.Block.template(
        candidate_pool.BLOCK_VIEW @ omitted_mechanisms.maximum_over_offsets() @ pin_newest_block()
        @ candidate_pool.KEEP_BLOCKS @ candidate_pool.COVER,
        title=text.POOL_TITLE, fill_color=candidate_pool.POOL_COLOUR,
        description=text.PINNED_SELECT_CANDIDATE_BLOCKS_OF_ONE_TOKEN_DESCRIPTION,
        references=(model_lines(583, 610), model_lines(597, 599), model_lines(602, 605),
                    model_lines(607), model_lines(609, 610)))


PINNED_POOL_BODY = select_candidate_blocks_of_one_token()


def select_candidate_blocks_of_every_token() -> cat.BroadcastedCategory:
    '''The same selection written out over every token, which the box is confirmed
    against. It lifts the body the box holds, because `cat.Block.template` mints a
    fresh tag on every call.'''
    return over((x,), PINNED_POOL_BODY)


PINNED_CANDIDATE_POOL = discovering_broadcasts.broadcast_block_over_axes(
    PINNED_POOL_BODY, (x,), ((0,),), candidate_pool.POOL_BOX)
PINNED_POOL_CONFIRMATION = discovering_broadcasts.confirm_broadcast_expansion(
    select_candidate_blocks_of_every_token(), PINNED_CANDIDATE_POOL)
