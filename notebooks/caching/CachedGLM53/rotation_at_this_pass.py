# Claude Opus 5.5 (1M context), effort 40.
'''The rotary embedding of GLM-5.3 at the positions of the tokens of one pass.

Token `i_x` of a pass stands at position `|P| + i_x` of the sequence, because the `|P|`
tokens of the earlier passes stand before it. The reference passes that position to the
rotary table: `GlmMoeDsaModel.forward` counts the tokens the cache already holds and
adds the count to the position of every token of the pass, at lines 702 to 705 of
`modeling_glm_moe_dsa.py`.

`table_at_the_positions_of_this_pass` writes that addition as a read. The table of turns
is written over every cached position `P + x`, and a view named `pass` reads it at
`|P| + i_x`, so a figure shows the offset on a wire of its own. The rest of each
rotation box is the box of `notebooks/sota/GLM53/rotary_embedding.py`, and the boxes
here are built with its `rotation_box`, so they draw under the same title and colour.

    ROTATE_QUERY_KEY_CHANNELS_AT_THIS_PASS   R[x, p] -> R[x, p]
    ROTATE_INDEXER_CHANNELS_AT_THIS_PASS     R[x, d] -> R[x, d]

A key is turned before it is cached, as the reference turns it at line 405 and caches
it at line 409, so a cached key carries the turn of the pass that computed it and no
later pass turns it again.
'''
from __future__ import annotations

import algebra.einops_simplification as einops_simplification
import algebra.write_index_notation as write_index_notation
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import data_structure.Term as fd
import deepseek.data_structure as dst
import deepseek.registries.standard_expansions as rotary_expansions

from notebooks.caching.CachedGLM53.cached_axes import CACHED_TOKENS, P
from notebooks.caching.CachedGLM53.wording import TEXT as cached_text
from notebooks.sota.DeepSeekV41Flash.construction_idioms import hold, over
from notebooks.sota.GLM53.declared_axes import R, dbar, p, t, x
from notebooks.sota.GLM53.released_constants import ROTARY_BASE
from notebooks.sota.GLM53.rotary_embedding import (
    COMPLEX, PAIR_SHAPE, UNTURNED_INDEXER_SHAPE, cut_indexer_channels,
    join_indexer_channels, rotation_box)
from notebooks.sota.GLM53.block_titles_and_descriptions import TEXT as text

POSITIONS_OF_THIS_PASS_NAME = 'pass'


def token_positions_of_this_pass() -> sc.StrideMorphism:
    '''`x -> P + x`: token `i_x` of the pass at position `|P| + i_x`.'''
    return sc.StrideMorphism(
        _dom=(x,),
        _cod_stride_shift=((CACHED_TOKENS, (nm.Integer(1),), P.local_size()),),
        name=fd.DynamicName(POSITIONS_OF_THIS_PASS_NAME))


def positions_of_this_pass() -> cat.StrideCategory:
    '''`(x, t) -> (P + x, t)`: the token positions of the pass beside every pair at
    itself.'''
    return token_positions_of_this_pass() * cat.ProdObject((t,)).identity()


def table_at_the_positions_of_this_pass() -> cat.BroadcastedCategory:
    '''`-> C[x, t]`: the table of turns over every cached position, read at the
    positions of the tokens of this pass.'''
    return (dst.Rotary.template(CACHED_TOKENS, t, base=ROTARY_BASE)
            @ ops.View.template(base=COMPLEX, reindexing=positions_of_this_pass(),
                                name=POSITIONS_OF_THIS_PASS_NAME))


def rotate_pairs_at_this_pass() -> cat.BroadcastedCategory:
    '''The turned channels of every token of the pass read as pairs, each pair
    multiplied by the factor of its position, and the pairs written back as
    channels.'''
    return ((over((x,), dst.PairsAsComplex.template(base=R, channels=p, pairs=t))
             * table_at_the_positions_of_this_pass())
            @ einops_simplification.einsum(
                (PAIR_SHAPE, PAIR_SHAPE), PAIR_SHAPE, COMPLEX)
            @ over((x,), dst.Decomplex.template(base=R, pairs=t, channels=p)))


def rotate_first_indexer_channels_at_this_pass() -> cat.BroadcastedCategory:
    return (cut_indexer_channels()
            @ (rotate_pairs_at_this_pass()
               * hold(cat.Array(R, UNTURNED_INDEXER_SHAPE)))
            @ join_indexer_channels())


def pair_formula_at_this_pass(offset: str) -> str:
    '''The turn of pair `i_t` of token `i_x` of the pass against the table `F`, which
    reads the table at the position `|P| + i_x`.'''
    token, pair = write_index_notation.axis_letters((x, t))
    position = write_index_notation.index_of(token)
    first = f'{offset}2 {write_index_notation.index_of(pair)}'
    real = f'{position}, {first}'
    imaginary = f'{position}, {first} + 1'
    factor = (f'{rotary_expansions.TABLE_SYMBOL}[\\lvert P \\rvert + {position}, '
              f'{write_index_notation.index_of(pair)}]')
    return (rf'y[{real}] + \mathrm{{i}}\, y[{imaginary}] = {factor}\, '
            rf'\big(v[{real}] + \mathrm{{i}}\, v[{imaginary}]\big)')


def indexer_rotation_formula_at_this_pass() -> str:
    token, turned, unturned = write_index_notation.axis_letters((x, p, dbar))
    position = write_index_notation.index_of(token)
    passed = (f'{position}, {write_index_notation.element_count((turned,))} + '
              f'{write_index_notation.index_of(unturned)}')
    return (rf'\begin{{gathered}} {pair_formula_at_this_pass("")} \\ '
            rf'y[{passed}] = v[{passed}] \end{{gathered}}')


ROTATE_QUERY_KEY_CHANNELS_AT_THIS_PASS = rotation_box(
    rotate_pairs_at_this_pass(), text.ROTARY_TITLE, pair_formula_at_this_pass(''),
    cached_text.QUERY_KEY_ROTATION_AT_THIS_PASS_DESCRIPTION)
ROTATE_INDEXER_CHANNELS_AT_THIS_PASS = rotation_box(
    rotate_first_indexer_channels_at_this_pass(), text.INDEXER_ROTARY_TITLE,
    indexer_rotation_formula_at_this_pass(),
    cached_text.INDEXER_ROTATION_AT_THIS_PASS_DESCRIPTION)
