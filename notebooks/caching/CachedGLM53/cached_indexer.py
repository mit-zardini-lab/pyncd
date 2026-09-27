# Claude Opus 5.5 (1M context), effort 40.
'''The lightning indexer of GLM-5.3 reading its keys from a cache.

The indexer of `notebooks/sota/GLM53/lightning_indexer.py` scores every query against
the indexer key of every token at or before it. The reference computes the indexer key
of the tokens of the pass alone, appends it to the indexer key cache of its layer with
`past_key_values.update_indexer`, and scores against the whole cache, at lines 228 to
239 of `modeling_glm_moe_dsa.py`. The cache is `update_indexer` of
`DynamicIndexedLayer`, lines 353 to 366 of `cache_utils.py`.

The key is projected, normalised and turned at the position of its token before it is
cached, so the cache holds 128 channels per token, turned. The queries are computed for
the tokens of the pass alone, and each counts back over every cached token: distance
`i_r` of query `i_x` reads position `|P| + i_x - i_r` of `P + x`.

A layer in the Shared mode runs no indexer, and the reference leaves the indexer key
cache of its layer empty. The 57 Shared layers therefore hold no indexer key.

    INDEXER_KEY_CACHE_BOX       R[x, d] -> R[P + x, d], the cache `idx`
    CACHED_INDEXER              QR[x, q], STATE[x, m] -> R[x, r|x],
                                with `|r| = |P| + |x|`
    SELECT_FROM_THE_CACHE       R[r|x] -> Nat(P + x)[s|x], the top-2048 over the cache
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

import caching.data_structure.Caching as Caching
from notebooks.caching.CachedGLM53.cached_axes import (
    CACHED_DISTANCES, CACHED_TOKENS, INDEXER_KEY_CACHE, P)
from notebooks.caching.CachedGLM53.reference_links import CACHE_UPDATE_REFERENCES
from notebooks.caching.CachedGLM53.rotation_at_this_pass import (
    ROTATE_INDEXER_CHANNELS_AT_THIS_PASS)
from notebooks.caching.CachedGLM53.wording import TEXT as cached_text
from notebooks.sota.DeepSeekV41Flash.construction_idioms import boxed, hold, over, route
from notebooks.sota.GLM53.declared_axes import (
    R, STATE, d, i, m, q, selected_tokens, x)
from notebooks.sota.GLM53.lightning_indexer import (
    BACK_VIEW_NAME, HEAD_WEIGHTS, HEAD_WEIGHT_SCALE_NAME, INDEXER_BOX, INDEXER_COLOUR,
    INDEXER_QUERIES, INDEXER_REFERENCES, SCORE_BOX, SCORE_REFERENCES, SCORE_SCALE_NAME,
    rectify, scale_by_inverse_square_root)
from notebooks.sota.GLM53.released_constants import KEY_NORM_EPSILON
from notebooks.sota.GLM53.rotary_embedding import (
    broadcast_between_positions_and_channels)
from notebooks.sota.GLM53.block_titles_and_descriptions import TEXT as text


def indexer_queries_at_this_pass() -> cat.BroadcastedCategory:
    '''`R[x, q] -> R[x, i, d]`: one indexer query per token of the pass and head,
    turned at the position of its token.'''
    return (over((x,), ops.Linear.template((q,), (i, d), 'q^{I}'))
            @ broadcast_between_positions_and_channels(
                ROTATE_INDEXER_CHANNELS_AT_THIS_PASS, (i,)))


def indexer_key_of_this_pass() -> cat.BroadcastedCategory:
    '''`R[x, m] -> R[x, d]`: the indexer key of every token of the pass, projected,
    normalised and turned at the position of its token.'''
    return ((x >> ops.Linear.template((m,), (d,), 'k^{I}'))
            @ over((x,), ops.LayerNorm.template((d,), epsilon=KEY_NORM_EPSILON))
            @ ROTATE_INDEXER_CHANNELS_AT_THIS_PASS)


INDEXER_KEY_CACHE_BOX = Caching.Caching.template(
    (x, d), CACHED_TOKENS, INDEXER_KEY_CACHE)


def read_back_over_the_cache[A: cat.Axis](
    rest: tuple[A, ...],
    base: cat.Datatype = R,
) -> cat.Broadcasted:
    '''An array over the cached tokens read at position `|P| + i_x - i_r`, `i_r`
    tokens back from query `i_x` of the pass, times the identity on `rest`. The
    distance axis leaves the view as `r|x`, live where `|P| + i_x - i_r >= 0`.'''
    return mark_sparse_domains.guarded_view(
        base=base,
        reindexing=(sc.StrideMorphism(
            _dom=(x, CACHED_DISTANCES),
            _cod_stride_shift=((CACHED_TOKENS, (nm.Integer(1), nm.Integer(-1)),
                                P.local_size()),),
            name=fd.DynamicName(BACK_VIEW_NAME)),
            cat.ProdObject(rest).identity()),
        name=BACK_VIEW_NAME)


CACHED_KEYS_READ_BACK = read_back_over_the_cache((d,))
cached_reach = CACHED_KEYS_READ_BACK.cod()[0].shape()[1]


def score_every_query_against_the_cache() -> cat.Block:
    '''The combined score of every query of the pass against each cached token at or
    before it, written as `lightning_indexer.score_every_query` writes it over the
    distances back over the cache.'''
    keys = cat.Array(R, (x, cached_reach, d))
    return cat.Block.template(
        (hold(INDEXER_QUERIES) * hold(keys)
         * (over((x,), ops.Linear.template((m,), (i,), 'w^{I}'))
            @ over((x, i), scale_by_inverse_square_root(i, HEAD_WEIGHT_SCALE_NAME))))
        @ ((ops.Einops.template('x i d, x r d -> x i r')
            @ scale_by_inverse_square_root(d, SCORE_SCALE_NAME)
            @ rectify())
           * hold(HEAD_WEIGHTS))
        @ ops.Einops.template('x i r, x i -> x r'),
        title=text.SCORE_TITLE, fill_color=INDEXER_COLOUR,
        description=cached_text.SCORE_AGAINST_THE_CACHE_DESCRIPTION,
        references=SCORE_REFERENCES)


SCORING_AGAINST_THE_CACHE = discovering_broadcasts.discover_broadcast_over_axes(
    score_every_query_against_the_cache(), (x,), SCORE_BOX)


def index_every_cached_token() -> cat.Block:
    '''`QR[x, q], STATE[x, m] -> R[x, r|x]`: the indexer queries of the pass, the
    indexer keys of the pass saved into the cache `idx` and the whole cache read back
    from every query, and the scoring box.'''
    return cat.Block.template(
        (indexer_queries_at_this_pass()
         * (route((0, 0), (STATE,))
            @ ((indexer_key_of_this_pass() @ INDEXER_KEY_CACHE_BOX
                @ CACHED_KEYS_READ_BACK)
               * hold(STATE))))
        @ SCORING_AGAINST_THE_CACHE.candidate,
        title=text.INDEXER_TITLE, fill_color=INDEXER_COLOUR,
        description=cached_text.CACHED_INDEXER_DESCRIPTION,
        references=(*INDEXER_REFERENCES, *CACHE_UPDATE_REFERENCES))


CACHED_INDEXER = boxed(index_every_cached_token(), INDEXER_BOX)

SELECT_FROM_THE_CACHE = dst.TopK.template(
    k=selected_tokens, axis=cached_reach, form=dst.SelectionForm.ONLY_SELECTION)
cached_selected, = SELECT_FROM_THE_CACHE.cod()[0].shape()
CACHED_SELECTION = cat.Array(
    SELECT_FROM_THE_CACHE.cod()[0].datatype, (x, cached_selected))
