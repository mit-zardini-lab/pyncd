# Claude Opus 5.5 (1M context), effort 40.
'''The multi-head latent attention of GLM-5.3 reading its keys and values from a cache.

The reference caches two arrays per token and layer, and it caches them before the
latent is expanded into the keys and the values of the heads: the normalised latent of
512 channels and the turned key of 64 channels, at lines 407 to 409 of
`modeling_glm_moe_dsa.py`. `past_key_values.update(k_pass, k_rot, layer_idx)` hands the
latent to the slot a `DynamicLayer` names `keys` and the turned key to the slot it
names `values`, and each slot is concatenated onto separately, at lines 129 to 148 of
`cache_utils.py`. The expansion `expand_kv` then runs on every cached token, at line
413.

Here each of the two arrays passes through a `Caching` of its own, `lat` and `rot`, and
everything after the caches stands on the cached tokens `P + x`: the expansion of the
latent by `W^{Kb}` and `W^{Vb}`, and the join of the unturned channels of every head
to the one turned key of the token onto `a`. The queries stand on the tokens of the pass `x`, and so does
everything from the gathers on, because every query of the pass reads its own 2048
tokens. The gathers read the keys and the values back over the cache, at
`|P| + i_x - i_r`.

    query_path_at_this_pass()         STATE[x, m] -> QUERIES[x, h, a], QR[x, q]
    query_without_low_rank_at_this_pass()
                                      STATE[x, m] -> QUERIES[x, h, a]
    LATENT_CACHE_BOX, TURNED_KEY_CACHE_BOX
                                      the caches `lat` and `rot`
    cached_keys_and_values()          STATE[x, m] -> KEYS[P + x, h, a],
                                      VALUES[P + x, h, u]
    GATHER_CACHED_KEYS, GATHER_CACHED_VALUES
                                      Nat(P + x)[x, s|x], R[P + x, h, ...]
                                      -> R[x, h, s|x, ...]
    CACHED_CORE                       the core over the slots selected from the cache
    attend_to_selected_cached_tokens()
                                      the gathers, the core and the output projection
'''
from __future__ import annotations

import advanced_axis_dynamics.data_structure.Operators as aops
import algebra.discovering_broadcasts as discovering_broadcasts
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Operators as ops
import deepseek.data_structure as dst

import caching.data_structure.Caching as Caching
from notebooks.caching.CachedGLM53.cached_axes import (
    CACHED_LATENT, CACHED_TOKENS, LATENT_CACHE, TURNED_KEY_CACHE)
from notebooks.caching.CachedGLM53.cached_indexer import (
    CACHED_SELECTION, cached_selected, read_back_over_the_cache)
from notebooks.caching.CachedGLM53.reference_links import (
    LATENT_CACHE_REFERENCES, POSITION_REFERENCES)
from notebooks.caching.CachedGLM53.rotation_at_this_pass import (
    ROTATE_QUERY_KEY_CHANNELS_AT_THIS_PASS)
from notebooks.caching.CachedGLM53.wording import TEXT as cached_text
from notebooks.sota.DeepSeekV41Flash.construction_idioms import boxed, hold, over, route
from notebooks.sota.GLM53.declared_axes import (
    QUERIES, QUERY_LOW_RANK, R, STATE, a, c, h, m, n, p, q, u, x)
from notebooks.sota.GLM53.lightning_indexer import scale_by_inverse_square_root
from notebooks.sota.GLM53.multi_latent_attention import (
    CORE_BOX, CORE_COLOUR, CORE_REFERENCES, GATHER_BOX, GATHER_COLOUR,
    GATHER_REFERENCES, KEY_VALUE_COLOUR, KEY_VALUE_REFERENCES, QUERY_COLOUR,
    QUERY_REFERENCES, SCORE_SCALE_NAME, join_key_channels_at_every_head,
    low_rank_query, output_projection)
from notebooks.sota.GLM53.released_constants import LATENT_NORM_EPSILON
from notebooks.sota.GLM53.rotary_embedding import (
    broadcast_between_positions_and_channels)
from notebooks.sota.GLM53.block_titles_and_descriptions import TEXT as text

CACHED_KEYS = cat.Array(R, (CACHED_TOKENS, h, a))
CACHED_VALUES = cat.Array(R, (CACHED_TOKENS, h, u))
CACHED_UNROTATED_KEYS = cat.Array(R, (CACHED_TOKENS, h, n))
CACHED_ROTATED_KEY_SHARED_BY_THE_HEADS = cat.Array(R, (CACHED_TOKENS, p))
SELECTED_CACHED_KEYS = cat.Array(R, (x, h, cached_selected, a))
SELECTED_CACHED_VALUES = cat.Array(R, (x, h, cached_selected, u))


def expand_queries_at_this_pass() -> cat.BroadcastedCategory:
    '''`QR[x, q] -> QUERIES[x, h, a]`: the queries of the tokens of the pass, the
    turned channels turned at their positions `|P| + i_x`.'''
    return (route((0, 0), (QUERY_LOW_RANK,))
            @ ((x >> ops.Linear.template((q,), (h, n), 'W^{Qn}'))
               * ((x >> ops.Linear.template((q,), (h, p), 'W^{Qr}'))
                  @ broadcast_between_positions_and_channels(
                      ROTATE_QUERY_KEY_CHANNELS_AT_THIS_PASS, (h,))))
            @ aops.ConcatenateAxes.template(((x, h, n), (x, h, p)), concatenated=a))


def query_path_at_this_pass() -> cat.Block:
    return cat.Block.template(
        low_rank_query()
        @ route((0, 0), (QUERY_LOW_RANK,))
        @ (expand_queries_at_this_pass() * hold(QUERY_LOW_RANK)),
        title=text.QUERY_TITLE, fill_color=QUERY_COLOUR,
        description=cached_text.QUERY_PATH_AT_THIS_PASS_DESCRIPTION,
        references=(*QUERY_REFERENCES, *POSITION_REFERENCES))


def query_without_low_rank_at_this_pass() -> cat.Block:
    return cat.Block.template(
        low_rank_query() @ expand_queries_at_this_pass(),
        title=text.QUERY_TITLE, fill_color=QUERY_COLOUR,
        description=cached_text.QUERY_WITHOUT_LOW_RANK_AT_THIS_PASS_DESCRIPTION,
        references=(*QUERY_REFERENCES, *POSITION_REFERENCES))


LATENT_CACHE_BOX = Caching.Caching.template((x, c), CACHED_TOKENS, LATENT_CACHE)
TURNED_KEY_CACHE_BOX = Caching.Caching.template((x, p), CACHED_TOKENS, TURNED_KEY_CACHE)


def latent_of_this_pass() -> cat.BroadcastedCategory:
    '''`STATE[x, m] -> R[x, c]`: the normalised latent of every token of the pass.'''
    return ((x >> ops.Linear.template((m,), (c,), 'W^{KVa}'))
            @ over((x,), ops.Normalize.template((c,), epsilon=LATENT_NORM_EPSILON)))


def turned_key_of_this_pass() -> cat.BroadcastedCategory:
    '''`STATE[x, m] -> R[x, p]`: the turned key of every token of the pass, turned at
    its position.'''
    return ((x >> ops.Linear.template((m,), (p,), 'W^{Kr}'))
            @ ROTATE_QUERY_KEY_CHANNELS_AT_THIS_PASS)


def expand_cached_latent() -> cat.BroadcastedCategory:
    '''`R[P + x, c] -> R[P + x, h, n], R[P + x, h, u]`: the unturned key channels and
    the value of every head, expanded from the latent of every cached token.'''
    return (route((0, 0), (CACHED_LATENT,))
            @ ((CACHED_TOKENS >> ops.Linear.template((c,), (h, n), 'W^{Kb}'))
               * (CACHED_TOKENS >> ops.Linear.template((c,), (h, u), 'W^{Vb}'))))


def cached_keys_and_values() -> cat.Block:
    '''`STATE[x, m] -> KEYS[P + x, h, a], VALUES[P + x, h, u]`: the latent and the
    turned key of the pass saved into the caches `lat` and `rot`, and the keys and the
    values of every head expanded from the whole of both caches.'''
    return cat.Block.template(
        route((0, 0), (STATE,))
        @ ((latent_of_this_pass() @ LATENT_CACHE_BOX @ expand_cached_latent())
           * (turned_key_of_this_pass() @ TURNED_KEY_CACHE_BOX))
        @ route((0, 2, 1), (CACHED_UNROTATED_KEYS, CACHED_VALUES,
                            CACHED_ROTATED_KEY_SHARED_BY_THE_HEADS))
        @ (join_key_channels_at_every_head(CACHED_TOKENS) * hold(CACHED_VALUES)),
        title=text.KEYS_AND_VALUES_TITLE, fill_color=KEY_VALUE_COLOUR,
        description=cached_text.CACHED_KEYS_AND_VALUES_DESCRIPTION,
        references=(*KEY_VALUE_REFERENCES, *LATENT_CACHE_REFERENCES))


def read_at_selected_cached_tokens[A: cat.Axis](
    channels: A,
    description: str,
) -> cat.Broadcasted:
    '''`Nat(P + x)[x, s|x], R[P + x, h, channels] -> R[x, h, s|x, channels]`: the
    cached array of every head read back from each query of the pass, and the gather of
    the token at each slot's distance.'''
    T = cat.WeaveMode.TILED
    read_back = read_back_over_the_cache((h, channels))
    distances = read_back.cod()[0].shape()[1]
    degree = (x, h, cached_selected, channels)
    return boxed(cat.Block.template(
        (hold(CACHED_SELECTION) * read_back)
        @ cat.Broadcasted(
            operator=dst.IndexSelect(),
            input_weaves=(cat.Weave(CACHED_SELECTION.datatype, (T, T)),
                          cat.Weave(R, (T, distances, T, T))),
            output_weaves=(cat.Weave(R, (T, T, T, T)),),
            reindexings=(cat.Rearrangement((0, 2), degree),
                         cat.Rearrangement((0, 1, 3), degree))),
        title=text.GATHER_TITLE, fill_color=GATHER_COLOUR, description=description,
        references=GATHER_REFERENCES), GATHER_BOX)


GATHER_CACHED_KEYS = read_at_selected_cached_tokens(
    a, cached_text.GATHER_CACHED_KEYS_DESCRIPTION)
GATHER_CACHED_VALUES = read_at_selected_cached_tokens(
    u, cached_text.GATHER_CACHED_VALUES_DESCRIPTION)


def attend_over_every_query_and_head() -> cat.Block:
    '''The softmax of every query of the pass and head over the tokens selected from
    the cache, written as `multi_latent_attention.attend_over_every_query_and_head`
    writes it over the slots of the cached selection.'''
    return cat.Block.template(
        (hold(QUERIES) * hold(SELECTED_CACHED_KEYS) * hold(SELECTED_CACHED_VALUES))
        @ (ops.Einops.template('x h a, x h s a -> x h s')
           * hold(SELECTED_CACHED_VALUES))
        @ (over((x, h, cached_selected),
                scale_by_inverse_square_root(a, SCORE_SCALE_NAME))
           * hold(SELECTED_CACHED_VALUES))
        @ (over((x, h), ops.SoftMax.template()) * hold(SELECTED_CACHED_VALUES))
        @ ops.Einops.template('x h s, x h s u -> x h u'),
        title=text.CORE_TITLE, fill_color=CORE_COLOUR,
        description=text.CORE_DESCRIPTION, references=CORE_REFERENCES)


CACHED_CORE = discovering_broadcasts.discover_broadcast_over_axes(
    attend_over_every_query_and_head(), (x, h), CORE_BOX)


def attend_to_selected_cached_tokens() -> cat.BroadcastedCategory:
    '''`QUERIES, SELECTION, KEYS, SELECTION, VALUES -> STATE`: the two gathers over the
    cache, the core and the output projection.'''
    return ((hold(QUERIES) * GATHER_CACHED_KEYS * GATHER_CACHED_VALUES)
            @ CACHED_CORE.candidate
            @ output_projection())
