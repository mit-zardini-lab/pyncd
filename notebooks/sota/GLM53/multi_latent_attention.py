# Claude Opus 5.5 (1M context), effort 40.
'''The multi-head latent attention of GLM-5.3, read at the tokens a selection keeps.

The attention is `GlmMoeDsaAttention` of the reference. The query passes through a
low rank of 2048 channels, and every key and value is expanded from one latent of 512
channels per token. Each of the 64 heads reads a query and a key of 256 channels, of
which 192 are not turned and 64 are turned by the rotary embedding, and a value of 256
channels. The turned part of the key is one vector per token, shared by every head.

The reference computes each weight as one linear map and cuts its results into parts
with `torch.split`. A linear map followed by a cut of its results is one linear map per
part, so each part has a weight of its own here, named after the part:

    q_b_proj              W^{Qn} onto the unturned channels, W^{Qr} onto the turned ones
    kv_a_proj_with_mqa    W^{KVa} onto the latent, W^{Kr} onto the turned key
    kv_b_proj             W^{Kb} onto the unturned key channels, W^{Vb} onto the values

The unturned and turned parts of a query and of a key are joined onto the axis `a` of
256 channels with `aops.ConcatenateAxes`, which is the `torch.cat` of the reference.

The reference expands the key and the value of every token and then masks every token
left out by the selection. Here the selection is read by a gather, which reads the key
and the value of every head at each selected token. The attention core is one box
computed once per query and per head, so the query, the gathered keys and the gathered
values all carry the query axis and the head axis at their head.

    query_path()                STATE[x, m] -> QUERIES[x, h, a], QR[x, q]
    query_without_low_rank()    STATE[x, m] -> QUERIES[x, h, a]
    keys_and_values()           STATE[x, m] -> KEYS[x, h, a], VALUES[x, h, u]
    GATHER_KEYS, GATHER_VALUES  Nat(x)[x, s|x], R[x, h, ...] -> R[x, h, s|x, ...]
    CORE                        the box computed once per query and head
    output_projection()         R[x, h, u] -> STATE[x, m]
'''
from __future__ import annotations

import advanced_axis_dynamics.data_structure.Operators as aops
import algebra.discovering_broadcasts as discovering_broadcasts
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Operators as ops
import deepseek.data_structure as dst

from notebooks.sota.DeepSeekV41Flash.construction_idioms import boxed, hold, over, route
from notebooks.sota.GLM53.declared_axes import (
    LATENT, QUERIES, QUERY_LOW_RANK, R, ROTATED_KEYS, STATE, UNROTATED_KEYS, VALUES,
    a, c, h, m, n, p, q, u, x)
from notebooks.sota.GLM53.lightning_indexer import (
    SELECTION, read_back_from_each_token, s, scale_by_inverse_square_root)
from notebooks.sota.GLM53.reference_links import modeling_lines
from notebooks.sota.GLM53.released_constants import LATENT_NORM_EPSILON
from notebooks.sota.GLM53.rotary_embedding import (
    ROTATE_QUERY_KEY_CHANNELS, broadcast_between_positions_and_channels)
from notebooks.sota.GLM53.block_titles_and_descriptions import TEXT as text

QUERY_COLOUR = '#E8DFF0'
KEY_VALUE_COLOUR = '#DDE8D6'
GATHER_COLOUR = '#D9E7F5'
CORE_COLOUR = '#C5BEDF'
GATHER_BOX = 'Gth'
CORE_BOX = 'Core'
SCORE_SCALE_NAME = '\\lvert a \\rvert^{-1/2} x'
REPEAT_VIEW_NAME = '\\mathrm{Repeat}'

QUERY_REFERENCES = (modeling_lines(335, 337), modeling_lines(394, 396),
                    modeling_lines(403, 405), modeling_lines(411))
KEY_VALUE_REFERENCES = (modeling_lines(339, 349), modeling_lines(398, 405),
                        modeling_lines(362, 379), modeling_lines(413))
GATHER_REFERENCES = (modeling_lines(430, 443),)
CORE_REFERENCES = (modeling_lines(268, 290), modeling_lines(357),
                   modeling_lines(445, 458))
OUTPUT_REFERENCES = (modeling_lines(351, 355), modeling_lines(460, 461))


def low_rank_query() -> cat.BroadcastedCategory:
    '''`STATE[x, m] -> QR[x, q]`: the hidden state projected onto the query low rank
    and normalised.'''
    return ((x >> ops.Linear.template((m,), (q,), 'W^{Qa}'))
            @ over((x,), ops.Normalize.template((q,), epsilon=LATENT_NORM_EPSILON)))


def expand_queries() -> cat.BroadcastedCategory:
    '''`QR[x, q] -> QUERIES[x, h, a]`: the unturned and the turned channels of every
    query head, the turned channels turned at the position of the token once per head,
    and the two parts joined.'''
    return (route((0, 0), (QUERY_LOW_RANK,))
            @ ((x >> ops.Linear.template((q,), (h, n), 'W^{Qn}'))
               * ((x >> ops.Linear.template((q,), (h, p), 'W^{Qr}'))
                  @ broadcast_between_positions_and_channels(
                      ROTATE_QUERY_KEY_CHANNELS, (h,))))
            @ aops.ConcatenateAxes.template(((x, h, n), (x, h, p)), concatenated=a))


def query_path() -> cat.Block:
    '''The queries, with a copy of the low rank handed on to the indexer, which reads
    the same low rank.'''
    return cat.Block.template(
        low_rank_query()
        @ route((0, 0), (QUERY_LOW_RANK,))
        @ (expand_queries() * hold(QUERY_LOW_RANK)),
        title=text.QUERY_TITLE, fill_color=QUERY_COLOUR,
        description=text.QUERY_PATH_DESCRIPTION, references=QUERY_REFERENCES)


def query_without_low_rank() -> cat.Block:
    '''The queries alone, for a layer that runs no indexer.'''
    return cat.Block.template(
        low_rank_query() @ expand_queries(),
        title=text.QUERY_TITLE, fill_color=QUERY_COLOUR,
        description=text.QUERY_WITHOUT_LOW_RANK_DESCRIPTION,
        references=QUERY_REFERENCES)


def repeat_over_heads() -> cat.Broadcasted:
    '''The one turned key of a token copied to every head, which is the `expand` of
    the reference.'''
    return ops.View.template(
        reindexing=cat.Rearrangement((0, 2), (x, h, p)), name=REPEAT_VIEW_NAME)


def keys_and_values() -> cat.Block:
    '''`STATE[x, m] -> KEYS[x, h, a], VALUES[x, h, u]`: the latent of every token,
    normalised and expanded into the unturned channels of every key head and into
    every value head, and the turned key of every token, turned and copied to every
    head.'''
    return cat.Block.template(
        route((0, 0), (STATE,))
        @ (((x >> ops.Linear.template((m,), (c,), 'W^{KVa}'))
            @ over((x,), ops.Normalize.template((c,), epsilon=LATENT_NORM_EPSILON))
            @ route((0, 0), (LATENT,))
            @ ((x >> ops.Linear.template((c,), (h, n), 'W^{Kb}'))
               * (x >> ops.Linear.template((c,), (h, u), 'W^{Vb}'))))
           * ((x >> ops.Linear.template((m,), (p,), 'W^{Kr}'))
              @ ROTATE_QUERY_KEY_CHANNELS
              @ repeat_over_heads()))
        @ route((0, 2, 1), (UNROTATED_KEYS, VALUES, ROTATED_KEYS))
        @ (aops.ConcatenateAxes.template(((x, h, n), (x, h, p)), concatenated=a)
           * hold(VALUES)),
        title=text.KEYS_AND_VALUES_TITLE, fill_color=KEY_VALUE_COLOUR,
        description=text.KEYS_AND_VALUES_DESCRIPTION, references=KEY_VALUE_REFERENCES)


def read_at_selected_tokens[A: cat.Axis](
    channels: A,
    description: str,
) -> cat.Broadcasted:
    '''`Nat(x)[x, s|x], R[x, h, channels] -> R[x, h, s|x, channels]`: the array of every
    head read back from each query, and a `dst.IndexSelect` broadcast over the queries,
    the heads, the slots and the channels, which reads the token at each slot's
    distance. An empty slot reads the unit.'''
    T = cat.WeaveMode.TILED
    read_back = read_back_from_each_token((h, channels))
    distances = read_back.cod()[0].shape()[1]
    degree = (x, h, s, channels)
    return boxed(cat.Block.template(
        (hold(SELECTION) * read_back)
        @ cat.Broadcasted(
            operator=dst.IndexSelect(),
            input_weaves=(cat.Weave(SELECTION.datatype, (T, T)),
                          cat.Weave(R, (T, distances, T, T))),
            output_weaves=(cat.Weave(R, (T, T, T, T)),),
            reindexings=(cat.Rearrangement((0, 2), degree),
                         cat.Rearrangement((0, 1, 3), degree))),
        title=text.GATHER_TITLE, fill_color=GATHER_COLOUR, description=description,
        references=GATHER_REFERENCES), GATHER_BOX)


GATHER_KEYS = read_at_selected_tokens(a, text.GATHER_KEYS_DESCRIPTION)
GATHER_VALUES = read_at_selected_tokens(u, text.GATHER_VALUES_DESCRIPTION)
SELECTED_KEYS = cat.Array(R, (x, h, s, a))
SELECTED_VALUES = cat.Array(R, (x, h, s, u))


def attend_over_every_query_and_head() -> cat.Block:
    '''The softmax of every query and head over the selected tokens, written out over
    every query and head: the score of the query against every selected key, the scale
    `|a|^{-1/2}`, the softmax over the selected slots and the sum of the values under
    the softmax.'''
    return cat.Block.template(
        (hold(QUERIES) * hold(SELECTED_KEYS) * hold(SELECTED_VALUES))
        @ (ops.Einops.template('x h a, x h s a -> x h s') * hold(SELECTED_VALUES))
        @ (over((x, h, s), scale_by_inverse_square_root(a, SCORE_SCALE_NAME))
           * hold(SELECTED_VALUES))
        @ (over((x, h), ops.SoftMax.template()) * hold(SELECTED_VALUES))
        @ ops.Einops.template('x h s, x h s u -> x h u'),
        title=text.CORE_TITLE, fill_color=CORE_COLOUR,
        description=text.CORE_DESCRIPTION, references=CORE_REFERENCES)


CORE = discovering_broadcasts.discover_broadcast_over_axes(
    attend_over_every_query_and_head(), (x, h), CORE_BOX)


def output_projection() -> cat.BroadcastedCategory:
    '''`R[x, h, u] -> STATE[x, m]`: the values of the 64 heads mapped back onto the
    hidden width.'''
    return x >> ops.Linear.template((h, u), (m,), 'W^{O}')


def attend_to_selected_tokens() -> cat.BroadcastedCategory:
    '''`QUERIES, SELECTION, KEYS, SELECTION, VALUES -> STATE`: the two gathers, the
    core and the output projection, which every layer runs whatever the source of its
    selection.'''
    return ((hold(QUERIES) * GATHER_KEYS * GATHER_VALUES)
            @ CORE.candidate
            @ output_projection())

