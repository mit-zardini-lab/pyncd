'''The lightning indexer of DeepSeek-V4.1-Flash with the rotation, the FP4 round trip
and the scale the released indexer applies.

Written by Claude Fable 5.1, reasoning effort 80.

The released indexer rotates the last 64 of the 128 channels of its keys and of its
queries, rounds both through FP4 in place, and multiplies the per-head weights by
`(|d| |i|)^{-1/2}`. `notebooks.sota.DeepSeekV41Flash.lightning_indexer` leaves the three
out. This module writes them in and keeps the domain and the codomain of the indexer
boxes of that module, `QR, KI, state -> R[x, r|x]`, so a mode composes `INDEXER_e` and
`INDEXER_d` of this module where it composed those.

The rotation of the indexer query depends on the token, and the scoring is one box
computed once per query, whose body holds no token axis. The rotation can be written in
two ways. The table can be a fourth operand of the scoring box, read at the token. The
`q^{I}` projection and its rotation can instead stand outside the box, which then reads
the rotated queries `R[x, i, d]` in place of the query low rank. This module takes the
second form, for three reasons. The rotation is then the box the five other rotation
sites draw, with its table inside it, where the first form would send a wire of complex
numbers into the scoring box and would write the cut of the channels, the pairs and the
product out inside the scoring body. The projection, the rotation and the round trip
stand in the order of the released lines L550 to L552. The scoring body keeps the
contraction, the rectifier and the head weights it has in the base model.

    rotated_index_keys        CKV -> KI: the key projection, its norm, the rotation at
                              the first token of each entry's group, the FP4 round trip
    rotated_indexer_queries   QR -> R[x, i, d]: the query projection, the rotation at
                              the token, the FP4 round trip
    score_every_query         the scoring written out over every query, with the scale
                              on the head weights
    INDEXER_e, INDEXER_d      the boxed indexers of the encoder and of the decoder

The keys read per query, the axis of reachable entries, the two selections and the
gather are those of `lightning_indexer`, so the selection a mode composes after an
indexer of this module is the one it composes in the base model.

`ARITHMETIC_ROLES` and `ARITHMETIC_REFERENCES` are the rows the explanation tables of
the integrated model take for the scale.
'''
from __future__ import annotations

import algebra.discovering_broadcasts as discovering_broadcasts
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd

import notebooks.sota.DeepSeekV41Flash.lightning_indexer as lightning_indexer
from notebooks.sota.DeepSeekV41Flash.construction_idioms import boxed, hold, over
from notebooks.sota.DeepSeekV41Flash.declared_axes import (
    R, B, b, d, i, m, q, state, x)
from notebooks.sota.DeepSeekV41Flash.reference_links import model_lines
from notebooks.sota.DeepSeekV41Flash.quantised_caches import (
    INDEXER_ROUND_TRIP)
from notebooks.sota.DeepSeekV41Flash.rotary_embedding import (
    ROTATE_DECODER_INDEXER_KEYS, ROTATE_ENCODER_INDEXER_KEYS, ROTATE_INDEXER_QUERIES,
    broadcast_between_positions_and_channels)
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

INDEXER_QUERIES = cat.Array(R, (x, i, d))
HEAD_WEIGHTS = cat.Array(R, (x, i))
HEAD_WEIGHT_SCALE_NAME = '(\\lvert d \\rvert \\lvert i \\rvert)^{-1/2} x'


def rotated_index_keys[A: cat.Axis](
    entry_axis: A,
    rotate_keys: cat.Broadcasted,
) -> cat.BroadcastedCategory:
    '''The shared indexer keys as the released key cache holds them: projected from
    the unrotated entries, normalised, rotated by `rotate_keys` and rounded through
    FP4.'''
    return (lightning_indexer.index_keys(entry_axis)
            @ rotate_keys
            @ over((entry_axis,), INDEXER_ROUND_TRIP))


def encoder_index_keys() -> cat.BroadcastedCategory:
    '''`R[b, c] -> R[b, d]`, with key `i_b` rotated at token `|a| i_b`.'''
    return rotated_index_keys(b, ROTATE_ENCODER_INDEXER_KEYS)


def decoder_index_keys() -> cat.BroadcastedCategory:
    '''`R[B, c] -> R[B, d]`, with key `i_B` rotated at token `i_B`.'''
    return rotated_index_keys(B, ROTATE_DECODER_INDEXER_KEYS)


def rotated_indexer_queries() -> cat.BroadcastedCategory:
    '''One indexer query per token and head, projected from the query low rank, rotated
    at the token's position once per head, and rounded through FP4.'''
    return (over((x,), ops.Linear.template((q,), (i, d), 'q^{I}'))
            @ broadcast_between_positions_and_channels(ROTATE_INDEXER_QUERIES, (i,))
            @ over((x, i), INDEXER_ROUND_TRIP))


def scale_head_weights() -> cat.Broadcasted:
    '''The factor `(|d| |i|)^{-1/2}` the released indexer multiplies its per-head
    weights by.'''
    return ops.Arithmetic.template(
        nm.x * nm.Power.template(d.local_size() * i.local_size(),
                                 nm.Integer(-1) / nm.Integer(2)),
        name=HEAD_WEIGHT_SCALE_NAME)


def score_every_query[A: cat.Axis](reach: A) -> cat.Block:
    '''Every query's combined score against each entry it has reached, from the rotated
    indexer queries, the keys already read per query and the hidden state.

    Each of the 32 indexer heads scores the query against the entry and the score is
    rectified. The rectified scores are summed under a per-token per-head weight, which
    is the reference's `weights_proj` read off the 5120-wide hidden state and multiplied
    by `(|d| |i|)^{-1/2}`, so the combine is a contraction between two wires.
    '''
    keys = cat.Array(R, (x, reach, d))
    return cat.Block.template(
        (hold(INDEXER_QUERIES) * hold(keys)
         * (over((x,), ops.Linear.template((m,), (i,), 'w^{I}'))
            @ over((x, i), scale_head_weights())))
        @ ((ops.Einops.template('x i d, x r d -> x i r') @ lightning_indexer.rectify())
           * hold(HEAD_WEIGHTS))
        @ ops.Einops.template('x i r, x i -> x r'),
        title=text.SCORE_TITLE,
        fill_color=lightning_indexer.INDEXER_COLOUR,
        description=text.ROTATED_SCORE_EVERY_QUERY_DESCRIPTION,
        references=(model_lines(513), model_lines(515), model_lines(555, 557)))


def rotated_relative_indexer(
    base: lightning_indexer.RelativeIndexer,
) -> lightning_indexer.RelativeIndexer:
    '''The indexer of the integrated model over the entries `base` reads. The keys read
    per query and the axis of reachable entries are those of `base`, and the scoring box
    reads the rotated indexer queries.'''
    scoring = discovering_broadcasts.discover_broadcast_over_axes(
        score_every_query(base.reach), (x,), lightning_indexer.SCORE_BOX)
    return lightning_indexer.RelativeIndexer(
        keys_per_query=base.keys_per_query,
        reach=base.reach,
        scoring=scoring,
        block=cat.Block.template(
            (rotated_indexer_queries() * base.keys_per_query * hold(state))
            @ scoring.candidate,
            title=text.INDEXER_TITLE,
            fill_color=lightning_indexer.INDEXER_COLOUR,
            description=text.ROTATED_RELATIVE_INDEXER_DESCRIPTION,
            references=(model_lines(550, 552), model_lines(563, 565),
                        model_lines(567), model_lines(577, 580))))


ENCODER_INDEXER = rotated_relative_indexer(lightning_indexer.ENCODER_INDEXER)
DECODER_INDEXER = rotated_relative_indexer(lightning_indexer.DECODER_INDEXER)
INDEXER_e = boxed(ENCODER_INDEXER.block, lightning_indexer.INDEXER_BOX)
INDEXER_d = boxed(DECODER_INDEXER.block, lightning_indexer.INDEXER_BOX)

ARITHMETIC_ROLES: dict[str, str] = {
    HEAD_WEIGHT_SCALE_NAME: (
        text.INDEXER_HEAD_SCALE_ROLE),
}

ARITHMETIC_REFERENCES: dict[str, fd.Prod[cat.CodeReference]] = {
    HEAD_WEIGHT_SCALE_NAME: (model_lines(513), model_lines(555)),
}
