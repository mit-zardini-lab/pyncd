# Claude Opus 5.5 (1M context), effort 40.
'''The lightning indexer of GLM-5.3 and the top-2048 selection it drives.

The indexer is `GlmMoeDsaIndexer` of the reference. For every query it scores each
token at or before the query with 32 heads of width 128, and a top-2048 keeps the best
of them. The indexer query is projected from the query low rank, which the attention
also reads, and the indexer
key is projected from the normalised hidden state, normalised by a layer normalisation
and shared by the 32 heads. The first 64 channels of both are turned by the rotary
embedding.

Causality is written on the keys. `read_back_from_each_token` reads the keys of the
tokens counted back from each query, so distance `i_r` of query `i_x` reads token
`i_x - i_r`. A distance that counts back past the first token is a negative index and
reads the universal unit, so `mark_sparse_domains.mark_sparse_domain` marks the
distance axis `r|x`, live where `i_x - i_r >= 0`. The reference adds the causal mask to
the scores before its top-k, which sets the score of every later token to minus
infinity. The read is marked once, as `READ_BACK`, so the indexer keys and the keys and
values of the attention carry one distance axis. The CausalSlide then finds one read
asked of the latent by the key branch and the value branch, and moves it back past the
copy of the latent.

The scoring is one box computed once per query. `score_every_query` writes it out over
every query, and `discovering_broadcasts.discover_broadcast_over_axes` deletes the query
axis from the body and confirms the box by broadcasting it back.

    INDEXER    QR[x, q], STATE[x, m] -> R[x, r|x]
    SELECT     R[r|x] -> Nat(x)[s|x], the top-2048 in the `ONLY_SELECTION` form

The top-2048 hands out the distances of the kept tokens alone, because the reference
reads the positions returned by `topk` and discards the values. A query with fewer
than 2048 tokens at or before it keeps every one of them, which is the
`min(self.index_topk, ...)` of the reference, and the selected axis `s|x` is live on as
many slots as there are tokens at or before the query.
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

from notebooks.sota.DeepSeekV41Flash.construction_idioms import boxed, hold, over, route
from notebooks.sota.GLM53.declared_axes import (
    R, STATE, d, i, m, q, r, selected_tokens, x)
from notebooks.sota.GLM53.reference_links import modeling_lines
from notebooks.sota.GLM53.released_constants import KEY_NORM_EPSILON
from notebooks.sota.GLM53.rotary_embedding import (
    ROTATE_INDEXER_CHANNELS, broadcast_between_positions_and_channels)
from notebooks.sota.GLM53.block_titles_and_descriptions import TEXT as text

INDEXER_COLOUR = '#D9E7F5'
INDEXER_BOX = 'Idx'
SCORE_BOX = 'Sco'
INDEXER_QUERIES = cat.Array(R, (x, i, d))
HEAD_WEIGHTS = cat.Array(R, (x, i))
SCORE_SCALE_NAME = '\\lvert d \\rvert^{-1/2} x'
HEAD_WEIGHT_SCALE_NAME = '\\lvert i \\rvert^{-1/2} x'
BACK_VIEW_NAME = '\\mathrm{Back}'

SCORE_REFERENCES = (modeling_lines(239, 244),)
INDEXER_REFERENCES = (modeling_lines(164, 193), modeling_lines(222, 253),
                      modeling_lines(415, 424))


def indexer_queries() -> cat.BroadcastedCategory:
    '''`R[x, q] -> R[x, i, d]`: one indexer query per token and head, projected from
    the query low rank and turned at the position of its token once per head.'''
    return (over((x,), ops.Linear.template((q,), (i, d), 'q^{I}'))
            @ broadcast_between_positions_and_channels(ROTATE_INDEXER_CHANNELS, (i,)))


def indexer_keys() -> cat.BroadcastedCategory:
    '''`R[x, m] -> R[x, d]`: one indexer key per token, shared by every indexer head,
    projected from the hidden state, normalised and turned.'''
    return ((x >> ops.Linear.template((m,), (d,), 'k^{I}'))
            @ over((x,), ops.LayerNorm.template((d,), epsilon=KEY_NORM_EPSILON))
            @ ROTATE_INDEXER_CHANNELS)


READ_BACK = mark_sparse_domains.mark_sparse_domain(sc.StrideMorphism(
    _dom=(x, r),
    _cod_stride_shift=((x, (nm.Integer(1), nm.Integer(-1)), nm.Integer(0)),),
    name=fd.DynamicName(BACK_VIEW_NAME)))
'''The read of token `i_x - i_r` at distance `i_r` of each query `i_x`, with the
distance axis marked `r|x`, live where `i_x - i_r >= 0`.'''


def read_back_from_each_token[A: cat.Axis](
    rest: tuple[A, ...],
    base: cat.Datatype = R,
) -> cat.Broadcasted:
    '''An array over the tokens read at token `i_x - i_r`, `i_r` tokens back from each
    query, times the identity on `rest`. The distance axis leaves the view as the
    `r|x` of `READ_BACK`.'''
    return ops.View.template(
        base=base, reindexing=(READ_BACK, cat.ProdObject(rest).identity()),
        name=BACK_VIEW_NAME)


KEYS_READ_BACK = read_back_from_each_token((d,))
reach = KEYS_READ_BACK.cod()[0].shape()[1]


def rectify() -> cat.Broadcasted:
    '''The rectifier, written as the input times the indicator of its sign, so that a
    derived backward pass carries the indicator rather than a primed name.'''
    return ops.Arithmetic.template(nm.FreeInput() * nm.IsPositive())


def scale_by_inverse_square_root[A: cat.Axis](axis: A, name: str) -> cat.Broadcasted:
    '''The factor one over the square root of the size of `axis`.'''
    return ops.Arithmetic.template(
        nm.x * nm.Power.template(axis.local_size(), nm.Integer(-1) / nm.Integer(2)),
        name=name)


def score_every_query() -> cat.Block:
    '''Every query's combined score against each token at or before it, from the indexer
    queries, the keys already read back from every query and the hidden state.

    Each of the 32 heads scores the query against the key, the score is multiplied by
    `|d|^{-1/2}`, and the product is rectified. A weight read off the hidden state by
    `w^{I}` and multiplied by `|i|^{-1/2}`, one number per token and head, says how much
    each head's score counts, so the combine is a contraction between two wires.
    '''
    keys = cat.Array(R, (x, reach, d))
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
        description=text.SCORE_EVERY_QUERY_DESCRIPTION,
        references=SCORE_REFERENCES)


SCORING = discovering_broadcasts.discover_broadcast_over_axes(
    score_every_query(), (x,), SCORE_BOX)


def index_every_token() -> cat.Block:
    '''`QR[x, q], STATE[x, m] -> R[x, r|x]`: the indexer queries, the keys read back
    from every query, and the scoring box.'''
    return cat.Block.template(
        (indexer_queries()
         * (route((0, 0), (STATE,))
            @ ((indexer_keys() @ KEYS_READ_BACK) * hold(STATE))))
        @ SCORING.candidate,
        title=text.INDEXER_TITLE, fill_color=INDEXER_COLOUR,
        description=text.INDEXER_DESCRIPTION, references=INDEXER_REFERENCES)


INDEXER_BLOCK = index_every_token()
INDEXER = boxed(INDEXER_BLOCK, INDEXER_BOX)

SELECTED_SLOTS = reach.selected_slots(
    selected_tokens, fd.DynamicName('s', code_form='selected_tokens'))
SELECT = dst.TopK.template(
    k=selected_tokens, axis=reach, form=dst.SelectionForm.ONLY_SELECTION,
    selected_axis=SELECTED_SLOTS)
s, = SELECT.cod()[0].shape()
SELECTION = cat.Array(SELECT.cod()[0].datatype, (x, s))
INDEXER_SCORES = cat.Array(R, (x, reach))

