# Claude Opus 5.5 (1M context), effort 40.
'''The rotary embedding of GLM-5.3, one box for each kind of vector it turns.

The reference turns four arrays: the rotated part of every query head and the one
rotated key a token shares between its heads, in the attention, and the indexer query
of every indexer head and the indexer key of every token, in the indexer. Every turn
reads the same table, `GlmMoeDsaRotaryEmbedding`, whose frequencies are the default
RoPE frequencies at the base `rope_theta`. The configuration of the checkpoint writes
`head_dim: 192`, and `GlmMoeDsaConfig.__post_init__` replaces it with
`qk_rope_head_dim`, so the table has 32 frequencies for the 64 turned channels.

Each rotation is one titled block boxed as an operator. `dst.PairsAsComplex` reads the
turned channels two at a time as complex numbers, the first channel of a pair as the
real part. The reference reads the same interleaved layout with `q[..., 0::2]` and
`q[..., 1::2]`. A product over the `dst.Complex` datatype multiplies every pair by the
factor held by the table for its position and its pair, and `dst.Decomplex` writes the
pairs back as reals. The table is a `dst.Rotary`, which draws as a circle with RoPE
written above it.

    ROTATE_QUERY_KEY_CHANNELS   R[x, p] -> R[x, p]   every channel turned
    ROTATE_INDEXER_CHANNELS     R[x, d] -> R[x, d]   the first |p| channels turned

The attention keeps its turned channels on an axis of their own, `p`, because the
reference projects them with rows of their own and joins them to the unturned channels
only after the turn. The indexer turns the first 64 of its 128 channels in place, so its
box cuts the channel axis with `aops.DeconcatenateAxes`, turns the first part and joins
the two parts back onto `d` with `aops.ConcatenateAxes`.

A box is written over one vector per token, which is the smallest array that holds the
position axis. `broadcast_between_positions_and_channels` computes a box once per head
of an array `[x, h, p]` or `[x, i, d]`, because `over` would put the head axis ahead of
the token axis.

The reference writes the turned pairs back as two halves, the 32 real parts and then
the 32 imaginary parts, where `dst.Decomplex` writes each pair back in place. The query
and the key are written back in the same order, so every score is unchanged, and the
block description states the difference.
'''
from __future__ import annotations

import advanced_axis_dynamics.data_structure.AxisConcatenation as AxisConcatenation
import advanced_axis_dynamics.data_structure.Operators as aops
import advanced_axis_dynamics.registries.standard_expansions  # noqa: F401 - a rule
import algebra.einops_simplification as einops_simplification
import algebra.write_index_notation as write_index_notation
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import deepseek.data_structure as dst
import deepseek.registries.standard_expansions as rotary_expansions
from websocket_transfer.auxiliary_information import OperatorRole

from notebooks.display.explain_operators import OperatorExplanation
from notebooks.sota.DeepSeekV41Flash.construction_idioms import boxed, hold, over
from notebooks.sota.GLM53.declared_axes import R, d, dbar, p, t, x
from notebooks.sota.GLM53.reference_links import (
    checkpoint_config_lines, configuration_lines, modeling_lines)
from notebooks.sota.GLM53.released_constants import ROTARY_BASE
from notebooks.sota.GLM53.block_titles_and_descriptions import TEXT as text

ROTATION_BOX = 'Rot'
ROTATION_COLOUR = '#E6EEF5'
COMPLEX = dst.Complex(R)
PAIR_SHAPE = (x, t)
TURNED_SHAPE = (x, p)
UNTURNED_INDEXER_SHAPE = (x, dbar)

TABLE_REFERENCES = (modeling_lines(86, 104), modeling_lines(106, 122),
                    configuration_lines(150, 152), checkpoint_config_lines(211, 214))
ROTATION_REFERENCES = (modeling_lines(125, 161), *TABLE_REFERENCES)


def table_of_turns() -> cat.Broadcasted[dst.Complex, cat.Axis, dst.Rotary]:
    '''The factor every pair of channels of every token is multiplied by.'''
    return dst.Rotary.template(x, t, base=ROTARY_BASE)


def rotate_pairs() -> cat.BroadcastedCategory:
    '''The turned channels of every token read as pairs, each pair multiplied by the
    factor of its token, and the pairs written back as channels.'''
    return ((over((x,), dst.PairsAsComplex.template(base=R, channels=p, pairs=t))
             * table_of_turns())
            @ einops_simplification.einsum(
                (PAIR_SHAPE, PAIR_SHAPE), PAIR_SHAPE, COMPLEX)
            @ over((x,), dst.Decomplex.template(base=R, pairs=t, channels=p)))


def cut_indexer_channels() -> cat.Broadcasted:
    '''An indexer vector of every token cut into the turned channels, which come
    first, and the channels left alone by the rotation.'''
    return aops.DeconcatenateAxes.template(
        (TURNED_SHAPE, UNTURNED_INDEXER_SHAPE), concatenated=d)


def join_indexer_channels() -> cat.Broadcasted:
    '''The turned channels and the channels left alone joined back onto `d`.'''
    return aops.ConcatenateAxes.template(
        (TURNED_SHAPE, UNTURNED_INDEXER_SHAPE), concatenated=d)


def rotate_first_indexer_channels() -> cat.BroadcastedCategory:
    return (cut_indexer_channels()
            @ (rotate_pairs() * hold(cat.Array(R, UNTURNED_INDEXER_SHAPE)))
            @ join_indexer_channels())


def pair_formula(offset: str) -> str:
    '''The turn of pair `i_t` of token `i_x` against the table `F`, for the pairs
    starting `offset` channels into the vector.'''
    token, pair = write_index_notation.axis_letters((x, t))
    position = write_index_notation.index_of(token)
    first = f'{offset}2 {write_index_notation.index_of(pair)}'
    real = f'{position}, {first}'
    imaginary = f'{position}, {first} + 1'
    factor = write_index_notation.read_at(
        rotary_expansions.TABLE_SYMBOL, (token, pair))
    return (rf'y[{real}] + \mathrm{{i}}\, y[{imaginary}] = {factor}\, '
            rf'\big(v[{real}] + \mathrm{{i}}\, v[{imaginary}]\big)')


def indexer_rotation_formula() -> str:
    '''The turn of the first `|p|` channels and the channels passed through after
    them.'''
    token, turned, unturned = write_index_notation.axis_letters((x, p, dbar))
    position = write_index_notation.index_of(token)
    passed = (f'{position}, {write_index_notation.element_count((turned,))} + '
              f'{write_index_notation.index_of(unturned)}')
    return (rf'\begin{{gathered}} {pair_formula("")} \\ '
            rf'y[{passed}] = v[{passed}] \end{{gathered}}')


def rotation_box(body: cat.BroadcastedCategory, title: str, formula: str,
                 description: str) -> cat.Broadcasted:
    return boxed(cat.Block.template(
        body, title=title, fill_color=ROTATION_COLOUR, formula=formula,
        description=description, references=ROTATION_REFERENCES), ROTATION_BOX)


def broadcast_between_positions_and_channels[A: cat.Axis](
    box: cat.Broadcasted,
    between: tuple[A, ...],
) -> cat.Broadcasted:
    '''A rotation box over `[x, channels]` computed once per index of `between`, on
    arrays `[x, *between, channels]`.'''
    tiled = (cat.WeaveMode.TILED,) * len(between)

    def with_tiled_slots(weave: cat.Weave) -> cat.Weave:
        positions, channels = weave._shape
        return cat.Weave(weave.datatype, (positions, *tiled, channels))

    return cat.Broadcasted(
        operator=box.operator,
        input_weaves=tuple(map(with_tiled_slots, box.input_weaves)),
        output_weaves=tuple(map(with_tiled_slots, box.output_weaves)),
        reindexings=tuple(cat.ProdObject(tuple(between)).identity()
                          for _ in box.input_weaves))


ROTATE_QUERY_KEY_CHANNELS = rotation_box(
    rotate_pairs(), text.ROTARY_TITLE, pair_formula(''),
    text.QUERY_KEY_ROTATION_DESCRIPTION)
ROTATE_INDEXER_CHANNELS = rotation_box(
    rotate_first_indexer_channels(), text.INDEXER_ROTARY_TITLE,
    indexer_rotation_formula(), text.INDEXER_ROTATION_DESCRIPTION)


def explain_channel_join(target: cat.Broadcasted) -> OperatorExplanation | None:
    '''The box over an `aops.ConcatenateAxes` that joins two runs of channels onto a
    declared channel axis. The model holds three: the query heads and the key heads
    are joined onto `a`, and a turned indexer vector onto `d`.'''
    operator = target.operator
    whole = operator.concatenated_axis()
    if isinstance(whole, AxisConcatenation.ConcatenatedAxis):
        return None
    channels, first, second = write_index_notation.axis_letters(
        (whole, *operator.parts()))
    start = write_index_notation.element_count((first,))
    return OperatorExplanation(
        title=r'\text{Concatenation of the Channels}',
        formula=(
            rf'y[{write_index_notation.index_of(first)}] = '
            rf'{write_index_notation.read_at("v", (first,))}, \qquad '
            rf'y[{start} + {write_index_notation.index_of(second)}] = '
            rf'{write_index_notation.read_at("w", (second,))}, \qquad '
            rf'|{channels}| = |{first}| + |{second}|'),
        description=text.CHANNEL_JOIN_DESCRIPTION,
        references=(modeling_lines(233, 234), modeling_lines(376, 378),
                    modeling_lines(411)))


def explain_channel_cut(target: cat.Broadcasted) -> OperatorExplanation:
    '''The box over the `aops.DeconcatenateAxes` that cuts an indexer vector into the
    turned channels and the channels left alone by the rotation.'''
    channels, first, second = write_index_notation.axis_letters(
        (target.operator.concatenated_axis(), *target.operator.parts()))
    start = write_index_notation.element_count((first,))
    return OperatorExplanation(
        title=r'\text{Cut of the Channels}',
        formula=(
            rf'v[{write_index_notation.index_of(first)}] = '
            rf'{write_index_notation.read_at("y", (first,))}, \qquad '
            rf'w[{write_index_notation.index_of(second)}] = '
            rf'y[{start} + {write_index_notation.index_of(second)}]'),
        description=text.CHANNEL_CUT_DESCRIPTION,
        references=(modeling_lines(226), modeling_lines(229)))


TABLE_ROLES: dict[str, OperatorRole] = {
    dst.table_name(dst.Rotary, None).to_bodies(): OperatorRole(
        role=text.ROTARY_TABLE_ROLE, references=TABLE_REFERENCES),
}

OPERATOR_EXPLANATIONS: dict[type[cat.Operator], OperatorExplanation] = {
    dst.PairsAsComplex: OperatorExplanation(
        title=r'\text{Pairs as Complex Numbers}',
        formula=r'w[i_{t}] = v[2 i_{t}] + \mathrm{i}\, v[2 i_{t} + 1]',
        description=text.PAIRS_AS_COMPLEX_DESCRIPTION,
        references=(modeling_lines(156, 157),)),
    dst.Decomplex: OperatorExplanation(
        title=r'\text{Complex Numbers as Pairs}',
        formula=(r'y[2 i_{t}] = \mathrm{Re}\, w[i_{t}], \qquad '
                 r'y[2 i_{t} + 1] = \mathrm{Im}\, w[i_{t}]'),
        description=text.DECOMPLEX_DESCRIPTION,
        references=(modeling_lines(159, 160),)),
}
