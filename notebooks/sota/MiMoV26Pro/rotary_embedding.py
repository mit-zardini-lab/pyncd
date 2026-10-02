# Claude Opus 5.5 (1M context), effort 40.
'''The rotary embedding of MiMo-V2.6-Pro, one box for each of the two tables of turns.

The reference turns the first 64 of the 192 channels of every query head and every key
head, and passes the other 128 through. `MiMoV2RotaryEmbedding` builds one table for the
full attention layers at the base `rope_theta` and one for the sliding window layers at
the base `swa_rope_theta`. Each table holds 32 frequencies, and the reference repeats
them over two halves of 32 channels with `torch.cat((freqs, freqs))`.

`apply_rotary_pos_emb` turns the 64 channels with `rotate_half`, so channel `i_t` of the
first half and channel `|t| + i_t` of the second half are one pair, turned at frequency
`i_t`, the first channel as the real part and the second as the imaginary part. The
projections of `grouped_query_attention` write the two halves into their rows as the
axis `c`, per the ruling on groupings in `obsidian/06-practice/Representing Models.md`,
so a box reads the turned channels of one token as `R[c, t]`, where entry `(i_c, i_t)`
is channel `i_t + |t| i_c` of the reference. The covariant view named Pairs writes the
two halves in the order of the pairs, `i_p = 2 i_t + i_c`, which `dst.PairsAsComplex`
reads, the pairs are multiplied by the table, and `dst.Decomplex` writes them back as
channels in the order of the pairs:

    y[2 i_t] + i y[2 i_t + 1] = F[i_x, i_t] (v[0, i_t] + i v[1, i_t])

The reference writes the turned pairs back as two halves. The box writes each pair back
in place, the query and the key are written in the same order, and the order of the
channels of a head changes no score, so every score is unchanged. The cache of the pass
over new tokens holds the same numbers as the cache of the reference, in the order of
the pairs.

    ROTATE_FULL_ATTENTION_CHANNELS      R[x, c, t] -> R[x, p]   at the base beta
    ROTATE_SLIDING_WINDOW_CHANNELS      R[x, c, t] -> R[x, p]   at the base beta_w

A box is written over the turned channels of one vector per token, which is the
smallest array that holds the position axis, and `rotate_every_head` computes it once
per head.
'''
from __future__ import annotations

import advanced_axis_dynamics.data_structure.Operators as aops
import algebra.einops_simplification as einops_simplification
import algebra.write_index_notation as write_index_notation
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.StrideCategory as sc
import data_structure.Term as fd
import deepseek.data_structure as dst
import deepseek.registries.standard_expansions as rotary_expansions
from websocket_transfer.auxiliary_information import OperatorRole

from notebooks.display.explain_operators import OperatorExplanation
from notebooks.sota.DeepSeekV41Flash.construction_idioms import boxed, over
from notebooks.sota.MiMoV26Pro.declared_axes import R, c, p, t, x
from notebooks.sota.MiMoV26Pro.reference_links import (
    checkpoint_config_lines, configuration_lines, modeling_lines)
from notebooks.sota.MiMoV26Pro.released_constants import (
    ROTARY_BASE, WINDOW_ROTARY_BASE)
from notebooks.sota.MiMoV26Pro.block_titles_and_descriptions import TEXT as text

ROTATION_BOX = 'Rot'
WINDOW_ROTATION_BOX = 'RotW'
ROTATION_COLOUR = '#E6EEF5'
COMPLEX = dst.Complex(R)
PAIR_SHAPE = (x, t)
HALVES_SHAPE = (c, t)
PAIRS_VIEW_NAME = '\\mathrm{Pairs}'

PAIRS = sc.StrideMorphism(
    _dom=(c, t),
    _cod_stride_shift=((p, (nm.Integer(1), nm.Integer(2)), nm.Integer(0)),),
    name=fd.DynamicName(PAIRS_VIEW_NAME))
'''Entry `(i_c, i_t)` of the two halves written at channel `2 i_t + i_c`, the order of
the pairs `dst.PairsAsComplex` reads.'''

TABLE_REFERENCES = (modeling_lines(445, 505), modeling_lines(1596, 1597),
                    modeling_lines(1660, 1661), configuration_lines(133, 135),
                    checkpoint_config_lines(359, 365), checkpoint_config_lines(373))
ROTATION_REFERENCES = (modeling_lines(47, 60), modeling_lines(253),
                       modeling_lines(305, 310), *TABLE_REFERENCES)


def table_of_turns(
    base: nm.FreeNumeric,
) -> cat.Broadcasted[dst.Complex, cat.Axis, dst.Rotary]:
    '''The factor every pair of channels of every token is multiplied by.'''
    return dst.Rotary.template(x, t, base=base)


def read_the_halves_as_pairs() -> cat.BroadcastedCategory:
    '''`R[x, c, t] -> Complex(R)[x, t]`: the two halves of every token written in the
    order of the pairs, and each pair read as one complex number.'''
    return (over((x,), aops.CovariantView.template(PAIRS, name=PAIRS_VIEW_NAME))
            @ over((x,), dst.PairsAsComplex.template(base=R, channels=p, pairs=t)))


def rotate_pairs(base: nm.FreeNumeric) -> cat.BroadcastedCategory:
    '''The turned channels of every token read as pairs, each pair multiplied by the
    factor of its token, and the pairs written back as channels.'''
    return ((read_the_halves_as_pairs() * table_of_turns(base))
            @ einops_simplification.einsum(
                (PAIR_SHAPE, PAIR_SHAPE), PAIR_SHAPE, COMPLEX)
            @ over((x,), dst.Decomplex.template(base=R, pairs=t, channels=p)))


def rotation_formula() -> str:
    '''The turn of pair `i_t` of token `i_x` against the table `F`, reading the real
    part from the first half and the imaginary part from the second.'''
    token, half, pair = write_index_notation.axis_letters((x, c, t))
    position = write_index_notation.index_of(token)
    index = write_index_notation.index_of(pair)
    factor = write_index_notation.read_at(rotary_expansions.TABLE_SYMBOL, (token, pair))
    return (rf'y[{position}, 2 {index}] + \mathrm{{i}}\, y[{position}, 2 {index} + 1] '
            rf'= {factor}\, \big(v[{position}, 0, {index}] + \mathrm{{i}}\, '
            rf'v[{position}, 1, {index}]\big)')


def rotation_box(base: nm.FreeNumeric, title: str, description: str,
                 short_name: str) -> cat.Broadcasted:
    return boxed(cat.Block.template(
        rotate_pairs(base), title=title, fill_color=ROTATION_COLOUR,
        formula=rotation_formula(), description=description,
        references=ROTATION_REFERENCES), short_name)


ROTATE_FULL_ATTENTION_CHANNELS = rotation_box(
    ROTARY_BASE, text.ROTARY_TITLE, text.FULL_ROTATION_DESCRIPTION, ROTATION_BOX)
ROTATE_SLIDING_WINDOW_CHANNELS = rotation_box(
    WINDOW_ROTARY_BASE, text.WINDOW_ROTARY_TITLE, text.WINDOW_ROTATION_DESCRIPTION,
    WINDOW_ROTATION_BOX)


def rotate_every_head[A: cat.Axis](box: cat.Broadcasted,
                                   heads: tuple[A, ...]) -> cat.Broadcasted:
    '''`box`, over `[x, *channels]`, computed once per index of `heads`, on arrays
    `[x, *heads, *channels]`. `over` prefixes the axes it broadcasts over, which would
    put them ahead of the positions.'''
    tiled = (cat.WeaveMode.TILED,) * len(heads)

    def with_tiled_slots(weave: cat.Weave) -> cat.Weave:
        positions, *channels = weave._shape
        return cat.Weave(weave.datatype, (positions, *tiled, *channels))

    return cat.Broadcasted(
        operator=box.operator,
        input_weaves=tuple(map(with_tiled_slots, box.input_weaves)),
        output_weaves=tuple(map(with_tiled_slots, box.output_weaves)),
        reindexings=tuple(cat.ProdObject(tuple(heads)).identity()
                          for _ in box.input_weaves))


def explain_pairs(target: cat.Broadcasted) -> OperatorExplanation:
    '''The box over the covariant view named Pairs, written from the row of its
    reindexing.'''
    half, pair = write_index_notation.axis_letters(HALVES_SHAPE)
    return OperatorExplanation(
        title=r'\text{Pairs}',
        formula=(rf'y[2 {write_index_notation.index_of(pair)} + '
                 rf'{write_index_notation.index_of(half)}] = '
                 rf'{write_index_notation.read_at("v", (half, pair))}'),
        description=text.PAIRS_MERGE_DESCRIPTION,
        references=(modeling_lines(47, 51), modeling_lines(58, 59)))


def explain_channel_join(target: cat.Broadcasted) -> OperatorExplanation:
    '''The box over the `aops.ConcatenateAxes` that joins the turned channels and the
    channels passed through onto the width of a query head or a key head.'''
    channels, first, second = write_index_notation.axis_letters(
        (target.operator.concatenated_axis(), *target.operator.parts()))
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
        references=(modeling_lines(306, 310),))


TABLE_ROLES: dict[str, OperatorRole] = {
    dst.table_name(dst.Rotary, None).to_bodies(): OperatorRole(
        role=text.ROTARY_TABLE_ROLE, references=TABLE_REFERENCES),
}

OPERATOR_EXPLANATIONS: dict[type[cat.Operator], OperatorExplanation] = {
    dst.PairsAsComplex: OperatorExplanation(
        title=r'\text{Pairs as Complex Numbers}',
        formula=r'w[i_{t}] = v[2 i_{t}] + \mathrm{i}\, v[2 i_{t} + 1]',
        description=text.PAIRS_AS_COMPLEX_DESCRIPTION,
        references=(modeling_lines(47, 51), modeling_lines(58, 59))),
    dst.Decomplex: OperatorExplanation(
        title=r'\text{Complex Numbers as Pairs}',
        formula=(r'y[2 i_{t}] = \mathrm{Re}\, w[i_{t}], \qquad '
                 r'y[2 i_{t} + 1] = \mathrm{Im}\, w[i_{t}]'),
        description=text.DECOMPLEX_DESCRIPTION,
        references=(modeling_lines(58, 59),)),
}
