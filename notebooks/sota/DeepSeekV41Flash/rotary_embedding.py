'''The rotary embedding of DeepSeek-V4.1-Flash, one box for each array it rotates.

Written by Claude Fable 5.1, reasoning effort 80.

A layer of the released model that holds compressed entries rotates six arrays: the
query of every head, the window latent of every token, the compressed entries, the keys
and the queries of the indexer, and the attention output, which it rotates back. Every
rotation acts on the last `|z|` channels of a vector and leaves the channels ahead of
them alone.

Each rotation is one titled block boxed as an operator. `aops.DeconcatenateAxes` cuts
the channel axis into the channels left alone and the channels rotated.
`dst.PairsAsComplex` reads the rotated channels two at a time as complex numbers. A
product over the `dst.Complex` datatype multiplies every pair by the factor the table
holds for its position and its pair. `dst.Decomplex` writes the pairs back as reals, and
`aops.ConcatenateAxes` joins the two runs onto the declared channel axis, so a box over
`c` returns `c` and a box over `d` returns `d`. The table is a `dst.Rotary` or a
`dst.YarnRotary`, which draws as a circle. This module imports
`deepseek.registries.standard_expansions`, so the inspection box over the circle writes
the table out.

`RotaryKind` names the two tables of the released model. Layers 0 and 1 hold no
compressed entries and rotate with `dst.Rotary` at `WINDOW_ROTARY_BASE`. Every layer
from 2 to 39 rotates with `dst.YarnRotary` at `ROTARY_BASE`, and only those layers hold
entries and an indexer.

A box is written over one vector per position, which is the smallest array that holds
the position axis, and a caller broadcasts it over any other axis:

    ROTATE_TOKEN_LATENTS[kind]          R[x, c] -> R[x, c]
    ROTATE_ATTENTION_OUTPUT_BACK[kind]  R[x, c] -> R[x, c]   conjugated
    ROTATE_ENCODER_ENTRIES              R[b, c] -> R[b, c]   stride `|a|`
    ROTATE_DECODER_ENTRIES              R[B, c] -> R[B, c]
    ROTATE_INDEXER_QUERIES              R[x, d] -> R[x, d]
    ROTATE_ENCODER_INDEXER_KEYS         R[b, d] -> R[b, d]   stride `|a|`
    ROTATE_DECODER_INDEXER_KEYS         R[B, d] -> R[B, d]

The window latents compose `ROTATE_TOKEN_LATENTS[kind]` as it stands. The query composes
it under `over((h,), ...)`, and the attention output composes
`ROTATE_ATTENTION_OUTPUT_BACK[kind]` under the same lift. A box with the stride `|a|`
rotates the vector at index `i_b` by the position of token `|a| i_b`, the first token of
the group the entry stands for.

`over` prefixes the axes it broadcasts over. The indexer query is `R[x, i, d]`, with the
heads between the tokens and the channels as the released `[b, s, i, d]` has them, and
the scoring box that reads it needs the token axis at its head.
`broadcast_between_positions_and_channels` therefore tiles a box over axes that stand
after the position axis, which is the broadcast the released
`freqs_cis.view(1, s, 1, d / 2)` states.

The product against the table is built by `einops_simplification.einsum` over the
declared axes. A letter of an `ops.Einops.template` mints a fresh axis, and composition
can keep the fresh axis in place of a declared one.

Four `cat.DefinedExpression`s state the mathematics of the table, each checked by
`validate_omitted_mechanisms.py`. `deconcatenation_defined` pairs the cut of the token
latent with its standard expansion, a copy and one affine view per part.
`rotation_defined` pairs the YaRN table with its standard expansion, which arranges the
positions and the pairs and exponentiates the angle of each. `yarn_ramp_defined` and
`yarn_frequencies_defined` pair a generic operator named `r` and one named
`\\theta'` with the two steps of that expansion that produce them, which
`deepseek.registries.standard_expansions` builds, so the ramp and the frequencies are
written in one place and read here.

`OPERATOR_EXPLANATIONS` holds the rows the explanation tables of the integrated model
take for `dst.PairsAsComplex` and `dst.Decomplex`, and `explain_channel_join` is the row
for the concatenation that closes a box. The row of the base model for
`aops.ConcatenateAxes` names the slot axes of the attention core. The row of the
integrated table therefore calls `explain_channel_join` first and returns the base
model's row where that function returns `None`. `TABLE_ROLES` says what each of the two
tables is for, under the name the class declares, which is the row a figure shows beside
the circle. A `dst.Rotary` takes no `OperatorExplanation`, because its standard
expansion supplies its formula and its description.
'''
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import advanced_axis_dynamics.data_structure.AxisConcatenation as AxisConcatenation
import advanced_axis_dynamics.data_structure.Operators as aops
import advanced_axis_dynamics.registries.standard_expansions  # noqa: F401 - the deconcatenation's rule
import algebra.define_by_expansion as define_by_expansion
import algebra.einops_simplification as einops_simplification
import algebra.write_index_notation as write_index_notation
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd
import deepseek.data_structure as dst
import deepseek.registries.standard_expansions as rotary_expansions
from websocket_transfer.auxiliary_information import OperatorRole

import notebooks.sota.DeepSeekV41Flash.omitted_mechanisms as omitted_mechanisms
from notebooks.display.explain_operators import OperatorExplanation
from notebooks.sota.DeepSeekV41Flash.construction_idioms import boxed, hold, over
from notebooks.sota.DeepSeekV41Flash.declared_axes import B, R, a, b, c, d, x
from notebooks.sota.DeepSeekV41Flash.reference_links import (
    inference_config_lines, model_lines)
from notebooks.sota.DeepSeekV41Flash.released_constants import (
    ROTARY_BASE, WINDOW_ROTARY_BASE, YARN_FACTOR, YARN_RAMP_END, YARN_RAMP_START)
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

COMPLEX = omitted_mechanisms.COMPLEX
t = omitted_mechanisms.t
z = omitted_mechanisms.z
zbar = omitted_mechanisms.zbar
dbar = fd.DynamicName('\\bar{d}', code_form='unrotated_indexer_width').capture(
    cat.RawAxis(_size=d.local_size() - z.local_size()))

ROTATION_BOX = 'Rot'
INVERSE_ROTATION_BOX = 'Rot^{-1}'
ROTATION_COLOUR = '#E6EEF5'
UNIT_STRIDE = nm.Integer(1)
VALUE_OF_EVERY_PAIR = cat.Array(R, (t,))



class RotaryKind(Enum):
    '''Which of the two tables of the released model a layer rotates with. `PLAIN` is
    the table of layers 0 and 1, base 10000 with no YaRN. The three DSpark draft
    blocks, layers 40 to 42, have the ratio 0 and take the same table. `YARN` is the
    table of every layer that holds compressed entries, base 160000 with YaRN.'''
    PLAIN = 'plain'
    YARN = 'yarn'


@dataclass(frozen=True)
class RotatedChannels[A: cat.Axis]:
    '''A channel axis as the run of channels a rotation leaves alone followed by the
    run it rotates. The rotated run is `z` for every array of the model.'''
    channels: A
    left_alone: A
    rotated: A


LATENT_CHANNELS = RotatedChannels(channels=c, left_alone=zbar, rotated=z)
INDEXER_CHANNELS = RotatedChannels(channels=d, left_alone=dbar, rotated=z)


@dataclass(frozen=True)
class RotationSite[A: cat.Axis]:
    '''One array the released code rotates. The fields give the axis that carries its
    positions, the token position one step along that axis stands for, whether the
    turn is conjugated, the cut of its channels, and the short name, the title, the
    description and the references of its box.

    `conjugated` says that the site multiplies by the conjugate of the table rather
    than by the table, which turns each point back through the angle of its position.
    '''
    positions: A
    position_stride: nm.Numeric
    conjugated: bool
    channels: RotatedChannels[A]
    box_name: str
    title: str
    description: str
    references: fd.Prod[cat.CodeReference]

    def left_alone_shape(self) -> tuple[A, A]:
        return (self.positions, self.channels.left_alone)

    def rotated_shape(self) -> tuple[A, A]:
        return (self.positions, self.channels.rotated)

    def pair_shape(self) -> tuple[A, A]:
        return (self.positions, t)


def rotary_table[A: cat.Axis](
    kind: RotaryKind,
    site: RotationSite[A],
) -> cat.Broadcasted[dst.Complex, A, dst.Rotary]:
    '''The table of factors `site` multiplies its pairs by, over the positions of the
    site and the pairs `t`.'''
    if kind is RotaryKind.PLAIN:
        return dst.Rotary.template(
            site.positions, t, base=WINDOW_ROTARY_BASE,
            position_stride=site.position_stride)
    return dst.YarnRotary.template(
        site.positions, t, base=ROTARY_BASE, position_stride=site.position_stride,
        factor=YARN_FACTOR, ramp_start=YARN_RAMP_START, ramp_end=YARN_RAMP_END)


def turns_of_every_position[A: cat.Axis](
    kind: RotaryKind,
    site: RotationSite[A],
) -> cat.BroadcastedCategory:
    '''The factor `site` multiplies each pair by: the table itself where the site
    turns forward, and the table followed by the conjugate of every entry where it
    turns back. A conjugated site holds no table of its own, because the conjugate of
    a turn is the turn through the same angle in the other direction.'''
    table = rotary_table(kind, site)
    if not site.conjugated:
        return table
    return table @ dst.conjugate_complex_values(table.cod()[0])


def rotate_pairs[A: cat.Axis](
    kind: RotaryKind,
    site: RotationSite[A],
) -> cat.BroadcastedCategory:
    '''The rotated channels of every position read as pairs, each pair multiplied by
    the factor of its position, and the pairs written back as channels.'''
    rotated = site.channels.rotated
    return ((over((site.positions,),
                  dst.PairsAsComplex.template(base=R, channels=rotated, pairs=t))
             * turns_of_every_position(kind, site))
            @ einops_simplification.einsum(
                (site.pair_shape(), site.pair_shape()), site.pair_shape(), COMPLEX)
            @ over((site.positions,),
                   dst.Decomplex.template(base=R, pairs=t, channels=rotated)))


def cut_rotated_channels[A: cat.Axis](site: RotationSite[A]) -> cat.Broadcasted:
    '''The vector of every position cut into the channels the rotation leaves alone
    and the channels it rotates.'''
    return aops.DeconcatenateAxes.template(
        (site.left_alone_shape(), site.rotated_shape()),
        concatenated=site.channels.channels)


def join_rotated_channels[A: cat.Axis](site: RotationSite[A]) -> cat.Broadcasted:
    '''The channels left alone and the rotated channels joined back onto the channel
    axis the vector arrived on.'''
    return aops.ConcatenateAxes.template(
        (site.left_alone_shape(), site.rotated_shape()),
        concatenated=site.channels.channels)


def rotate_last_channels[A: cat.Axis](
    kind: RotaryKind,
    site: RotationSite[A],
) -> cat.BroadcastedCategory:
    '''One vector per position with its last channels rotated, on the channel axis it
    arrived on.'''
    return (cut_rotated_channels(site)
            @ (hold(cat.Array(R, site.left_alone_shape())) * rotate_pairs(kind, site))
            @ join_rotated_channels(site))


def rotation_formula[A: cat.Axis](site: RotationSite[A]) -> str:
    '''The rotation of `v` into `y` in index notation: the pair equation against the
    table `F`, and the channels that pass through.'''
    positions, left_alone, pairs = write_index_notation.axis_letters(
        (site.positions, site.channels.left_alone, t))
    position = write_index_notation.index_of(positions)
    offset = write_index_notation.element_count((left_alone,))
    pair = write_index_notation.index_of(pairs)
    real = rf'{position}, {offset} + 2 {pair}'
    imaginary = rf'{position}, {offset} + 2 {pair} + 1'
    factor = write_index_notation.read_at(
        rotary_expansions.TABLE_SYMBOL, (positions, pairs))
    turned = (rf'y[{real}] + \mathrm{{i}}\, y[{imaginary}] = {factor}\, '
              rf'\big(v[{real}] + \mathrm{{i}}\, v[{imaginary}]\big)')
    unchanged = (f'{write_index_notation.read_at("y", (positions, left_alone))} = '
                 f'{write_index_notation.read_at("v", (positions, left_alone))}')
    return rf'\begin{{gathered}} {turned} \\ {unchanged} \end{{gathered}}'


def rotation_box[A: cat.Axis](
    kind: RotaryKind,
    site: RotationSite[A],
) -> cat.Broadcasted:
    return boxed(cat.Block.template(
        rotate_last_channels(kind, site),
        title=site.title, fill_color=ROTATION_COLOUR,
        formula=rotation_formula(site), description=site.description,
        references=site.references), site.box_name)


def broadcast_between_positions_and_channels[A: cat.Axis](
    box: cat.Broadcasted,
    between: tuple[A, ...],
) -> cat.Broadcasted:
    '''A rotation box over `[positions, channels]` computed once per index of
    `between`, on arrays `[positions, *between, channels]`. `over` prefixes the axes
    it broadcasts over, which would put them ahead of the positions.'''
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


TOKEN_LATENTS = RotationSite(
    positions=x, position_stride=UNIT_STRIDE,
    conjugated=False, channels=LATENT_CHANNELS,
    box_name=ROTATION_BOX, title=text.TOKEN_TITLE,
    description=text.TOKEN_LATENTS_DESCRIPTION,
    references=(model_lines(392, 406), model_lines(771, 772), model_lines(705, 706),
                model_lines(368, 389), model_lines(680, 687)))

ATTENTION_OUTPUT = RotationSite(
    positions=x, position_stride=UNIT_STRIDE,
    conjugated=True, channels=LATENT_CHANNELS,
    box_name=INVERSE_ROTATION_BOX, title=text.ROTARY_OUTPUT_TITLE,
    description=text.ATTENTION_OUTPUT_DESCRIPTION,
    references=(model_lines(780, 781), model_lines(392, 395), model_lines(398, 399)))

ENCODER_ENTRIES = RotationSite(
    positions=b, position_stride=a.local_size(),
    conjugated=False, channels=LATENT_CHANNELS,
    box_name=ROTATION_BOX, title=text.ENCODER_ENTRY_TITLE,
    description=text.ENCODER_ENTRIES_DESCRIPTION,
    references=(model_lines(749, 758), model_lines(392, 406), model_lines(85, 87)))

DECODER_ENTRIES = RotationSite(
    positions=B, position_stride=UNIT_STRIDE,
    conjugated=False, channels=LATENT_CHANNELS,
    box_name=ROTATION_BOX, title=text.DECODER_ENTRY_TITLE,
    description=text.DECODER_ENTRIES_DESCRIPTION,
    references=(model_lines(749, 758), model_lines(392, 406)))

INDEXER_QUERIES = RotationSite(
    positions=x, position_stride=UNIT_STRIDE,
    conjugated=False, channels=INDEXER_CHANNELS,
    box_name=ROTATION_BOX, title=text.INDEXER_QUERY_TITLE,
    description=text.INDEXER_QUERIES_DESCRIPTION,
    references=(model_lines(550, 551), model_lines(733, 734), model_lines(392, 406)))

ENCODER_INDEXER_KEYS = RotationSite(
    positions=b, position_stride=a.local_size(),
    conjugated=False, channels=INDEXER_CHANNELS,
    box_name=ROTATION_BOX, title=text.ENCODER_KEY_TITLE,
    description=text.ENCODER_INDEXER_KEYS_DESCRIPTION,
    references=(model_lines(537, 545), model_lines(392, 406)))

DECODER_INDEXER_KEYS = RotationSite(
    positions=B, position_stride=UNIT_STRIDE,
    conjugated=False, channels=INDEXER_CHANNELS,
    box_name=ROTATION_BOX, title=text.DECODER_KEY_TITLE,
    description=text.DECODER_INDEXER_KEYS_DESCRIPTION,
    references=(model_lines(537, 545), model_lines(392, 406)))

ROTATE_TOKEN_LATENTS: dict[RotaryKind, cat.Broadcasted] = {
    kind: rotation_box(kind, TOKEN_LATENTS) for kind in RotaryKind}
ROTATE_ATTENTION_OUTPUT_BACK: dict[RotaryKind, cat.Broadcasted] = {
    kind: rotation_box(kind, ATTENTION_OUTPUT) for kind in RotaryKind}
ROTATE_ENCODER_ENTRIES = rotation_box(RotaryKind.YARN, ENCODER_ENTRIES)
ROTATE_DECODER_ENTRIES = rotation_box(RotaryKind.YARN, DECODER_ENTRIES)
ROTATE_INDEXER_QUERIES = rotation_box(RotaryKind.YARN, INDEXER_QUERIES)
ROTATE_ENCODER_INDEXER_KEYS = rotation_box(RotaryKind.YARN, ENCODER_INDEXER_KEYS)
ROTATE_DECODER_INDEXER_KEYS = rotation_box(RotaryKind.YARN, DECODER_INDEXER_KEYS)

def yarn_table() -> cat.Broadcasted[dst.Complex, cat.Axis, dst.YarnRotary]:
    '''The table of turns of a layer that holds compressed entries, which is the
    table inside `ROTATE_TOKEN_LATENTS[RotaryKind.YARN]`.'''
    return rotary_table(RotaryKind.YARN, TOKEN_LATENTS)


def ramp_of_every_pair() -> cat.Broadcasted:
    '''The YaRN ramp over the pairs as one operator. `yarn_ramp_defined` states what
    the operator holds.'''
    return omitted_mechanisms.generic_operator(
        rotary_expansions.RAMP_SYMBOL, (), (VALUE_OF_EVERY_PAIR,))


def frequencies_of_every_pair() -> cat.Broadcasted:
    '''The YaRN frequency of every pair as one operator.
    `yarn_frequencies_defined` states what the operator holds.'''
    return omitted_mechanisms.generic_operator(
        rotary_expansions.YARN_FREQUENCY_SYMBOL, (), (VALUE_OF_EVERY_PAIR,))


def yarn_ramp_defined() -> cat.DefinedExpression:
    '''The ramp of every pair, which is the clamp to the unit interval that the
    expansion of the table applies to the index of the pair. An `ops.Arrange` over
    the pairs supplies the index.'''
    table = rotary_expansions.rotary_table_of(yarn_table())
    pair_indices = ops.Arrange.template(table.pairs)
    return cat.DefinedExpression.template(
        ramp_of_every_pair(),
        pair_indices @ rotary_expansions.clamp_pair_index_to_ramp(
            table, pair_indices.cod()[0]))


def yarn_frequencies_defined() -> cat.DefinedExpression:
    '''The frequency of every pair, which is the power of the base times the factor
    that interpolates along the ramp between one and the reciprocal of the YaRN
    factor, as the expansion of the table reads it.'''
    return cat.DefinedExpression.template(
        frequencies_of_every_pair(),
        rotary_expansions.yarn_frequencies(
            rotary_expansions.rotary_table_of(yarn_table())))


def rotation_defined() -> cat.DefinedExpression:
    '''The table of turns beside its standard expansion, which arranges the positions
    and the pairs, reads the frequency of every pair, multiplies the position by the
    frequency and exponentiates the angle of every position and pair.'''
    return define_by_expansion.define_by_standard_expansion(yarn_table())


def deconcatenation_defined() -> cat.DefinedExpression:
    '''The cut of the token latent beside its standard expansion, which is a copy of
    the latent followed by one affine view per part.'''
    return define_by_expansion.define_by_standard_expansion(
        cut_rotated_channels(TOKEN_LATENTS))


def explain_channel_join(target: cat.Broadcasted) -> OperatorExplanation | None:
    '''The row of an explanation table for the `aops.ConcatenateAxes` that closes a
    rotation box, which joins two runs of channels onto a declared channel axis. A
    concatenation onto an `AxisConcatenation.ConcatenatedAxis`, as the slots of the
    attention core are joined, is left to the row of the base model.'''
    operator = target.operator
    if not isinstance(operator, aops.ConcatenateAxes):
        return None
    whole = operator.concatenated_axis()
    if isinstance(whole, AxisConcatenation.ConcatenatedAxis):
        return None
    channels, *parts = write_index_notation.axis_letters((whole, *operator.parts()))
    first, second = parts
    start = write_index_notation.element_count((first,))
    return OperatorExplanation(
        title=r'\text{Concatenation of the Channels}',
        formula=(
            rf'y[{write_index_notation.index_of(first)}] = '
            rf'{write_index_notation.read_at("u", (first,))}, \qquad '
            rf'y[{start} + {write_index_notation.index_of(second)}] = '
            rf'{write_index_notation.read_at("r", (second,))}, \qquad '
            rf'|{channels}| = |{first}| + |{second}|'),
        description=text.EXPLAIN_CHANNEL_JOIN_DESCRIPTION,
        references=(model_lines(404, 405),))


PLAIN_TABLE_NAME = dst.table_name(dst.Rotary, None).to_bodies()
YARN_TABLE_NAME = dst.table_name(dst.YarnRotary, None).to_bodies()

TABLE_ROLES: dict[str, OperatorRole] = {
    PLAIN_TABLE_NAME: OperatorRole(
        role=(
            text.WINDOW_ROTARY_TABLE_ROLE),
        references=(model_lines(369, 389), inference_config_lines(30))),
    YARN_TABLE_NAME: OperatorRole(
        role=(
            text.YARN_ROTARY_TABLE_ROLE),
        references=(model_lines(369, 389), inference_config_lines(51),
                    inference_config_lines(31))),
}

OPERATOR_EXPLANATIONS: dict[type[cat.Operator], OperatorExplanation] = {
    dst.PairsAsComplex: OperatorExplanation(
        title=r'\text{Pairs as Complex Numbers}',
        formula=r'p[i_{t}] = v[2 i_{t}] + \mathrm{i}\, v[2 i_{t} + 1]',
        description=text.PAIRS_AS_COMPLEX_DESCRIPTION,
        references=(model_lines(397),)),
    dst.Decomplex: OperatorExplanation(
        title=r'\text{Complex Numbers as Pairs}',
        formula=(r'y[2 i_{t}] = \mathrm{Re}\, p[i_{t}], \qquad '
                 r'y[2 i_{t} + 1] = \mathrm{Im}\, p[i_{t}]'),
        description=text.DECOMPLEX_DESCRIPTION,
        references=(model_lines(404),)),
}
