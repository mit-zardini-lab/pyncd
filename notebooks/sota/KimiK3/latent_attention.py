# Claude Opus 5.5 (1M context), effort 40.
'''The gated multi-head latent attention of Kimi K3, which runs in 24 of the 93 layers.

The attention is `KimiMLAAttention` of the reference, the latent attention of
DeepSeek-V3 with two changes. The configuration sets `mla_use_nope`, so no channel of a
query or a key is turned by a rotary embedding, and the reference holds no rotary
table. It also sets `mla_use_output_gate`, so the output of every head is multiplied by
a sigmoid gate read off the hidden state before the output projection.

The query passes through a low rank of 1536 channels and an RMS normalisation, and one
projection gives every head 192 channels. The reference cuts those channels into 128
and 64 with `torch.split` and joins them again with `torch.cat` in the same order, which
changes no value, so the query is one projection onto the axis `a`. The key of a head
has two parts. 128 channels are expanded per head from a latent of 512 channels, and 64
channels are projected from the hidden state once per token and copied to every head,
which is the `k_rot` of the reference, named for a rotation that this configuration
does not apply. The value of every head is expanded from the same latent.

The reference computes `kv_a_proj_with_mqa` and `kv_b_proj` as one weight each and cuts
each result with `torch.split`. A linear map followed by a cut of its results is one
linear map per part, so the expression holds one weight per part:

    kv_a_proj_with_mqa    W^{KVa} onto the latent, W^{Kr} onto the shared key channels
    kv_b_proj             W^{Kb} onto the latent key channels, W^{Vb} onto the values

The two parts of a key are joined onto the axis `a` of 192 channels with
`aops.ConcatenateAxes`, which is the `torch.cat` of the reference.

A query attends to every token at or before it. The keys and the values of every head
are read back from each query at token `i_x - i_r`, through the view `Back`, whose
distance axis `r|x` holds a value where `i_x - i_r >= 0`. The core is one box computed
once per query and head.

    query_path()          STATE[x, m] -> QUERIES[x, h, a]
    keys_and_values()     STATE[x, m] -> KEYS[x, h, a], VALUES[x, h, u]
    CORE                  the box computed once per query and head
    output_gate()         STATE[x, m] -> R[x, h, u]
'''
from __future__ import annotations

import advanced_axis_dynamics.algebra.mark_sparse_domains as mark_sparse_domains
import advanced_axis_dynamics.data_structure.Operators as aops
import algebra.discovering_broadcasts as discovering_broadcasts
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import data_structure.Term as fd

from notebooks.sota.DeepSeekV41Flash.construction_idioms import boxed, hold, over, route
from notebooks.sota.DeepSeekV41Flash.custom_operations import sigmoid
from notebooks.sota.KimiK3.declared_axes import (
    HEAD_OUTPUTS, KEYS, LATENT, LATENT_KEYS, QUERIES, R, SHARED_KEYS, STATE, VALUES,
    a, c, h, m, n, p, q, r, u, x)
from notebooks.sota.KimiK3.reference_links import checkpoint_config_lines, modeling_lines
from notebooks.sota.KimiK3.released_constants import LATENT_NORM_EPSILON
from notebooks.sota.KimiK3.block_titles_and_descriptions import TEXT as text

QUERY_COLOUR = '#E8DFF0'
KEY_VALUE_COLOUR = '#DDE8D6'
CORE_COLOUR = '#C5BEDF'
GATE_COLOUR = '#F7E0EF'
ATTENTION_COLOUR = '#FFE2BB'
CORE_BOX = 'Core'
ATTENTION_BOX = 'MLA'
SCORE_SCALE_NAME = 'x / \\sqrt{\\lvert a \\rvert}'
REPEAT_VIEW_NAME = '\\mathrm{Repeat}'
BACK_VIEW_NAME = '\\mathrm{Back}'

QUERY_REFERENCES = (modeling_lines(364, 373), modeling_lines(418, 424),
                    modeling_lines(439), checkpoint_config_lines(199))
KEY_VALUE_REFERENCES = (modeling_lines(378, 389), modeling_lines(426, 440),
                        checkpoint_config_lines(59), checkpoint_config_lines(200, 201),
                        checkpoint_config_lines(266))
CORE_REFERENCES = (modeling_lines(311, 332), modeling_lines(357, 359),
                   modeling_lines(446, 466), modeling_lines(1173, 1180))
GATE_REFERENCES = (modeling_lines(398, 401), modeling_lines(468, 473),
                   checkpoint_config_lines(174))
ATTENTION_REFERENCES = (modeling_lines(335, 474), checkpoint_config_lines(173, 174),
                        checkpoint_config_lines(181), checkpoint_config_lines(188))


READ_BACK = mark_sparse_domains.mark_sparse_domain(sc.StrideMorphism(
    _dom=(x, r),
    _cod_stride_shift=((x, (nm.Integer(1), nm.Integer(-1)), nm.Integer(0)),),
    name=fd.DynamicName(BACK_VIEW_NAME)))
'''The read of token `i_x - i_r` at distance `i_r` of each query `i_x`, with the
distance axis marked `r|x`, live where `i_x - i_r >= 0`.'''


def read_back_from_each_token[A: cat.Axis](rest: tuple[A, ...]) -> cat.Broadcasted:
    '''An array over the tokens read at token `i_x - i_r`, `i_r` tokens back from each
    query, times the identity on `rest`.'''
    return ops.View.template(
        reindexing=(READ_BACK, cat.ProdObject(rest).identity()), name=BACK_VIEW_NAME)


KEYS_READ_BACK = read_back_from_each_token((h, a))
reach = KEYS_READ_BACK.cod()[0].shape()[1]


def query_path() -> cat.Block:
    '''`STATE[x, m] -> QUERIES[x, h, a]`: the hidden state projected onto the query low
    rank, normalised, and projected onto every head.'''
    return cat.Block.template(
        (x >> ops.Linear.template((m,), (q,), 'W^{Qa}'))
        @ over((x,), ops.Normalize.template((q,), epsilon=LATENT_NORM_EPSILON))
        @ (x >> ops.Linear.template((q,), (h, a), 'W^{Qb}')),
        title=text.QUERY_TITLE, fill_color=QUERY_COLOUR,
        description=text.QUERY_PATH_DESCRIPTION, references=QUERY_REFERENCES)


def repeat_over_heads() -> cat.Broadcasted:
    '''The one shared key of a token copied to every head, which is the `expand` of
    the reference.'''
    return ops.View.template(
        reindexing=cat.Rearrangement((0, 2), (x, h, p)), name=REPEAT_VIEW_NAME)


def keys_and_values() -> cat.Block:
    '''`STATE[x, m] -> KEYS[x, h, a], VALUES[x, h, u]`: the latent of every token,
    normalised and expanded into the latent key channels of every head and into every
    value head, and the shared key channels of every token, copied to every head and
    joined to the latent key channels.'''
    return cat.Block.template(
        route((0, 0), (STATE,))
        @ (((x >> ops.Linear.template((m,), (c,), 'W^{KVa}'))
            @ over((x,), ops.Normalize.template((c,), epsilon=LATENT_NORM_EPSILON))
            @ route((0, 0), (LATENT,))
            @ ((x >> ops.Linear.template((c,), (h, n), 'W^{Kb}'))
               * (x >> ops.Linear.template((c,), (h, u), 'W^{Vb}'))))
           * ((x >> ops.Linear.template((m,), (p,), 'W^{Kr}'))
              @ repeat_over_heads()))
        @ route((0, 2, 1), (LATENT_KEYS, VALUES, SHARED_KEYS))
        @ (aops.ConcatenateAxes.template(((x, h, n), (x, h, p)), concatenated=a)
           * hold(VALUES)),
        title=text.KEYS_AND_VALUES_TITLE, fill_color=KEY_VALUE_COLOUR,
        description=text.KEYS_AND_VALUES_DESCRIPTION, references=KEY_VALUE_REFERENCES)


def scale_by_inverse_square_root[A: cat.Axis](axis: A, name: str) -> cat.Broadcasted:
    '''The factor one over the square root of the size of `axis`.'''
    return ops.Arithmetic.template(nm.x / nm.SquareRoot(axis.local_size()), name=name)


def attend_over_every_query_and_head() -> cat.Block:
    '''The softmax of every query and head over the tokens at or before it, written out
    over every query and head: the score of the query against every key read back, the
    scale `1 / \\sqrt{|a|}`, the softmax over the distances and the sum of the values
    under the softmax.'''
    keys_back = cat.Array(R, (x, reach, h, a))
    values_back = cat.Array(R, (x, reach, h, u))
    return cat.Block.template(
        (hold(QUERIES) * hold(keys_back) * hold(values_back))
        @ (ops.Einops.template('x h a, x r h a -> x h r') * hold(values_back))
        @ (over((x, h, reach), scale_by_inverse_square_root(a, SCORE_SCALE_NAME))
           * hold(values_back))
        @ (over((x, h), ops.SoftMax.template()) * hold(values_back))
        @ ops.Einops.template('x h r, x r h u -> x h u'),
        title=text.CORE_TITLE, fill_color=CORE_COLOUR,
        description=text.CORE_DESCRIPTION, references=CORE_REFERENCES)


def attend_for_one_query_and_head() -> cat.Block:
    '''One query and one head: the score against the key at every distance, the scale,
    the softmax over the distances and the sum of the values under the softmax.'''
    keys_back = cat.Array(R, (reach, a))
    values_back = cat.Array(R, (reach, u))
    return cat.Block.template(
        (hold(cat.Array(R, (a,))) * hold(keys_back) * hold(values_back))
        @ (ops.Einops.template('a, r a -> r') * hold(values_back))
        @ (over((reach,), scale_by_inverse_square_root(a, SCORE_SCALE_NAME))
           * hold(values_back))
        @ (ops.SoftMax.template() * hold(values_back))
        @ ops.Einops.template('r, r u -> u'),
        title=text.CORE_TITLE, fill_color=CORE_COLOUR,
        description=text.CORE_DESCRIPTION, references=CORE_REFERENCES)


def computed_once_per_query_and_head(body: cat.Block) -> cat.Broadcasted:
    '''`body` as one box computed once per query and head, on the queries `[x, h, a]`
    and on the keys and the values read back, `[x, r|x, h, ...]`, whose head axis
    stands after the distance axis. The box tiles the positions of the query and the
    head in every weave.'''
    T = cat.WeaveMode.TILED
    degree = (x, h)
    queries, keys_back, values_back = body.dom()
    output, = body.cod()
    return cat.Broadcasted(
        operator=ops.BlockOperator(
            name=fd.DynamicName(CORE_BOX, settings=fd.DynamicNameSettings(bold=True)),
            block=body),
        input_weaves=(
            cat.Weave(R, (T, T, *queries.shape())),
            cat.Weave(R, (T, keys_back.shape()[0], T, keys_back.shape()[1])),
            cat.Weave(R, (T, values_back.shape()[0], T, values_back.shape()[1]))),
        output_weaves=(cat.Weave(R, (T, T, *output.shape())),),
        reindexings=tuple(cat.ProdObject(degree).identity() for _ in range(3)))


CORE = computed_once_per_query_and_head(attend_for_one_query_and_head())
CORE_CONFIRMATION = discovering_broadcasts.confirm_broadcast_expansion(
    attend_over_every_query_and_head(), CORE)


def output_gate() -> cat.Block:
    '''`STATE[x, m] -> R[x, h, u]`: the sigmoid of a projection of the hidden state onto
    every channel of every head.'''
    return cat.Block.template(
        (x >> ops.Linear.template((m,), (h, u), 'W^{Gm}')) @ sigmoid(),
        title=text.OUTPUT_GATE_TITLE, fill_color=GATE_COLOUR,
        description=text.OUTPUT_GATE_DESCRIPTION, references=GATE_REFERENCES)


def gated_latent_attention() -> cat.Block:
    '''`STATE[x, m] -> STATE[x, m]`: the queries, the keys and the values, the keys and
    the values read back from every query, the core, the output gate and the output
    projection.'''
    return cat.Block.template(
        route((0, 0, 0), (STATE,))
        @ (query_path() * keys_and_values() * output_gate())
        @ (hold(QUERIES) * KEYS_READ_BACK * read_back_from_each_token((h, u))
           * hold(HEAD_OUTPUTS))
        @ (CORE * hold(HEAD_OUTPUTS))
        @ ops.Einops.template('x h u, x h u -> x h u')
        @ (x >> ops.Linear.template((h, u), (m,), 'W^{Om}')),
        title=text.MLA_TITLE, fill_color=ATTENTION_COLOUR,
        description=text.MLA_DESCRIPTION, references=ATTENTION_REFERENCES)


GATED_LATENT_ATTENTION = boxed(gated_latent_attention(), ATTENTION_BOX)
