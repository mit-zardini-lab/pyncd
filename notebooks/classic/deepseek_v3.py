# Claude Opus 5.5 (1M context), reasoning effort 40.
'''DeepSeek-V3 at the sizes of the hand-drawn diagram, as a morphism in Br.

The model reads the token identifiers of a prompt and returns one score per token of the
vocabulary at every position. It is written from `inference/model.py` of
`deepseek-ai/DeepSeek-V3`, pinned in `reference_links.py`, at the sizes of
`inference/configs/config_16B.json`, which are the sizes the hand-drawn diagram labels.
That configuration sets no low rank for the queries, so the queries are projected from
the hidden state directly. It leaves the score function of the router at its default,
the softmax, and the diagram draws the sigmoid gate of `config_671B.json`, which this
module writes. At `config_16B.json` the router has no correction bias, one expert group
and a route scale of one, so the sigmoid gate is the selection of the six highest
sigmoid scores followed by their division by their sum.

The embedding is followed by `D` dense layers and `L` layers holding the mixture of
experts, so the released `n_layers` is `D + L`. Every layer applies multi-head latent
attention, boxed as `At`, and then its feed-forward sublayer, each after an RMSNorm and
inside a residual connection.

A query and a key of a head are two runs of channels laid end to end by an
`aops.ConcatenateAxes`: `d_n` channels that carry no position and `d_r` channels that
the rotary embedding turns. The rotated key channels are projected once per token and
repeated for every head. The keys and the values are read back from every query through
the view named `Mask`, whose slot axis `w` is as long as the sequence, so the slot axis
leaves the view as `w|x` and the mask of the released code is stated as a read.

The rotary embedding is boxed as `RoPE` over the `d_r` channels of one token, and the
queries compute the box once per head. The rotary table is a `dst.YarnRotary`. The
released code applies YaRN whenever its
maximum sequence length, 16384 by default, exceeds the original 4096, and it then
multiplies the softmax scale by `mu^2`. The ramp ends 10 and 23 are the correction range
the released `find_correction_range` computes for `beta_fast = 32`, `beta_slow = 1`,
`d_r = 64` and the base 10000.
'''
from __future__ import annotations

import advanced_axis_dynamics.data_structure.Operators as aops
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd
import deepseek.data_structure as dst
from websocket_transfer.auxiliary_information import OperatorRole

from notebooks.classic.deepseek_v3_wording import TEXT as text
from notebooks.classic.reference_links import (
    deepseek_lines, deepseek_role, released_config_lines, small_config_lines)
from notebooks.classic.shared_mechanisms import (
    ROTARY_BOX, broadcast_between_positions_and_channels, contract,
    read_back_from_every_position, residual_connection, rotate_channel_pairs,
    sigmoid_weighted_input)
from notebooks.display.explain_operators import (
    ExplanationOfOperator, OperatorExplanation, explain_named_arithmetic)
from notebooks.display.explain_reindexings import ReindexingExplanation
from notebooks.sota.DeepSeekV41Flash.construction_idioms import (
    boxed, hold, l1_norm_over, over, route)
from notebooks.sota.DeepSeekV41Flash.declared_axes import selection_count

R = cat.Reals()
MINUS_HALF = nm.Integer(-1) / nm.Integer(2)
HALF = nm.Integer(1) / nm.Integer(2)

x = cat.RawAxis.named('x', code_form='tokens')
w = fd.DynamicName('w', code_form='earlier_tokens').capture(
    cat.RawAxis(_size=x.local_size()))
m = cat.RawAxis.named('m', code_form='model_width')
h = cat.RawAxis.named('h', code_form='heads')
latent = cat.RawAxis.named('\\ell', code_form='key_value_latent')
dn = cat.RawAxis.named(fd.DynamicName(
    'd', fd.DynamicName('n'), code_form='unrotated_channels'))
dr = cat.RawAxis.named(fd.DynamicName(
    'd', fd.DynamicName('r'), code_form='rotated_channels'))
dv = cat.RawAxis.named(fd.DynamicName(
    'd', fd.DynamicName('v'), code_form='value_width'))
d = fd.DynamicName('d', code_form='query_key_width').capture(
    cat.RawAxis(_size=dn.local_size() + dr.local_size()))
t = fd.DynamicName('t', code_form='rotary_pairs').capture(
    cat.RawAxis(_size=dr.local_size() * HALF))
u = cat.RawAxis.named('u', code_form='dense_width')
n = cat.RawAxis.named('n', code_form='routed_experts')
f = cat.RawAxis.named('f', code_form='expert_width')
SHARED_COUNT = nm.FreeNumeric.named('S')
s = fd.DynamicName('s', code_form='shared_width').capture(
    cat.RawAxis(_size=SHARED_COUNT * f.local_size()))
v = cat.RawAxis.named(fd.DynamicName(
    'v', settings=fd.DynamicNameSettings(overline=True), code_form='vocabulary'))
selected_count = selection_count('k', 'experts_per_token')

DENSE_LAYER_COUNT = nm.FreeNumeric.named('D')
MIXTURE_LAYER_COUNT = nm.FreeNumeric.named('L')
ROTARY_BASE = nm.FreeNumeric.named('\\beta')
YARN_FACTOR = nm.FreeNumeric.named('\\kappa')
YARN_RAMP_START = nm.FreeNumeric.named('\\mathrm{lo}')
YARN_RAMP_END = nm.FreeNumeric.named('\\mathrm{hi}')
ATTENTION_FACTOR = nm.FreeNumeric.named('\\mu')
NORM_EPSILON = nm.FreeNumeric.named('\\epsilon')

TOKEN = cat.Natural(v.local_size())
STATE = cat.Array(R, (x, m))
LATENT = cat.Array(R, (x, latent))
QUERIES = cat.Array(R, (x, h, d))
KEYS_UNROTATED = cat.Array(R, (x, h, dn))
KEYS_ROTATED = cat.Array(R, (x, h, dr))
VALUES = cat.Array(R, (x, h, dv))

MODEL_COLOUR = '#FFFFFF'
EMBEDDING_COLOUR = '#FCE0E1'
LAYER_COLOUR = '#FFFFFF'
RESIDUAL_COLOUR = '#F1F4C1'
ATTENTION_COLOUR = '#FFFDFA'
QUERY_COLOUR = '#FFF4E2'
KEY_VALUE_COLOUR = '#EFE1FF'
ROTARY_COLOUR = '#BDDCFF'
CORE_COLOUR = '#FFE2BB'
MLP_COLOUR = '#EDFCFF'
MOE_COLOUR = '#F8FFED'
GATE_COLOUR = '#DFF7C3'
EXPERTS_COLOUR = '#B1E7FF'
OUTPUT_COLOUR = '#DBDFEF'

ATTENTION_BOX = 'At'
MLP_BOX = 'MLP'
MOE_BOX = 'MoE'
MASK_NAME = '\\mathrm{Mask}'
DIAGONAL_NAME = '\\mathrm{Diagonal}'
SCORE_SCALE_NAME = '|d|^{-1/2} \\mu^{2} x'
SIGMOID_NAME = '\\sigma'
HEAD_NAME = 'W^{\\mathrm{head}}'

ROTARY_FORMULA = (
    r'y[i_{x}, 2 i_{t}] + \mathrm{i}\, y[i_{x}, 2 i_{t} + 1] = '
    r"e^{\mathrm{i}\, i_{x}\, \theta'[i_{t}]}\, "
    r'\big(v[i_{x}, 2 i_{t}] + \mathrm{i}\, v[i_{x}, 2 i_{t} + 1]\big)')
CORE_FORMULA = (
    r'\mathrm{Attention}(Q, K, V) = '
    r'\mathrm{softmax}\left(\frac{\mu^{2}\, Q K^{\top}}{\sqrt{|d|}}\right) V,'
    r'\quad \mu = 0.1\, m_{c} \ln \kappa + 1')
GATE_FORMULA = (
    r'g[i_{n}] = \frac{\sigma(s[i_{n}])}{\sum_{j \in \mathrm{TopK}(s)} \sigma(s[j])}'
    r'\ \text{for}\ i_{n} \in \mathrm{TopK}(s),\quad s = W^{g} x')
MOE_FORMULA = (
    r'y = \sum_{i_{n} \in \mathrm{TopK}(s)} g[i_{n}]\, E_{i_{n}}(x) '
    r'+ \sum_{i \in S} E^{\mathrm{shared}}_{i}(x)')
SWIGLU_FORMULA = (
    r'E(x) = W_{2}\big(\mathrm{SiLU}(W_{1}\, x) \odot W_{3}\, x\big)')
EMBEDDING_FORMULA = r'y[i_{m}] = E[x, i_{m}]'
PAIRS_AS_COMPLEX_FORMULA = r'z[i_{t}] = v[2 i_{t}] + \mathrm{i}\, v[2 i_{t} + 1]'
DECOMPLEX_FORMULA = r'y[2 i_{t}] + \mathrm{i}\, y[2 i_{t} + 1] = z[i_{t}]'
TOP_K_FORMULA = (
    r'y[i_{n}] = s[i_{n}] \text{ where } s[i_{n}] \text{ is among the } |k| '
    r'\text{ largest of } s')
CONCATENATE_FORMULA = (
    r'y[i_{d}] = a[i_{d}] \text{ for } i_{d} < |d_{n}|,\quad '
    r'y[|d_{n}| + i_{d_{r}}] = b[i_{d_{r}}]')


def scale_by(factor: nm.Numeric, name: str) -> cat.Broadcasted:
    return ops.Arithmetic.template(nm.x * factor, name=name)


def rms_norm_over(width: cat.RawAxis) -> cat.BroadcastedCategory:
    return x >> ops.Normalize.template((width,), epsilon=NORM_EPSILON)


def swiglu(width: cat.RawAxis) -> cat.BroadcastedCategory:
    '''`W_2(SiLU(W_1 z) * W_3 z)` at every token, through an inner width of `width`.'''
    return (route((0, 0), (STATE,))
            @ ((x >> ops.Linear.template((m,), (width,), 'W_{3}'))
               * ((x >> ops.Linear.template((m,), (width,), 'W_{1}'))
                  @ sigmoid_weighted_input()))
            @ contract(((x, width), (x, width)), (x, width))
            @ (x >> ops.Linear.template((width,), (m,), 'W_{2}')))


# ==========================================================================
# Multi-head latent attention.
# ==========================================================================
def yarn_table() -> cat.Broadcasted[dst.Complex, cat.Axis, dst.YarnRotary]:
    return dst.YarnRotary.template(
        x, t, base=ROTARY_BASE, factor=YARN_FACTOR,
        ramp_start=YARN_RAMP_START, ramp_end=YARN_RAMP_END)


ROTARY_EMBEDDING = boxed(cat.Block.template(
    rotate_channel_pairs(yarn_table(), x, (), dr, t),
    title=text.ROTARY_TITLE, fill_color=ROTARY_COLOUR,
    description=text.ROTARY_DESCRIPTION, formula=ROTARY_FORMULA,
    references=(deepseek_lines(297, 375), deepseek_lines(378, 393),
                deepseek_lines(56), deepseek_lines(81, 85))), ROTARY_BOX)


def rotary_embedding(heads: tuple[cat.RawAxis, ...]) -> cat.Broadcasted:
    '''The `d_r` channels of every token turned by the YaRN table of its token, once
    per index of `heads`.'''
    return broadcast_between_positions_and_channels(ROTARY_EMBEDDING, heads)


def join_query_channels() -> cat.Broadcasted:
    return aops.ConcatenateAxes.template(((x, h, dn), (x, h, dr)), concatenated=d)


def query_generation() -> cat.Block:
    return cat.Block.template(
        route((0, 0), (STATE,))
        @ ((x >> ops.Linear.template((m,), (h, dn), 'W^{Q}'))
           * ((x >> ops.Linear.template((m,), (h, dr), 'W^{QR}'))
              @ rotary_embedding((h,))))
        @ join_query_channels(),
        title=text.QUERY_GENERATION_TITLE, fill_color=QUERY_COLOUR,
        description=text.QUERY_GENERATION_DESCRIPTION,
        references=(deepseek_lines(424, 425), deepseek_lines(461, 467),
                    small_config_lines(13)))


def key_value_generation() -> cat.Block:
    '''The keys of every head, as their two runs of channels joined, and the values.'''
    unrotated_and_values = (
        (x >> ops.Linear.template((m,), (latent,), 'W^{DKV}'))
        @ rms_norm_over(latent)
        @ route((0, 0), (LATENT,))
        @ ((x >> ops.Linear.template((latent,), (h, dn), 'W^{UK}'))
           * (x >> ops.Linear.template((latent,), (h, dv), 'W^{UV}'))))
    rotated = ((x >> ops.Linear.template((m,), (dr,), 'W^{KR}'))
               @ rotary_embedding(())
               @ contract(((x, dr),), (x, h, dr)))
    return cat.Block.template(
        route((0, 0), (STATE,))
        @ (unrotated_and_values * rotated)
        @ route((0, 2, 1), (KEYS_UNROTATED, VALUES, KEYS_ROTATED))
        @ (aops.ConcatenateAxes.template(((x, h, dn), (x, h, dr)), concatenated=d)
           * hold(VALUES)),
        title=text.KV_GENERATION_TITLE, fill_color=KEY_VALUE_COLOUR,
        description=text.KV_GENERATION_DESCRIPTION,
        references=(deepseek_lines(430, 432), deepseek_lines(468, 476)))


def scaled_dot_product_attention() -> cat.Block:
    queries = (x, h, d)
    keys = (x, w, h, d)
    values = (x, w, h, dv)
    scores = (h, x, w)
    return cat.Block.template(
        ((contract((queries, keys), scores)
          @ scale_by(d.local_size() ** MINUS_HALF * ATTENTION_FACTOR ** nm.Integer(2),
                     SCORE_SCALE_NAME)
          @ ops.SoftMax.template())
         * hold(cat.Array(R, values)))
        @ contract((scores, values), (x, h, dv)),
        title=text.CORE_TITLE, fill_color=CORE_COLOUR,
        description=text.CORE_DESCRIPTION, formula=CORE_FORMULA,
        references=(deepseek_lines(434, 437), deepseek_lines(479),
                    deepseek_lines(488, 492)))


def attention_from_queries(queries: cat.Block) -> cat.BroadcastedCategory:
    '''Multi-head latent attention of the hidden state, with the query of every head
    returned by `queries`, the keys and the values generated from the latent and read
    back from every query through the causal mask, and `W^{O}` after the core.'''
    return (route((0, 0), (STATE,))
            @ (queries * key_value_generation())
            @ (hold(QUERIES)
               * read_back_from_every_position(x, w, (h, d), MASK_NAME)
               * read_back_from_every_position(x, w, (h, dv), MASK_NAME))
            @ scaled_dot_product_attention()
            @ (x >> ops.Linear.template((h, dv), (m,), 'W^{O}')))


def multi_head_latent_attention() -> cat.Block:
    return cat.Block.template(
        attention_from_queries(query_generation()),
        title=text.ATTENTION_TITLE, fill_color=ATTENTION_COLOUR,
        description=text.ATTENTION_DESCRIPTION,
        references=(deepseek_lines(396, 497), deepseek_lines(789),
                    small_config_lines(8),
                    small_config_lines(13, 17)))


# ==========================================================================
# The feed-forward sublayers.
# ==========================================================================
def multi_layer_perceptron() -> cat.Block:
    return cat.Block.template(
        swiglu(u),
        title=text.MLP_TITLE, fill_color=MLP_COLOUR,
        description=text.MLP_DESCRIPTION, formula=SWIGLU_FORMULA,
        references=(deepseek_lines(500, 532), small_config_lines(4)))


def selection() -> cat.Broadcasted:
    return dst.TopK.template(
        k=selected_count, axis=n,
        name=fd.DynamicName('k/n', code_form='selected_experts'))


def gate(selected: cat.Broadcasted, chosen: dst.SparseAxis) -> cat.Block:
    return cat.Block.template(
        (x >> ops.Linear.template((m,), (n,), 'W^{g}'))
        @ ops.Arithmetic.template(nm.Sigmoid(nm.x), name=SIGMOID_NAME)
        @ selected
        @ l1_norm_over((x, chosen), 1),
        title=text.GATE_TITLE, fill_color=GATE_COLOUR,
        description=text.GATE_DESCRIPTION, formula=GATE_FORMULA,
        references=(deepseek_lines(566, 598),
                    released_config_lines(15),
                    small_config_lines(12)))


def swiglu_of_the_chosen_experts(chosen: dst.SparseAxis) -> cat.BroadcastedCategory:
    '''The SwiGLU of every expert a token chose, through the inner width `f`. Each
    projection holds the weights of all `n` experts, and the diagonal after `W_{2}`
    keeps the output of the expert each slot chose.'''
    first = (x >> ops.Linear.template((m,), (n, f), 'W_{1}')) @ sigmoid_weighted_input()
    third = x >> ops.Linear.template((m,), (n, f), 'W_{3}')
    down = (over((x, chosen), ops.Linear.template((f,), (n, m), 'W_{2}'))
            @ ops.View.template(
                reindexing=cat.Rearrangement((0, 1, 1, 2), (x, chosen, m)),
                name=DIAGONAL_NAME))
    return (route((0, 0), (STATE,)) @ (third * first)
            @ contract(((x, chosen, f), (x, chosen, f)), (x, chosen, f))
            @ down)


def routed_experts(chosen: dst.SparseAxis) -> cat.Block:
    return cat.Block.template(
        swiglu_of_the_chosen_experts(chosen),
        title=text.ROUTED_EXPERTS_TITLE, fill_color=EXPERTS_COLOUR,
        description=text.ROUTED_EXPERTS_DESCRIPTION, formula=SWIGLU_FORMULA,
        references=(deepseek_lines(601, 633), deepseek_lines(684, 689)))


def shared_experts() -> cat.Block:
    return cat.Block.template(
        swiglu(s),
        title=text.SHARED_EXPERTS_TITLE, fill_color=EXPERTS_COLOUR,
        description=text.SHARED_EXPERTS_DESCRIPTION, formula=SWIGLU_FORMULA,
        references=(deepseek_lines(667), deepseek_lines(690),
                    small_config_lines(10)))


def mixture_from_gate(gate_block: cat.Block, routed_block: cat.Block,
                      shared_block: cat.Block,
                      chosen: dst.SparseAxis) -> cat.BroadcastedCategory:
    '''The routed experts on the slots `chosen` weighted by the gates of `gate_block`
    and summed over the slots, added to the shared experts.'''
    return (route((0, 0, 0), (STATE,))
            @ (gate_block * routed_block * shared_block)
            @ (contract(((x, chosen), (x, chosen, m)), (x, m)) * hold(STATE))
            @ ops.AdditionOp.template())


def deepseek_mixture_of_experts() -> cat.Block:
    selected = selection()
    chosen = selected.cod()[0].shape()[-1]
    return cat.Block.template(
        mixture_from_gate(gate(selected, chosen), routed_experts(chosen),
                          shared_experts(), chosen),
        title=text.MOE_TITLE, fill_color=MOE_COLOUR,
        description=text.MOE_DESCRIPTION, formula=MOE_FORMULA,
        references=(deepseek_lines(636, 693),
                    small_config_lines(9, 11)))


# ==========================================================================
# The layers and the model.
# ==========================================================================
ATTENTION = boxed(multi_head_latent_attention(), ATTENTION_BOX)
MLP = boxed(multi_layer_perceptron(), MLP_BOX)
MOE = boxed(deepseek_mixture_of_experts(), MOE_BOX)


def residual_after_norm(sublayer: cat.BroadcastedCategory, line: int) -> cat.Block:
    return residual_connection(
        rms_norm_over(m) @ sublayer, STATE, text.RESIDUAL_TITLE,
        text.RESIDUAL_DESCRIPTION, RESIDUAL_COLOUR, (deepseek_lines(line),))


def dense_layers() -> cat.Block:
    return cat.Block.template(
        residual_after_norm(ATTENTION, 733) @ residual_after_norm(MLP, 734),
        title=text.DENSE_LAYER_TITLE, fill_color=LAYER_COLOUR,
        repetition=DENSE_LAYER_COUNT, description=text.DENSE_LAYER_DESCRIPTION,
        references=(deepseek_lines(706, 735), small_config_lines(7)))


def mixture_layers() -> cat.Block:
    return cat.Block.template(
        residual_after_norm(ATTENTION, 733) @ residual_after_norm(MOE, 734),
        title=text.MOE_LAYER_TITLE, fill_color=LAYER_COLOUR,
        repetition=MIXTURE_LAYER_COUNT,
        description=text.MOE_LAYER_DESCRIPTION,
        references=(deepseek_lines(706, 735), small_config_lines(6)))


def embedding() -> cat.Block:
    return cat.Block.template(
        x >> ops.Embedding.template(TOKEN, (m,)),
        title=text.EMBEDDING_TITLE, fill_color=EMBEDDING_COLOUR,
        description=text.EMBEDDING_DESCRIPTION,
        references=(deepseek_lines(89, 129), deepseek_lines(764),
                    small_config_lines(2)))


def output_projection() -> cat.Block:
    return cat.Block.template(
        rms_norm_over(m) @ (x >> ops.Linear.template((m,), (v,), HEAD_NAME)),
        title=text.OUTPUT_TITLE, fill_color=OUTPUT_COLOUR,
        description=text.OUTPUT_DESCRIPTION,
        references=(deepseek_lines(768, 769), deepseek_lines(792, 793)))


def deepseek_v3() -> cat.Block:
    return cat.Block.template(
        embedding() @ dense_layers() @ mixture_layers() @ output_projection(),
        title=text.MODEL_TITLE, fill_color=MODEL_COLOUR,
        description=text.MODEL_DESCRIPTION,
        references=(deepseek_lines(738, 798), small_config_lines(1, 19)))


# ==========================================================================
# What the inspection boxes show over the operators and the views.
# ==========================================================================
SILU_NAME = sigmoid_weighted_input().operator.name.to_bodies()

ARITHMETIC_ROLES: dict[str, str] = {
    SCORE_SCALE_NAME: text.SCORE_SCALE_ROLE,
    SILU_NAME: text.SILU_ROLE,
    SIGMOID_NAME: text.SIGMOID_ROLE,
}

ARITHMETIC_REFERENCES: dict[str, tuple[cat.CodeReference, ...]] = {
    SCORE_SCALE_NAME: (deepseek_lines(434, 437),),
    SILU_NAME: (deepseek_lines(623, 633),),
    SIGMOID_NAME: (deepseek_lines(577, 580),),
}

OPERATOR_EXPLANATIONS: dict[type[cat.Operator], ExplanationOfOperator] = {
    ops.Embedding: OperatorExplanation(
        title=r'\text{Embedding}', formula=EMBEDDING_FORMULA,
        description=text.EMBEDDING_OPERATOR_DESCRIPTION,
        references=(deepseek_lines(107, 129),)),
    dst.PairsAsComplex: OperatorExplanation(
        title=r'\text{Pairs as Complex}', formula=PAIRS_AS_COMPLEX_FORMULA,
        description=text.PAIRS_AS_COMPLEX_DESCRIPTION,
        references=(deepseek_lines(390),)),
    dst.Decomplex: OperatorExplanation(
        title=r'\text{Decomplex}', formula=DECOMPLEX_FORMULA,
        description=text.DECOMPLEX_DESCRIPTION,
        references=(deepseek_lines(392),)),
    dst.TopK: OperatorExplanation(
        title=r'\text{TopK}', formula=TOP_K_FORMULA,
        description=text.TOP_K_DESCRIPTION,
        references=(deepseek_lines(593),)),
    aops.ConcatenateAxes: OperatorExplanation(
        title=r'\text{Concatenate}', formula=CONCATENATE_FORMULA,
        description=text.CONCATENATE_DESCRIPTION,
        references=(deepseek_lines(472), deepseek_lines(476))),
    ops.Arithmetic: explain_named_arithmetic(ARITHMETIC_ROLES, ARITHMETIC_REFERENCES),
}


def role_key(name: str) -> str:
    '''The text a table of roles is keyed by, the bodies of the name, which joins a
    subscript with no underscore, so `W_{1}` is keyed `W{1}`.'''
    return fd.DynamicName.from_str(name).to_bodies()


OPERATOR_ROLES: dict[str, OperatorRole] = {
    'W^{Q}': deepseek_role(text.QUERY_ROLE, 425),
    'W^{QR}': deepseek_role(text.ROTARY_QUERY_ROLE, 425),
    'W^{DKV}': deepseek_role(text.LATENT_ROLE, 430),
    'W^{KR}': deepseek_role(text.ROTARY_KEY_ROLE, 430),
    'W^{UK}': deepseek_role(text.KEY_ROLE, 432),
    'W^{UV}': deepseek_role(text.VALUE_ROLE, 432),
    'W^{O}': deepseek_role(text.OUTPUT_ROLE, 433),
    'W^{g}': deepseek_role(text.GATE_ROLE, 563),
    role_key('W_{1}'): deepseek_role(text.FIRST_PROJECTION_ROLE, 518),
    role_key('W_{3}'): deepseek_role(text.THIRD_PROJECTION_ROLE, 520),
    role_key('W_{2}'): deepseek_role(text.SECOND_PROJECTION_ROLE, 519),
    HEAD_NAME: deepseek_role(text.HEAD_ROLE, 769),
    'RMSNorm': deepseek_role(text.RMS_NORM_ROLE, 270, 294),
    '\\mathrm{YaRN}': deepseek_role(text.YARN_TABLE_ROLE, 366, 374),
}

OPERATOR_REFERENCES: dict[type[cat.Operator], tuple[cat.CodeReference, ...]] = {
    ops.SoftMax: (deepseek_lines(490),),
    ops.L1Norm: (deepseek_lines(595, 596),),
}

REINDEXING_EXPLANATIONS: dict[str, ReindexingExplanation] = {
    MASK_NAME: ReindexingExplanation(
        description=text.MASK_VIEW_DESCRIPTION,
        references=(deepseek_lines(789), deepseek_lines(488, 489))),
    DIAGONAL_NAME: ReindexingExplanation(
        description=text.DIAGONAL_VIEW_DESCRIPTION,
        references=(deepseek_lines(688, 689),)),
}
