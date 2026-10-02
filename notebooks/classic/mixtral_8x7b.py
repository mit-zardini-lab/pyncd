# Claude Opus 5.5 (1M context), reasoning effort 40.
'''Mixtral-8x7B, as a morphism in Br.

The model reads the token identifiers of a prompt and returns the probability of every
token of the vocabulary at every position. It is written from the reference
implementation `mistralai/mistral-inference` and the configuration of the released
weights, both pinned in `reference_links.py`.

The token identifiers are embedded and pass through `N` decoder layers. Each layer is a
residual connection around an RMSNorm and the attention, followed by a residual
connection around an RMSNorm and the sparse mixture of experts. A final RMSNorm, the
output projection and a softmax over the vocabulary finish the model.

The attention is grouped-query attention. The query projection produces the 32 query
heads as `h'` groups of `g` heads, head `|g| i_{h'} + i_g` of the released code being
head `i_g` of group `i_{h'}`, which is the pairing made by `repeat_interleave` there. The
released code projects onto 32 heads and reshapes them. The expression writes the
grouping into the projection, per the ruling on groupings in
`obsidian/06-practice/Representing Models.md`, so no view groups the heads. The keys
and the values carry `h'` alone, so every head of a group reads the same key-value
head. The queries and the keys are turned by the rotary embedding,
boxed as `RoPE` over one vector per token and computed once per head, before the keys
are read back from every query through the view named Mask, whose
slot `i_w` holds the token `i_w` before the query. The slot axis is as long as the
sequence, because the released configuration sets no sliding window, and the view marks
it as `w|x`.

The mixture of experts scores the eight experts with `W^{g}`, keeps two with a
`ds.TopK` whose result rides the sparse axis `k/n`, and takes the softmax over the two.
Each expert projection is one `ops.Linear` producing the expert axis, and composition
aligns that axis with `k/n`, so a projection reads the weights of the chosen experts.
The down projection produces the expert axis a second time and the view named Diagonal
keeps the output of the chosen expert, per the ruling on the down projection of a
mixture in `obsidian/06-practice/Representing Models.md`. Every named view is named by
a whole word, per the ruling of 2026-09-26 recorded in the same note.
'''
from __future__ import annotations

import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd
import deepseek.data_structure as dst
from websocket_transfer.auxiliary_information import OperatorRole

from notebooks.classic.mixtral_8x7b_wording import TEXT as text
from notebooks.classic.reference_links import (
    mistral_lines, mixtral_config_lines, mixtral_role)
from notebooks.classic.shared_mechanisms import (
    ROTARY_BOX, broadcast_between_positions_and_channels, contract,
    read_back_from_every_position, residual_connection, rotate_channel_pairs,
    sigmoid_weighted_input)
from notebooks.display.explain_operators import (
    ExplanationOfOperator, OperatorExplanation, explain_named_arithmetic)
from notebooks.display.explain_reindexings import ReindexingExplanation
from notebooks.sota.DeepSeekV41Flash.construction_idioms import (
    boxed, hold, over, route)
from notebooks.sota.DeepSeekV41Flash.declared_axes import selection_count

R = cat.Reals()
HALF = nm.Integer(1) / nm.Integer(2)

x = cat.RawAxis.named('x', code_form='tokens')
w = fd.DynamicName('w', code_form='earlier_tokens').capture(
    cat.RawAxis(_size=x.local_size()))
m = cat.RawAxis.named('m', code_form='model_width')
kv_heads = cat.RawAxis.named("h'", code_form='key_value_heads')
g = cat.RawAxis.named('g', code_form='queries_per_key_value_head')
d = cat.RawAxis.named('d', code_form='head_width')
t = fd.DynamicName('t', code_form='rotary_pairs').capture(
    cat.RawAxis(_size=d.local_size() * HALF))
f = cat.RawAxis.named('f', code_form='expert_width')
n = cat.RawAxis.named('n', code_form='experts')
v = cat.RawAxis.named(fd.DynamicName(
    'v', settings=fd.DynamicNameSettings(overline=True), code_form='vocabulary'))
selected_count = selection_count('k', 'experts_per_token')

LAYER_COUNT = nm.FreeNumeric.named('N')
ROTARY_BASE = nm.FreeNumeric.named('\\beta')
NORM_EPSILON = nm.FreeNumeric.named('\\epsilon')

TOKEN = cat.Natural(v.local_size())
STATE = cat.Array(R, (x, m))

MODEL_COLOUR = '#FFFFFF'
EMBEDDING_COLOUR = '#FCE0E1'
LAYER_COLOUR = '#FFFFFF'
RESIDUAL_COLOUR = '#F1F4C1'
ATTENTION_COLOUR = '#FFE2BB'
CORE_COLOUR = '#C5BEDF'
ROTARY_COLOUR = '#BDDCFF'
MIXTURE_COLOUR = '#DFF7C3'
ROUTER_COLOUR = '#DFF7C3'
EXPERTS_COLOUR = '#B1E7FF'
OUTPUT_COLOUR = '#DBDFEF'

MASK_NAME = '\\mathrm{Mask}'
DIAGONAL_NAME = '\\mathrm{Diagonal}'
SCORE_SCALE_NAME = 'x / \\sqrt{|d|}'
OUTPUT_PROJECTION_NAME = 'W^{\\mathrm{out}}'
FIRST_EXPERT_NAME = 'W_{1}'
SECOND_EXPERT_NAME = 'W_{2}'
THIRD_EXPERT_NAME = 'W_{3}'

ROTARY_FORMULA = (
    r'y[i_{x}, 2 i_{t}] + \mathrm{i}\, y[i_{x}, 2 i_{t} + 1] = '
    r'e^{\mathrm{i}\, i_{x}\, \beta^{-i_{t}/|t|}}\, '
    r'\big(v[i_{x}, 2 i_{t}] + \mathrm{i}\, v[i_{x}, 2 i_{t} + 1]\big)')
CORE_FORMULA = (
    r'\mathrm{Attention}(Q, K, V) = '
    r'\mathrm{softmax}\left(\frac{Q K^{\top}}{\sqrt{|d|}}\right) V')
MIXTURE_FORMULA = (
    r'y = \sum_{i \in \mathrm{TopK}(W^{g} x)} '
    r'\mathrm{softmax}(\mathrm{TopK}(W^{g} x))[i]\, E_{i}(x)')
EXPERT_FORMULA = (
    r'E_{i}(x) = W_{2, i}\big(\mathrm{SiLU}(W_{1, i}\, x) \odot W_{3, i}\, x\big)')
EMBEDDING_FORMULA = r'y[i_{m}] = E[x, i_{m}]'
PAIRS_AS_COMPLEX_FORMULA = r'z[i_{t}] = v[2 i_{t}] + \mathrm{i}\, v[2 i_{t} + 1]'
DECOMPLEX_FORMULA = r'y[2 i_{t}] + \mathrm{i}\, y[2 i_{t} + 1] = z[i_{t}]'
TOP_K_FORMULA = (
    r'y[i_{n}] = s[i_{n}] \text{ where } s[i_{n}] \text{ is among the } |k| '
    r'\text{ largest of } s')


def scale_by(factor: nm.Numeric, name: str) -> cat.Broadcasted:
    return ops.Arithmetic.template(nm.x * factor, name=name)


def rms_norm() -> cat.BroadcastedCategory:
    return x >> ops.Normalize.template((m,), epsilon=NORM_EPSILON)


# ==========================================================================
# Attention.
# ==========================================================================
ROTARY_EMBEDDING = boxed(cat.Block.template(
    rotate_channel_pairs(dst.Rotary.template(x, t, base=ROTARY_BASE), x, (), d, t),
    title=text.ROTARY_TITLE, fill_color=ROTARY_COLOUR,
    description=text.ROTARY_DESCRIPTION, formula=ROTARY_FORMULA,
    references=(mistral_lines('rope.py', 6, 10), mistral_lines('rope.py', 13, 23),
                mistral_lines('transformer.py', 115, 116),
                mixtral_config_lines(21))), ROTARY_BOX)


def rotary_embedding(head_axes: tuple[cat.RawAxis, ...]) -> cat.Broadcasted:
    '''The vector of every head of every token turned by the table of its token.'''
    return broadcast_between_positions_and_channels(ROTARY_EMBEDDING, head_axes)


def read_earlier_tokens() -> cat.Broadcasted:
    return read_back_from_every_position(x, w, (kv_heads, d), MASK_NAME)


def scaled_dot_product_attention() -> cat.Block:
    queries = (x, kv_heads, g, d)
    keys = (x, w, kv_heads, d)
    scores = (kv_heads, g, x, w)
    return cat.Block.template(
        ((contract((queries, keys), scores)
          @ scale_by(nm.Integer(1) / nm.SquareRoot(d.local_size()), SCORE_SCALE_NAME)
          @ ops.SoftMax.template())
         * hold(cat.Array(R, keys)))
        @ contract((scores, keys), queries),
        title=text.CORE_TITLE, fill_color=CORE_COLOUR,
        description=text.CORE_DESCRIPTION, formula=CORE_FORMULA,
        references=(mistral_lines('transformer_layers.py', 83, 88),))


def attention() -> cat.Block:
    query = ((x >> ops.Linear.template((m,), (kv_heads, g, d), 'W^{Q}'))
             @ rotary_embedding((kv_heads, g)))
    key = ((x >> ops.Linear.template((m,), (kv_heads, d), 'W^{K}'))
           @ rotary_embedding((kv_heads,)) @ read_earlier_tokens())
    value = ((x >> ops.Linear.template((m,), (kv_heads, d), 'W^{V}'))
             @ read_earlier_tokens())
    return cat.Block.template(
        route((0, 0, 0), (STATE,))
        @ (query * key * value)
        @ scaled_dot_product_attention()
        @ (x >> ops.Linear.template((kv_heads, g, d), (m,), 'W^{O}')),
        title=text.ATTENTION_TITLE, fill_color=ATTENTION_COLOUR,
        description=text.ATTENTION_DESCRIPTION,
        references=(mistral_lines('transformer_layers.py', 31, 93),
                    mixtral_config_lines(14), mixtral_config_lines(17),
                    mixtral_config_lines(23)))


# ==========================================================================
# The sparse mixture of experts.
# ==========================================================================
def selection() -> cat.Broadcasted:
    '''The two highest of the eight expert scores, on a sparse axis over the experts.'''
    return dst.TopK.template(
        k=selected_count, axis=n,
        name=fd.DynamicName('k/n', code_form='selected_experts'))


def router(selected: cat.Broadcasted) -> cat.Block:
    return cat.Block.template(
        (x >> ops.Linear.template((m,), (n,), 'W^{g}'))
        @ selected @ ops.SoftMax.template(),
        title=text.ROUTER_TITLE, fill_color=ROUTER_COLOUR,
        description=text.ROUTER_DESCRIPTION,
        references=(mistral_lines('moe.py', 25, 27),))


def experts(chosen: dst.SparseAxis) -> cat.Block:
    '''The SwiGLU of every chosen expert, with the diagonal that keeps the output of
    the expert each slot chose.'''
    first = ((x >> ops.Linear.template((m,), (n, f), FIRST_EXPERT_NAME))
             @ sigmoid_weighted_input())
    third = x >> ops.Linear.template((m,), (n, f), THIRD_EXPERT_NAME)
    down = (over((x, chosen), ops.Linear.template((f,), (n, m), SECOND_EXPERT_NAME))
            @ ops.View.template(
                reindexing=cat.Rearrangement((0, 1, 1, 2), (x, chosen, m)),
                name=DIAGONAL_NAME))
    return cat.Block.template(
        route((0, 0), (STATE,)) @ (third * first)
        @ contract(((x, chosen, f), (x, chosen, f)), (x, chosen, f))
        @ down,
        title=text.EXPERTS_TITLE, fill_color=EXPERTS_COLOUR,
        description=text.EXPERTS_DESCRIPTION, formula=EXPERT_FORMULA,
        references=(mistral_lines('transformer_layers.py', 96, 106),
                    mixtral_config_lines(11)))


def sparse_mixture_of_experts() -> cat.Block:
    selected = selection()
    chosen = selected.cod()[0].shape()[-1]
    return cat.Block.template(
        route((0, 0), (STATE,))
        @ (router(selected) * experts(chosen))
        @ contract(((x, chosen), (x, chosen, m)), (x, m)),
        title=text.MIXTURE_TITLE, fill_color=MIXTURE_COLOUR,
        description=text.MIXTURE_DESCRIPTION, formula=MIXTURE_FORMULA,
        references=(mistral_lines('moe.py', 22, 32), mixtral_config_lines(15),
                    mixtral_config_lines(18)))


# ==========================================================================
# The layers and the model.
# ==========================================================================
def decoder_layer() -> cat.Block:
    return cat.Block.template(
        residual_connection(
            rms_norm() @ attention(), STATE, text.ATTENTION_RESIDUAL_TITLE,
            text.RESIDUAL_DESCRIPTION, RESIDUAL_COLOUR,
            (mistral_lines('transformer_layers.py', 165, 166),))
        @ residual_connection(
            rms_norm() @ sparse_mixture_of_experts(), STATE,
            text.MIXTURE_RESIDUAL_TITLE, text.RESIDUAL_DESCRIPTION, RESIDUAL_COLOUR,
            (mistral_lines('transformer_layers.py', 167, 168),)),
        title=text.LAYER_TITLE, fill_color=LAYER_COLOUR, repetition=LAYER_COUNT,
        description=text.LAYER_DESCRIPTION,
        references=(mistral_lines('transformer_layers.py', 123, 169),
                    mixtral_config_lines(16)))


def input_embedding() -> cat.Block:
    return cat.Block.template(
        x >> ops.Embedding.template(TOKEN, (m,)),
        title=text.EMBEDDING_TITLE, fill_color=EMBEDDING_COLOUR,
        description=text.EMBEDDING_DESCRIPTION,
        references=(mistral_lines('transformer.py', 57), mixtral_config_lines(28)))


def output_projection() -> cat.Block:
    return cat.Block.template(
        rms_norm()
        @ (x >> ops.Linear.template((m,), (v,), OUTPUT_PROJECTION_NAME))
        @ ops.SoftMax.template(),
        title=text.OUTPUT_TITLE, fill_color=OUTPUT_COLOUR,
        description=text.OUTPUT_DESCRIPTION,
        references=(mistral_lines('transformer.py', 78, 79),
                    mistral_lines('transformer.py', 233, 240)))


def mixtral() -> cat.Block:
    return cat.Block.template(
        input_embedding() @ decoder_layer() @ output_projection(),
        title=text.MODEL_TITLE, fill_color=MODEL_COLOUR,
        description=text.MODEL_DESCRIPTION,
        references=(mixtral_config_lines(1, 29),))


# ==========================================================================
# What the inspection boxes show over the operators and the views.
# ==========================================================================
SILU_NAME = sigmoid_weighted_input().operator.name.to_bodies()

ARITHMETIC_ROLES: dict[str, str] = {
    SCORE_SCALE_NAME: text.SCORE_SCALE_ROLE,
    SILU_NAME: text.SILU_ROLE,
}

ARITHMETIC_REFERENCES: dict[str, tuple[cat.CodeReference, ...]] = {
    SCORE_SCALE_NAME: (mistral_lines('transformer_layers.py', 48),),
    SILU_NAME: (mistral_lines('transformer_layers.py', 105, 106),),
}

OPERATOR_EXPLANATIONS: dict[type[cat.Operator], ExplanationOfOperator] = {
    ops.Embedding: OperatorExplanation(
        title=r'\text{Embedding}', formula=EMBEDDING_FORMULA,
        description=text.EMBEDDING_OPERATOR_DESCRIPTION,
        references=(mistral_lines('transformer.py', 57),)),
    dst.PairsAsComplex: OperatorExplanation(
        title=r'\text{Pairs as Complex}', formula=PAIRS_AS_COMPLEX_FORMULA,
        description=text.PAIRS_AS_COMPLEX_DESCRIPTION,
        references=(mistral_lines('rope.py', 18, 19),)),
    dst.Decomplex: OperatorExplanation(
        title=r'\text{Decomplex}', formula=DECOMPLEX_FORMULA,
        description=text.DECOMPLEX_DESCRIPTION,
        references=(mistral_lines('rope.py', 21, 22),)),
    dst.TopK: OperatorExplanation(
        title=r'\text{TopK}', formula=TOP_K_FORMULA,
        description=text.TOP_K_DESCRIPTION,
        references=(mistral_lines('moe.py', 26),)),
    ops.Arithmetic: explain_named_arithmetic(ARITHMETIC_ROLES, ARITHMETIC_REFERENCES),
}

def table_key(name: str) -> str:
    '''The text a table keys a named operator by, which is the text of the name the
    operator parses from `name`. A subscript is written without its underscore, so
    `W_{1}` is keyed as `W{1}`.'''
    return fd.DynamicName.from_str(name).to_bodies()


OPERATOR_ROLES: dict[str, OperatorRole] = {
    'W^{Q}': mixtral_role(text.QUERY_ROLE, 'transformer_layers.py', 51),
    'W^{K}': mixtral_role(text.KEY_ROLE, 'transformer_layers.py', 52),
    'W^{V}': mixtral_role(text.VALUE_ROLE, 'transformer_layers.py', 53),
    'W^{O}': mixtral_role(text.OUTPUT_ROLE, 'transformer_layers.py', 54),
    'W^{g}': mixtral_role(text.GATE_ROLE, 'transformer_layers.py', 152),
    table_key(FIRST_EXPERT_NAME): mixtral_role(
        text.FIRST_EXPERT_ROLE, 'transformer_layers.py', 101),
    table_key(THIRD_EXPERT_NAME): mixtral_role(
        text.THIRD_EXPERT_ROLE, 'transformer_layers.py', 103),
    table_key(SECOND_EXPERT_NAME): mixtral_role(
        text.SECOND_EXPERT_ROLE, 'transformer_layers.py', 102),
    OUTPUT_PROJECTION_NAME: mixtral_role(text.HEAD_ROLE, 'transformer.py', 79),
    'RMSNorm': OperatorRole(
        role=text.RMS_NORM_ROLE,
        references=(mistral_lines('transformer_layers.py', 109, 120),
                    mixtral_config_lines(20))),
    '\\mathrm{RoPE}': OperatorRole(
        role=text.ROTARY_TABLE_ROLE,
        references=(mistral_lines('rope.py', 6, 10), mixtral_config_lines(21))),
}

OPERATOR_REFERENCES: dict[type[cat.Operator], tuple[cat.CodeReference, ...]] = {
    ops.SoftMax: (mistral_lines('moe.py', 27),
                  mistral_lines('transformer.py', 239, 240)),
}

REINDEXING_EXPLANATIONS: dict[str, ReindexingExplanation] = {
    MASK_NAME: ReindexingExplanation(
        description=text.MASK_VIEW_DESCRIPTION,
        references=(mixtral_config_lines(12), mixtral_config_lines(23))),
    DIAGONAL_NAME: ReindexingExplanation(
        description=text.DIAGONAL_VIEW_DESCRIPTION,
        references=(mistral_lines('moe.py', 28, 31),)),
}
