# Claude Opus 5.5 (1M context), reasoning effort 40.
'''The four attention mechanisms of the tutorial pages, each as a morphism in Br.

`scaled_dot_product_attention` is dot-product attention with its scores multiplied by
`|d|^{-1/2}`. It reads queries over `q`, and keys and values over `x`, and it has no
weights.

`attention_with_weights_and_residual` reads one state over the tokens `x` and the width
`m`. Three linear maps project the state into a query, a key and a value of `d` channels,
the scaled dot-product attention combines them, `W^O` projects the result back to the
width `m`, and a residual connection adds it to the state. `multi_head_attention` gives
the projections a head axis `h`, which every operation of the attention is broadcast
over, and `W^O` reads the heads and the channels together. `grouped_query_attention`
writes the query heads as `h'` groups of `g` heads, with one key head and one value head
for each group, as `notebooks/classic/mixtral_8x7b.py` writes them.

The three models that read a state are causal. The keys and the values are read back from
every token through the view named `\\mathrm{Mask}`, built by
`shared_mechanisms.read_back_from_every_position`, whose slot `i_w` of token `i_x` holds
token `i_x - i_w`. A slot before the first token names a negative position and holds the
universal unit, which the softmax and the weighted sum skip, and the view marks the slot
axis as `w|x`. The models are built with the mask after the key and the value
projections, and a notebook displays each of them in the CausalSlide form that
`advanced_axis_dynamics.algebra.slide_causal_reads_backwards` returns.

Every size is a free symbol, because the pages state the mechanisms and no released
model. `obsidian/06-practice/Representing Models.md` states the rules these follow.
'''
from __future__ import annotations

import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd
from websocket_transfer.auxiliary_information import OperatorRole

from notebooks.classic.reference_links import (
    mistral_lines, transformer_role, transformer_section)
from notebooks.classic.shared_mechanisms import (
    contract, read_back_from_every_position, residual_connection)
from notebooks.display.explain_operators import (
    ExplanationOfOperator, explain_named_arithmetic)
from notebooks.display.explain_reindexings import ReindexingExplanation
from notebooks.sota.DeepSeekV41Flash.construction_idioms import hold, route
from notebooks.website.tutorial.tutorial_wording import TEXT as text

R = cat.Reals()
MINUS_HALF = nm.Integer(-1) / nm.Integer(2)

queries = cat.RawAxis.named('q', code_form='queries')
keys = cat.RawAxis.named('x', code_form='keys')
key_width = cat.RawAxis.named('d', code_form='key_width')
value_width = cat.RawAxis.named('v', code_form='value_width')

tokens = cat.RawAxis.named('x', code_form='tokens')
earlier_tokens = fd.DynamicName('w', code_form='earlier_tokens').capture(
    cat.RawAxis(_size=tokens.local_size()))
model_width = cat.RawAxis.named('m', code_form='model_width')
head_width = cat.RawAxis.named('d', code_form='head_width')
heads = cat.RawAxis.named('h', code_form='heads')
key_value_heads = cat.RawAxis.named("h'", code_form='key_value_heads')
heads_per_group = cat.RawAxis.named('g', code_form='queries_per_key_value_head')

STATE = cat.Array(R, (tokens, model_width))

CORE_COLOUR = '#C5BEDF'
ATTENTION_COLOUR = '#FFE2BB'
RESIDUAL_COLOUR = '#F1F4C1'

SCORE_SCALE_NAME = '|d|^{-1/2} x'
MASK_VIEW_NAME = '\\mathrm{Mask}'

CORE_FORMULA = (
    r'\mathrm{Attention}(Q, K, V) = '
    r'\mathrm{softmax}\left(\frac{Q K^{\top}}{\sqrt{|d|}}\right) V')
MULTI_HEAD_FORMULA = (
    r'\mathrm{MultiHead}(z) = \sum_{i_{h} \in h} '
    r'\mathrm{Attention}(z W^{Q}[i_{h}], z W^{K}[i_{h}], z W^{V}[i_{h}])\, '
    r'W^{O}[i_{h}]')
GROUPED_QUERY_FORMULA = (
    r"\mathrm{GQA}(z) = \sum_{i_{h'} \in h'} \sum_{i_{g} \in g} "
    r"\mathrm{Attention}(z W^{Q}[i_{h'}, i_{g}], z W^{K}[i_{h'}], "
    r"z W^{V}[i_{h'}])\, W^{O}[i_{h'}, i_{g}]")

GROUPED_QUERY_PAPER_URL = 'https://arxiv.org/pdf/2305.13245v3'


def grouped_query_section(section: str) -> cat.CodeReference:
    '''A section of *GQA: Training Generalized Multi-Query Transformer Models from
    Multi-Head Checkpoints*, version 3, read on 2026-09-27.'''
    return cat.CodeReference(
        label=f'GQA: Training Generalized Multi-Query Transformer Models, '
              f'Section {section}',
        url=GROUPED_QUERY_PAPER_URL)


def scale_scores(scored_width: cat.RawAxis) -> cat.Broadcasted:
    '''Every score multiplied by `|d|^{-1/2}`, for the width `d` the dot product sums
    over, named with the power rather than with a root, because a root in the name of an
    operator does not draw.'''
    return ops.Arithmetic.template(
        nm.x * scored_width.local_size() ** MINUS_HALF, name=SCORE_SCALE_NAME)


def scaled_dot_product[A: cat.Axis](
    query_shape: tuple[A, ...], key_shape: tuple[A, ...], value_shape: tuple[A, ...],
    score_shape: tuple[A, ...], result_shape: tuple[A, ...], scored_width: A,
) -> cat.Block:
    '''softmax(Q K^T |d|^{-1/2}) V over the shapes given, where `scored_width` is the
    axis `d` the dot product sums over. The scores end with the axis of the keys, which
    the softmax normalises over, and every contraction is written over the declared
    axes.'''
    return cat.Block.template(
        ((contract((query_shape, key_shape), score_shape)
          @ scale_scores(scored_width)
          @ ops.SoftMax.template())
         * hold(cat.Array(R, value_shape)))
        @ contract((score_shape, value_shape), result_shape),
        title=text.CORE_TITLE, fill_color=CORE_COLOUR,
        description=text.CORE_DESCRIPTION, formula=CORE_FORMULA,
        references=(transformer_section('3.2.1'),))


def scaled_dot_product_attention() -> cat.Block:
    '''Queries `[q, d]`, keys `[x, d]` and values `[x, v]` into one vector `[q, v]` for
    every query.'''
    return scaled_dot_product(
        (queries, key_width), (keys, key_width), (keys, value_width),
        (queries, keys), (queries, value_width), key_width)


def project(name: str, produced: tuple[cat.RawAxis, ...]) -> cat.BroadcastedCategory:
    '''The state of every token projected by the weight `name` onto `produced`.'''
    return tokens >> ops.Linear.template((model_width,), produced, name)


def read_earlier_tokens(carried: tuple[cat.RawAxis, ...]) -> cat.Broadcasted:
    '''An array `[x, *carried]` read back from every token, slot `i_w` holding the token
    `i_w` before it, which returns `[x, w|x, *carried]`.'''
    return read_back_from_every_position(
        tokens, earlier_tokens, carried, MASK_VIEW_NAME)


def project_and_mask(name: str, produced: tuple[cat.RawAxis, ...],
                     ) -> cat.BroadcastedCategory:
    '''The projection `name`, read back from every token.'''
    return project(name, produced) @ read_earlier_tokens(produced)


def attention_block(
    per_query: tuple[cat.RawAxis, ...], per_key: tuple[cat.RawAxis, ...],
    score_prefix: tuple[cat.RawAxis, ...], title: str, description: str,
    formula: str | None, references: tuple[cat.CodeReference, ...],
) -> cat.Block:
    '''The state copied into the query, key and value projections, the keys and values
    read back from every token, the scaled dot-product attention, and `W^O`. Each query
    holds the axes `per_query` after the token, each key and value the axes `per_key`,
    and `score_prefix` names the head axes the scores carry ahead of the tokens.'''
    query_shape = (tokens, *per_query)
    key_shape = (tokens, earlier_tokens, *per_key)
    score_shape = (*score_prefix, tokens, earlier_tokens)
    return cat.Block.template(
        route((0, 0, 0), (STATE,))
        @ (project('W^{Q}', per_query)
           * project_and_mask('W^{K}', per_key)
           * project_and_mask('W^{V}', per_key))
        @ scaled_dot_product(
            query_shape, key_shape, key_shape, score_shape, query_shape, head_width)
        @ (tokens >> ops.Linear.template(per_query, (model_width,), 'W^{O}')),
        title=title, fill_color=ATTENTION_COLOUR, description=description,
        formula=formula, references=references)


def causal_self_attention() -> cat.Block:
    return attention_block(
        (head_width,), (head_width,), (), text.SELF_ATTENTION_TITLE,
        text.SELF_ATTENTION_DESCRIPTION, None,
        (transformer_section('3.2.2'), transformer_section('3.2.3')))


def attention_with_weights_and_residual() -> cat.Block:
    return residual_connection(
        causal_self_attention(), STATE, text.RESIDUAL_TITLE, text.RESIDUAL_DESCRIPTION,
        RESIDUAL_COLOUR, (transformer_section('3.1'),))


def multi_head_attention() -> cat.Block:
    return attention_block(
        (heads, head_width), (heads, head_width), (heads,), text.MULTI_HEAD_TITLE,
        text.MULTI_HEAD_DESCRIPTION, MULTI_HEAD_FORMULA,
        (transformer_section('3.2.2'), transformer_section('3.2.3')))


def grouped_query_attention() -> cat.Block:
    return attention_block(
        (key_value_heads, heads_per_group, head_width), (key_value_heads, head_width),
        (key_value_heads, heads_per_group), text.GROUPED_QUERY_TITLE,
        text.GROUPED_QUERY_DESCRIPTION, GROUPED_QUERY_FORMULA,
        (grouped_query_section('2.2'),
         mistral_lines('transformer_layers.py', 16, 19),
         mistral_lines('transformer_layers.py', 83, 88)))


# ==========================================================================
# What the inspection boxes show over the operators and the views.
# ==========================================================================
OPERATOR_EXPLANATIONS: dict[type[cat.Operator], ExplanationOfOperator] = {
    ops.Arithmetic: explain_named_arithmetic(
        {SCORE_SCALE_NAME: text.SCORE_SCALE_ROLE},
        {SCORE_SCALE_NAME: (transformer_section('3.2.1'),)}),
}

REINDEXING_EXPLANATIONS: dict[str, ReindexingExplanation] = {
    MASK_VIEW_NAME: ReindexingExplanation(
        description=text.MASK_VIEW_DESCRIPTION,
        references=(transformer_section('3.2.3'),)),
}

OPERATOR_REFERENCES: dict[type[cat.Operator], tuple[cat.CodeReference, ...]] = {
    ops.SoftMax: (transformer_section('3.2.1'),),
}

SINGLE_HEAD_ROLES: dict[str, OperatorRole] = {
    'W^{Q}': transformer_role(text.QUERY_ROLE, '3.2.2'),
    'W^{K}': transformer_role(text.KEY_ROLE, '3.2.2'),
    'W^{V}': transformer_role(text.VALUE_ROLE, '3.2.2'),
    'W^{O}': transformer_role(text.OUTPUT_ROLE, '3.2.2'),
}

MULTI_HEAD_ROLES: dict[str, OperatorRole] = {
    'W^{Q}': transformer_role(text.HEAD_QUERY_ROLE, '3.2.2'),
    'W^{K}': transformer_role(text.HEAD_KEY_ROLE, '3.2.2'),
    'W^{V}': transformer_role(text.HEAD_VALUE_ROLE, '3.2.2'),
    'W^{O}': transformer_role(text.HEAD_OUTPUT_ROLE, '3.2.2'),
}

GROUPED_QUERY_ROLES: dict[str, OperatorRole] = {
    'W^{Q}': OperatorRole(role=text.GROUPED_QUERY_ROLE, references=(
        grouped_query_section('2.2'), mistral_lines('transformer_layers.py', 51))),
    'W^{K}': OperatorRole(role=text.GROUPED_KEY_ROLE, references=(
        grouped_query_section('2.2'), mistral_lines('transformer_layers.py', 52))),
    'W^{V}': OperatorRole(role=text.GROUPED_VALUE_ROLE, references=(
        grouped_query_section('2.2'), mistral_lines('transformer_layers.py', 53))),
    'W^{O}': OperatorRole(role=text.GROUPED_OUTPUT_ROLE, references=(
        mistral_lines('transformer_layers.py', 54),)),
}
