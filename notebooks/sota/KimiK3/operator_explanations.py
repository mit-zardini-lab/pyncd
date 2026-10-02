# Claude Opus 5.5 (1M context), effort 40.
'''What an inspection box shows over an operator of Kimi K3.

A block of the model carries its own title, description and references. An operator
carries none, and neither does a reindexing, so the four tables here supply them to the
display, as `notebooks/sota/GLM53/operator_explanations.py` does for GLM-5.3.

`OPERATOR_EXPLANATIONS` explains each operator that no rule writes out in primitives:
the top-16 of the router, the read of the scores at the chosen experts, the embedding,
the join of the two parts of a key, the writes at an entry of the attention residuals
and at a token of the scan, and the elementwise maps whose name hides part of their
formula.

`OPERATOR_REFERENCES` gives the reference lines an expanded operator stands for. An RMS
normalisation and a softmax are written out by `algebra.registries.standard_expansions`,
so their boxes draw the expansion and take their links from this table.

`OPERATOR_ROLES` says what each weight of the model is for, by the text of the weight's
name, with the line of `modeling_kimi_linear.py` that declares it.

`REINDEXING_EXPLANATIONS` explains the named views, by the text of the name.

`with_explanation_tables` writes the four tables into the fields of a
`DiagramSettings` that the display reads.
'''
from __future__ import annotations

import dataclasses

import advanced_axis_dynamics.data_structure.Operators as aops
import algebra.write_index_notation as write_index_notation
import data_structure.Category as cat
import data_structure.Operators as ops
import data_structure.Term as fd
import deepseek.data_structure as dst
from websocket_transfer.auxiliary_information import OperatorRole

import notebooks.display.notebook_diagrams as notebook_diagrams
from notebooks.display.explain_operators import (
    ExplanationOfOperator, OperatorExplanation, explain_named_arithmetic)
from notebooks.display.explain_reindexings import ReindexingExplanation
from notebooks.sota.DeepSeekV41Flash.custom_operations import (
    indicator, sigmoid, sigmoid_weighted_input)
from notebooks.sota.KimiK3 import (
    attention_residuals, delta_attention, delta_rule_scan, feed_forward,
    latent_attention, latent_mixture_of_experts)
from notebooks.sota.KimiK3.declared_axes import TOKEN_COUNTER, x
from notebooks.sota.KimiK3.reference_links import (
    L2_NORM_KERNEL, RECURRENCE, declared_at, modeling_lines, wrapper_lines)
from notebooks.sota.KimiK3.block_titles_and_descriptions import TEXT as text


def name_of(node: cat.Broadcasted) -> str:
    '''The text a table keys an operator by: the bodies of its name.'''
    return node.operator.name.to_bodies()


def table_key(name: str) -> str:
    '''The text a table keys a named operator by, from the LaTeX of its name.'''
    return fd.DynamicName.from_str(name).to_bodies()


def whole_name_key(name: str) -> str:
    '''The text a table keys an operator by whose name was built whole, with no cut at
    an underscore.'''
    return fd.DynamicName(name).to_bodies()


def explain_top_k(target: cat.Broadcasted) -> OperatorExplanation:
    '''The box over the top-16 of the router. `s` is the biased scores and `y` the
    values handed out.'''
    scored, = write_index_notation.axis_letters(target.input_weaves[0].target().shape())
    score = write_index_notation.read_at('s', (scored,))
    kept = (rf'{score} \text{{ is among the }} {target.operator.k.to_latex()} '
            rf'\text{{ largest of }} s')
    return OperatorExplanation(
        title=r'\text{TopK}',
        formula=(rf'{write_index_notation.read_at("y", (scored,))} = {score} '
                 rf'\text{{ where }} {kept}'),
        description=text.TOP_K_EXPERTS_DESCRIPTION,
        references=(modeling_lines(723), modeling_lines(747, 749)))


def explain_key_join(target: cat.Broadcasted) -> OperatorExplanation:
    '''The box over the join of the latent key channels and the shared key channels of
    a head onto `a`.'''
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
        description=text.KEY_JOIN_DESCRIPTION,
        references=(modeling_lines(440),))


def explain_covariant_view(target: cat.Broadcasted) -> OperatorExplanation:
    '''The box over a write at one position: an output of a sublayer written at the
    entry of its block, the embedding written at entry 0, or the output of one token of
    the scan written at the token. The position written is the row of the reindexing.'''
    reindexing = target.operator.reindexing
    (axis, _, position), = reindexing._cod_stride_shift
    letter, = write_index_notation.axis_letters((axis,))
    written = position.to_latex()
    at_token = position == TOKEN_COUNTER
    return OperatorExplanation(
        formula=(rf'y[{written}] = v, \qquad y[{write_index_notation.index_of(letter)}] '
                 rf'= \varnothing \text{{ elsewhere}}'),
        description=(text.WRITE_AT_TOKEN_DESCRIPTION if at_token
                     else text.WRITE_AT_ENTRY_DESCRIPTION),
        references=(RECURRENCE,) if at_token else attention_residuals.LIST_REFERENCES)


ARITHMETIC_ROLES: dict[str, str] = {
    latent_attention.SCORE_SCALE_NAME: text.ATTENTION_SCALE_ROLE,
    delta_attention.QUERY_SCALE_NAME: text.QUERY_SCALE_ROLE,
    name_of(sigmoid()): text.SIGMOID_ROLE,
    name_of(sigmoid_weighted_input()): text.SILU_ROLE,
    name_of(indicator()): text.INDICATOR_ROLE,
    whole_name_key(feed_forward.SITU_GATE_NAME): text.SITU_GATE_ROLE,
    whole_name_key(feed_forward.SITU_UP_NAME): text.SITU_UP_ROLE,
    whole_name_key(delta_attention.DECAY_NAME): text.DECAY_ROLE,
    table_key(delta_attention.EXPONENTIAL_NAME): text.EXPONENTIAL_ROLE,
    whole_name_key(delta_attention.INVERSE_LENGTH_NAME): text.INVERSE_LENGTH_ROLE,
    table_key(delta_rule_scan.NEGATE_NAME): text.NEGATE_ROLE,
}

ARITHMETIC_REFERENCES: dict[str, fd.Prod[cat.CodeReference]] = {
    latent_attention.SCORE_SCALE_NAME: (modeling_lines(359), modeling_lines(324)),
    delta_attention.QUERY_SCALE_NAME: delta_attention.QUERY_SCALE_REFERENCES,
    name_of(sigmoid()): (modeling_lines(712), modeling_lines(471),
                         *delta_attention.STRENGTH_REFERENCES[-1:]),
    name_of(sigmoid_weighted_input()): delta_attention.CONVOLUTION_REFERENCES,
    whole_name_key(feed_forward.SITU_GATE_NAME): feed_forward.SITU_REFERENCES,
    whole_name_key(feed_forward.SITU_UP_NAME): feed_forward.SITU_REFERENCES,
    whole_name_key(delta_attention.DECAY_NAME): delta_attention.DECAY_REFERENCES,
    table_key(delta_attention.EXPONENTIAL_NAME): delta_attention.DECAY_REFERENCES,
    whole_name_key(delta_attention.INVERSE_LENGTH_NAME): (L2_NORM_KERNEL,),
    table_key(delta_rule_scan.NEGATE_NAME): (RECURRENCE,),
}

OPERATOR_EXPLANATIONS: dict[type[cat.Operator], ExplanationOfOperator] = {
    dst.TopK: explain_top_k,
    ops.Arithmetic: explain_named_arithmetic(ARITHMETIC_ROLES, ARITHMETIC_REFERENCES),
    dst.Select: OperatorExplanation(
        title=r'\text{Select}',
        formula=(r'\mathrm{Select}(w, p)[j] = w[j]\, p[j] '
                 r'\text{ at each live position } j'),
        description=text.SELECT_DESCRIPTION,
        references=(modeling_lines(750),)),
    ops.Embedding: OperatorExplanation(
        title=r'\text{Embedding}',
        formula=r'E(t) = W[t]',
        description=text.EMBEDDING_DESCRIPTION,
        references=(modeling_lines(1096, 1097),)),
    aops.ConcatenateAxes: explain_key_join,
    aops.CovariantView: explain_covariant_view,
}

OPERATOR_REFERENCES: dict[type[cat.Operator], fd.Prod[cat.CodeReference]] = {
    ops.Normalize: (modeling_lines(226, 236),),
    ops.SoftMax: (modeling_lines(328), modeling_lines(1086)),
}

OPERATOR_ROLES: dict[str, OperatorRole] = {
    table_key('W^{Qa}'): declared_at(text.QUERY_DOWN_PROJECTION_ROLE, 365, 367),
    table_key('W^{Qb}'): declared_at(text.QUERY_UP_PROJECTION_ROLE, 369, 373),
    table_key('W^{KVa}'): declared_at(text.LATENT_PROJECTION_ROLE, 378, 382),
    table_key('W^{Kr}'): declared_at(text.SHARED_KEY_PROJECTION_ROLE, 378, 382),
    table_key('W^{Kb}'): declared_at(text.KEY_EXPANSION_ROLE, 384, 389),
    table_key('W^{Vb}'): declared_at(text.VALUE_EXPANSION_ROLE, 384, 389),
    table_key('W^{Gm}'): declared_at(text.LATENT_OUTPUT_GATE_ROLE, 398, 401),
    table_key('W^{Om}'): declared_at(text.LATENT_OUTPUT_PROJECTION_ROLE, 390, 394),
    table_key('W^{Q}'): declared_at(text.DELTA_QUERY_PROJECTION_ROLE, 498, 499),
    table_key('W^{K}'): declared_at(text.DELTA_KEY_PROJECTION_ROLE, 500, 501),
    table_key('W^{V}'): declared_at(text.DELTA_VALUE_PROJECTION_ROLE, 502),
    table_key('W^{Cq}'): declared_at(text.QUERY_TAPS_ROLE, 504, 508),
    table_key('W^{Ck}'): declared_at(text.KEY_TAPS_ROLE, 509, 513),
    table_key('W^{Cv}'): declared_at(text.VALUE_TAPS_ROLE, 514, 518),
    table_key('W^{Fa}'): declared_at(text.DECAY_DOWN_PROJECTION_ROLE, 523),
    table_key('W^{Fb}'): declared_at(text.DECAY_UP_PROJECTION_ROLE, 524),
    table_key('b^{\\Delta}'): declared_at(text.DECAY_BIAS_ROLE, 526, 527),
    table_key('A_{\\log}'): declared_at(text.DECAY_RATE_ROLE, 520, 521),
    table_key('W^{\\beta}'): declared_at(text.STRENGTH_PROJECTION_ROLE, 529),
    table_key('W^{Gk}'): declared_at(text.DELTA_OUTPUT_GATE_ROLE, 533, 534),
    table_key('W^{Ok}'): declared_at(text.DELTA_OUTPUT_PROJECTION_ROLE, 541),
    table_key('W^{R}'): declared_at(text.ROUTER_ROLE, 689, 691),
    table_key(latent_mixture_of_experts.ROUTER_BIAS_NAME): declared_at(
        text.CORRECTION_BIAS_ROLE, 693, 695),
    table_key('W^{Ld}'): declared_at(text.LATENT_DOWN_PROJECTION_ROLE, 804, 806),
    table_key('W^{Lu}'): declared_at(text.LATENT_UP_PROJECTION_ROLE, 807, 809),
    table_key('W^{G}'): declared_at(text.EXPERT_GATE_PROJECTION_ROLE, 249),
    table_key('W^{U}'): declared_at(text.EXPERT_UP_PROJECTION_ROLE, 251),
    table_key('W^{D}'): declared_at(text.EXPERT_DOWN_PROJECTION_ROLE, 250),
    table_key('W^{Gs}'): declared_at(text.SHARED_GATE_PROJECTION_ROLE, 279, 280),
    table_key('W^{Us}'): declared_at(text.SHARED_UP_PROJECTION_ROLE, 281, 282),
    table_key('W^{Ds}'): declared_at(text.SHARED_DOWN_PROJECTION_ROLE, 283, 284),
    table_key('W^{G0}'): declared_at(text.DENSE_GATE_PROJECTION_ROLE, 279, 280),
    table_key('W^{U0}'): declared_at(text.DENSE_UP_PROJECTION_ROLE, 281, 282),
    table_key('W^{D0}'): declared_at(text.DENSE_DOWN_PROJECTION_ROLE, 283, 284),
    table_key(attention_residuals.ATTENTION_INPUT_WEIGHT): declared_at(
        text.ATTENTION_MIX_WEIGHT_ROLE, 910, 915),
    table_key(attention_residuals.FEED_FORWARD_INPUT_WEIGHT): declared_at(
        text.FEED_FORWARD_MIX_WEIGHT_ROLE, 912, 917),
    table_key(attention_residuals.OUTPUT_INPUT_WEIGHT): declared_at(
        text.OUTPUT_MIX_WEIGHT_ROLE, 1105, 1108),
    table_key('W^{L}'): declared_at(text.OUTPUT_HEAD_ROLE, 1247, 1248),
    table_key('E'): OperatorRole(
        role=text.EMBEDDING_TABLE_ROLE,
        references=(modeling_lines(1096, 1097), wrapper_lines(927, 928))),
}

REINDEXING_EXPLANATIONS: dict[str, ReindexingExplanation] = {
    table_key(latent_attention.BACK_VIEW_NAME): ReindexingExplanation(
        description=text.BACK_VIEW_DESCRIPTION,
        references=(modeling_lines(1173, 1180), modeling_lines(454, 463))),
    table_key(latent_attention.REPEAT_VIEW_NAME): ReindexingExplanation(
        description=text.REPEAT_VIEW_DESCRIPTION,
        references=(modeling_lines(435, 437),)),
    table_key(latent_mixture_of_experts.DIAGONAL_VIEW_NAME): ReindexingExplanation(
        description=text.DIAGONAL_VIEW_DESCRIPTION,
        references=(modeling_lines(840, 874),)),
    table_key(delta_attention.TAPS_VIEW_NAME): ReindexingExplanation(
        description=text.TAPS_VIEW_DESCRIPTION,
        references=delta_attention.CONVOLUTION_REFERENCES),
    table_key(delta_attention.DEPTHWISE_VIEW_NAME): ReindexingExplanation(
        description=text.DEPTHWISE_VIEW_DESCRIPTION,
        references=delta_attention.CONVOLUTION_REFERENCES),
    attention_residuals.EMBEDDING_ENTRY.to_latex(): ReindexingExplanation(
        description=text.EMBEDDING_ENTRY_VIEW_DESCRIPTION,
        references=(modeling_lines(987, 998),)),
    delta_rule_scan.row_at(x, TOKEN_COUNTER).name.to_bodies(): ReindexingExplanation(
        description=text.TOKEN_VIEW_DESCRIPTION, references=(RECURRENCE,)),
}


def with_explanation_tables(
    settings: notebook_diagrams.DiagramSettings,
) -> notebook_diagrams.DiagramSettings:
    '''`settings` with the four tables of this module in the fields the display
    reads.'''
    return dataclasses.replace(
        settings,
        operator_explanations=OPERATOR_EXPLANATIONS,
        operator_references=OPERATOR_REFERENCES,
        operator_roles=OPERATOR_ROLES,
        reindexing_explanations=REINDEXING_EXPLANATIONS)

