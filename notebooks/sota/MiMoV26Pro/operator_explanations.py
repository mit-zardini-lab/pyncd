# Claude Opus 5.5 (1M context), effort 40.
'''What an inspection box shows over an operator of MiMo-V2.6-Pro.

A block of the model carries its own title, description and references. An operator
carries none, and neither does a reindexing, so the four tables here supply them to the
display, as `notebooks/sota/GLM53/operator_explanations.py` does for GLM-5.3.

`OPERATOR_EXPLANATIONS` explains each operator that no rule writes out in primitives:
the top-8 over the experts, the read of the unbiased scores at the kept experts, the
embedding, the join of the channels of a head, the covariant view that writes the two
halves of the turned channels in the order of the pairs, the reading of pairs as
complex numbers and back, and the elementwise maps whose name hides part of their
formula.

`OPERATOR_REFERENCES` gives the reference lines an expanded operator stands for. An RMS
normalisation, a softmax, a division by a sum and a linear map are written out by
`algebra.registries.standard_expansions`, so their boxes draw the expansion and take
their links from this table.

`OPERATOR_ROLES` says what each weight of the model is for, by the text of the weight's
name, with the line of `modeling_mimo_v2.py` that declares it. The table of turns takes
its row from `rotary_embedding.TABLE_ROLES`.

`REINDEXING_EXPLANATIONS` explains the three named views, by the text of the name.

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
    indicator, reciprocal, sigmoid, sigmoid_weighted_input)
from notebooks.sota.MiMoV26Pro import (
    grouped_query_attention, mixture_of_experts, rotary_embedding)
from notebooks.sota.MiMoV26Pro.reference_links import (
    SLIDING_WINDOW_MASK_REFERENCES, declared_at, modeling_lines)
from notebooks.sota.MiMoV26Pro.block_titles_and_descriptions import TEXT as text


def name_of(node: cat.Broadcasted) -> str:
    '''The text a table keys an operator by: the bodies of its name.'''
    return node.operator.name.to_bodies()


def table_key(name: str) -> str:
    '''The text a table keys a named operator by, from the LaTeX of its name.'''
    return fd.DynamicName.from_str(name).to_bodies()


def explain_top_k(target: cat.Broadcasted) -> OperatorExplanation:
    '''The box over the top-8 over the experts. `s` is the biased scores and `y` the
    values handed out.'''
    operator = target.operator
    scored, = write_index_notation.axis_letters(target.input_weaves[0].target().shape())
    score = write_index_notation.read_at('s', (scored,))
    kept = (rf'{score} \text{{ is among the }} {operator.k.to_latex()} '
            rf'\text{{ largest of }} s')
    return OperatorExplanation(
        title=r'\text{TopK}',
        formula=(rf'{write_index_notation.read_at("y", (scored,))} = {score} '
                 rf'\text{{ where }} {kept}'),
        description=text.TOP_K_EXPERTS_DESCRIPTION,
        references=(modeling_lines(164), modeling_lines(175)))


ARITHMETIC_ROLES: dict[str, str] = {
    grouped_query_attention.SCORE_SCALE_NAME: text.ATTENTION_SCALE_ROLE,
    grouped_query_attention.VALUE_SCALE_NAME: text.VALUE_SCALE_ROLE,
    name_of(sigmoid()): text.ROUTER_SIGMOID_ROLE,
    name_of(sigmoid_weighted_input()): text.SILU_ROLE,
    name_of(indicator()): text.INDICATOR_ROLE,
    name_of(reciprocal()): text.RECIPROCAL_ROLE,
}

ARITHMETIC_REFERENCES: dict[str, fd.Prod[cat.CodeReference]] = {
    grouped_query_attention.SCORE_SCALE_NAME: (modeling_lines(84), modeling_lines(261)),
    grouped_query_attention.VALUE_SCALE_NAME: (modeling_lines(267),
                                               modeling_lines(302, 303)),
    name_of(sigmoid()): (modeling_lines(156, 157),),
    name_of(sigmoid_weighted_input()): (modeling_lines(129), modeling_lines(132)),
    name_of(indicator()): (modeling_lines(175, 176),),
    name_of(reciprocal()): (modeling_lines(89, 97),),
}


OPERATOR_EXPLANATIONS: dict[type[cat.Operator], ExplanationOfOperator] = {
    dst.TopK: explain_top_k,
    ops.Arithmetic: explain_named_arithmetic(ARITHMETIC_ROLES, ARITHMETIC_REFERENCES),
    dst.Select: OperatorExplanation(
        title=r'\text{Select}',
        formula=(r'\mathrm{Select}(w, p)[j] = w[j]\, p[j] '
                 r'\text{ at each live position } j'),
        description=text.SELECT_DESCRIPTION,
        references=(modeling_lines(176),)),
    ops.Embedding: OperatorExplanation(
        title=r'\text{Embedding}',
        formula=r'E(t) = W[t]',
        description=text.EMBEDDING_DESCRIPTION,
        references=(modeling_lines(1584),)),
    aops.ConcatenateAxes: rotary_embedding.explain_channel_join,
    aops.CovariantView: rotary_embedding.explain_pairs,
    **rotary_embedding.OPERATOR_EXPLANATIONS,
}

OPERATOR_REFERENCES: dict[type[cat.Operator], fd.Prod[cat.CodeReference]] = {
    ops.Normalize: (modeling_lines(105, 117),),
    ops.SoftMax: (modeling_lines(93, 94),),
    ops.L1Norm: (modeling_lines(180, 182),),
}

OPERATOR_ROLES: dict[str, OperatorRole] = {
    table_key('W^{Qr}'): declared_at(text.QUERY_TURNED_PROJECTION_ROLE, 278, 283),
    table_key('W^{Qn}'): declared_at(text.QUERY_UNTURNED_PROJECTION_ROLE, 278, 283),
    table_key('W^{Kr}'): declared_at(text.KEY_TURNED_PROJECTION_ROLE, 278, 283),
    table_key('W^{Kn}'): declared_at(text.KEY_UNTURNED_PROJECTION_ROLE, 278, 283),
    table_key('W^{V}'): declared_at(text.VALUE_PROJECTION_ROLE, 278, 283),
    table_key('W^{O}'): declared_at(text.OUTPUT_PROJECTION_ROLE, 288),
    table_key(grouped_query_attention.SINK_NAME): declared_at(text.SINK_ROLE, 268, 275),
    table_key('W^{R}'): declared_at(text.ROUTER_ROLE, 148),
    table_key(mixture_of_experts.ROUTER_BIAS_NAME): declared_at(
        text.CORRECTION_BIAS_ROLE, 149, 150),
    table_key('W^{G}'): declared_at(text.EXPERT_GATE_PROJECTION_ROLE, 126),
    table_key('W^{U}'): declared_at(text.EXPERT_UP_PROJECTION_ROLE, 127),
    table_key('W^{D}'): declared_at(text.EXPERT_DOWN_PROJECTION_ROLE, 128),
    table_key('W^{Gd}'): declared_at(text.DENSE_GATE_PROJECTION_ROLE, 126),
    table_key('W^{Ud}'): declared_at(text.DENSE_UP_PROJECTION_ROLE, 127),
    table_key('W^{Dd}'): declared_at(text.DENSE_DOWN_PROJECTION_ROLE, 128),
    table_key('W^{L}'): declared_at(text.OUTPUT_HEAD_ROLE, 1704),
    **rotary_embedding.TABLE_ROLES,
}

REINDEXING_EXPLANATIONS: dict[str, ReindexingExplanation] = {
    grouped_query_attention.BACK_VIEW_NAME: ReindexingExplanation(
        description=text.BACK_VIEW_DESCRIPTION,
        references=(modeling_lines(84, 87), modeling_lines(1652))),
    grouped_query_attention.WINDOW_VIEW_NAME: ReindexingExplanation(
        description=text.WINDOW_VIEW_DESCRIPTION,
        references=(modeling_lines(262), modeling_lines(1654, 1657),
                    *SLIDING_WINDOW_MASK_REFERENCES)),
    mixture_of_experts.DIAGONAL_VIEW_NAME: ReindexingExplanation(
        description=text.DIAGONAL_VIEW_DESCRIPTION,
        references=(modeling_lines(196, 210),)),
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
