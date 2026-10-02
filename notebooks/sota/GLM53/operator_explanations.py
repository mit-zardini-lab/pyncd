# Claude Opus 5.5 (1M context), effort 40.
'''What an inspection box shows over an operator of GLM-5.3.

A block of the model carries its own title, description and references. An operator
carries none, and neither does a reindexing, so the four tables here supply them to the
display, as `notebooks/sota/DeepSeekV41Flash/operator_explanations.py` does for that
model.

`OPERATOR_EXPLANATIONS` explains each operator that no rule writes out in primitives:
the two top-k selections, the two reads at selected positions, the embedding, the cut
and the join of the channels, the reading of pairs as complex numbers and back, and the
elementwise maps whose name hides part of their formula. `explain_top_k` writes the
formula from the form of the selection, so the top-2048 over the tokens and the top-8
over the experts open two different boxes.

`OPERATOR_REFERENCES` gives the reference lines an expanded operator stands for. An RMS
normalisation, a layer normalisation and a linear map are written out by
`algebra.registries.standard_expansions`, so their boxes draw the expansion and take
their links from this table.

`OPERATOR_ROLES` says what each weight of the model is for, by the text of the weight's
name, with the line of `modeling_glm_moe_dsa.py` that declares it. The table of turns
takes its row from `rotary_embedding.TABLE_ROLES`.

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
    indicator, sigmoid, sigmoid_weighted_input)
from notebooks.sota.GLM53 import (
    lightning_indexer, mixture_of_experts, multi_latent_attention, rotary_embedding)
from notebooks.sota.GLM53.reference_links import declared_at, modeling_lines
from notebooks.sota.GLM53.block_titles_and_descriptions import TEXT as text


def name_of(node: cat.Broadcasted) -> str:
    '''The text a table keys an operator by: the bodies of its name.'''
    return node.operator.name.to_bodies()


def table_key(name: str) -> str:
    '''The text a table keys a named operator by, from the LaTeX of its name.'''
    return fd.DynamicName.from_str(name).to_bodies()


def explain_top_k(target: cat.Broadcasted) -> OperatorExplanation:
    '''The box over a top-k selection, written from the form the selection hands its
    results out in. `s` is the scores, `y` the values handed out and `p` the positions
    handed out.'''
    operator = target.operator
    scored, = write_index_notation.axis_letters(target.input_weaves[0].target().shape())
    selected, = write_index_notation.axis_letters(
        target.output_weaves[0].target().shape())
    score = write_index_notation.read_at('s', (scored,))
    kept = (rf'{score} \text{{ is among the }} {operator.k.to_latex()} '
            rf'\text{{ largest of }} s')
    if operator.form is dst.SelectionForm.ONLY_SELECTION:
        position = write_index_notation.read_at('p', (selected,))
        return OperatorExplanation(
            title=r'\text{TopK}',
            formula=(rf'\{{{position}\}} = '
                     rf'\{{{write_index_notation.index_of(scored)} : {kept}\}}'),
            description=text.TOP_K_TOKENS_DESCRIPTION,
            references=(modeling_lines(246, 253),))
    return OperatorExplanation(
        title=r'\text{TopK}',
        formula=(rf'{write_index_notation.read_at("y", (scored,))} = {score} '
                 rf'\text{{ where }} {kept}'),
        description=text.TOP_K_EXPERTS_DESCRIPTION,
        references=(modeling_lines(498), modeling_lines(513)))


ARITHMETIC_ROLES: dict[str, str] = {
    multi_latent_attention.SCORE_SCALE_NAME: text.ATTENTION_SCALE_ROLE,
    lightning_indexer.SCORE_SCALE_NAME: text.INDEXER_SCORE_SCALE_ROLE,
    lightning_indexer.HEAD_WEIGHT_SCALE_NAME: text.INDEXER_HEAD_SCALE_ROLE,
    mixture_of_experts.ROUTE_SCALE_NAME: text.ROUTE_SCALE_ROLE,
    name_of(sigmoid()): text.ROUTER_SIGMOID_ROLE,
    name_of(sigmoid_weighted_input()): text.SILU_ROLE,
    name_of(lightning_indexer.rectify()): text.RECTIFIER_ROLE,
    name_of(indicator()): text.INDICATOR_ROLE,
}

ARITHMETIC_REFERENCES: dict[str, fd.Prod[cat.CodeReference]] = {
    multi_latent_attention.SCORE_SCALE_NAME: (modeling_lines(281), modeling_lines(357)),
    lightning_indexer.SCORE_SCALE_NAME: (modeling_lines(193), modeling_lines(239)),
    lightning_indexer.HEAD_WEIGHT_SCALE_NAME: (modeling_lines(243),),
    mixture_of_experts.ROUTE_SCALE_NAME: (modeling_lines(518),),
    name_of(sigmoid()): (modeling_lines(497),),
    name_of(sigmoid_weighted_input()): (modeling_lines(477), modeling_lines(554)),
    name_of(lightning_indexer.rectify()): (modeling_lines(240),),
}

OPERATOR_EXPLANATIONS: dict[type[cat.Operator], ExplanationOfOperator] = {
    dst.TopK: explain_top_k,
    ops.Arithmetic: explain_named_arithmetic(ARITHMETIC_ROLES, ARITHMETIC_REFERENCES),
    dst.Select: OperatorExplanation(
        title=r'\text{Select}',
        formula=(r'\mathrm{Select}(w, p)[j] = w[j]\, p[j] '
                 r'\text{ at each live position } j'),
        description=text.SELECT_DESCRIPTION,
        references=(modeling_lines(514),)),
    dst.IndexSelect: OperatorExplanation(
        title=r'\text{IndexSelect}',
        formula=r'\mathrm{IndexSelect}(i, p) = p[i]',
        description=text.INDEX_SELECT_DESCRIPTION,
        references=(modeling_lines(430, 443),)),
    ops.Embedding: OperatorExplanation(
        title=r'\text{Embedding}',
        formula=r'E(t) = W[t]',
        description=text.EMBEDDING_DESCRIPTION,
        references=(modeling_lines(669),)),
    aops.ConcatenateAxes: rotary_embedding.explain_channel_join,
    aops.DeconcatenateAxes: rotary_embedding.explain_channel_cut,
    **rotary_embedding.OPERATOR_EXPLANATIONS,
}

OPERATOR_REFERENCES: dict[type[cat.Operator], fd.Prod[cat.CodeReference]] = {
    ops.Normalize: (modeling_lines(47, 62),),
    ops.LayerNorm: (modeling_lines(191), modeling_lines(228)),
}

OPERATOR_ROLES: dict[str, OperatorRole] = {
    table_key('W^{Qa}'): declared_at(text.QUERY_DOWN_PROJECTION_ROLE, 335),
    table_key('W^{Qn}'): declared_at(text.QUERY_UNTURNED_PROJECTION_ROLE, 337),
    table_key('W^{Qr}'): declared_at(text.QUERY_TURNED_PROJECTION_ROLE, 337),
    table_key('W^{KVa}'): declared_at(text.LATENT_PROJECTION_ROLE, 339, 343),
    table_key('W^{Kr}'): declared_at(text.KEY_TURNED_PROJECTION_ROLE, 339, 343),
    table_key('W^{Kb}'): declared_at(text.KEY_EXPANSION_ROLE, 345, 349),
    table_key('W^{Vb}'): declared_at(text.VALUE_EXPANSION_ROLE, 345, 349),
    table_key('W^{O}'): declared_at(text.OUTPUT_PROJECTION_ROLE, 351, 355),
    table_key('q^{I}'): declared_at(text.INDEXER_QUERY_PROJECTION_ROLE, 189),
    table_key('k^{I}'): declared_at(text.INDEXER_KEY_PROJECTION_ROLE, 190),
    table_key('w^{I}'): declared_at(text.INDEXER_HEAD_WEIGHTS_ROLE, 192),
    table_key('W^{R}'): declared_at(text.ROUTER_ROLE, 487),
    table_key(mixture_of_experts.ROUTER_BIAS_NAME): declared_at(
        text.CORRECTION_BIAS_ROLE, 492),
    table_key('W^{G}'): declared_at(text.EXPERT_GATE_PROJECTION_ROLE, 531),
    table_key('W^{U}'): declared_at(text.EXPERT_UP_PROJECTION_ROLE, 531),
    table_key('W^{D}'): declared_at(text.EXPERT_DOWN_PROJECTION_ROLE, 532),
    table_key('W^{Gs}'): declared_at(text.SHARED_GATE_PROJECTION_ROLE, 471),
    table_key('W^{Us}'): declared_at(text.SHARED_UP_PROJECTION_ROLE, 472),
    table_key('W^{Ds}'): declared_at(text.SHARED_DOWN_PROJECTION_ROLE, 473),
    table_key('W^{Gd}'): declared_at(text.DENSE_GATE_PROJECTION_ROLE, 471),
    table_key('W^{Ud}'): declared_at(text.DENSE_UP_PROJECTION_ROLE, 472),
    table_key('W^{Dd}'): declared_at(text.DENSE_DOWN_PROJECTION_ROLE, 473),
    table_key('W^{L}'): declared_at(text.OUTPUT_HEAD_ROLE, 753),
    **rotary_embedding.TABLE_ROLES,
}

REINDEXING_EXPLANATIONS: dict[str, ReindexingExplanation] = {
    lightning_indexer.BACK_VIEW_NAME: ReindexingExplanation(
        description=text.BACK_VIEW_DESCRIPTION,
        references=(modeling_lines(246, 250), modeling_lines(430, 443))),
    mixture_of_experts.DIAGONAL_VIEW_NAME: ReindexingExplanation(
        description=text.DIAGONAL_VIEW_DESCRIPTION,
        references=(modeling_lines(551, 555),)),
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
