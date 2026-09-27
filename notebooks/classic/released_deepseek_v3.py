# Claude Opus 5.5 (1M context), reasoning effort 40.
'''The released DeepSeek-V3, at the sizes of `inference/configs/config_671B.json`, as a
morphism in Br.

`deepseek_v3.py` writes the model at the sizes of `config_16B.json`, which the
hand-drawn diagram labels. The released checkpoint is the model of `config_671B.json`,
and that configuration changes two mechanisms of the code the two share, both pinned in
`reference_links.py`.

The queries are projected through a low rank. `q_lora_rank = 1536` sends the hidden
state through `wq_a`, an RMSNorm and `wq_b`, which the expression names `W^{DQ}`, the
RMSNorm over `\\ell_q`, and `W^{UQ}` beside `W^{QR}`.

The gate chooses its experts among the best groups. The configuration sets eight expert
groups, four kept groups, the sigmoid score function and a route scale of 2.5, and a
model of width 7168 holds a correction bias. The bias is added to one copy of the
scores. The 256 experts are read as 8 groups of 32 through the view named `Group`, each
group is scored by the sum of its two best biased scores, and the four best groups are
kept as positions, per the ruling that a selection whose values are not read hands out
its positions. `dst.merge_selected_positions` writes the positions of the 128 experts of
the kept groups, the biased scores are read there, and the best eight of them choose the
experts on the sparse axis `k/n`. The indicator of those eight biased scores selects the
unbiased scores, as the gates of GLM-5.3 and DeepSeek-V4.1-Flash do. Every correction
bias of the checkpoint lies between 2.02 and 8.05, read from the headers and the data of
its shards on 2026-09-27, so every biased score is positive and the indicator is one at
every chosen expert. The router of the released code is one module, `Gate`, and it
computes the same map at every token, so the expression writes its body for one token
and boxes it as `Gate`, computed once per token and confirmed against the body lifted
over the tokens. The derivation of the cached pass passes a box computed once per token
without entering it.

The keys and the values, the rotary embedding, the attention core, the experts and the
layers are the functions of `deepseek_v3.py`, and the descriptions that name no size
are read from `deepseek_v3_wording.json`.

`decode_form` is the model in the CausalSlide form, which is the standard form of a
displayed expression with a causal read, per
`obsidian/06-practice/Representing Models.md`. The tables at the end give the inspection
boxes of the page their rows, and `with_explanation_tables` writes them into the
settings of a figure.
'''
from __future__ import annotations

import dataclasses
import math

import advanced_axis_dynamics.algebra.slide_causal_reads_backwards as slide_causal_reads_backwards
import advanced_axis_dynamics.data_structure.Operators as aops
import algebra.discovering_broadcasts as discovering_broadcasts
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import data_structure.Term as fd
import deepseek.data_structure as dst
import term_utilities.generate_config as generate_config
import term_utilities.term_utilities as term_utilities
from websocket_transfer.auxiliary_information import OperatorRole

import notebooks.classic.deepseek_v3 as deepseek_v3
import notebooks.display.notebook_diagrams as notebook_diagrams
from notebooks.classic.deepseek_v3 import (
    DENSE_LAYER_COUNT, MIXTURE_LAYER_COUNT, QUERIES, SHARED_COUNT, STATE, TOKEN, d, dn,
    dr, dv, f, h, latent, m, n, s, u, v, x)
from notebooks.classic.deepseek_v3_wording import TEXT as generic_text
from notebooks.classic.reference_links import (
    deepseek_lines, deepseek_role, released_config_lines)
from notebooks.classic.released_deepseek_v3_wording import TEXT as text
from notebooks.display.explain_operators import (
    ExplanationOfOperator, OperatorExplanation, explain_named_arithmetic)
from notebooks.display.explain_reindexings import ReindexingExplanation
from notebooks.sota.DeepSeekV41Flash.construction_idioms import (
    boxed, hold, l1_norm_over, over, route)
from notebooks.sota.DeepSeekV41Flash.custom_operations import indicator, weights

R = cat.Reals()
TILED = cat.WeaveMode.TILED

query_latent = cat.RawAxis.named(
    fd.DynamicName('\\ell', fd.DynamicName('q'), code_form='query_latent'))
g = cat.RawAxis.named('g', code_form='expert_groups')
j = cat.RawAxis.named('j', code_form='experts_per_group')
c = cat.RawAxis.named('c', code_form='candidate_experts')
best_scores = cat.RawAxis.named('b', code_form='best_scores_of_a_group')
kept_groups = cat.RawAxis.named('p', code_form='kept_groups')
chosen_expert_count = deepseek_v3.selected_count
ROUTE_SCALE = nm.FreeNumeric.named('\\gamma')

QUERY_LATENT = cat.Array(R, (x, query_latent))
TOKEN_SCORES = cat.Array(R, (n,))
CANDIDATE_POSITIONS = cat.Array(cat.Natural(n.local_size()), (c,))
CANDIDATE_SCORES = cat.Array(R, (c,))

GATE_BOX = 'Gate'
GATE_COLOUR = deepseek_v3.GATE_COLOUR
GROUP_VIEW_NAME = '\\mathrm{Group}'
CANDIDATE_VIEW_NAME = '\\mathrm{Candidate}'
ROUTER_BIAS_NAME = '\\mathrm{bias}'
ROUTE_SCALE_NAME = '\\gamma x'
SIGMOID_NAME = deepseek_v3.SIGMOID_NAME
GATE_FORMULA = (
    r'g[i_{n}] = \gamma\, \frac{\sigma(s[i_{n}])}{\sum_{i \in K} \sigma(s[i])}'
    r'\ \text{for}\ i_{n} \in K,\quad s = W^{g} x')

RELEASED_SIZES: dict[str, int] = {
    'v': 129280, 'm': 7168, 'h': 128, '\\ellq': 1536, '\\ell': 512, 'dn': 128,
    'dr': 64, 'dv': 128, 'u': 18432, 'n': 256, 'g': 8, 'j': 32, 'p': 4, 'b': 2,
    'c': 128, 'k': 8, 'f': 2048, 'S': 1, 'D': 3, 'L': 58}
'''The sizes of `config_671B.json`, keyed by the bodies of the name of each symbol.
`c` is the number of experts in the kept groups, `p` times `j`.'''

GROUP_SPLIT = sc.StrideMorphism(
    _dom=(g, j),
    _cod_stride_shift=((n, (j.local_size(), nm.Integer(1)), nm.Integer(0)),),
    name=fd.DynamicName(GROUP_VIEW_NAME))
'''Expert `i_j` of group `i_g` is expert `|j| i_g + i_j`, which is the
`scores.view(x.size(0), self.n_groups, -1)` of the released gate.'''


# ==========================================================================
# Multi-head latent attention with the queries through a low rank.
# ==========================================================================
def query_generation_through_a_low_rank() -> cat.Block:
    return cat.Block.template(
        (x >> ops.Linear.template((m,), (query_latent,), 'W^{DQ}'))
        @ deepseek_v3.rms_norm_over(query_latent)
        @ route((0, 0), (QUERY_LATENT,))
        @ ((x >> ops.Linear.template((query_latent,), (h, dn), 'W^{UQ}'))
           * ((x >> ops.Linear.template((query_latent,), (h, dr), 'W^{QR}'))
              @ deepseek_v3.rotary_embedding((h,))))
        @ deepseek_v3.join_query_channels(),
        title=text.QUERY_GENERATION_TITLE, fill_color=deepseek_v3.QUERY_COLOUR,
        description=text.QUERY_GENERATION_DESCRIPTION,
        references=(deepseek_lines(426, 429), deepseek_lines(463, 467),
                    released_config_lines(16)))


def multi_head_latent_attention() -> cat.Block:
    return cat.Block.template(
        deepseek_v3.attention_from_queries(query_generation_through_a_low_rank()),
        title=text.ATTENTION_TITLE, fill_color=deepseek_v3.ATTENTION_COLOUR,
        description=text.ATTENTION_DESCRIPTION,
        references=(deepseek_lines(396, 497), deepseek_lines(789),
                    released_config_lines(8), released_config_lines(16, 20)))


# ==========================================================================
# The gate, limited to the best groups of experts, computed once per token.
# ==========================================================================
def add_correction_bias() -> cat.BroadcastedCategory:
    '''The correction bias added to a copy of one token's scores. The bias is one
    number per expert, so the addition is broadcast over the experts and both of its
    operands are read at the expert.'''
    return ((hold(TOKEN_SCORES) * weights(ROUTER_BIAS_NAME, (n,)))
            @ over((n,), ops.AdditionOp.template()))


GROUP_VIEW = ops.View.template(reindexing=(GROUP_SPLIT,), name=GROUP_VIEW_NAME)
BEST_SCORES_OF_A_GROUP = over((g,), dst.TopK.template(
    k=best_scores.local_size(), axis=j, selected_axis=best_scores,
    form=dst.SelectionForm.ONLY_WEIGHTS))
KEEP_GROUPS = dst.TopK.template(
    k=kept_groups.local_size(), axis=g, selected_axis=kept_groups,
    form=dst.SelectionForm.ONLY_SELECTION)
CANDIDATES_OF_THE_KEPT_GROUPS = dst.merge_selected_positions(
    GROUP_SPLIT, kept_groups, cat.ProdObject(()), c, name=CANDIDATE_VIEW_NAME)
CHOOSE_EXPERTS = dst.TopK.template(
    k=chosen_expert_count, axis=c, positions_of=n,
    name=fd.DynamicName('k/n', code_form='chosen_experts'))
chosen = CHOOSE_EXPERTS.cod()[0].shape()[0]


def score_every_group() -> cat.BroadcastedCategory:
    '''One token's biased scores read as groups, each group scored by the sum of its
    best `|b|` biased scores.'''
    return (GROUP_VIEW @ BEST_SCORES_OF_A_GROUP
            @ deepseek_v3.contract(((g, best_scores),), (g,)))


def read_at_the_candidates() -> cat.Broadcasted:
    '''One token's biased scores read at the positions of the candidate experts, one
    read per candidate, broadcast over the candidates.'''
    return cat.Broadcasted(
        operator=dst.IndexSelect(),
        input_weaves=(cat.Weave(cat.Natural(n.local_size()), (TILED,)),
                      cat.Weave(R, (n,))),
        output_weaves=(cat.Weave(R, (TILED,)),),
        reindexings=(cat.ProdObject((c,)).identity(),
                     cat.Rearrangement((), (c,))))


def choose_among_the_best_groups() -> cat.BroadcastedCategory:
    '''One token's biased scores turned into its eight chosen experts: the four best
    groups kept, the experts of those groups laid out as candidates, and the best eight
    candidates chosen on the sparse axis `k/n`.'''
    return (route((0, 0), (TOKEN_SCORES,))
            @ ((score_every_group() @ KEEP_GROUPS @ CANDIDATES_OF_THE_KEPT_GROUPS)
               * hold(TOKEN_SCORES))
            @ route((0, 0, 1), (CANDIDATE_POSITIONS, TOKEN_SCORES))
            @ (hold(CANDIDATE_POSITIONS) * read_at_the_candidates())
            @ route((1, 0), (CANDIDATE_POSITIONS, CANDIDATE_SCORES))
            @ CHOOSE_EXPERTS)


def gate_of_one_token() -> cat.Block:
    return cat.Block.template(
        ops.Linear.template((m,), (n,), 'W^{g}')
        @ ops.Arithmetic.template(nm.Sigmoid(nm.x), name=SIGMOID_NAME)
        @ route((0, 0), (TOKEN_SCORES,))
        @ ((add_correction_bias() @ choose_among_the_best_groups() @ indicator())
           * hold(TOKEN_SCORES))
        @ dst.Select.template(chosen, n)
        @ l1_norm_over((chosen,), 0)
        @ ops.Arithmetic.template(nm.x * ROUTE_SCALE, name=ROUTE_SCALE_NAME),
        title=text.GATE_TITLE, fill_color=GATE_COLOUR,
        description=text.GATE_DESCRIPTION, formula=GATE_FORMULA,
        references=(deepseek_lines(535, 598), released_config_lines(11, 15)))


GATE_BODY = gate_of_one_token()
GATE = discovering_broadcasts.broadcast_block_over_axes(
    GATE_BODY, (x,), ((0,),), GATE_BOX)
GATE_CONFIRMATION = discovering_broadcasts.confirm_broadcast_expansion(
    over((x,), GATE_BODY), GATE)
'''The gate computes the same map at every token, so it is one box computed once per
token, and the box is confirmed against its body lifted over the tokens.'''


# ==========================================================================
# The feed-forward sublayers.
# ==========================================================================
def multi_layer_perceptron() -> cat.Block:
    return cat.Block.template(
        deepseek_v3.swiglu(u),
        title=text.MLP_TITLE, fill_color=deepseek_v3.MLP_COLOUR,
        description=text.MLP_DESCRIPTION, formula=deepseek_v3.SWIGLU_FORMULA,
        references=(deepseek_lines(500, 532), released_config_lines(4)))


def routed_experts() -> cat.Block:
    return cat.Block.template(
        deepseek_v3.swiglu_of_the_chosen_experts(chosen),
        title=text.ROUTED_EXPERTS_TITLE, fill_color=deepseek_v3.EXPERTS_COLOUR,
        description=text.ROUTED_EXPERTS_DESCRIPTION,
        formula=deepseek_v3.SWIGLU_FORMULA,
        references=(deepseek_lines(601, 633), deepseek_lines(684, 689),
                    released_config_lines(5)))


def shared_expert() -> cat.Block:
    return cat.Block.template(
        deepseek_v3.swiglu(s),
        title=text.SHARED_EXPERTS_TITLE, fill_color=deepseek_v3.EXPERTS_COLOUR,
        description=text.SHARED_EXPERTS_DESCRIPTION,
        formula=deepseek_v3.SWIGLU_FORMULA,
        references=(deepseek_lines(667), deepseek_lines(690),
                    released_config_lines(10)))


def deepseek_mixture_of_experts() -> cat.Block:
    return cat.Block.template(
        deepseek_v3.mixture_from_gate(GATE, routed_experts(), shared_expert(), chosen),
        title=text.MOE_TITLE, fill_color=deepseek_v3.MOE_COLOUR,
        description=text.MOE_DESCRIPTION, formula=deepseek_v3.MOE_FORMULA,
        references=(deepseek_lines(636, 693), released_config_lines(9, 11)))


# ==========================================================================
# The layers and the model.
# ==========================================================================
ATTENTION = boxed(multi_head_latent_attention(), deepseek_v3.ATTENTION_BOX)
MLP = boxed(multi_layer_perceptron(), deepseek_v3.MLP_BOX)
MOE = boxed(deepseek_mixture_of_experts(), deepseek_v3.MOE_BOX)


def dense_layers() -> cat.Block:
    return cat.Block.template(
        deepseek_v3.residual_after_norm(ATTENTION, 733)
        @ deepseek_v3.residual_after_norm(MLP, 734),
        title=text.DENSE_LAYER_TITLE, fill_color=deepseek_v3.LAYER_COLOUR,
        repetition=DENSE_LAYER_COUNT, description=text.DENSE_LAYER_DESCRIPTION,
        references=(deepseek_lines(706, 735), released_config_lines(7)))


def mixture_layers() -> cat.Block:
    return cat.Block.template(
        deepseek_v3.residual_after_norm(ATTENTION, 733)
        @ deepseek_v3.residual_after_norm(MOE, 734),
        title=text.MOE_LAYER_TITLE, fill_color=deepseek_v3.LAYER_COLOUR,
        repetition=MIXTURE_LAYER_COUNT, description=text.MOE_LAYER_DESCRIPTION,
        references=(deepseek_lines(706, 735), released_config_lines(6, 7)))


def embedding() -> cat.Block:
    return cat.Block.template(
        x >> ops.Embedding.template(TOKEN, (m,)),
        title=text.EMBEDDING_TITLE, fill_color=deepseek_v3.EMBEDDING_COLOUR,
        description=text.EMBEDDING_DESCRIPTION,
        references=(deepseek_lines(89, 129), deepseek_lines(764),
                    released_config_lines(2)))


def output_projection() -> cat.Block:
    return cat.Block.template(
        deepseek_v3.rms_norm_over(m)
        @ (x >> ops.Linear.template((m,), (v,), deepseek_v3.HEAD_NAME)),
        title=text.OUTPUT_TITLE, fill_color=deepseek_v3.OUTPUT_COLOUR,
        description=text.OUTPUT_DESCRIPTION,
        references=(deepseek_lines(768, 769), deepseek_lines(792, 793)))


def released_deepseek_v3() -> cat.Block:
    return cat.Block.template(
        embedding() @ dense_layers() @ mixture_layers() @ output_projection(),
        title=text.MODEL_TITLE, fill_color=deepseek_v3.MODEL_COLOUR,
        description=text.MODEL_DESCRIPTION,
        references=(deepseek_lines(738, 798), released_config_lines(1, 22)))


MODEL = released_deepseek_v3()


def decode_form(model: cat.Block = MODEL) -> cat.BroadcastedCategory:
    '''`model` in the CausalSlide form, with every causal read moved back to the copy
    whose other branches read its operand unmasked.'''
    return slide_causal_reads_backwards.slide_causal_reads_backwards(model)


def released_assigned_sizes(term: fd.GeneralTerm = MODEL) -> dict[str, int]:
    '''The released size of every symbol of `term` that `RELEASED_SIZES` names, keyed
    by the bodies of the symbol's name.'''
    config = generate_config.NumericConfig.template(term)
    config.assign_values(**{name: size for name, size in RELEASED_SIZES.items()
                            if name in names_of_symbols(term)})
    return config.assigned_integers_by_name()


def names_of_symbols(term: fd.GeneralTerm) -> frozenset[str]:
    return frozenset(symbol.uid._name.to_bodies()
                     for symbol in term_utilities.type_search(nm.FreeNumeric, term)
                     if symbol.uid._name is not None)


# ==========================================================================
# What the inspection boxes show over the operators and the views.
# ==========================================================================
def explain_selection(target: cat.Broadcasted) -> OperatorExplanation:
    '''The box over one of the three selections of the gate, told apart by the form
    each hands its results out in.'''
    operator = target.operator
    match operator.form:
        case dst.SelectionForm.ONLY_WEIGHTS:
            return OperatorExplanation(
                title=r'\text{TopK}',
                formula=(r'\{y[i_{b}]\} = \{s[i_{j}] : s[i_{j}] \text{ is among the } '
                         r'|b| \text{ largest of } s\}'),
                description=text.TOP_TWO_DESCRIPTION,
                references=(deepseek_lines(589),))
        case dst.SelectionForm.ONLY_SELECTION:
            return OperatorExplanation(
                title=r'\text{TopK}',
                formula=(r'\{p[i_{p}]\} = \{i_{g} : s[i_{g}] \text{ is among the } '
                         r'|p| \text{ largest of } s\}'),
                description=text.KEPT_GROUPS_DESCRIPTION,
                references=(deepseek_lines(590, 592),))
    return OperatorExplanation(
        title=r'\text{TopK}',
        formula=(r'y[q[i_{c}]] = s[i_{c}] \text{ where } s[i_{c}] '
                 r'\text{ is among the } |k| \text{ largest of } s'),
        description=text.CHOSEN_EXPERTS_DESCRIPTION,
        references=(deepseek_lines(593),))


SILU_NAME = deepseek_v3.SILU_NAME
INDICATOR_NAME = indicator().operator.name.to_bodies()

ARITHMETIC_ROLES: dict[str, str] = {
    deepseek_v3.SCORE_SCALE_NAME: text.SCORE_SCALE_ROLE,
    SILU_NAME: generic_text.SILU_ROLE,
    SIGMOID_NAME: generic_text.SIGMOID_ROLE,
    INDICATOR_NAME: text.INDICATOR_ROLE,
    ROUTE_SCALE_NAME: text.ROUTE_SCALE_ROLE,
}

ARITHMETIC_REFERENCES: dict[str, tuple[cat.CodeReference, ...]] = {
    deepseek_v3.SCORE_SCALE_NAME: (deepseek_lines(434, 437),),
    SILU_NAME: (deepseek_lines(623, 633),),
    SIGMOID_NAME: (deepseek_lines(577, 580),),
    INDICATOR_NAME: (deepseek_lines(593, 594),),
    ROUTE_SCALE_NAME: (deepseek_lines(597), released_config_lines(14)),
}

OPERATOR_EXPLANATIONS: dict[type[cat.Operator], ExplanationOfOperator] = {
    **deepseek_v3.OPERATOR_EXPLANATIONS,
    dst.TopK: explain_selection,
    ops.Arithmetic: explain_named_arithmetic(ARITHMETIC_ROLES, ARITHMETIC_REFERENCES),
    dst.IndexSelect: OperatorExplanation(
        title=r'\text{IndexSelect}',
        formula=r'y[i_{c}] = s[q[i_{c}]]',
        description=text.INDEX_SELECT_DESCRIPTION,
        references=(deepseek_lines(592),)),
    dst.Select: OperatorExplanation(
        title=r'\text{Select}',
        formula=(r'y[i_{k}] = w[i_{k}]\, s[i_{k}] '
                 r'\text{ at each live position } i_{k} \text{ of } k/n'),
        description=text.SELECT_DESCRIPTION,
        references=(deepseek_lines(594),)),
    dst.MergedPositions: OperatorExplanation(
        formula=r'q[i_{p}, i_{j}] = |j|\, p[i_{p}] + i_{j}',
        description=text.MERGED_POSITIONS_DESCRIPTION,
        references=(deepseek_lines(585), deepseek_lines(591, 592))),
    aops.CovariantView: OperatorExplanation(
        formula=r'y[|j|\, i_{p} + i_{j}] = q[i_{p}, i_{j}]',
        description=text.CANDIDATE_VIEW_DESCRIPTION,
        references=(deepseek_lines(592),)),
}

OPERATOR_ROLES: dict[str, OperatorRole] = {
    **{name: role for name, role in deepseek_v3.OPERATOR_ROLES.items()
       if name not in ('W^{Q}',)},
    'W^{DQ}': deepseek_role(text.QUERY_DOWN_ROLE, 427),
    'W^{UQ}': deepseek_role(text.QUERY_UP_ROLE, 429),
    'W^{QR}': deepseek_role(text.ROTARY_QUERY_ROLE, 429),
    ROUTER_BIAS_NAME: deepseek_role(text.CORRECTION_BIAS_ROLE, 564),
    'E': deepseek_role(text.EMBEDDING_TABLE_ROLE, 105),
}

OPERATOR_REFERENCES: dict[type[cat.Operator], tuple[cat.CodeReference, ...]] = (
    deepseek_v3.OPERATOR_REFERENCES)

REINDEXING_EXPLANATIONS: dict[str, ReindexingExplanation] = {
    **deepseek_v3.REINDEXING_EXPLANATIONS,
    GROUP_VIEW_NAME: ReindexingExplanation(
        description=text.GROUP_VIEW_DESCRIPTION,
        references=(deepseek_lines(585),)),
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


# ==========================================================================
# Reading the model.
# ==========================================================================
class NameIsNotABoxOfTheModel(ValueError):
    '''A name asked of a model and carried by no box of it.'''


def box_named(name: str, term: fd.GeneralTerm) -> cat.Broadcasted:
    '''The first box of `term` whose operator carries `name`.'''
    for node in term_utilities.type_search(cat.Broadcasted, term):
        if (isinstance(node.operator, ops.BlockOperator)
                and node.operator.name is not None
                and node.operator.name.to_bodies() == name):
            return node
    raise NameIsNotABoxOfTheModel(f'{name} names no box of the model')


def correction_dimension(rotations: float, width: int = 64, base: float = 10000.0,
                         original_length: int = 4096) -> float:
    '''The released `find_correction_dim`: the pair whose angular velocity turns
    `rotations` times over `original_length` positions.'''
    return (width * math.log(original_length / (rotations * 2 * math.pi))
            / (2 * math.log(base)))


def yarn_ramp(beta_fast: float = 32, beta_slow: float = 1, width: int = 64
              ) -> tuple[int, int]:
    '''The pairs at which the YaRN ramp starts and ends, as the released
    `find_correction_range` computes them.'''
    return (max(math.floor(correction_dimension(beta_fast, width)), 0),
            min(math.ceil(correction_dimension(beta_slow, width)), width - 1))


def attention_factor(configured_factor: float = 1.0,
                     yarn_factor: float = 40.0) -> float:
    '''The factor `mu` YaRN multiplies the softmax scale by twice, for the default
    `mscale` of 1, which `config_671B.json` leaves in place.'''
    return 0.1 * configured_factor * math.log(yarn_factor) + 1.0


def score_scale(query_key_width: int = 192) -> float:
    return query_key_width ** -0.5 * attention_factor() ** 2
