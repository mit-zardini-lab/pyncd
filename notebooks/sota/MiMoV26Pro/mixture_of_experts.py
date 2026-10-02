# Claude Opus 5.5 (1M context), effort 40.
'''The mixture of experts of MiMo-V2.6-Pro: 384 routed experts, eight kept per token,
and no shared expert.

The mixture is `MiMoV2MoE` of the reference, and it stands in every layer from 1 to 69.
The router is `MiMoV2MoEGate`. It scores every expert with a sigmoid of a linear map of
the hidden state. A learned correction bias is added to one copy of the scores, and the
eight largest biased scores choose the experts. The unbiased scores are read at the
eight chosen experts and divided by their sum plus `\\varepsilon_{\\mathrm{r}}`. The
bias therefore changes which experts run and never scales the output of an expert. The
configuration names the method `noaux_tc`.

The reference first limits the choice to the best groups of experts. The configuration
sets `n_group` and `topk_group` to 1, so the one group holds every expert and is always
kept, and the group step changes no choice. The reference multiplies the gates by
`routed_scaling_factor`, which the configuration leaves unset and the router replaces by
1.0. Both are left out of the expression.

Everything from `W^{R}` to the eight normalised gates is one box, `Gate`, whose chosen
experts are marked by `[x > 0]` of their biased scores. The reference gathers the
unbiased scores at the chosen experts and tests no sign, so the two agree wherever every
chosen expert has a positive biased score. Each routed expert is a SwiGLU whose weights
produce the expert axis, and the down projection is followed by the diagonalisation
that keeps the entries where the slot and the expert agree. One contraction over the
eight live slots multiplies the outputs of the experts by the gates and sums them, and
the reference multiplies them at the same place, after `down_proj`.

Every operation of the mixture is the same map at every token, so the mixture is one
box computed once per token, confirmed against its body lifted over the tokens.
'''
from __future__ import annotations

import algebra.discovering_broadcasts as discovering_broadcasts
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Operators as ops
import data_structure.Term as fd
import deepseek.data_structure as dst

from notebooks.sota.DeepSeekV41Flash.construction_idioms import (
    boxed, hold, l1_norm_over, over, route)
from notebooks.sota.DeepSeekV41Flash.custom_operations import (
    indicator, multiply_along, sigmoid, sigmoid_weighted_input, weights)
from notebooks.sota.MiMoV26Pro.declared_axes import (
    R, TOKEN_STATE, e, f, m, selected_experts, x)
from notebooks.sota.MiMoV26Pro.feed_forward import FEED_FORWARD_REFERENCES
from notebooks.sota.MiMoV26Pro.reference_links import (
    checkpoint_config_lines, modeling_lines)
from notebooks.sota.MiMoV26Pro.released_constants import ROUTER_EPSILON
from notebooks.sota.MiMoV26Pro.block_titles_and_descriptions import TEXT as text

GATE_BOX = 'Gate'
MIXTURE_BOX = 'MoE'
GATE_COLOUR = '#F7E0EF'
EXPERTS_COLOUR = '#C1E8F7'
MIXTURE_COLOUR = '#FFE2BB'
ROUTER_BIAS_NAME = '\\mathrm{bias}'
DIAGONAL_VIEW_NAME = '\\mathrm{Diagonal}'

ROUTER_REFERENCES = (modeling_lines(135, 184), checkpoint_config_lines(198, 203),
                     checkpoint_config_lines(366, 367),
                     checkpoint_config_lines(376, 377))
EXPERT_REFERENCES = (*FEED_FORWARD_REFERENCES, modeling_lines(191, 193),
                     modeling_lines(196, 210))
MIXTURE_REFERENCES = (modeling_lines(187, 217), modeling_lines(405, 409))


def add_correction_bias() -> cat.BroadcastedCategory:
    '''The correction bias added to a copy of one token's scores. The bias is one
    number per expert, so the addition is broadcast over the experts and both of its
    operands are read at the expert.'''
    scores = cat.Array(R, (e,))
    return ((hold(scores) * weights(ROUTER_BIAS_NAME, (e,)))
            @ over((e,), ops.AdditionOp.template()))


def score_and_choose() -> tuple[cat.BroadcastedCategory, cat.Axis]:
    '''One token's unbiased scores read at the eight experts chosen by the biased
    scores, and the sparse expert axis `k/e` that carries them.'''
    scores = cat.Array(R, (e,))
    choose = dst.TopK.template(
        k=selected_experts, axis=e,
        name=fd.DynamicName('k/e', code_form='selected_experts'))
    chosen_experts = choose.cod()[0].shape()[0]
    return ((ops.Linear.template((m,), (e,), 'W^{R}') @ sigmoid()
             @ route((0, 0), (scores,))
             @ ((add_correction_bias() @ choose @ indicator()) * hold(scores))
             @ dst.Select.template(chosen_experts, e)),
            chosen_experts)


def expert_gate() -> tuple[cat.Broadcasted, cat.Axis]:
    '''The router as one named box, from one token's hidden state to the eight gates,
    and the sparse expert axis that carries the gates.'''
    chosen, chosen_experts = score_and_choose()
    return boxed(cat.Block.template(
        chosen @ l1_norm_over((chosen_experts,), 0, epsilon=ROUTER_EPSILON),
        title=text.GATE_TITLE, fill_color=GATE_COLOUR,
        description=text.GATE_DESCRIPTION, references=ROUTER_REFERENCES),
        GATE_BOX), chosen_experts


def down_projection[A: cat.Axis](chosen_experts: A) -> cat.BroadcastedCategory:
    '''`W^{D}` produces the expert axis as the other two weights do, so its implicit
    weight holds all 384 down projections. The diagonalisation after it keeps the
    entries where the slot and the expert agree.'''
    return (over((chosen_experts,), ops.Linear.template((f,), (e, m), 'W^{D}'))
            @ ops.View.template(
                reindexing=cat.Rearrangement((0, 0, 1), (chosen_experts, m)),
                name=DIAGONAL_VIEW_NAME))


def routed_experts[A: cat.Axis](chosen_experts: A) -> cat.Block:
    '''The eight chosen experts of one token, each a SwiGLU with its own weights.'''
    return cat.Block.template(
        route((0, 0), (TOKEN_STATE,))
        @ ((ops.Linear.template((m,), (e, f), 'W^{G}') @ sigmoid_weighted_input())
           * ops.Linear.template((m,), (e, f), 'W^{U}'))
        @ multiply_along(f)
        @ down_projection(chosen_experts),
        title=text.EXPERTS_TITLE, fill_color=EXPERTS_COLOUR,
        description=text.ROUTED_EXPERTS_DESCRIPTION, references=EXPERT_REFERENCES)


def combine() -> cat.Broadcasted:
    '''The gates against the outputs of the chosen experts: one contraction over the
    eight live slots, which is the multiplication and the sum together.'''
    return ops.Einops.template('k, k m -> m')


def mix_one_token() -> cat.Block:
    '''One token through the gate box and the eight experts it chose, with the outputs
    of the experts combined under the gates.'''
    gate, chosen_experts = expert_gate()
    return cat.Block.template(
        route((0, 0), (TOKEN_STATE,))
        @ (gate * routed_experts(chosen_experts))
        @ combine(),
        title=text.MIXTURE_TITLE, fill_color=MIXTURE_COLOUR,
        description=text.MIXTURE_DESCRIPTION, references=MIXTURE_REFERENCES)


MIXTURE_BODY = mix_one_token()
MIXTURE = discovering_broadcasts.broadcast_block_over_axes(
    MIXTURE_BODY, (x,), ((0,),), MIXTURE_BOX)
MIXTURE_CONFIRMATION = discovering_broadcasts.confirm_broadcast_expansion(
    over((x,), MIXTURE_BODY), MIXTURE)
