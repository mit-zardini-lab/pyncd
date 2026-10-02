# Claude Opus 5.5 (1M context), effort 40.
'''The mixture of experts of Kimi K3: 896 routed experts that read and write a latent of
3584 channels, sixteen kept per token, and two shared experts.

The mixture is `KimiSparseMoeBlock` of the reference, and it stands in every layer from
1 to 92. The router is `KimiMoEGate`. It scores every expert with a sigmoid of a linear
map of the hidden state. A learned correction bias is added to one copy of the scores,
and the sixteen largest biased scores choose the experts. The unbiased scores are read
at the sixteen chosen experts and divided by their sum plus
`\\varepsilon_{\\mathrm{r}}`. The configuration names the method `noaux_tc`, and sets
one group of experts and `routed_scaling_factor` to 1, so the group step keeps every
expert and the final scaling changes no gate. Both are left out of the expression.

The routed experts do not read the hidden state. `routed_expert_down_proj` maps the
hidden state of 7168 channels onto a latent of 3584, every chosen expert is a SiTU-GLU
from the latent to the latent, the gated sum of the experts is normalised by an RMS
normalisation over the latent, and `routed_expert_up_proj` maps it back onto the hidden
width. Moonshot calls the arrangement a latent mixture of experts. An expert parallel
server sends the latent to the processor holding an expert and receives the latent
back, which is half the width of the hidden state.

Each routed expert's weights produce the expert axis, so the implicit weight of each
projection holds all 896 experts, and a diagonalisation after the down projection keeps
the output of the expert chosen at each slot. The gates land on the outputs of the
experts as one contraction over the sixteen live slots, which the reference computes in
FP32 with `mul_` and `sum` after the experts. The shared experts are one SiTU-GLU of
width 6144, which is `moe_intermediate_size` times `num_shared_experts`, reading the
hidden state, and their output is added to the output of the routed experts.

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
    indicator, multiply_along, sigmoid, weights)
from notebooks.sota.KimiK3.declared_axes import (
    R, TOKEN_STATE, e, f, l, m, selected_experts, t, x)
from notebooks.sota.KimiK3.feed_forward import (
    FEED_FORWARD_REFERENCES, situ_gate, situ_glu, situ_up)
from notebooks.sota.KimiK3.reference_links import checkpoint_config_lines, modeling_lines
from notebooks.sota.KimiK3.released_constants import NORM_EPSILON, ROUTER_EPSILON
from notebooks.sota.KimiK3.block_titles_and_descriptions import TEXT as text

GATE_BOX = 'Gate'
MIXTURE_BOX = 'MoE'
GATE_COLOUR = '#F7E0EF'
EXPERTS_COLOUR = '#C1E8F7'
SHARED_EXPERT_COLOUR = '#DFF0D8'
MIXTURE_COLOUR = '#FFE2BB'
ROUTER_BIAS_NAME = '\\mathrm{bias}'
DIAGONAL_VIEW_NAME = '\\mathrm{Diagonal}'

ROUTER_REFERENCES = (modeling_lines(666, 759), checkpoint_config_lines(178, 179),
                     checkpoint_config_lines(184, 186), checkpoint_config_lines(247),
                     checkpoint_config_lines(258, 259))
EXPERT_REFERENCES = (modeling_lines(242, 270), modeling_lines(786, 795),
                     modeling_lines(840, 874))
LATENT_REFERENCES = (modeling_lines(776, 781), modeling_lines(803, 813),
                     modeling_lines(821, 832), checkpoint_config_lines(64),
                     checkpoint_config_lines(246))
MIXTURE_REFERENCES = (modeling_lines(762, 838), modeling_lines(893, 898),
                      checkpoint_config_lines(176), checkpoint_config_lines(185, 186),
                      checkpoint_config_lines(191))


def add_correction_bias() -> cat.BroadcastedCategory:
    '''The correction bias added to a copy of one token's scores, one number per
    expert.'''
    scores = cat.Array(R, (e,))
    return ((hold(scores) * weights(ROUTER_BIAS_NAME, (e,)))
            @ over((e,), ops.AdditionOp.template()))


def score_and_choose() -> tuple[cat.BroadcastedCategory, cat.Axis]:
    '''One token's unbiased scores read at the sixteen experts chosen by the biased
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
    '''The router as one named box, from one token's hidden state to the sixteen gates
    divided by their sum, and the sparse expert axis that carries the gates.'''
    chosen, chosen_experts = score_and_choose()
    return boxed(cat.Block.template(
        chosen @ l1_norm_over((chosen_experts,), 0, epsilon=ROUTER_EPSILON),
        title=text.GATE_TITLE, fill_color=GATE_COLOUR,
        description=text.GATE_DESCRIPTION, references=ROUTER_REFERENCES),
        GATE_BOX), chosen_experts


def down_projection[A: cat.Axis](chosen_experts: A) -> cat.BroadcastedCategory:
    '''`W^{D}` produces the expert axis as the other two weights do, so its implicit
    weight holds all 896 down projections. The diagonalisation after it keeps the
    entries where the slot and the expert agree.'''
    return (over((chosen_experts,), ops.Linear.template((f,), (e, l), 'W^{D}'))
            @ ops.View.template(
                reindexing=cat.Rearrangement((0, 0, 1), (chosen_experts, l)),
                name=DIAGONAL_VIEW_NAME))


def routed_experts[A: cat.Axis](chosen_experts: A) -> cat.Block:
    '''The sixteen chosen experts of one token, each a SiTU-GLU from the latent to the
    latent with its own weights.'''
    latent = cat.Array(R, (l,))
    return cat.Block.template(
        route((0, 0), (latent,))
        @ ((ops.Linear.template((l,), (e, f), 'W^{G}') @ situ_gate())
           * (ops.Linear.template((l,), (e, f), 'W^{U}') @ situ_up()))
        @ multiply_along(f)
        @ down_projection(chosen_experts),
        title=text.EXPERTS_TITLE, fill_color=EXPERTS_COLOUR,
        description=text.ROUTED_EXPERTS_DESCRIPTION, references=EXPERT_REFERENCES)


def combine() -> cat.Broadcasted:
    '''The gates against the outputs of the chosen experts: one contraction over the
    sixteen live slots, which is the multiplication and the sum together.'''
    return ops.Einops.template('k, k l -> l')


def through_the_latent[A: cat.Axis](chosen_experts: A) -> cat.Block:
    '''`[k/e], [m] -> [m]`: the hidden state projected onto the latent, the chosen
    experts, the gated sum, the RMS normalisation of the latent and the projection
    back onto the hidden width.'''
    gates = cat.Array(R, (chosen_experts,))
    return cat.Block.template(
        (hold(gates) * (ops.Linear.template((m,), (l,), 'W^{Ld}')
                        @ routed_experts(chosen_experts)))
        @ combine()
        @ ops.Normalize.template((l,), epsilon=NORM_EPSILON)
        @ ops.Linear.template((l,), (m,), 'W^{Lu}'),
        title=text.LATENT_TITLE, fill_color=EXPERTS_COLOUR,
        description=text.LATENT_DESCRIPTION, references=LATENT_REFERENCES)


def shared_experts() -> cat.Block:
    return cat.Block.template(
        situ_glu((m,), t, 'W^{Gs}', 'W^{Us}', 'W^{Ds}'),
        title=text.SHARED_EXPERT_TITLE, fill_color=SHARED_EXPERT_COLOUR,
        description=text.SHARED_EXPERT_DESCRIPTION,
        references=(*FEED_FORWARD_REFERENCES, modeling_lines(797, 801),
                    checkpoint_config_lines(191)))


def mix_one_token() -> cat.Block:
    '''One token through the gate box, the experts it chose through the latent, and the
    shared experts, with the two outputs added.'''
    gate, chosen_experts = expert_gate()
    return cat.Block.template(
        route((0, 0, 0), (TOKEN_STATE,))
        @ (gate * hold(TOKEN_STATE) * shared_experts())
        @ (through_the_latent(chosen_experts) * hold(TOKEN_STATE))
        @ ops.AdditionOp.template(),
        title=text.MIXTURE_TITLE, fill_color=MIXTURE_COLOUR,
        description=text.MIXTURE_DESCRIPTION, references=MIXTURE_REFERENCES)


MIXTURE_BODY = mix_one_token()
MIXTURE = discovering_broadcasts.broadcast_block_over_axes(
    MIXTURE_BODY, (x,), ((0,),), MIXTURE_BOX)
MIXTURE_CONFIRMATION = discovering_broadcasts.confirm_broadcast_expansion(
    over((x,), MIXTURE_BODY), MIXTURE)
