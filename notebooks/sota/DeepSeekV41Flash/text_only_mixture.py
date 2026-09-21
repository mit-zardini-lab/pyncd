# Claude Opus 5 (1M context), effort high.
'''DeepSeekMoE of the text-only DeepSeek-V4.1-Flash, whose router reads the hidden
state and nothing else.

`notebooks.sota.DeepSeekV41Flash.clamped_mixture_of_experts` states the mixture as the
released multimodal model computes it. The released gate holds two correction biases,
`bias` for a text token and `bias_vl` for a token inside an image span, and the
modality of the token selects between them, so that the load of the experts is
balanced for each kind of token separately. A model that reads text alone reaches the
first of the two at every token, so the bias here is one learned number per expert and
the mixture reads the hidden state alone.

    add_correction_bias   the bias added to the copy of the scores that chooses the
                          six experts, with no modality to select it
    router, expert_gate   the gate box, which reads one token's hidden state
    mix_one_token, MIXTURE
                          the mixture of one token, and the box that computes it once
                          per token from the hidden states

The clamps of the two expert branches, the router's temperature, the epsilon of the
gate normalisation and the route scale are those of `clamped_mixture_of_experts`,
because none of them reads the modality. The sublayer body is the box itself, where
`notebooks.sota.DeepSeekV41Flash.tempered_mixture` puts a grab of
`SLOT_MODALITY` beside it, so the mixture sublayer of the text-only model touches no
tape slot.
'''
from __future__ import annotations

import algebra.discovering_broadcasts as discovering_broadcasts
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Operators as ops
import deepseek.data_structure as dst

from notebooks.sota.DeepSeekV41Flash.construction_idioms import (
    boxed, hold, over, route)
from notebooks.sota.DeepSeekV41Flash.custom_operations import (
    indicator, sqrt_softplus, weights)
from notebooks.sota.DeepSeekV41Flash.declared_axes import e, kexp, m, x
from notebooks.sota.DeepSeekV41Flash.mixture_of_experts import (
    GATE_BOX, GATE_COLOUR, MIXTURE_BOX, ROUTER_BIAS_NAME, TOKEN_STATE, combine)
from notebooks.sota.DeepSeekV41Flash.reference_links import model_lines
from notebooks.sota.DeepSeekV41Flash.clamped_mixture_of_experts import (
    EXPERT_SCORES, normalise_gates, routed_experts, shared_expert, temper_scores)
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text


def add_correction_bias() -> cat.BroadcastedCategory:
    '''The auxiliary-loss-free correction bias added to a copy of the scores. The bias
    is one learned number per expert and the body is one token's, so the addition is
    broadcast over the experts and both of its operands are read at the expert.'''
    return ((hold(EXPERT_SCORES) * weights(ROUTER_BIAS_NAME, (e,)))
            @ over((e,), ops.AdditionOp.template()))


def router() -> tuple[cat.BroadcastedCategory, cat.Axis]:
    '''One token's gates from its hidden state, and this layer's sparse expert axis.
    The logits are divided by the temperature before the score function, the biased
    copy of the scores chooses the six experts and the unbiased copy weights them.'''
    sel = dst.TopK.template(k=kexp, axis=e, name='k/e')
    ke = sel.cod()[0].shape()[0]
    scores = (ops.Linear.template((m,), (e,), 'W^{R}') @ temper_scores()
              @ sqrt_softplus())
    picked = (scores
              @ route((0, 0), (EXPERT_SCORES,))
              @ ((add_correction_bias() @ sel @ indicator()) * hold(EXPERT_SCORES))
              @ dst.Select.template(ke, e))
    return picked, ke


def expert_gate() -> tuple[cat.Broadcasted, cat.Axis]:
    '''The gate as one named box, and this layer's sparse expert axis. The box reads
    one token's hidden state and returns the six gates the experts are weighted by,
    which is what the released `Gate` module returns.'''
    picked, ke = router()
    return boxed(cat.Block.template(
        picked @ normalise_gates(ke), title=text.GATE_TITLE, fill_color=GATE_COLOUR,
        description=text.TEXT_ONLY_MIXTURE_EXPERT_GATE_DESCRIPTION,
        references=(model_lines(809, 827), model_lines(811), model_lines(818, 820),
                    model_lines(825))), GATE_BOX), ke


def mix_one_token() -> cat.Block:
    '''One token through the gate box, the six experts it chose and the shared
    expert.'''
    gate, ke = expert_gate()
    return cat.Block.template(
        route((0, 0, 0), (TOKEN_STATE,))
        @ (gate * routed_experts(ke) * shared_expert())
        @ (combine() * hold(TOKEN_STATE))
        @ ops.AdditionOp.template(),
        title=text.MIXTURE_TITLE, fill_color='#FFE2BB',
        description=text.TEXT_ONLY_MIXTURE_MIX_ONE_TOKEN_DESCRIPTION,
        references=(model_lines(889, 904),))


MIXTURE_BODY = mix_one_token()


def mix_over_all_tokens() -> cat.BroadcastedCategory:
    '''The same mixture written out over every token, which the box is confirmed
    against. It lifts the body the box holds, because `cat.Block.template` mints a
    fresh tag per call and the comparison reads the tags of the expert blocks.'''
    return over((x,), MIXTURE_BODY)


MIXTURE = discovering_broadcasts.broadcast_block_over_axes(
    MIXTURE_BODY, (x,), ((0,),), MIXTURE_BOX)
MIXTURE_CONFIRMATION = discovering_broadcasts.confirm_broadcast_expansion(
    mix_over_all_tokens(), MIXTURE)
