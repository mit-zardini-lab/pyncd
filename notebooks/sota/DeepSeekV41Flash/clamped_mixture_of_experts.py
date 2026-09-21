# Claude Opus 5 (1M context), effort high.
'''DeepSeekMoE of DeepSeek-V4.1-Flash with the SwiGLU clamps, the router's temperature,
the correction bias chosen by the token's modality and the epsilon of the gate
normalisation.

Written by Claude Fable 5.1, reasoning effort 80, as
`notebooks.sota.DeepSeekV41Flash.tempered_mixture`. Split into this base
module by Claude Opus 5 (1M context), effort high, on 2026-09-19, so that
`omitted_mechanisms` and the integrated model read one mixture.

`notebooks.sota.DeepSeekV41Flash.mixture_of_experts` states the mixture with those four
operations left out. This module states the mixture with them. `down_projection` and
`combine` are imported from that module unchanged, and the titles, the box names and the
fill colours are the ones that module declares.

    clamp_from_above, clamp_on_both_sides
                            the two limits `Expert.forward` applies before the SiLU
    clamped_swiglu          one token through the two clamped branches and their product
    routed_experts, shared_expert
                            the experts of `mixture_of_experts` on a clamped SwiGLU
    temper_scores           the router's logits divided by the temperature
    bias_of_modality        a `Linear` whose input is the token's modality, an index
                            that selects one of two bias vectors
    add_correction_bias     that bias added to the copy of the scores which chooses the
                            six experts
    router, normalise_gates, expert_gate
                            the gate box, which reads one token's hidden state and its
                            modality
    mix_one_token           the mixture of one token, which every token runs

`notebooks.sota.DeepSeekV41Flash.tempered_mixture` broadcasts `mix_one_token`
over the tokens and feeds the modalities from the tape.
`notebooks.sota.DeepSeekV41Flash.text_only_mixture` builds a router whose
correction bias is one learned number per expert, from the experts, the temperature and
the gate normalisation of this module.

`ARITHMETIC_ROLES` and `ARITHMETIC_REFERENCES` say what each elementwise map this
module names is for, by the text of its name, in the form
`notebooks.sota.DeepSeekV41Flash.operator_explanations` holds its own. `WEIGHT_ROLES`
and `WEIGHT_REFERENCES` hold the sentence and the released lines of the one weight this
module adds, from which the explanations module builds the
`websocket_transfer.auxiliary_information.OperatorRole`, so this module imports no
backend.

An elementwise map is named with a `fd.DynamicName` built from the whole string, because
`fd.DynamicName.from_str` cuts a string at its first underscore.
'''
from __future__ import annotations

import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd
import deepseek.data_structure as dst

from notebooks.sota.DeepSeekV41Flash.construction_idioms import (
    boxed, hold, l1_norm_over, over, route)
from notebooks.sota.DeepSeekV41Flash.custom_operations import (
    indicator, multiply_along, sigmoid_weighted_input, sqrt_softplus)
from notebooks.sota.DeepSeekV41Flash.declared_axes import R, e, f, kexp, m, x
from notebooks.sota.DeepSeekV41Flash.mixture_of_experts import (
    GATE_BOX, GATE_COLOUR, TOKEN_STATE, combine, down_projection)
from notebooks.sota.DeepSeekV41Flash.reference_links import (
    inference_config_lines, model_lines)
from notebooks.sota.DeepSeekV41Flash.released_constants import (
    MODALITY, ROUTER_EPSILON, ROUTER_TEMPERATURE, SWIGLU_LIMIT)
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text


MODALITY_OF_ONE_TOKEN = cat.Array(MODALITY, ())
MODALITY_OF_EVERY_TOKEN = cat.Array(MODALITY, (x,))
EXPERT_SCORES = cat.Array(R, (e,))

CLAMP_FROM_ABOVE_NAME = '\\min(x, \\lambda)'
TEMPER_SCORES_NAME = 'x / \\tau'
ROUTE_SCALE_NAME = '\\tfrac{3}{2}x'
MODALITY_BIAS_NAME = '\\mathrm{bias}^{\\mu}'


def clamp_from_above() -> cat.Broadcasted:
    '''The smaller of a value and the SwiGLU limit, written as the value minus the
    part of it above the limit.'''
    return ops.Arithmetic.template(nm.x - nm.RectifiedLinear(nm.x - SWIGLU_LIMIT),
                                   name=fd.DynamicName(CLAMP_FROM_ABOVE_NAME))


def clamp_on_both_sides() -> cat.Broadcasted:
    return ops.Arithmetic.template(nm.Clamp(nm.x, -SWIGLU_LIMIT, SWIGLU_LIMIT))


CLAMP_ON_BOTH_SIDES_NAME = clamp_on_both_sides().operator.name.to_bodies()


def clamped_swiglu(
    gate_projection: cat.Broadcasted,
    up_projection: cat.Broadcasted,
) -> cat.BroadcastedCategory:
    '''One token through a SwiGLU with the two limits of the released expert. The
    result of `gate_projection` is limited from above and passes through the SiLU, the
    result of `up_projection` is limited on both sides, and the two are multiplied
    along the expert width.'''
    return (route((0, 0), (TOKEN_STATE,))
            @ ((gate_projection @ clamp_from_above() @ sigmoid_weighted_input())
               * (up_projection @ clamp_on_both_sides()))
            @ multiply_along(f))


def routed_experts(ke: cat.Axis) -> cat.Block:
    '''One routed expert whole: a clamped SwiGLU and the down-projection with its
    diagonal.'''
    return cat.Block.template(
        clamped_swiglu(ops.Linear.template((m,), (e, f), 'W^{G}'),
                       ops.Linear.template((m,), (e, f), 'W^{U}'))
        @ down_projection(ke),
        title=text.EXPERTS_TITLE, fill_color='#C1E8F7',
        description=text.TEMPERED_ROUTED_EXPERTS_DESCRIPTION,
        references=(model_lines(841, 851), model_lines(845, 847),
                    inference_config_lines(19)))


def shared_expert() -> cat.Block:
    '''The expert every token runs, at the same hidden width, with the same two limits
    and with no router.'''
    return cat.Block.template(
        clamped_swiglu(ops.Linear.template((m,), (f,), 'W^{Gs}'),
                       ops.Linear.template((m,), (f,), 'W^{Us}'))
        @ ops.Linear.template((f,), (m,), 'W^{Ds}'),
        title=text.SHARED_TITLE, fill_color='#DFF0D8',
        description=text.TEMPERED_SHARED_EXPERT_DESCRIPTION,
        references=(model_lines(887), model_lines(845, 847), model_lines(903)))


def temper_scores() -> cat.Broadcasted:
    return ops.Arithmetic.template(nm.x / ROUTER_TEMPERATURE,
                                   name=fd.DynamicName(TEMPER_SCORES_NAME))


def bias_of_modality() -> cat.Broadcasted:
    '''The correction bias of every expert for one token, chosen by the token's
    modality. The input is an index, zero for a text token and one for a token inside
    an image span, and a `Linear` whose input is an index selects one of its weight
    vectors, so the weight holds the two bias vectors of the released gate.'''
    return ops.Linear.template((), (e,), MODALITY_BIAS_NAME, datatype=MODALITY,
                               output_datatype=R)


def add_correction_bias() -> cat.BroadcastedCategory:
    '''The bias of the token's modality added to a copy of the scores. The body is one
    token's, so the addition is broadcast over the experts and both of its operands
    are read at the expert.'''
    return ((hold(EXPERT_SCORES) * bias_of_modality())
            @ over((e,), ops.AdditionOp.template()))


def router() -> tuple[cat.BroadcastedCategory, cat.Axis]:
    '''One token's gates from its hidden state and its modality, and this layer's
    sparse expert axis. The logits are divided by the temperature before the score
    function, the biased copy of the scores chooses the six experts and the unbiased
    copy weights them.'''
    sel = dst.TopK.template(k=kexp, axis=e, name='k/e')
    ke = sel.cod()[0].shape()[0]
    scores = (ops.Linear.template((m,), (e,), 'W^{R}') @ temper_scores()
              @ sqrt_softplus())
    picked = ((scores * hold(MODALITY_OF_ONE_TOKEN))
              @ route((0, 1, 0), (EXPERT_SCORES, MODALITY_OF_ONE_TOKEN))
              @ ((add_correction_bias() @ sel @ indicator()) * hold(EXPERT_SCORES))
              @ dst.Select.template(ke, e))
    return picked, ke


def normalise_gates(ke: cat.Axis) -> cat.BroadcastedCategory:
    '''The gates divided by their sum plus the router's epsilon and scaled by the
    route scale, which closes the box `expert_gate` builds. The division is one
    `ops.L1Norm` carrying that epsilon.'''
    return (l1_norm_over((ke,), 0, epsilon=ROUTER_EPSILON)
            @ ops.Arithmetic.template(nm.x * nm.Integer(3) / nm.Integer(2),
                                      name=ROUTE_SCALE_NAME))


def expert_gate() -> tuple[cat.Broadcasted, cat.Axis]:
    '''The gate as one named box, and this layer's sparse expert axis. The box reads
    one token's hidden state and its modality and returns the six gates the experts are
    weighted by, which is what the released `Gate` module returns.'''
    picked, ke = router()
    return boxed(cat.Block.template(
        picked @ normalise_gates(ke), title=text.GATE_TITLE, fill_color=GATE_COLOUR,
        description=text.TEMPERED_EXPERT_GATE_DESCRIPTION,
        references=(model_lines(809, 827), model_lines(811), model_lines(818, 820),
                    model_lines(825))), GATE_BOX), ke


def mix_one_token() -> cat.Block:
    '''One token through the gate box, the six experts it chose and the shared expert.
    The modality is the second operand and the gate box alone reads it.'''
    gate, ke = expert_gate()
    return cat.Block.template(
        route((0, 1, 0, 0), (TOKEN_STATE, MODALITY_OF_ONE_TOKEN))
        @ (gate * routed_experts(ke) * shared_expert())
        @ (combine() * hold(TOKEN_STATE))
        @ ops.AdditionOp.template(),
        title=text.MIXTURE_TITLE, fill_color='#FFE2BB',
        description=text.TEMPERED_MIX_ONE_TOKEN_DESCRIPTION,
        references=(model_lines(889, 904), model_lines(1249)))


ARITHMETIC_ROLES: dict[str, str] = {
    CLAMP_FROM_ABOVE_NAME: (
        text.GATE_LIMIT_ROLE),
    CLAMP_ON_BOTH_SIDES_NAME: (
        text.UP_LIMIT_ROLE),
    TEMPER_SCORES_NAME: (
        text.TEMPERATURE_ROLE),
}

ARITHMETIC_REFERENCES: dict[str, fd.Prod[cat.CodeReference]] = {
    CLAMP_FROM_ABOVE_NAME: (model_lines(847), inference_config_lines(19)),
    CLAMP_ON_BOTH_SIDES_NAME: (model_lines(846), inference_config_lines(19)),
    TEMPER_SCORES_NAME: (model_lines(811), model_lines(67)),
}

WEIGHT_ROLES: dict[str, str] = {
    MODALITY_BIAS_NAME: (
        text.MODALITY_BIAS_ROLE),
}

WEIGHT_REFERENCES: dict[str, fd.Prod[cat.CodeReference]] = {
    MODALITY_BIAS_NAME: (model_lines(806, 807), model_lines(818, 820)),
}
