'''DeepSeekMoE as DeepSeek-V4.1-Flash runs it: 384 routed experts top-6, plus one shared.

Written by Claude Opus 5, effort high.

The router scores every expert, the correction bias is added on one copy of the scores
and chooses the six, and a `dst.Select` reads the other copy at the slots the biased copy
chose, so the bias changes which experts run and never scales what one returns. The
gates land on the outputs of the experts as one dot product over the six live slots,
which is the right-hand side of the identity that a sum of weighted expert outputs is
the sum of the experts applied to weighted inputs.

Everything from `W^{R}` to the six normalised gates is one named box, which
`expert_gate` builds. It reads one token's hidden state and hands out the gates on the
sparse axis `k/e`. The reference's `Gate` module returns the same thing, as a weight
array beside the index array that says which expert each weight belongs to, and the
sparse axis carries the two together. Boxing them means the scoring, the bias, the
selection, the normalisation and the route scale are read as one piece of the mixture,
and the three expert projections beside it are read as another.

Every operation of the mixture is the same map at every token, so the mixture is one box
computed once per token. `mix_one_token` is the body and it holds no token axis: the
router's scores are `[e]`, the bias addition is broadcast over the experts with both
operands read at the expert, and the selection, the experts and the combine each run on
one token's own arrays. `mix_over_all_tokens` writes the same map out over every token,
so that `discovering_broadcasts.confirm_broadcast_expansion` can check the box against
it.
'''
from __future__ import annotations

import algebra.discovering_broadcasts as discovering_broadcasts
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import deepseek.data_structure as dst

from notebooks.sota.DeepSeekV41Flash.construction_idioms import (
    boxed, hold, l1_norm_over, over, route)
from notebooks.sota.DeepSeekV41Flash.custom_operations import (
    indicator, multiply_along, sigmoid_weighted_input, sqrt_softplus, weights)
from notebooks.sota.DeepSeekV41Flash.declared_axes import R, e, f, kexp, m, x
from notebooks.sota.DeepSeekV41Flash.reference_links import model_lines
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

GATE_BOX = 'Gate'
GATE_COLOUR = '#F7E0EF'
MIXTURE_BOX = 'MoE'
ROUTER_BIAS_NAME = '\\mathrm{bias}'

TOKEN_STATE = cat.Array(R, (m,))


def add_correction_bias() -> cat.BroadcastedCategory:
    '''The auxiliary-loss-free correction bias added to a copy of the scores. The bias is
    one number per expert and the body is one token's, so the addition is broadcast over
    the experts and both of its operands are read at the expert.'''
    scores = cat.Array(R, (e,))
    return ((hold(scores) * weights('\\mathrm{bias}', (e,)))
            @ over((e,), ops.AdditionOp.template()))


def router() -> tuple[cat.BroadcastedCategory, cat.Axis]:
    '''One token's gates and this layer's sparse expert axis, which `expert_gate` puts
    in a box together with the normalisation. The bias chooses the six experts and the
    unbiased scores weight them.'''
    scores = cat.Array(R, (e,))
    sel = dst.TopK.template(k=kexp, axis=e, name='k/e')
    ke = sel.cod()[0].shape()[0]
    picked = (ops.Linear.template((m,), (e,), 'W^{R}') @ sqrt_softplus()
              @ route((0, 0), (scores,))
              @ ((add_correction_bias() @ sel @ indicator()) * hold(scores))
              @ dst.Select.template(ke, e))
    return picked, ke


def normalise_gates(ke: cat.Axis) -> cat.BroadcastedCategory:
    '''The gates divided by their sum and scaled by the route scale, which closes the
    box `expert_gate` builds. There is one axis, so the sum is over position 0 and
    nothing is broadcast over.'''
    return (l1_norm_over((ke,), 0)
            @ ops.Arithmetic.template(nm.x * nm.Integer(3) / nm.Integer(2),
                                      name='\\tfrac{3}{2}x'))


def expert_gate() -> tuple[cat.Broadcasted, cat.Axis]:
    '''The gate as one named box, and this layer's sparse expert axis. The box runs from
    one token's hidden state to the six gates the experts are weighted by, which is what
    the reference's `Gate` module returns, so it holds `router` and `normalise_gates`
    together.'''
    picked, ke = router()
    return boxed(cat.Block.template(picked @ normalise_gates(ke),
                                    title=text.GATE_TITLE, fill_color=GATE_COLOUR,
        description=text.EXPERT_GATE_DESCRIPTION,
        references=(model_lines(822, 826),)),
                 GATE_BOX), ke


def down_projection(ke: cat.Axis) -> cat.BroadcastedCategory:
    '''`W^{D}` produces the expert axis the way the other two weights do, so its implicit
    weight tensor holds all 384 down-projections. The diagonalisation after it keeps the
    entries where the slot and the expert agree.'''
    return (over((ke,), ops.Linear.template((f,), (e, m), 'W^{D}'))
            @ ops.View.template(
                reindexing=cat.Rearrangement((0, 0, 1), (ke, m)),
                name='diag'))


def routed_experts(ke: cat.Axis) -> cat.Block:
    '''One expert whole: a SwiGLU and the down-projection with its diagonal.'''
    gate = ops.Linear.template((m,), (e, f), 'W^{G}') @ sigmoid_weighted_input()
    up = ops.Linear.template((m,), (e, f), 'W^{U}')
    return cat.Block.template(
        route((0, 0), (TOKEN_STATE,)) @ (gate * up) @ multiply_along(f)
        @ down_projection(ke),
        title=text.EXPERTS_TITLE, fill_color='#C1E8F7',
        description=text.ROUTED_EXPERTS_DESCRIPTION,
        references=(model_lines(846, 847),))


def shared_expert() -> cat.Block:
    '''The expert every token runs, at the same hidden width and with no router.'''
    return cat.Block.template(
        route((0, 0), (TOKEN_STATE,))
        @ ((ops.Linear.template((m,), (f,), 'W^{Gs}') @ sigmoid_weighted_input())
           * ops.Linear.template((m,), (f,), 'W^{Us}'))
        @ multiply_along(f)
        @ ops.Linear.template((f,), (m,), 'W^{Ds}'),
        title=text.SHARED_TITLE, fill_color='#DFF0D8',
        description=text.SHARED_EXPERT_DESCRIPTION,
        references=(model_lines(846, 847),))


def combine() -> cat.Broadcasted:
    '''The gates against what the experts produced: one einops dot product over the six
    live slots, which is the multiply and the sum together.'''
    return ops.Einops.template('k, k m -> m')


def mix_one_token() -> cat.Block:
    '''One token through the gate box, the six experts it chose, the shared expert and
    the residual.'''
    gate, ke = expert_gate()
    return cat.Block.template(
        route((0, 0, 0), (TOKEN_STATE,))
        @ (gate * routed_experts(ke) * shared_expert())
        @ (combine() * hold(TOKEN_STATE))
        @ ops.AdditionOp.template(),
        title=text.MIXTURE_TITLE, fill_color='#FFE2BB',
        description=text.MIX_ONE_TOKEN_DESCRIPTION,
        references=(model_lines(822, 826),))


MIXTURE_BODY = mix_one_token()


def mix_over_all_tokens() -> cat.BroadcastedCategory:
    '''The same mixture written out over every token, which the box is confirmed against.
    It lifts the body the box holds rather than building a second one, because
    `cat.Block.template` mints a fresh tag per call and the comparison reads the tags of
    the expert blocks.'''
    return over((x,), MIXTURE_BODY)


MIXTURE = discovering_broadcasts.broadcast_block_over_axes(
    MIXTURE_BODY, (x,), ((0,),), MIXTURE_BOX)
MIXTURE_CONFIRMATION = discovering_broadcasts.confirm_broadcast_expansion(
    mix_over_all_tokens(), MIXTURE)
