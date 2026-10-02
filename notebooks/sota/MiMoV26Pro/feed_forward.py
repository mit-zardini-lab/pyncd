# Claude Opus 5.5 (1M context), effort 40.
'''The SwiGLU feed-forward map of MiMo-V2.6-Pro, which is the dense MLP of layer 0 and
the body of every routed expert.

The map is `MiMoV2MLP` of the reference: a gate projection and an up projection of one
token's hidden state, the product of the SiLU of the gate with the up projection, and a
down projection back onto the hidden width. `moe_layer_freq` gives layer 0 this map at
the width `intermediate_size`, 16384, in place of a mixture of experts, and every routed
expert is the same map at the width `moe_intermediate_size`, 2048.

Every operation of the map is the same at every token, so the dense MLP is one box
computed once per token, built by `discovering_broadcasts.broadcast_block_over_axes`
from the body of one token and confirmed against the body lifted over the tokens.
'''
from __future__ import annotations

import algebra.discovering_broadcasts as discovering_broadcasts
import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Operators as ops

from notebooks.sota.DeepSeekV41Flash.construction_idioms import over, route
from notebooks.sota.DeepSeekV41Flash.custom_operations import (
    multiply_along, sigmoid_weighted_input)
from notebooks.sota.MiMoV26Pro.declared_axes import TOKEN_STATE, d, m, x
from notebooks.sota.MiMoV26Pro.reference_links import (
    checkpoint_config_lines, modeling_lines)
from notebooks.sota.MiMoV26Pro.block_titles_and_descriptions import TEXT as text

DENSE_COLOUR = '#DFF0D8'
DENSE_BOX = 'MLP'
FEED_FORWARD_REFERENCES = (modeling_lines(120, 132),)


def swiglu[A: cat.Axis](
    width: A,
    gate_name: str,
    up_name: str,
    down_name: str,
) -> cat.BroadcastedCategory:
    '''One token's hidden state through a gate projection and an up projection onto
    `width`, the product of the SiLU of the gate with the up projection, and the down
    projection back onto the hidden width.'''
    gate = ops.Linear.template((m,), (width,), gate_name) @ sigmoid_weighted_input()
    return (route((0, 0), (TOKEN_STATE,))
            @ (gate * ops.Linear.template((m,), (width,), up_name))
            @ multiply_along(width)
            @ ops.Linear.template((width,), (m,), down_name))


DENSE_BODY = cat.Block.template(
    swiglu(d, 'W^{Gd}', 'W^{Ud}', 'W^{Dd}'),
    title=text.DENSE_MLP_TITLE, fill_color=DENSE_COLOUR,
    description=text.DENSE_MLP_DESCRIPTION,
    references=(*FEED_FORWARD_REFERENCES, modeling_lines(405, 409),
                checkpoint_config_lines(121), checkpoint_config_lines(126, 197)))
DENSE_MLP = discovering_broadcasts.broadcast_block_over_axes(
    DENSE_BODY, (x,), ((0,),), DENSE_BOX)
DENSE_MLP_CONFIRMATION = discovering_broadcasts.confirm_broadcast_expansion(
    over((x,), DENSE_BODY), DENSE_MLP)
