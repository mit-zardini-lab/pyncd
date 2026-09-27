# Claude Opus 5.5 (1M context), effort 40.
'''The SwiGLU feed-forward map of GLM-5.3, which is the dense MLP of the first three
layers and the body of the shared expert.

The map is `GlmMoeDsaMLP` of the reference: a gate projection and an up projection of
one token's hidden state, the product of the SiLU of the gate with the up projection,
and a down projection back onto the hidden width. `config.mlp_layer_types` gives the
first three layers this map at the width `intermediate_size`, 12288, in place of a
mixture of experts. The shared expert of every later layer is the same map at the width
`moe_intermediate_size` times `n_shared_experts`, which is 2048.

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
from notebooks.sota.GLM53.declared_axes import TOKEN_STATE, g, m, x
from notebooks.sota.GLM53.reference_links import modeling_lines
from notebooks.sota.GLM53.block_titles_and_descriptions import TEXT as text

DENSE_COLOUR = '#DFF0D8'
DENSE_BOX = 'MLP'
FEED_FORWARD_REFERENCES = (modeling_lines(465, 478),)


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
    swiglu(g, 'W^{Gd}', 'W^{Ud}', 'W^{Dd}'),
    title=text.DENSE_MLP_TITLE, fill_color=DENSE_COLOUR,
    description=text.DENSE_MLP_DESCRIPTION,
    references=(*FEED_FORWARD_REFERENCES, modeling_lines(592)))
DENSE_MLP = discovering_broadcasts.broadcast_block_over_axes(
    DENSE_BODY, (x,), ((0,),), DENSE_BOX)
DENSE_MLP_CONFIRMATION = discovering_broadcasts.confirm_broadcast_expansion(
    over((x,), DENSE_BODY), DENSE_MLP)
