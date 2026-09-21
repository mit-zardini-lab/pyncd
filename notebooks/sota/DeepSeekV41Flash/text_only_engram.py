# Claude Opus 5 (1M context), effort high.
'''Engram of the text-only DeepSeek-V4.1-Flash, which reads the token identifiers and
nothing else.

`notebooks.sota.DeepSeekV41Flash.omitted_mechanisms.engram_expression` is the whole
mechanism with its two operands open, from the token map through the written-out hash,
the table and the two projections to the gate. This module wraps it in a box that grabs
the token identifiers from `SLOT_IDS` and grabs nothing else.

`notebooks.sota.DeepSeekV41Flash.engram_modules` states Engram as the released
multimodal model computes it. It differs here in one operation: the released code sets
the gate of an image token to zero by multiplying the gate by one minus the modality of
the token, and a model that reads text alone holds no image token.
'''
from __future__ import annotations

import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import para.data_structure.Para as Para
import para.data_structure.ParaBlockOperator as ParaBlockOperator

from notebooks.sota.DeepSeekV41Flash.construction_idioms import hold, para_boxed
from notebooks.sota.DeepSeekV41Flash.declared_axes import COLLAPSE, X
from notebooks.sota.DeepSeekV41Flash.omitted_mechanisms import (
    ENGRAM_COLOUR, TOKEN_IDS, engram_expression, named_for_layer)
from notebooks.sota.DeepSeekV41Flash.reference_links import (
    inference_config_lines, model_lines)
from notebooks.sota.DeepSeekV41Flash.engram_modules import ENGRAM_BOX
from notebooks.sota.DeepSeekV41Flash.integrated_axes_and_slots import SLOT_IDS
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text


def engram_block(layer: int) -> cat.Block:
    '''The module of `layer` as one titled block from the residual to the residual.
    The grab stands at the top level of the block, where
    `ParaBlockOperator.expose_tape_as_ports` looks for it.'''
    return cat.Block.template(
        (hold(X) * Para.Grab(tape=SLOT_IDS, size=TOKEN_IDS))
        @ engram_expression(layer),
        title=f'\\text{{Engram of Layer {layer}}}', fill_color=ENGRAM_COLOUR,
        formula=('X\'[i_{x}, i_{n}, i_{m}] = X[i_{x}, i_{n}, i_{m}] '
                 '+ g[i_{x}, i_{n}]\\, v[i_{x}, i_{m}]'),
        description=text.TEXT_ONLY_ENGRAM_BLOCK_DESCRIPTION.format(layer=layer),
        references=(model_lines(328, 365), model_lines(1249, 1252),
                    model_lines(1262, 1263), inference_config_lines(43, 48),
                    inference_config_lines(63, 64)))


def engram_of_layer(layer: int) -> ParaBlockOperator.WrappedBox:
    '''The module of `layer` as a box from the four-stream residual to the four-stream
    residual, which stands between two layer blocks beside the collapse vector. The
    box records the one slot it grabs.'''
    return para_boxed(engram_block(layer), named_for_layer(ENGRAM_BOX, layer))


def engram_beside_the_collapse_vector(layer: int) -> cat.Morphism:
    '''The Engram of `layer` on the residual, with the collapse vector the layer
    before it predicted passed on unchanged.'''
    return hold(COLLAPSE) * engram_of_layer(layer)
