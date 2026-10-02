# Claude Opus 5.5 (1M context), effort 40.
'''The 70 layers of MiMo-V2.6-Pro, as repetition blocks over the boxed modes.

Every layer is `MiMoV2DecoderLayer` of the reference: an attention sublayer and a
feed-forward sublayer, each inside a residual that normalises its input with an RMS
normalisation and adds its output back. `hybrid_layer_pattern` gives each layer an
attention mode, 0 for full attention and 1 for a sliding window, and `moe_layer_freq`
gives each layer a feed-forward map, 0 for the dense MLP and 1 for the mixture of
experts. The full attention layers are 0, 7, 15, 23, 31, 39, 47, 55, 62 and 69, and
layer 0 alone holds the dense MLP:

    layer 0                     full attention, dense MLP
    layers 1 to 7               6 sliding window layers, then 1 full attention layer
    layers 8 to 55              6 groups of 7 sliding window layers and 1 full
                                attention layer
    layers 56 to 69             2 groups of 6 sliding window layers and 1 full
                                attention layer

Every layer after layer 0 holds the mixture of experts. The counts add to
1 + (6 + 1) + 6 (7 + 1) + 2 (6 + 1) = 70, of which 60 are sliding window layers.

The residual is the only wire between the layers, so every repeated block returns its
own domain.
'''
from __future__ import annotations

import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops

from notebooks.sota.DeepSeekV41Flash.construction_idioms import hold, over, route
from notebooks.sota.MiMoV26Pro.attention_modes import FULL, WINDOW
from notebooks.sota.MiMoV26Pro.declared_axes import STATE, m, x
from notebooks.sota.MiMoV26Pro.feed_forward import DENSE_MLP
from notebooks.sota.MiMoV26Pro.mixture_of_experts import MIXTURE
from notebooks.sota.MiMoV26Pro.reference_links import (
    checkpoint_config_lines, modeling_lines)
from notebooks.sota.MiMoV26Pro.released_constants import NORM_EPSILON
from notebooks.sota.MiMoV26Pro.block_titles_and_descriptions import TEXT as text

WINDOW_LAYERS_OF_THE_SHORT_GROUPS = 6
WINDOW_LAYERS_OF_THE_LONG_GROUPS = 7
LONG_GROUPS = 6
LAST_SHORT_GROUPS = 2
RELEASED_LAYER_COUNT = 70

RESIDUAL_COLOUR = '#F1F4C1'
LAYER_COLOUR = '#EFEFF4'
WINDOW_LAYER_COLOUR = '#F3F3F4'

LAYER_REFERENCES = (modeling_lines(394, 442),)
PLAN_REFERENCES = (checkpoint_config_lines(47, 118), checkpoint_config_lines(126, 197),
                   checkpoint_config_lines(204), modeling_lines(400, 409),
                   modeling_lines(1585, 1594))


def residual_sublayer(sublayer: cat.BroadcastedCategory) -> cat.Block:
    '''`STATE -> STATE`: the hidden state normalised, passed through `sublayer`, and
    added back to itself.'''
    return cat.Block.template(
        route((0, 0), (STATE,))
        @ ((over((x,), ops.Normalize.template((m,), epsilon=NORM_EPSILON)) @ sublayer)
           * hold(STATE))
        @ ops.AdditionOp.template(),
        title=text.RESIDUAL_TITLE, fill_color=RESIDUAL_COLOUR,
        description=text.RESIDUAL_DESCRIPTION,
        references=(modeling_lines(410, 411), modeling_lines(424, 441)))


def layer(attention: cat.BroadcastedCategory,
          feed_forward: cat.BroadcastedCategory) -> cat.BroadcastedCategory:
    '''One layer: an attention sublayer and a feed-forward sublayer, each inside the
    residual.'''
    return residual_sublayer(attention) @ residual_sublayer(feed_forward)


def window_layers(count: int) -> cat.Block:
    return cat.Block.template(
        layer(WINDOW, MIXTURE), title=text.WINDOW_LAYERS_TITLE, repetition=count,
        fill_color=WINDOW_LAYER_COLOUR, description=text.WINDOW_LAYERS_DESCRIPTION,
        references=LAYER_REFERENCES)


def hybrid_group(window_layer_count: int, repetition: int, title: str,
                 description: str) -> cat.Block:
    '''A run of sliding window layers followed by one full attention layer, repeated
    `repetition` times.'''
    return cat.Block.template(
        window_layers(window_layer_count) @ layer(FULL, MIXTURE),
        title=title, repetition=repetition, fill_color=LAYER_COLOUR,
        description=description, references=PLAN_REFERENCES)


first_layer = cat.Block.template(
    layer(FULL, DENSE_MLP), title=text.FIRST_LAYER_TITLE, fill_color=LAYER_COLOUR,
    description=text.FIRST_LAYER_DESCRIPTION, references=PLAN_REFERENCES)

first_group = hybrid_group(
    WINDOW_LAYERS_OF_THE_SHORT_GROUPS, 1, text.FIRST_GROUP_TITLE,
    text.FIRST_GROUP_DESCRIPTION)

long_groups = hybrid_group(
    WINDOW_LAYERS_OF_THE_LONG_GROUPS, LONG_GROUPS, text.LONG_GROUPS_TITLE,
    text.LONG_GROUPS_DESCRIPTION)

last_groups = hybrid_group(
    WINDOW_LAYERS_OF_THE_SHORT_GROUPS, LAST_SHORT_GROUPS, text.LAST_GROUPS_TITLE,
    text.LAST_GROUPS_DESCRIPTION)

layer_stack = first_layer @ first_group @ long_groups @ last_groups


class RepetitionIsNotAnInteger(ValueError):
    '''A repeated block whose repetition is a symbol, so its runs cannot be counted.'''


def runs_of_blocks_titled(title: str, term: object, repetitions: int = 1) -> int:
    '''How many times a block titled `title` runs in `term`. Each occurrence counts
    once for every iteration of the repeated blocks around it, so a block inside a
    block of repetition 7 inside a block of repetition 6 counts 42 times.'''
    match term:
        case cat.Composed(content=parts) | cat.ProductOfMorphisms(content=parts):
            return sum(runs_of_blocks_titled(title, part, repetitions)
                       for part in parts)
        case cat.Block(body=body, block_tag=block_tag):
            if not isinstance(block_tag.repetition, nm.Integer):
                raise RepetitionIsNotAnInteger(
                    f'the repetition {block_tag.repetition.to_latex()} is no integer')
            aesthetics = block_tag.aesthetics
            titled = aesthetics is not None and aesthetics.title == title
            inside = repetitions * block_tag.repetition._value
            return (inside if titled else 0) + runs_of_blocks_titled(
                title, body, inside)
        case cat.Broadcasted(operator=ops.BlockOperator(block=block)):
            return runs_of_blocks_titled(title, block, repetitions)
        case _:
            return 0


def layer_count(term: cat.Morphism) -> int:
    '''The number of layers `term` runs, read off its blocks: every layer holds one
    attention sublayer, of full attention or of a sliding window.'''
    return (runs_of_blocks_titled(text.FULL_TITLE, term)
            + runs_of_blocks_titled(text.WINDOW_TITLE, term))
