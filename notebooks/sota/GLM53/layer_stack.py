# Claude Opus 5.5 (1M context), effort 40.
'''The 78 layers of GLM-5.3, as repetition blocks over the boxed modes.

Every layer is `GlmMoeDsaDecoderLayer` of the reference: an attention sublayer and a
feed-forward sublayer, each inside a residual that normalises its input with an RMS
normalisation and adds its output back. `config.indexer_types` gives each layer an
attention mode and `config.mlp_layer_types` gives each layer a feed-forward map:

    layers 0 and 1      Full,   dense MLP
    layer 2             Full,   dense MLP, and its selection is read by layers 3 to 5
    layers 3 to 5       Shared, mixture of experts
    layers 6 + 4 l      Full,   mixture of experts, for l from 0 to 17
    layers 7 + 4 l to 9 + 4 l
                        Shared, mixture of experts, reading the selection of
                        layer 6 + 4 l

The counts add to 2 + (1 + 3) + 18 (1 + 3) = 78. The configuration writes the attention
modes as a list, generated in the reference from `index_topk_freq` of 4 and
`index_skip_topk_offset` of 3: layer `i` runs the Full mode when `max(i - 2, 0)` is a
multiple of 4.

The residual is the only wire between the layers. The selection shared by an IndexShare
group travels on the tape slot `sel`, per `attention_modes`, so every repeated block
returns its own domain. The repeated group carries the counter `l`, so each iteration
of the group writes a member of the slot and the Shared layers of the same iteration
read it.
'''
from __future__ import annotations

import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import para.data_structure.ParaWrap as para_wrap

from notebooks.sota.DeepSeekV41Flash.construction_idioms import hold, over, route
from notebooks.sota.GLM53.attention_modes import (
    FULL, FULL_PUBLISHING, FULL_PUBLISHING_IN_GROUP, SHARED, SHARED_IN_GROUP)
from notebooks.sota.GLM53.declared_axes import GROUP_COUNTER_NAME, STATE, m, x
from notebooks.sota.GLM53.feed_forward import DENSE_MLP
from notebooks.sota.GLM53.mixture_of_experts import MIXTURE
from notebooks.sota.GLM53.reference_links import (
    checkpoint_config_lines, configuration_lines, modeling_lines)
from notebooks.sota.GLM53.released_constants import NORM_EPSILON
from notebooks.sota.GLM53.block_titles_and_descriptions import TEXT as text

LAYERS_BEFORE_THE_FIRST_GROUP = 2
SHARED_LAYERS_PER_GROUP = 3
REPEATED_GROUPS = 18
RELEASED_LAYER_COUNT = 78

RESIDUAL_COLOUR = '#F1F4C1'
LAYER_COLOUR = '#EFEFF4'
SHARED_LAYER_COLOUR = '#F3F3F4'

LAYER_REFERENCES = (modeling_lines(586, 627),)
PLAN_REFERENCES = (configuration_lines(135, 156), checkpoint_config_lines(14),
                   checkpoint_config_lines(21, 23), checkpoint_config_lines(26, 105),
                   checkpoint_config_lines(110, 189), modeling_lines(722, 733))


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
        references=(modeling_lines(594, 595), modeling_lines(608, 626)))


def layer(attention: cat.BroadcastedCategory,
          feed_forward: cat.BroadcastedCategory) -> cat.BroadcastedCategory:
    '''One layer: an attention sublayer and a feed-forward sublayer, each inside the
    residual.'''
    return residual_sublayer(attention) @ residual_sublayer(feed_forward)


def shared_layers(attention: cat.BroadcastedCategory) -> cat.Block:
    return cat.Block.template(
        layer(attention, MIXTURE), title=text.SHARED_LAYER_TITLE,
        repetition=SHARED_LAYERS_PER_GROUP, fill_color=SHARED_LAYER_COLOUR,
        description=text.SHARED_LAYERS_DESCRIPTION, references=LAYER_REFERENCES)


layers_before_the_first_group = cat.Block.template(
    layer(FULL, DENSE_MLP), title=text.DENSE_LAYER_TITLE,
    repetition=LAYERS_BEFORE_THE_FIRST_GROUP, fill_color=LAYER_COLOUR,
    description=text.DENSE_LAYERS_DESCRIPTION, references=PLAN_REFERENCES)

first_group = cat.Block.template(
    layer(FULL_PUBLISHING, DENSE_MLP) @ shared_layers(SHARED),
    title=text.FIRST_GROUP_TITLE, fill_color=LAYER_COLOUR,
    description=text.FIRST_GROUP_DESCRIPTION, references=PLAN_REFERENCES)

repeated_groups = cat.Block.template(
    layer(FULL_PUBLISHING_IN_GROUP, MIXTURE) @ shared_layers(SHARED_IN_GROUP),
    title=text.GROUP_TITLE, repetition=REPEATED_GROUPS, fill_color=LAYER_COLOUR,
    description=text.REPEATED_GROUPS_DESCRIPTION, index_name=GROUP_COUNTER_NAME,
    references=PLAN_REFERENCES)

layer_stack = layers_before_the_first_group @ first_group @ repeated_groups


class RepetitionIsNotAnInteger(ValueError):
    '''A repeated block whose repetition is a symbol, so its runs cannot be counted.'''


def runs_of_blocks_titled(title: str, term: object, repetitions: int = 1) -> int:
    '''How many times a block titled `title` runs in `term`. Each occurrence counts
    once for every iteration of the repeated blocks around it, so a block inside a
    block of repetition 3 inside a block of repetition 18 counts 54 times.'''
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
        case para_wrap.ParaWrap(body=body):
            return runs_of_blocks_titled(title, body, repetitions)
        case cat.Broadcasted(operator=ops.BlockOperator(block=block)):
            return runs_of_blocks_titled(title, block, repetitions)
        case _:
            return 0


def layer_count(term: cat.Morphism) -> int:
    '''The number of layers `term` runs, read off its blocks: every layer holds one
    attention sublayer, in the Full mode or in the Shared mode.'''
    return (runs_of_blocks_titled(text.FULL_TITLE, term)
            + runs_of_blocks_titled(text.SHARED_TITLE, term))
