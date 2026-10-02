# Claude Opus 5.5 (1M context), effort 40.
'''The 93 layers of Kimi K3, as repeated blocks over the two kinds of attention and the
blocks of the attention residuals.

Every layer is `KimiDecoderLayer` of the reference: an attention sublayer and a
feed-forward sublayer. Each sublayer reads the mix of the attention residuals, normalises
it with an RMS normalisation, and adds its output to the entry of its block, per
`attention_residuals`. `config.is_kda_layer` gives layer `i` the delta attention when
`i + 1` is listed in `kda_layers`, and the latent attention otherwise, and
`first_k_dense_replace` of 1 gives layer 0 the dense MLP and every later layer the
mixture of experts:

    layer 0                   delta attention, dense MLP, reads the embedding directly
    layers 1, 2               delta attention, mixture of experts
    layer 3 + 4 i             latent attention, mixture of experts, for i from 0 to 22
    layers 4 + 4 i to 6 + 4 i delta attention, mixture of experts, for i from 0 to 21
    layer 92                  latent attention, mixture of experts

The configuration lists 69 delta layers and 24 latent layers, the last two latent layers
standing next to each other. The blocks of the attention residuals hold twelve layers,
which are three groups of four, so the plan is written as:

    first block, entry 1      layer 0, two delta layers, one latent layer, and two
                              groups of three delta layers and one latent layer
    six blocks, entry o + 2   three groups of four, for the counter o from 0 to 5
    last block, entry 8       two groups of four and one latent layer

The counts add to 12 + 6 x 12 + 9 = 93 layers, of which 3 + 6 + 6 x 9 + 6 = 69 are
delta layers.
'''
from __future__ import annotations

import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import para.data_structure.ParaWrap as para_wrap

from notebooks.sota.DeepSeekV41Flash.construction_idioms import hold, over, route
from notebooks.sota.KimiK3.attention_residuals import (
    ATTENTION_INPUT_WEIGHT, FEED_FORWARD_INPUT_WEIGHT, add_at_entry, mix_of,
    read_the_embedding)
from notebooks.sota.KimiK3.declared_axes import (
    BLOCK_COUNTER, BLOCK_COUNTER_NAME, RESIDUAL_ENTRIES, m, x)
from notebooks.sota.KimiK3.delta_attention import DELTA_ATTENTION
from notebooks.sota.KimiK3.feed_forward import DENSE_MLP
from notebooks.sota.KimiK3.latent_attention import GATED_LATENT_ATTENTION
from notebooks.sota.KimiK3.latent_mixture_of_experts import MIXTURE
from notebooks.sota.KimiK3.reference_links import (
    checkpoint_config_lines, configuration_lines, modeling_lines)
from notebooks.sota.KimiK3.released_constants import NORM_EPSILON
from notebooks.sota.KimiK3.block_titles_and_descriptions import TEXT as text

DELTA_LAYERS_BEFORE_A_LATENT_LAYER = 3
GROUPS_PER_BLOCK = 3
GROUPS_IN_THE_FIRST_BLOCK_AFTER_ITS_FIRST_GROUP = 2
GROUPS_IN_THE_LAST_BLOCK = 2
REPEATED_BLOCKS = 6
RELEASED_LAYER_COUNT = 93
FIRST_BLOCK_ENTRY = nm.Integer(1)
LAST_BLOCK_ENTRY = nm.Integer(8)
REPEATED_BLOCK_ENTRY = BLOCK_COUNTER + nm.Integer(2)

RESIDUAL_COLOUR = '#F1F4C1'
LAYER_COLOUR = '#EFEFF4'
GROUP_COLOUR = '#F3F3F4'

LAYER_REFERENCES = (modeling_lines(877, 1046),)
PLAN_REFERENCES = (configuration_lines(152, 156), checkpoint_config_lines(26),
                   checkpoint_config_lines(46), checkpoint_config_lines(66, 165),
                   checkpoint_config_lines(187), modeling_lines(883, 900))


def normalise() -> cat.BroadcastedCategory:
    return over((x,), ops.Normalize.template((m,), epsilon=NORM_EPSILON))


def residual_sublayer(
    sublayer: cat.BroadcastedCategory,
    entry: nm.Numeric,
    input_reading: cat.BroadcastedCategory,
) -> cat.Block:
    '''`R[x, b, m] -> R[x, b, m]`: the input of `sublayer` read off the entries by
    `input_reading`, normalised, passed through `sublayer`, and added at `entry`.'''
    return cat.Block.template(
        route((0, 0), (RESIDUAL_ENTRIES,))
        @ (hold(RESIDUAL_ENTRIES) * (input_reading @ normalise() @ sublayer))
        @ add_at_entry(entry),
        title=text.RESIDUAL_TITLE, fill_color=RESIDUAL_COLOUR,
        description=text.RESIDUAL_DESCRIPTION,
        references=(modeling_lines(984, 1044), modeling_lines(901, 917)))


def layer(attention: cat.BroadcastedCategory,
          feed_forward: cat.BroadcastedCategory,
          entry: nm.Numeric) -> cat.BroadcastedCategory:
    '''One layer: the attention and the feed-forward map, each reading its own mix of
    the entries and adding its output at `entry`.'''
    return (residual_sublayer(attention, entry, mix_of(ATTENTION_INPUT_WEIGHT))
            @ residual_sublayer(feed_forward, entry, mix_of(FEED_FORWARD_INPUT_WEIGHT)))


def delta_layers(entry: nm.Numeric, repetition: int) -> cat.Block:
    return cat.Block.template(
        layer(DELTA_ATTENTION, MIXTURE, entry), title=text.DELTA_LAYER_TITLE,
        repetition=repetition, fill_color=LAYER_COLOUR,
        description=text.DELTA_LAYERS_DESCRIPTION, references=LAYER_REFERENCES)


def latent_layer(entry: nm.Numeric) -> cat.Block:
    return cat.Block.template(
        layer(GATED_LATENT_ATTENTION, MIXTURE, entry), title=text.LATENT_LAYER_TITLE,
        fill_color=LAYER_COLOUR, description=text.LATENT_LAYER_DESCRIPTION,
        references=LAYER_REFERENCES)


def groups_of_four(entry: nm.Numeric, repetition: int) -> cat.Block:
    '''`repetition` groups of three delta layers and one latent layer.'''
    return cat.Block.template(
        delta_layers(entry, DELTA_LAYERS_BEFORE_A_LATENT_LAYER) @ latent_layer(entry),
        title=text.GROUP_TITLE, repetition=repetition, fill_color=GROUP_COLOUR,
        description=text.GROUP_DESCRIPTION, references=PLAN_REFERENCES)


def first_layer() -> cat.Block:
    '''Layer 0: the delta attention reads the embedding directly, and the dense MLP
    reads the mix of the embedding and the output of the attention.'''
    return cat.Block.template(
        residual_sublayer(DELTA_ATTENTION, FIRST_BLOCK_ENTRY, read_the_embedding())
        @ residual_sublayer(DENSE_MLP, FIRST_BLOCK_ENTRY,
                            mix_of(FEED_FORWARD_INPUT_WEIGHT)),
        title=text.FIRST_LAYER_TITLE, fill_color=LAYER_COLOUR,
        description=text.FIRST_LAYER_DESCRIPTION,
        references=(*LAYER_REFERENCES, modeling_lines(987, 998)))


first_block = cat.Block.template(
    first_layer()
    @ delta_layers(FIRST_BLOCK_ENTRY, DELTA_LAYERS_BEFORE_A_LATENT_LAYER - 1)
    @ latent_layer(FIRST_BLOCK_ENTRY)
    @ groups_of_four(FIRST_BLOCK_ENTRY, GROUPS_IN_THE_FIRST_BLOCK_AFTER_ITS_FIRST_GROUP),
    title=text.FIRST_BLOCK_TITLE, fill_color=LAYER_COLOUR,
    description=text.FIRST_BLOCK_DESCRIPTION, references=PLAN_REFERENCES)

repeated_blocks = cat.Block.template(
    groups_of_four(REPEATED_BLOCK_ENTRY, GROUPS_PER_BLOCK),
    title=text.REPEATED_BLOCKS_TITLE, repetition=REPEATED_BLOCKS,
    fill_color=LAYER_COLOUR, description=text.REPEATED_BLOCKS_DESCRIPTION,
    index_name=BLOCK_COUNTER_NAME, references=PLAN_REFERENCES)

last_block = cat.Block.template(
    groups_of_four(LAST_BLOCK_ENTRY, GROUPS_IN_THE_LAST_BLOCK)
    @ latent_layer(LAST_BLOCK_ENTRY),
    title=text.LAST_BLOCK_TITLE, fill_color=LAYER_COLOUR,
    description=text.LAST_BLOCK_DESCRIPTION, references=PLAN_REFERENCES)

layer_stack = first_block @ repeated_blocks @ last_block


class RepetitionIsNotAnInteger(ValueError):
    '''A repeated block whose repetition is a symbol, so its runs cannot be counted.'''


def runs_of_boxes_named(short_name: str, term: object, repetitions: int = 1) -> int:
    '''How many times a box named `short_name` runs in `term`. Each occurrence counts
    once for every iteration of the repeated blocks around it, and a repeated block
    whose repetition is a symbol, which only the scan over the tokens is, is not
    entered.'''
    match term:
        case cat.Composed(content=parts) | cat.ProductOfMorphisms(content=parts):
            return sum(runs_of_boxes_named(short_name, part, repetitions)
                       for part in parts)
        case cat.Block(body=body, block_tag=block_tag):
            if not isinstance(block_tag.repetition, nm.Integer):
                return 0
            return runs_of_boxes_named(
                short_name, body, repetitions * block_tag.repetition._value)
        case para_wrap.ParaWrap(body=body):
            return runs_of_boxes_named(short_name, body, repetitions)
        case cat.Broadcasted(operator=ops.BlockOperator(block=block, name=name)):
            named = name is not None and name.to_bodies() == short_name
            return (repetitions if named else 0) + runs_of_boxes_named(
                short_name, block, repetitions)
        case _:
            return 0


def delta_layer_count(term: cat.Morphism) -> int:
    return runs_of_boxes_named('KDA', term)


def latent_layer_count(term: cat.Morphism) -> int:
    return runs_of_boxes_named('MLA', term)
