'''The layer plan of DeepSeek-V4.1-Flash, as repetition blocks over the boxed modes.

Written by Claude Opus 5, effort high.

Layers 0 and 1 are sliding window only. Layers 2 to 19 are the causal encoder at ratio
2, in three groups of a Full layer and five Reuse layers. Layers 20 to 39 are the
decoder at ratio 1, one Full layer with three Reuse layers below it and then four groups
of a Reindex layer and three Reuse layers.

The residual and the collapse vector are the only wires between layers, because every
value one layer publishes and another reads travels on a tape slot inside the boxes. A
repeated `cat.Block` therefore returns its own domain, which is the assertion the
notebook makes of every block here. A mode whose box holds grabs and drops is boxed by
`ParaBlockOperator`, which lists them on the operator. The sliding-window attention
touches no slot and is boxed by `ops.BlockOperator`, and the mixture arrives boxed from
`mixture_of_experts`, because its box is one computed once per token.

The two group blocks carry the name `GROUP_COUNTER_NAME` as the index of their loop,
because the slots their bodies write and read carry that counter. One iteration of an
encoder group writes the compressed entries and the selection its own Reuse layers read,
and one iteration of a Reindex group writes the selection its own Reuse layers read.
The decoder's Full layer stands outside both groups, so the three Reuse layers under it
grab what it wrote without an iteration, which is why the decoder has two Reuse boxes.
'''
from __future__ import annotations

import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat

from notebooks.sota.DeepSeekV41Flash.attention_modes import (
    decoder_full_attention, decoder_reuse_attention, encoder_full_attention,
    encoder_reuse_attention, reindex_attention, reindex_group_reuse_attention,
    window_attention)
from notebooks.sota.DeepSeekV41Flash.construction_idioms import boxed, para_boxed
from notebooks.sota.DeepSeekV41Flash.declared_axes import GROUP_COUNTER_NAME
from notebooks.sota.DeepSeekV41Flash.mixture_of_experts import MIXTURE
from notebooks.sota.DeepSeekV41Flash.single_pass_mhc import mhc_sublayer
from notebooks.sota.DeepSeekV41Flash.reference_links import (
    inference_config_lines, top_level_config_lines)
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

ENCODER_GROUPS = 3
ENCODER_REUSE = 5
DECODER_GROUPS = 4
DECODER_REUSE = 3
SWA_LAYERS = 2

SWA = boxed(window_attention(), 'SWA')
FULLE = para_boxed(encoder_full_attention(), 'Full')
FULLD = para_boxed(decoder_full_attention(), 'Full')
REINDEX = para_boxed(reindex_attention(), 'Rex')
REUSEE = para_boxed(encoder_reuse_attention(), 'Reu')
REUSED = para_boxed(decoder_reuse_attention(), 'Reu')
REUSED_IN_GROUP = para_boxed(reindex_group_reuse_attention(), 'Reu')


def layer(attention: cat.BroadcastedCategory) -> cat.BroadcastedCategory:
    '''One layer: an attention sublayer and a mixture sublayer, each inside the
    four-stream residual.'''
    return mhc_sublayer(attention) @ mhc_sublayer(MIXTURE)


swa_block = cat.Block.template(layer(SWA), title='\\text{Sliding Window Layer}',
                               description=text.SWA_BLOCK_DESCRIPTION,
                               repetition=SWA_LAYERS, fill_color='#EFEFF4',
                               references=(inference_config_lines(26, 28),))

encoder_reuse_block = cat.Block.template(
    layer(REUSEE), title='\\text{Reuse Layer}',
    description=text.ENCODER_REUSE_BLOCK_DESCRIPTION,
    repetition=ENCODER_REUSE, fill_color='#F3F3F4',
    references=(inference_config_lines(36, 41),))

encoder_group = cat.Block.template(
    layer(FULLE) @ encoder_reuse_block,
    title='\\text{Encoder Group}', repetition=ENCODER_GROUPS, fill_color='#EFEFF4',
    description=text.ENCODER_GROUP_DESCRIPTION,
    index_name=GROUP_COUNTER_NAME,
    references=(inference_config_lines(36, 41), top_level_config_lines(106),
                top_level_config_lines(112)))

decoder_reuse_block = cat.Block.template(
    layer(REUSED), title='\\text{Reuse Layer}',
    description=text.DECODER_REUSE_BLOCK_DESCRIPTION,
    repetition=DECODER_REUSE, fill_color='#F3F3F4',
    references=(inference_config_lines(36, 41),))

reindex_reuse_block = cat.Block.template(
    layer(REUSED_IN_GROUP), title='\\text{Reuse Layer}',
    description=text.REINDEX_REUSE_BLOCK_DESCRIPTION,
    repetition=DECODER_REUSE, fill_color='#F3F3F4',
    references=(inference_config_lines(36, 41),))

reindex_group = cat.Block.template(
    layer(REINDEX) @ reindex_reuse_block,
    title='\\text{Reindex Group}', repetition=DECODER_GROUPS, fill_color='#EFEFF4',
    description=text.REINDEX_GROUP_DESCRIPTION,
    index_name=GROUP_COUNTER_NAME,
    references=(inference_config_lines(52), top_level_config_lines(106),
                top_level_config_lines(112)))

decoder = layer(FULLD) @ decoder_reuse_block @ reindex_group
