'''The forty layers of the integrated DeepSeek-V4.1-Flash, divided where Engram and
DSpark stand between two layers.

Written by Claude Fable 5.1, reasoning effort 80.

`notebooks.sota.DeepSeekV41Flash.layer_stack` states the forty layers as four repeated
blocks. Two mechanisms of the released model act between two layers of one such block,
so the block is divided at each of them:

    layer 0                 a sliding-window layer
    Engram of layer 1       added to the residual that enters layer 1
    layer 1                 a sliding-window layer
    layers 2 to 13          two encoder groups, a repeated block with the counter `l`
    Engram of layer 14      added to the residual that enters layer 14
    layers 14 to 19         the third encoder group, written alone
    layers 20 to 23         the decoder's Full layer and its three Reuse layers
    layers 24 to 35         three Reindex groups, a repeated block with the counter `l`
    layers 36 to 39         the fourth Reindex group, written alone, with the stream
                            mean of DSpark tapped ahead of each of its Reuse layers

A group written alone stands in no repeated block, so no counter names the member of
the tape it writes. The slots of its group carry a constant member in place of the
counter: the third encoder group writes and reads member 2 of the entries and of the
selection, and the fourth Reindex group writes and reads member 3 of the selection,
which are the members a third and a fourth iteration would have written.

The three tapped Reuse layers are a repeated block with a counter of their own,
`REUSE_COUNTER`, which runs from zero, because `dspark_drafter.grab_stream_means` reads
the members 0, 1 and 2 of the slot the tap writes.

Every layer is an attention sublayer and a mixture sublayer, each inside the
four-stream residual of `mhc_with_epsilons.mhc_sublayer`. The mixture sublayer is
`tempered_mixture.MIXTURE_SUBLAYER_BODY`, which grabs the modality of every token from
the slot the input of the model writes. `layer_count` reads the number of layers off the
blocks, each counted once per iteration of the repeated blocks around it, and the
validator checks that the divided stack runs forty.
'''
from __future__ import annotations

import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import para.data_structure.Para as Para
import para.data_structure.ParaWrap as para_wrap

from notebooks.sota.DeepSeekV41Flash.construction_idioms import hold, para_boxed
from notebooks.sota.DeepSeekV41Flash.declared_axes import (
    COLLAPSE, GROUP_COUNTER_NAME, SLOT_CKVe, SLOT_SELd, SLOT_SELe)
from notebooks.sota.DeepSeekV41Flash.reference_links import (
    inference_config_lines, model_lines, top_level_config_lines)
from notebooks.sota.DeepSeekV41Flash.dspark_draft_chain import (
    FIRST_TARGET_LAYER, TARGET_LAYERS)
from notebooks.sota.DeepSeekV41Flash.dspark_drafter import tap_stream_mean
from notebooks.sota.DeepSeekV41Flash.engram_modules import engram_of_layer
from notebooks.sota.DeepSeekV41Flash.integrated_attention_modes import (
    FULLD, FULLE, REINDEX, REUSED, REUSED_IN_GROUP, REUSEE, SWA,
    encoder_full_attention, encoder_reuse_attention, reindex_attention,
    reindex_group_reuse_attention)
from notebooks.sota.DeepSeekV41Flash.mhc_with_epsilons import mhc_sublayer
from notebooks.sota.DeepSeekV41Flash.tempered_mixture import (
    MIXTURE_SUBLAYER_BODY)
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

ENCODER_GROUPS = 3
ENCODER_REUSE = 5
DECODER_GROUPS = 4
DECODER_REUSE = 3
SWA_LAYERS = 2
SUBLAYERS_OF_ONE_LAYER = 2
LAYERS_OF_THE_RELEASED_MODEL = 40

REPEATED_ENCODER_GROUPS = ENCODER_GROUPS - 1
REPEATED_REINDEX_GROUPS = DECODER_GROUPS - 1
THIRD_ENCODER_GROUP = nm.Integer(REPEATED_ENCODER_GROUPS)
FOURTH_REINDEX_GROUP = nm.Integer(REPEATED_REINDEX_GROUPS)

FIRST_ENGRAM_LAYER = 1
SECOND_ENGRAM_LAYER = SWA_LAYERS + REPEATED_ENCODER_GROUPS * (1 + ENCODER_REUSE)

REUSE_COUNTER_NAME = 'r'
REUSE_COUNTER = nm.FreeNumeric.named(REUSE_COUNTER_NAME)

GROUP_COLOUR = '#EFEFF4'
REUSE_COLOUR = '#F3F3F4'

LAYER_KINDS_REFERENCE = inference_config_lines(27, 28)
COMPRESSION_RATIOS_REFERENCE = inference_config_lines(52)
ENGRAM_IN_THE_LAYER_LOOP = model_lines(1262, 1263)
TAP_IN_THE_LAYER_LOOP = model_lines(1264, 1266)

FULLE_OF_THE_THIRD_GROUP = para_boxed(
    encoder_full_attention(Para.LoopSlot(SLOT_CKVe, THIRD_ENCODER_GROUP),
                           Para.LoopSlot(SLOT_SELe, THIRD_ENCODER_GROUP)), 'Full')
REUSEE_OF_THE_THIRD_GROUP = para_boxed(
    encoder_reuse_attention(Para.LoopSlot(SLOT_CKVe, THIRD_ENCODER_GROUP),
                            Para.LoopSlot(SLOT_SELe, THIRD_ENCODER_GROUP)), 'Reu')
REINDEX_OF_THE_FOURTH_GROUP = para_boxed(
    reindex_attention(Para.LoopSlot(SLOT_SELd, FOURTH_REINDEX_GROUP)), 'Rex')
REUSED_OF_THE_FOURTH_GROUP = para_boxed(
    reindex_group_reuse_attention(Para.LoopSlot(SLOT_SELd, FOURTH_REINDEX_GROUP)),
    'Reu')


def layer(attention: cat.BroadcastedCategory) -> cat.BroadcastedCategory:
    '''One layer: an attention sublayer and a mixture sublayer, each inside the
    four-stream residual. The mixture reads the modality of every token from the
    tape.'''
    return mhc_sublayer(attention) @ mhc_sublayer(MIXTURE_SUBLAYER_BODY)


def engram_beside_the_collapse_vector(layer_number: int) -> cat.Morphism:
    '''The Engram of `layer_number` on the residual, with the collapse vector the layer
    before it predicted passed on unchanged.'''
    return hold(COLLAPSE) * engram_of_layer(layer_number)


def window_layer(layer_number: int, engram_sentence: str) -> cat.Block:
    return cat.Block.template(
        layer(SWA), title=text.WINDOW_LAYER_TITLE, fill_color=GROUP_COLOUR,
        description=text.WINDOW_LAYER_DESCRIPTION.format(
            layer_number=layer_number, engram_sentence=engram_sentence),
        references=(inference_config_lines(26), COMPRESSION_RATIOS_REFERENCE,
                    ENGRAM_IN_THE_LAYER_LOOP))


first_window_layer = window_layer(
    0, text.FIRST_WINDOW_LAYER_SENTENCE)

second_window_layer = window_layer(
    FIRST_ENGRAM_LAYER,
    text.SECOND_WINDOW_LAYER_SENTENCE)

encoder_reuse_block = cat.Block.template(
    layer(REUSEE), title=text.REUSE_LAYER_TITLE,
    description=text.ENCODER_REUSE_BLOCK_DESCRIPTION,
    repetition=ENCODER_REUSE, fill_color=REUSE_COLOUR,
    references=(LAYER_KINDS_REFERENCE, model_lines(1166, 1170)))

encoder_groups = cat.Block.template(
    layer(FULLE) @ encoder_reuse_block,
    title=text.ENCODER_GROUP_TITLE, repetition=REPEATED_ENCODER_GROUPS,
    fill_color=GROUP_COLOUR,
    description=text.ENCODER_GROUPS_DESCRIPTION,
    index_name=GROUP_COUNTER_NAME,
    references=(LAYER_KINDS_REFERENCE, COMPRESSION_RATIOS_REFERENCE,
                top_level_config_lines(106), top_level_config_lines(112)))

third_group_reuse_block = cat.Block.template(
    layer(REUSEE_OF_THE_THIRD_GROUP), title=text.REUSE_LAYER_TITLE,
    description=text.THIRD_GROUP_REUSE_BLOCK_DESCRIPTION,
    repetition=ENCODER_REUSE, fill_color=REUSE_COLOUR,
    references=(LAYER_KINDS_REFERENCE, model_lines(1166, 1170)))

third_encoder_group = cat.Block.template(
    layer(FULLE_OF_THE_THIRD_GROUP) @ third_group_reuse_block,
    title=text.THIRD_ENCODER_GROUP_TITLE, fill_color=GROUP_COLOUR,
    description=text.THIRD_ENCODER_GROUP_DESCRIPTION,
    references=(LAYER_KINDS_REFERENCE, inference_config_lines(43),
                ENGRAM_IN_THE_LAYER_LOOP))

decoder_reuse_block = cat.Block.template(
    layer(REUSED), title=text.REUSE_LAYER_TITLE,
    description=text.DECODER_REUSE_BLOCK_DESCRIPTION,
    repetition=DECODER_REUSE, fill_color=REUSE_COLOUR,
    references=(LAYER_KINDS_REFERENCE, model_lines(1166, 1170)))

reindex_reuse_block = cat.Block.template(
    layer(REUSED_IN_GROUP), title=text.REUSE_LAYER_TITLE,
    description=text.REINDEX_REUSE_BLOCK_DESCRIPTION,
    repetition=DECODER_REUSE, fill_color=REUSE_COLOUR,
    references=(LAYER_KINDS_REFERENCE, model_lines(1166, 1170)))

reindex_groups = cat.Block.template(
    layer(REINDEX) @ reindex_reuse_block,
    title=text.REINDEX_GROUP_TITLE, repetition=REPEATED_REINDEX_GROUPS,
    fill_color=GROUP_COLOUR,
    description=text.REINDEX_GROUPS_DESCRIPTION,
    index_name=GROUP_COUNTER_NAME,
    references=(inference_config_lines(28), COMPRESSION_RATIOS_REFERENCE,
                top_level_config_lines(106), top_level_config_lines(112)))

tapped_reuse_block = cat.Block.template(
    tap_stream_mean(REUSE_COUNTER) @ layer(REUSED_OF_THE_FOURTH_GROUP),
    title=text.TAPPED_REUSE_LAYER_TITLE,
    description=text.TAPPED_REUSE_BLOCK_DESCRIPTION.format(
        first_layer=FIRST_TARGET_LAYER,
        last_layer=FIRST_TARGET_LAYER + TARGET_LAYERS - 1),
    repetition=TARGET_LAYERS, index_name=REUSE_COUNTER_NAME, fill_color=REUSE_COLOUR,
    references=(TAP_IN_THE_LAYER_LOOP, inference_config_lines(9),
                LAYER_KINDS_REFERENCE))

fourth_reindex_group = cat.Block.template(
    layer(REINDEX_OF_THE_FOURTH_GROUP) @ tapped_reuse_block,
    title=text.FOURTH_REINDEX_GROUP_TITLE, fill_color=GROUP_COLOUR,
    description=text.FOURTH_REINDEX_GROUP_DESCRIPTION,
    references=(inference_config_lines(28), inference_config_lines(9),
                TAP_IN_THE_LAYER_LOOP))

window_layers_with_engram = (first_window_layer
                             @ engram_beside_the_collapse_vector(FIRST_ENGRAM_LAYER)
                             @ second_window_layer)

encoder = (encoder_groups
           @ engram_beside_the_collapse_vector(SECOND_ENGRAM_LAYER)
           @ third_encoder_group)

decoder = (layer(FULLD) @ decoder_reuse_block @ reindex_groups @ fourth_reindex_group)

divided_stack = window_layers_with_engram @ encoder @ decoder


class RepetitionIsNotAnInteger(ValueError):
    '''A repeated block whose repetition is a symbol, so its runs cannot be counted.'''


def runs_of_blocks_titled(title: str, term: object, repetitions: int = 1) -> int:
    '''How many times a block titled `title` runs in `term`. Each occurrence counts
    once for every iteration of the repeated blocks around it, so a block inside a
    block of repetition 5 inside a block of repetition 2 counts ten times.'''
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
    '''The number of layers `term` runs, read off its blocks: every layer holds two
    sublayers inside the four-stream residual, one for the attention and one for the
    mixture.'''
    return runs_of_blocks_titled(text.MHC_TITLE, term) // SUBLAYERS_OF_ONE_LAYER
