# Claude Opus 5 (1M context), effort high.
'''The integrated DeepSeek-V4.1-Flash restricted to text, from the token identifiers of
a prompt to the probabilities of the next token.

`integrated_whole_model.v41_flash_integrated` is the released model whole: it reads an
image beside the prompt and it drafts five tokens with DSpark after the backbone has
produced one. This module is that model with the image pathway and DSpark taken out,
which leaves the algorithm a prompt of text goes through and nothing else. It reads one
array and returns one:

    the token identifiers   Nat(v)[x]
    the probabilities       R[x, v]

Every mechanism of the integrated model that acts on text is kept, and each is stated
as that model states it: the rotary embedding and YaRN, the score scales, the pinned
block, the FP4 and FP8 round trips of the caches, the SwiGLU clamps, the router
temperature, the epsilons of the hyper-connections and of the gates, and Engram before
layers 1 and 14.

Three things change, and each follows from there being no image:

    the inputs      the modality of every token is zero, so the model reads no
                    modality and writes no slot for one. The identifiers are copied,
                    one copy is embedded and `INPUTS` drops the other on `SLOT_IDS`,
                    which Engram grabs
    the mixture     the router reaches one of its two correction biases at every
                    token, so `text_only_mixture` holds the bias of a text token alone
    Engram          the hash reads no modality and the gate has no factor for one, in
                    `text_only_engram`

The layer plan is that of `divided_layer_stack` with the taps of DSpark removed, and
this module imports that module's counts, titles, colours and released lines so that
the two plans are described in one place. The encoder groups are still divided at the
third, because Engram is applied to the residual before layer 14 and a step that runs
before one group of three cannot stand inside the repeated block. The Reindex groups
are one repeated block of four, where the integrated model writes the fourth alone so
that its three Reuse layers can drop the stream mean DSpark reads. The layer count is
unchanged at forty.

`RELEASED_SIZES` is the mapping of `integrated_whole_model` without the sizes of the
image pathway and of DSpark, because no axis of either is in this model, and
`released_assigned_sizes` binds them on a term as that module's function does.
'''
from __future__ import annotations

import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Term as fd
import para.data_structure.Para as Para
import term_utilities.generate_config as gc

from notebooks.sota.DeepSeekV41Flash.construction_idioms import (
    hold, para_boxed, route)
from notebooks.sota.DeepSeekV41Flash.declared_axes import (
    GROUP_COUNTER_NAME, SLOT_CKVe, SLOT_SELe, X)
from notebooks.sota.DeepSeekV41Flash.omitted_mechanisms import TOKEN_IDS
from notebooks.sota.DeepSeekV41Flash.reference_links import (
    inference_config_lines, model_lines, top_level_config_lines)
from notebooks.sota.DeepSeekV41Flash.single_pass_mhc import collapse_streams
from notebooks.sota.DeepSeekV41Flash.whole_model import (
    expand_into_streams, initial_collapse, unembed)
from notebooks.sota.DeepSeekV41Flash.divided_layer_stack import (
    COMPRESSION_RATIOS_REFERENCE, DECODER_GROUPS, DECODER_REUSE, ENCODER_REUSE, ENGRAM_IN_THE_LAYER_LOOP, FIRST_ENGRAM_LAYER, GROUP_COLOUR, LAYER_KINDS_REFERENCE, REPEATED_ENCODER_GROUPS, REUSE_COLOUR, SECOND_ENGRAM_LAYER, THIRD_ENCODER_GROUP)
from notebooks.sota.DeepSeekV41Flash.integrated_attention_modes import (
    FULLD, FULLE, REINDEX, REUSED, REUSED_IN_GROUP, REUSEE, SWA,
    encoder_full_attention, encoder_reuse_attention)
from notebooks.sota.DeepSeekV41Flash.integrated_axes_and_slots import SLOT_IDS
from notebooks.sota.DeepSeekV41Flash.integrated_whole_model import (
    BASE_MODEL_SIZES, CACHE_GROUP_SIZES, ENGRAM_SIZES, INPUTS_BOX, INPUTS_COLOUR, ROTARY_SIZES, VOCABULARY_SIZES, embed_token_identifiers)
from notebooks.sota.DeepSeekV41Flash.mhc_with_epsilons import mhc_sublayer
from notebooks.sota.DeepSeekV41Flash.text_only_engram import (
    engram_beside_the_collapse_vector)
from notebooks.sota.DeepSeekV41Flash.text_only_mixture import MIXTURE
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

MODEL_INPUTS = (TOKEN_IDS,)

RELEASED_SIZES: dict[str, int] = {
    **BASE_MODEL_SIZES, **VOCABULARY_SIZES, **ROTARY_SIZES, **CACHE_GROUP_SIZES,
    **ENGRAM_SIZES}

FULLE_OF_THE_THIRD_GROUP = para_boxed(
    encoder_full_attention(Para.LoopSlot(SLOT_CKVe, THIRD_ENCODER_GROUP),
                           Para.LoopSlot(SLOT_SELe, THIRD_ENCODER_GROUP)), 'Full')
REUSEE_OF_THE_THIRD_GROUP = para_boxed(
    encoder_reuse_attention(Para.LoopSlot(SLOT_CKVe, THIRD_ENCODER_GROUP),
                            Para.LoopSlot(SLOT_SELe, THIRD_ENCODER_GROUP)), 'Reu')


def publish_identifiers() -> cat.Block:
    '''The token identifiers dropped onto their slot. The drop stands at the top level
    of the block, where `para_boxed` looks for it, and the block returns nothing on a
    wire.'''
    return cat.Block.template(
        Para.Drop(tape=SLOT_IDS, size=TOKEN_IDS),
        title=text.INPUTS_TITLE, fill_color=INPUTS_COLOUR,
        description=text.PUBLISH_IDENTIFIERS_DESCRIPTION,
        references=(model_lines(1241, 1243), model_lines(1249, 1252),
                    model_lines(1267)))


INPUTS = para_boxed(publish_identifiers(), INPUTS_BOX)


def read_inputs() -> cat.Morphism:
    '''From the token identifiers to the embedded tokens. The identifiers are copied,
    one copy is embedded and the other goes to `INPUTS`.'''
    return route((0, 0), MODEL_INPUTS) @ (embed_token_identifiers() * INPUTS)


def layer(attention: cat.BroadcastedCategory) -> cat.BroadcastedCategory:
    '''One layer: an attention sublayer and a mixture sublayer, each inside the
    four-stream residual. Neither sublayer reads a tape slot of its own.'''
    return mhc_sublayer(attention) @ mhc_sublayer(MIXTURE)


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
    title=text.REINDEX_GROUP_TITLE, repetition=DECODER_GROUPS, fill_color=GROUP_COLOUR,
    description=text.TEXT_ONLY_REINDEX_GROUPS_DESCRIPTION,
    index_name=GROUP_COUNTER_NAME,
    references=(inference_config_lines(28), COMPRESSION_RATIOS_REFERENCE,
                top_level_config_lines(106), top_level_config_lines(112)))

window_layers_with_engram = (
    first_window_layer
    @ engram_beside_the_collapse_vector(FIRST_ENGRAM_LAYER)
    @ second_window_layer)

encoder = (encoder_groups
           @ engram_beside_the_collapse_vector(SECOND_ENGRAM_LAYER)
           @ third_encoder_group)

decoder = layer(FULLD) @ decoder_reuse_block @ reindex_groups

text_only_stack = window_layers_with_engram @ encoder @ decoder

v41_flash_text_only = (
    read_inputs()
    @ expand_into_streams()
    @ (initial_collapse() * hold(X))
    @ text_only_stack
    @ collapse_streams()
    @ unembed())


def released_assigned_sizes(
    term: fd.GeneralTerm = v41_flash_text_only,
) -> dict[str, int]:
    '''The released size of every symbol of `term` that `RELEASED_SIZES` names, keyed
    by the bodies of the symbol's name. A figure passes the mapping as
    `assigned_sizes`, and the display writes each size onto the label of its axis while
    the term stays symbolic.'''
    config = gc.NumericConfig.template(term)
    config.assign_values(**RELEASED_SIZES)
    return config.assigned_integers_by_name()
