'''The integrated DeepSeek-V4.1-Flash end to end, from the inputs of a prompt that holds
one image to the probabilities and the five drafts of DSpark.

Written by Claude Fable 5.1, reasoning effort 80.

The model reads six arrays, in this order:

    the token identifiers             Nat(v)[x]
    the modality of every token       Nat(mu)[x], zero for text and one for a token of
                                      an image
    the token position of every cell  Nat(x)[H', W']
    the patches of the image          R[H, W, F]
    the position of every delimiter   Nat(x)[Delta]
    the kind of every delimiter       Nat(delta)[Delta]

It returns the probabilities of the next token at every position, `R[x, v]`, the five
draft tokens of DSpark and their five confidences.

The token identifiers and the modality are read again inside the model. Engram hashes
the identifiers before layers 1 and 14, the router of every layer chooses its correction
bias by the modality, and Engram sets its gate to zero at an image token. A sublayer
reads the hidden state alone, so the two arrays travel on the tape. The identifiers are
copied once, one copy goes to the embedding, and `INPUTS` is the box that drops the other
copy and the modality onto `SLOT_IDS` and `SLOT_MODALITY`. It is boxed by `para_boxed`,
so the two slots stand at the ports of the box, as the slots of an attention mode do.
The box returns nothing on a wire. A box that also passed the identifiers through could
not be built, because `construction_idioms.recycled_block` lifts a wire that passes
through a block out of the block.

The expansion into the four streams, the initial collapse vector, the collapse of the
streams and the output head are those of `notebooks.sota.DeepSeekV41Flash.whole_model`.
The embedding is written here, on the datatype of the identifiers the slot holds. The
image pathway stands between the embedding and the expansion, where the released
`merge_image_embeddings` runs.

`RELEASED_SIZES` holds the size the released configuration gives every axis and every
selection count, by the body of the symbol's name, and `released_assigned_sizes` binds
them on a term and returns the mapping a figure writes onto its labels. The token axis
and the patch grid of the image have no released size and stay symbolic. The two Engram
tables have row counts of their own, keyed `T1` and `T14`. The combined-stream
axis `N` is aligned with the stream axis `n` when the coefficients are composed into a
sublayer, so it takes the size of `n` and has no entry. A named constant is never
assigned, because the figures show the formulas with the constants in them.
`ALL_RELEASED_CONSTANTS` lists the released value of each constant with the line that
sets it.
'''
from __future__ import annotations

import functools
import operator

import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import data_structure.Operators as ops
import data_structure.Term as fd
import para.data_structure.Para as Para
import term_utilities.generate_config as gc

import notebooks.sota.DeepSeekV41Flash.dspark_draft_chain as dspark_draft_chain
import notebooks.sota.DeepSeekV41Flash.dspark_drafter as dspark_drafter
import notebooks.sota.DeepSeekV41Flash.engram_modules as engram_modules
import notebooks.sota.DeepSeekV41Flash.gumbel_max_sampler as gumbel_max_sampler
import notebooks.sota.DeepSeekV41Flash.omitted_mechanisms as omitted_mechanisms
from notebooks.sota.DeepSeekV41Flash.construction_idioms import (
    hold, over, para_boxed, route)
from notebooks.sota.DeepSeekV41Flash.declared_axes import X, m, x
from notebooks.sota.DeepSeekV41Flash.omitted_mechanisms import (
    IMAGE_POSITIONS, PATCHES, TOKEN_IDS)
from notebooks.sota.DeepSeekV41Flash.reference_links import (
    inference_config_lines, model_lines)
from notebooks.sota.DeepSeekV41Flash.single_pass_mhc import collapse_streams
from notebooks.sota.DeepSeekV41Flash.whole_model import (
    expand_into_streams, initial_collapse, unembed)
from notebooks.sota.DeepSeekV41Flash.divided_layer_stack import divided_stack
from notebooks.sota.DeepSeekV41Flash.integrated_axes_and_slots import (
    RELEASED_CONSTANTS, SLOT_IDS, SLOT_MODALITY, ReleasedConstant)
from notebooks.sota.DeepSeekV41Flash.tempered_mixture import (
    MODALITY_OF_EVERY_TOKEN)
from notebooks.sota.DeepSeekV41Flash.vision_pathway import (
    DELIMITER_KINDS, DELIMITER_POSITIONS, image_pathway_with_delimiters)
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

INPUTS_BOX = 'In'
INPUTS_COLOUR = '#FCE0E1'

IMAGE_OPERANDS = (IMAGE_POSITIONS, PATCHES, DELIMITER_POSITIONS, DELIMITER_KINDS)
MODEL_INPUTS = (TOKEN_IDS, MODALITY_OF_EVERY_TOKEN, *IMAGE_OPERANDS)

BASE_MODEL_SIZES: dict[str, int] = {
    'm': 5120, 'h': 64, 'c': 512, 'q': 1280, 'o': 1024, 'g': 8, 'j': 8, 'w': 128,
    'a': 2, 'u': 8, 'i': 32, 'd': 128, 'e': 384, 'f': 2304, 'n': 4, 'k': 6, 's': 512,
    'p': 2048, 'C': 16384}
VOCABULARY_SIZES: dict[str, int] = {'v': 129280}
ROTARY_SIZES: dict[str, int] = {'t': 32}
CACHE_GROUP_SIZES: dict[str, int] = {
    'E': 32, 'y': 16, '\\hat{y}': 32, '\\hat{E}': 4, '\\tilde{E}': 16}
ENGRAM_SIZES: dict[str, int] = {
    'L': 4, 'G': 3, 'K': 8, 'D': 256, '\\hat{v}': 99_092, '\\mu': 2,
    **{omitted_mechanisms.table_row_count(layer).uid._name.to_bodies(): rows
       for layer, rows in omitted_mechanisms.TABLE_ROWS_OF_LAYER.items()}}
VISION_SIZES: dict[str, int] = {'U': 3, 'F': 588, 'M': 1024, '\\delta': 3}

RELEASED_SIZES: dict[str, int] = {
    **BASE_MODEL_SIZES, **VOCABULARY_SIZES, **ROTARY_SIZES, **CACHE_GROUP_SIZES,
    **ENGRAM_SIZES, **VISION_SIZES, **dspark_draft_chain.RELEASED_SIZES}

ALL_RELEASED_CONSTANTS: tuple[ReleasedConstant, ...] = (
    *RELEASED_CONSTANTS, *gumbel_max_sampler.SAMPLER_RELEASED_CONSTANTS)


def publish_identifiers_and_modality() -> cat.Block:
    '''The token identifiers and the modality of every token, each dropped onto its
    slot. The two drops stand at the top level of the block, where `para_boxed` looks
    for them, and the block returns nothing on a wire.'''
    return cat.Block.template(
        Para.Drop(tape=SLOT_IDS, size=TOKEN_IDS)
        * Para.Drop(tape=SLOT_MODALITY, size=MODALITY_OF_EVERY_TOKEN),
        title=text.INPUTS_TITLE, fill_color=INPUTS_COLOUR,
        description=text.PUBLISH_IDENTIFIERS_AND_MODALITY_DESCRIPTION,
        references=(model_lines(1241, 1243), model_lines(1249, 1252),
                    model_lines(1267)))


INPUTS = para_boxed(publish_identifiers_and_modality(), INPUTS_BOX)


def embed_token_identifiers() -> cat.Block:
    '''The embedding of `whole_model.embed` over every token, on the datatype of
    `TOKEN_IDS`. `ops.Embedding.template` mints a vocabulary symbol per call, and the
    Engram modules grab `TOKEN_IDS` from the tape, so the embedding reads the array of
    that datatype and the slot holds the array the embedding reads.'''
    return cat.Block.template(
        over((x,), ops.Embedding.template(TOKEN_IDS.datatype, (m,))),
        title=text.EMBEDDING_TITLE, fill_color=INPUTS_COLOUR,
        description=text.EMBED_TOKEN_IDENTIFIERS_DESCRIPTION,
        references=(model_lines(174), model_lines(1201), model_lines(1253),
                    inference_config_lines(62)))


def beside_the_image_operands(morphism: cat.Morphism) -> cat.Morphism:
    '''`morphism` with the four operands of the image pathway passed on beside it.'''
    return functools.reduce(operator.mul, map(hold, IMAGE_OPERANDS), morphism)


def read_inputs() -> cat.Morphism:
    '''From the six inputs of the model to the embedded tokens and the four operands of
    the image pathway. The token identifiers are copied, one copy is embedded, and the
    other copy and the modality go to `INPUTS`.'''
    return (route((0, 0, *range(1, len(MODEL_INPUTS))), MODEL_INPUTS)
            @ beside_the_image_operands(embed_token_identifiers() * INPUTS))


v41_flash_integrated = (
    read_inputs()
    @ image_pathway_with_delimiters()
    @ expand_into_streams()
    @ (initial_collapse() * hold(X))
    @ divided_stack
    @ collapse_streams()
    @ unembed()
    @ dspark_drafter.probabilities_with_drafts())


def released_assigned_sizes(
    term: fd.GeneralTerm = v41_flash_integrated,
) -> dict[str, int]:
    '''The released size of every symbol of `term` that `RELEASED_SIZES` names, keyed
    by the bodies of the symbol's name. A figure passes the mapping as
    `assigned_sizes`, and the display writes each size onto the label of its axis while
    the term stays symbolic.'''
    config = gc.NumericConfig.template(term)
    config.assign_values(**RELEASED_SIZES)
    return config.assigned_integers_by_name()
