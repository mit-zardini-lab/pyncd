# Claude Opus 5.5 (1M context), effort 40.
'''The two attention modes of GLM-5.3, Full and Shared, which differ in the source of
their selection.

`config.indexer_types` of the reference gives every layer one of two modes. A layer in
the Full mode runs its own indexer and its own top-2048. A layer in the Shared mode runs
no indexer and attends to the tokens kept by the most recent Full layer, which is the
IndexShare of arXiv 2603.12201. The reference passes the selection from one layer to
the next as `prev_topk_indices`, and a Shared layer reads the value passed to it. Here
a Full layer whose selection is read later drops the selection onto the tape slot
`sel`, and each Shared layer grabs it from the slot, so every mode reads and returns the
hidden state alone.

    FULL                      runs its own indexer and drops nothing, layers 0 and 1
    FULL_PUBLISHING           drops sel, layer 2
    FULL_PUBLISHING_IN_GROUP  drops sel[l], the first layer of IndexShare group l
    SHARED                    grabs sel, layers 3 to 5
    SHARED_IN_GROUP           grabs sel[l], the other three layers of group l

Layers 0 and 1 run the Full mode and the layer after each of them runs the Full mode
as well, so their selections are read by no other layer. The reference returns them
from the layer all the same, and the next layer ignores them.
'''
from __future__ import annotations

import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import para.data_structure.Para as Para

from notebooks.sota.DeepSeekV41Flash.construction_idioms import (
    boxed, hold, over, para_boxed, route)
from notebooks.sota.GLM53.declared_axes import (
    GROUP_COUNTER, KEYS, QUERIES, QUERY_LOW_RANK, SLOT_SELECTION, STATE, VALUES, x)
from notebooks.sota.GLM53.lightning_indexer import INDEXER, SELECT, SELECTION
from notebooks.sota.GLM53.multi_latent_attention import (
    attend_to_selected_tokens, keys_and_values, query_path, query_without_low_rank)
from notebooks.sota.GLM53.reference_links import (
    checkpoint_config_lines, configuration_lines, modeling_lines)
from notebooks.sota.GLM53.block_titles_and_descriptions import TEXT as text

FULL_BOX = 'Full'
SHARED_BOX = 'Shared'
FULL_COLOUR = '#C5BEDF'
SHARED_COLOUR = '#B8D8CE'

MODE_REFERENCES = (modeling_lines(309, 316), modeling_lines(415, 428),
                   configuration_lines(135, 148), checkpoint_config_lines(21, 23),
                   checkpoint_config_lines(26, 105))


def select_tokens() -> cat.BroadcastedCategory:
    '''`STATE -> QUERIES, SELECTION, KEYS, VALUES`: the queries, the keys and the values
    of every head, and the top-2048 of the indexer of this layer.'''
    return (route((0, 0, 0), (STATE,))
            @ (query_path() * keys_and_values() * hold(STATE))
            @ route((0, 1, 4, 2, 3), (QUERIES, QUERY_LOW_RANK, KEYS, VALUES, STATE))
            @ (hold(QUERIES) * INDEXER * hold(KEYS) * hold(VALUES))
            @ (hold(QUERIES) * over((x,), SELECT) * hold(KEYS) * hold(VALUES)))


def full_attention() -> cat.Block:
    '''The Full mode of a layer whose selection is read by no later layer.'''
    return cat.Block.template(
        select_tokens()
        @ route((0, 1, 2, 1, 3), (QUERIES, SELECTION, KEYS, VALUES))
        @ attend_to_selected_tokens(),
        title=text.FULL_TITLE, fill_color=FULL_COLOUR,
        description=text.FULL_ATTENTION_DESCRIPTION, references=MODE_REFERENCES)


def full_attention_publishing(drop: Para.Drop) -> cat.Block:
    '''The Full mode of a layer whose selection is read by the Shared layers after it.
    The selection is dropped onto the slot named by `drop`.'''
    return cat.Block.template(
        select_tokens()
        @ route((0, 1, 2, 1, 3, 1), (QUERIES, SELECTION, KEYS, VALUES))
        @ (attend_to_selected_tokens() * hold(SELECTION))
        @ (hold(STATE) * drop),
        title=text.FULL_TITLE, fill_color=FULL_COLOUR,
        description=text.FULL_ATTENTION_PUBLISHING_DESCRIPTION,
        references=MODE_REFERENCES)


def shared_attention(selection_entry: Para.NamedEntry) -> cat.Block:
    '''The Shared mode: the selection is grabbed from the slot named by
    `selection_entry`, and the layer computes its own queries, keys and values.'''
    return cat.Block.template(
        (hold(STATE) * Para.grab_of(selection_entry, SELECTION))
        @ route((0, 0, 1), (STATE, SELECTION))
        @ (query_without_low_rank() * keys_and_values() * hold(SELECTION))
        @ route((0, 3, 1, 3, 2), (QUERIES, KEYS, VALUES, SELECTION))
        @ attend_to_selected_tokens(),
        title=text.SHARED_TITLE, fill_color=SHARED_COLOUR,
        description=text.SHARED_ATTENTION_DESCRIPTION, references=MODE_REFERENCES)


FULL = boxed(full_attention(), FULL_BOX)
FULL_PUBLISHING = para_boxed(
    full_attention_publishing(Para.Drop(tape=SLOT_SELECTION, size=SELECTION)), FULL_BOX)
FULL_PUBLISHING_IN_GROUP = para_boxed(
    full_attention_publishing(Para.LoopDrop(
        tape=SLOT_SELECTION, size=SELECTION, index=GROUP_COUNTER)), FULL_BOX)
SHARED = para_boxed(shared_attention(SLOT_SELECTION), SHARED_BOX)
SHARED_IN_GROUP = para_boxed(
    shared_attention(Para.LoopSlot(SLOT_SELECTION, GROUP_COUNTER)), SHARED_BOX)
