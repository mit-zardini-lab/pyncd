# Claude Opus 5.5 (1M context), effort 40.
'''The Full and Shared modes of GLM-5.3 reading their keys from caches.

The modes are the modes of `notebooks/sota/GLM53/attention_modes.py`, built from the
cached parts. A Full layer saves three arrays per token of the pass, the latent into
`lat`, the turned key into `rot` and the indexer key into `idx`. A Shared layer runs no
indexer and saves two, `lat` and `rot`.

The selection is not cached. A Full layer selects 2048 tokens for each query of the
pass, and the Shared layers of its group read that selection from the tape slot `sel`
within the same pass. The next pass selects again for its own queries, because the
reference passes `topk_indices` from layer to layer as `prev_topk_indices` and never
hands it to the cache, at lines 722 to 733 of `modeling_glm_moe_dsa.py`.

    FULL, FULL_PUBLISHING, FULL_PUBLISHING_IN_GROUP, SHARED, SHARED_IN_GROUP
                              the five modes of `attention_modes`, cached
'''
from __future__ import annotations

import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat
import para.data_structure.Para as Para

from notebooks.caching.CachedGLM53.cached_attention import (
    CACHED_KEYS, CACHED_VALUES, attend_to_selected_cached_tokens,
    cached_keys_and_values, query_path_at_this_pass,
    query_without_low_rank_at_this_pass)
from notebooks.caching.CachedGLM53.cached_indexer import (
    CACHED_INDEXER, CACHED_SELECTION, SELECT_FROM_THE_CACHE)
from notebooks.caching.CachedGLM53.wording import TEXT as cached_text
from notebooks.sota.DeepSeekV41Flash.construction_idioms import (
    boxed, hold, over, para_boxed, route)
from notebooks.sota.GLM53.attention_modes import (
    FULL_BOX, FULL_COLOUR, MODE_REFERENCES, SHARED_BOX, SHARED_COLOUR)
from notebooks.sota.GLM53.declared_axes import (
    GROUP_COUNTER, QUERIES, QUERY_LOW_RANK, SLOT_SELECTION, STATE, x)
from notebooks.sota.GLM53.block_titles_and_descriptions import TEXT as text


def select_tokens_from_the_cache() -> cat.BroadcastedCategory:
    '''`STATE -> QUERIES, SELECTION, KEYS, VALUES`: the queries of the pass, the keys
    and the values of every cached token, and the top-2048 of the indexer of this
    layer over the cache.'''
    return (route((0, 0, 0), (STATE,))
            @ (query_path_at_this_pass() * cached_keys_and_values() * hold(STATE))
            @ route((0, 1, 4, 2, 3),
                    (QUERIES, QUERY_LOW_RANK, CACHED_KEYS, CACHED_VALUES, STATE))
            @ (hold(QUERIES) * CACHED_INDEXER * hold(CACHED_KEYS) * hold(CACHED_VALUES))
            @ (hold(QUERIES) * over((x,), SELECT_FROM_THE_CACHE) * hold(CACHED_KEYS)
               * hold(CACHED_VALUES)))


def cached_full_attention() -> cat.Block:
    return cat.Block.template(
        select_tokens_from_the_cache()
        @ route((0, 1, 2, 1, 3),
                (QUERIES, CACHED_SELECTION, CACHED_KEYS, CACHED_VALUES))
        @ attend_to_selected_cached_tokens(),
        title=text.FULL_TITLE, fill_color=FULL_COLOUR,
        description=cached_text.CACHED_FULL_DESCRIPTION, references=MODE_REFERENCES)


def cached_full_attention_publishing(drop: Para.Drop) -> cat.Block:
    '''The cached Full mode of a layer whose selection the Shared layers after it read
    in the same pass, dropped onto the slot named by `drop`.'''
    return cat.Block.template(
        select_tokens_from_the_cache()
        @ route((0, 1, 2, 1, 3, 1),
                (QUERIES, CACHED_SELECTION, CACHED_KEYS, CACHED_VALUES))
        @ (attend_to_selected_cached_tokens() * hold(CACHED_SELECTION))
        @ (hold(STATE) * drop),
        title=text.FULL_TITLE, fill_color=FULL_COLOUR,
        description=cached_text.CACHED_FULL_PUBLISHING_DESCRIPTION,
        references=MODE_REFERENCES)


def cached_shared_attention(selection_entry: Para.NamedEntry) -> cat.Block:
    return cat.Block.template(
        (hold(STATE) * Para.grab_of(selection_entry, CACHED_SELECTION))
        @ route((0, 0, 1), (STATE, CACHED_SELECTION))
        @ (query_without_low_rank_at_this_pass() * cached_keys_and_values()
           * hold(CACHED_SELECTION))
        @ route((0, 3, 1, 3, 2),
                (QUERIES, CACHED_KEYS, CACHED_VALUES, CACHED_SELECTION))
        @ attend_to_selected_cached_tokens(),
        title=text.SHARED_TITLE, fill_color=SHARED_COLOUR,
        description=cached_text.CACHED_SHARED_DESCRIPTION, references=MODE_REFERENCES)


FULL = boxed(cached_full_attention(), FULL_BOX)
FULL_PUBLISHING = para_boxed(
    cached_full_attention_publishing(
        Para.Drop(tape=SLOT_SELECTION, size=CACHED_SELECTION)), FULL_BOX)
FULL_PUBLISHING_IN_GROUP = para_boxed(
    cached_full_attention_publishing(Para.LoopDrop(
        tape=SLOT_SELECTION, size=CACHED_SELECTION, index=GROUP_COUNTER)), FULL_BOX)
SHARED = para_boxed(cached_shared_attention(SLOT_SELECTION), SHARED_BOX)
SHARED_IN_GROUP = para_boxed(
    cached_shared_attention(Para.LoopSlot(SLOT_SELECTION, GROUP_COUNTER)), SHARED_BOX)
