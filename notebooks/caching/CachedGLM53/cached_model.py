# Claude Opus 5.5 (1M context), effort 40.
'''One pass of GLM-5.3 with every layer reading its keys from caches.

The layer plan, the residual, the feed-forward maps, the embedding and the output head
are those of `notebooks/sota/GLM53/`, and the attention modes are those of
`cached_attention_modes`. The pass reads the identifiers of its own tokens `x` and
returns their logits. Every other token enters only through the caches, whose results
stand on `P + x`.

    layers 0 and 1      cached Full,   dense MLP
    layer 2             cached Full,   dense MLP, dropping sel
    layers 3 to 5       cached Shared, mixture of experts
    layers 6 + 4 l      cached Full,   mixture of experts, dropping sel[l]
    layers 7 + 4 l to 9 + 4 l
                        cached Shared, mixture of experts, grabbing sel[l]

`CACHED_SIZES` gives the released size of every axis the configuration of the
checkpoint sizes. `P` and `x` are left out, because they change from pass to pass.
'''
from __future__ import annotations

import construction_helpers as ch  # noqa: F401 - the @, * and >> overloads
import data_structure.Category as cat

import notebooks.caching.CachedGLM53.cached_attention_modes as cached_attention_modes
from notebooks.sota.GLM53.declared_axes import GROUP_COUNTER_NAME
from notebooks.sota.GLM53.feed_forward import DENSE_MLP
from notebooks.sota.GLM53.layer_stack import (
    LAYERS_BEFORE_THE_FIRST_GROUP, LAYER_COLOUR, LAYER_REFERENCES, PLAN_REFERENCES,
    REPEATED_GROUPS, SHARED_LAYERS_PER_GROUP, SHARED_LAYER_COLOUR, layer)
from notebooks.sota.GLM53.mixture_of_experts import MIXTURE
from notebooks.sota.GLM53.whole_model import (
    RELEASED_SIZES, embed, output_logits, released_assigned_sizes)
from notebooks.sota.GLM53.block_titles_and_descriptions import TEXT as text


def shared_layers(attention: cat.BroadcastedCategory) -> cat.Block:
    return cat.Block.template(
        layer(attention, MIXTURE), title=text.SHARED_LAYER_TITLE,
        repetition=SHARED_LAYERS_PER_GROUP, fill_color=SHARED_LAYER_COLOUR,
        description=text.SHARED_LAYERS_DESCRIPTION, references=LAYER_REFERENCES)


layers_before_the_first_group = cat.Block.template(
    layer(cached_attention_modes.FULL, DENSE_MLP), title=text.DENSE_LAYER_TITLE,
    repetition=LAYERS_BEFORE_THE_FIRST_GROUP, fill_color=LAYER_COLOUR,
    description=text.DENSE_LAYERS_DESCRIPTION, references=PLAN_REFERENCES)

first_group = cat.Block.template(
    layer(cached_attention_modes.FULL_PUBLISHING, DENSE_MLP)
    @ shared_layers(cached_attention_modes.SHARED),
    title=text.FIRST_GROUP_TITLE, fill_color=LAYER_COLOUR,
    description=text.FIRST_GROUP_DESCRIPTION, references=PLAN_REFERENCES)

repeated_groups = cat.Block.template(
    layer(cached_attention_modes.FULL_PUBLISHING_IN_GROUP, MIXTURE)
    @ shared_layers(cached_attention_modes.SHARED_IN_GROUP),
    title=text.GROUP_TITLE, repetition=REPEATED_GROUPS, fill_color=LAYER_COLOUR,
    description=text.REPEATED_GROUPS_DESCRIPTION, index_name=GROUP_COUNTER_NAME,
    references=PLAN_REFERENCES)

cached_layer_stack = layers_before_the_first_group @ first_group @ repeated_groups

cached_glm53 = embed() @ cached_layer_stack @ output_logits()

CACHED_SIZES: dict[str, int] = RELEASED_SIZES


def cached_assigned_sizes(term: cat.Morphism = cached_glm53) -> dict[str, int]:
    '''The released size of every symbol of `term` that the configuration of the
    checkpoint names, for a figure's `assigned_sizes`.'''
    return released_assigned_sizes(term)
