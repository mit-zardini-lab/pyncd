# Claude Opus 5.5 (1M context), effort 40.
'''The quantities read off the cached GLM-5.3.

Every number here is computed from the expression of `cached_model` at the sizes of the
checkpoint, with `|P|` and `|x|` bound per pass.

    cached_entries_by_layer     the values each layer keeps per token, by cache
    widths_along_the_key_value_path
                                the values per token on the wires from the hidden
                                state to the keys and the values of every head, with
                                the two wires the reference caches marked
    attention_operations        the floating-point operations of every linear map and
                                contraction of the attention of one pass, read by
                                `caching.algebra.count_pass_operations`, symbolic in
                                `|P|` and `|x|`
    operations_by_part          the same summed into the parts of `AttentionPart`

`read_symbolic_work` of `performance_modeling/morphism_work.py` counts two operations
per multiply-add, and so does everything here.
'''
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import data_structure.Category as cat
import data_structure.Operators as ops

import caching.algebra.cache_contents as cache_contents
import caching.algebra.count_pass_operations as count_pass_operations
import notebooks.caching.CachedGLM53.cached_attention as cached_attention
from notebooks.caching.CachedGLM53.cached_model import CACHED_SIZES, cached_glm53
from notebooks.sota.GLM53.attention_modes import FULL_BOX, SHARED_BOX
from notebooks.sota.GLM53.lightning_indexer import SCORE_BOX
from notebooks.sota.GLM53.multi_latent_attention import CORE_BOX

EXPANSION_WEIGHTS = ('W^{Kb}', 'W^{Vb}')


def sizes_of_a_pass(past_tokens: int, tokens_of_this_pass: int) -> dict[str, int]:
    '''The sizes of the checkpoint with `|P|` and `|x|` bound for one pass.'''
    return {**CACHED_SIZES, 'P': past_tokens, 'x': tokens_of_this_pass}


@dataclass(frozen=True)
class LayerCaches:
    '''The values one layer keeps per token, by the name of each cache.'''
    layer: int
    mode: str
    entries: tuple[tuple[str, int], ...]

    def total(self) -> int:
        return sum(count for _, count in self.entries)


def cached_entries_by_layer(
    term: cat.Morphism = cached_glm53,
    sizes: Mapping[str, int] = CACHED_SIZES,
) -> tuple[LayerCaches, ...]:
    return tuple(
        LayerCaches(
            layer=layer, mode=mode,
            entries=tuple(sorted(
                (cache_contents.cache_name(cache),
                 cache_contents.entries_per_token(cache, sizes))
                for cache in caches)))
        for layer, (mode, caches)
        in enumerate(cache_contents.caches_by_outermost_box(term)))


def entries_per_token_of_array(array: cat.Array, sizes: Mapping[str, int]) -> int:
    '''The values `array` holds for one token, its token axis standing first.'''
    return math.prod(cache_contents.size_of(axis.local_size(), sizes)
                     for axis in tuple(array.shape())[1:])


@dataclass(frozen=True)
class WireWidth:
    '''The values per token of one or more wires of the key-value path.'''
    label: str
    entries: int
    is_cached: bool


def widths_along_the_key_value_path(
    sizes: Mapping[str, int] = CACHED_SIZES,
) -> tuple[WireWidth, ...]:
    '''The hidden state the block reads, the two arrays it caches, and the keys and
    the values of every head it returns, each per token.'''
    block = cached_attention.cached_keys_and_values()
    caches = tuple(cache_contents.caches_in_the_order_they_run(block))
    keys, values = block.cod()
    return (
        WireWidth('hidden state', entries_per_token_of_array(block.dom()[0], sizes),
                  is_cached=False),
        WireWidth('latent and turned key',
                  sum(cache_contents.entries_per_token(cache, sizes)
                      for cache in caches),
                  is_cached=True),
        WireWidth('keys and values of every head',
                  entries_per_token_of_array(keys, sizes)
                  + entries_per_token_of_array(values, sizes),
                  is_cached=False),
    )


class AttentionPart(Enum):
    '''The parts of the attention into which `operations_by_part` sums the
    operations.'''
    PROJECTIONS_OF_THE_PASS = 'projections of the tokens of the pass'
    EXPANSION_OF_THE_CACHE = 'expansion of every cached latent'
    INDEXER_SCORES = 'indexer scores against every cached key'
    ATTENTION_CORE = 'attention core over the 2048 selected tokens'


def part_of(node: count_pass_operations.NodeOperations) -> AttentionPart:
    if SCORE_BOX in node.boxes:
        return AttentionPart.INDEXER_SCORES
    if CORE_BOX in node.boxes:
        return AttentionPart.ATTENTION_CORE
    if (isinstance(node.node.operator, ops.Linear)
            and node.node.operator.name.to_text() in EXPANSION_WEIGHTS):
        return AttentionPart.EXPANSION_OF_THE_CACHE
    return AttentionPart.PROJECTIONS_OF_THE_PASS


def attention_operations(
    term: cat.Morphism = cached_glm53,
) -> tuple[count_pass_operations.NodeOperations, ...]:
    '''The operations of every attention sublayer of `term` in one pass.'''
    return tuple(node for node in count_pass_operations.operations_of_a_pass(term)
                 if node.boxes and node.boxes[0] in (FULL_BOX, SHARED_BOX))


ATTENTION_OPERATIONS = attention_operations()


def operations_by_part(
    past_tokens: int,
    tokens_of_this_pass: int,
    nodes: tuple[count_pass_operations.NodeOperations, ...] = ATTENTION_OPERATIONS,
) -> dict[AttentionPart, int]:
    '''The operations of each part of the attention of the 78 layers in one pass.'''
    sizes = sizes_of_a_pass(past_tokens, tokens_of_this_pass)
    totals = {part: 0 for part in AttentionPart}
    for node in nodes:
        totals[part_of(node)] += cache_contents.size_of(node.operations, sizes)
    return totals


def expansion_after_the_gather(past_tokens: int, tokens_of_this_pass: int) -> int:
    '''The operations of the expansion were it applied to the latents each query
    selects rather than to every cached latent: the expansion of one cached token,
    read off the expression, times `|x| |s|`.'''
    per_cached_token = operations_by_part(0, 1)[AttentionPart.EXPANSION_OF_THE_CACHE]
    return (per_cached_token * tokens_of_this_pass
            * CACHED_SIZES['s'])
