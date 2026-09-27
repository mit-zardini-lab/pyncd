# Claude Opus 5.5 (1M context), reasoning effort 40.
'''The pass of the released DeepSeek-V3 that reads its earlier tokens from caches,
derived from the model.

`caching.algebra.derive_cached_pass` carries the read of the new tokens back through
the model and places a cache wherever a causal read reaches an earlier token.
`placements_by_sliding_caches_back` derives every placement of the caches of the
attention, and `ATTENTION_PLACEMENTS` holds them. `NARROWEST` is the placement whose
caches hold the fewest values per token, which caches the normalised latent and the
turned key and computes `W^{UK}`, `W^{UV}`, the repeat of the turned key over the heads
and the join of the two runs of the key at every cached token.
`WRITTEN_ORDER_PASS` is the pass of the whole model with every attention placed that
way. Its dense and mixture layers share one attention box, so one placement serves all
61 layers.

In the order the model is written, that placement expands every cached latent into a
key and a value in every pass. `absorb_into_the_queries.with_linear_maps_absorbed`
contracts the queries against `W^{UK}` first and applies `W^{UV}` after the weighted
sum, where that order costs fewer operations at the sizes of the pass. The rewrite runs
inside one scope of the hypergraph, and the attention of the model is a box, so
`absorbed_inside_the_attention` applies it to the body of the box and returns the box
holding the rewritten body. `CACHED_PASS` is the result at one new token after 32,768
earlier tokens. It is the absorb mode of the released code, which that code runs by
default.

The block of the attention and the block of the model are the blocks of the derivation,
and their descriptions are rewritten to describe the pass, because the descriptions of
the model describe a pass over every token. `obsidian/08-caching/Deriving Caches by
Dragging the New Tokens.md` states the derivation and its proof.
'''
from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import caching.algebra.absorb_into_the_queries as absorb_into_the_queries
import caching.algebra.cache_contents as cache_contents
import caching.algebra.cost_cache_placements as cost_cache_placements
import caching.algebra.count_pass_operations as count_pass_operations
import caching.algebra.derive_cached_pass as derive_cached_pass
import caching.registries.standard_expansions  # noqa: F401 - the expansion of a cache
import data_structure.Category as cat
import data_structure.Operators as ops
import data_structure.Term as fd
import term_utilities.term_utilities as term_utilities

import notebooks.classic.released_deepseek_v3 as released_deepseek_v3
from notebooks.classic.deepseek_v3 import ATTENTION_BOX, x
from notebooks.classic.released_deepseek_v3_wording import TEXT as text

past_tokens = cat.RawAxis.named(
    fd.DynamicName('x', fd.DynamicName('old'), code_form='past_tokens'))
new_tokens = cat.RawAxis.named(
    fd.DynamicName('x', fd.DynamicName('new'), code_form='new_tokens'))

EARLIER_TOKENS = 32768
ONE_NEW_TOKEN_SIZES: dict[str, int] = {
    **released_deepseek_v3.RELEASED_SIZES, 'xold': EARLIER_TOKENS, 'xnew': 1}
'''The sizes of one decoding step after 32,768 earlier tokens, at which the order of the
contractions over the cache is chosen.'''

ATTENTION_PLACEMENTS: tuple[derive_cached_pass.CachedPass, ...] = (
    cost_cache_placements.placements_by_sliding_caches_back(
        released_deepseek_v3.multi_head_latent_attention(), x, past_tokens, new_tokens))

NARROWEST = cost_cache_placements.narrowest_placement(
    ATTENTION_PLACEMENTS, ONE_NEW_TOKEN_SIZES)

WRITTEN_ORDER_PASS = derive_cached_pass.derive_cached_pass(
    released_deepseek_v3.MODEL, x, past_tokens, new_tokens,
    NARROWEST.computed_over_the_cache)


def is_the_attention_box(node: object) -> bool:
    return (isinstance(node, cat.Broadcasted)
            and isinstance(node.operator, ops.BlockOperator)
            and node.operator.name is not None
            and node.operator.name.to_bodies() == ATTENTION_BOX)


def attention_boxes(term: object) -> tuple[cat.Broadcasted, ...]:
    '''Every distinct attention box of `term`.'''
    return tuple(node for node in term_utilities.type_search(cat.Broadcasted, term)
                 if is_the_attention_box(node))


def described_again(block: cat.Block, description: str, salt: str) -> cat.Block:
    '''`block` with `description` in place of its own, under a tag of its own, so an
    inspection box keyed by the tag shows the description of the block it opens
    over.'''
    tag = block.block_tag
    return block.reconstruct(block_tag=tag.reconstruct(
        uid=tag.uid.reconstruct(_id=fd.hash_id((tag.uid._id, salt))),
        aesthetics=tag.aesthetics.reconstruct(description=description)))


def with_operations_replaced[T: fd.GeneralTerm](
    target: T, replacements: Mapping[cat.Broadcasted, cat.Broadcasted],
) -> T:
    '''`target` with every operation `replacements` names replaced by the operation
    it maps to, each shared part visited once.'''
    memo: dict[int, tuple[object, object]] = {}

    def replaced(part: object) -> object:
        found = memo.get(id(part))
        if found is not None:
            return found[1]
        result = (replacements[part]
                  if isinstance(part, cat.Broadcasted) and part in replacements
                  else fd.deep_reconstruct(part, replaced)
                  if isinstance(part, (fd.Term, tuple)) else part)
        memo[id(part)] = (part, result)
        return result

    return replaced(target)  # type: ignore[return-value]


def absorbed_inside_the_attention(
    cached_pass: derive_cached_pass.CachedPass, sizes: Mapping[str, int],
) -> derive_cached_pass.CachedPass:
    '''`cached_pass` with the body of every attention box rewritten by
    `with_linear_maps_absorbed` at `sizes`, and the block of the box described as the
    attention of one pass.'''
    replacements: dict[cat.Broadcasted, cat.Broadcasted] = {}
    for box in attention_boxes(cached_pass.expression):
        block = box.operator.block
        absorbed = absorb_into_the_queries.with_linear_maps_absorbed(
            dataclasses.replace(cached_pass, expression=block), sizes)
        rewritten = described_again(
            block.reconstruct(body=absorbed.expression),
            text.CACHED_ATTENTION_DESCRIPTION, 'absorbed attention')
        replacements[box] = box.reconstruct(
            operator=box.operator.reconstruct(block=rewritten))
    return dataclasses.replace(cached_pass, expression=with_operations_replaced(
        cached_pass.expression, replacements))


def described_as_a_pass(
    cached_pass: derive_cached_pass.CachedPass,
) -> derive_cached_pass.CachedPass:
    '''`cached_pass` with the block of the model described as one pass over the new
    tokens.'''
    expression = cached_pass.expression
    if not isinstance(expression, cat.Block):
        return cached_pass
    return dataclasses.replace(cached_pass, expression=described_again(
        expression, text.CACHED_MODEL_DESCRIPTION, 'cached pass'))


CACHED_PASS = described_as_a_pass(
    absorbed_inside_the_attention(WRITTEN_ORDER_PASS, ONE_NEW_TOKEN_SIZES))


def cached_attention(cached_pass: derive_cached_pass.CachedPass = CACHED_PASS
                     ) -> cat.Broadcasted:
    '''The one attention box of `cached_pass`, which every layer runs.'''
    box, = attention_boxes(cached_pass.expression)
    return box


def cache_widths_of_the_attention(
    cached_pass: derive_cached_pass.CachedPass = CACHED_PASS,
    sizes: Mapping[str, int] = ONE_NEW_TOKEN_SIZES,
) -> dict[str, int]:
    '''The values each cache of one attention box holds per token, by the name of the
    cache.'''
    return {cache_contents.cache_name(cache):
            cache_contents.entries_per_token(cache, sizes)
            for cache in cache_contents.caches_in_the_order_they_run(
                cached_attention(cached_pass))}


def values_cached_per_token(
    cached_pass: derive_cached_pass.CachedPass = CACHED_PASS,
    sizes: Mapping[str, int] = ONE_NEW_TOKEN_SIZES,
) -> int:
    '''The values the caches of every layer hold together for one token.'''
    layers = sizes['D'] + sizes['L']
    return layers * sum(cache_widths_of_the_attention(cached_pass, sizes).values())


def attention_operations(
    cached_pass: derive_cached_pass.CachedPass,
    sizes: Mapping[str, int] = ONE_NEW_TOKEN_SIZES,
) -> int:
    '''The operations of the linear maps and the contractions of one attention box of
    `cached_pass` at `sizes`, two per multiply-add.'''
    return count_pass_operations.operations_at_sizes(
        tuple(count_pass_operations.operations_of_a_pass(
            cached_attention(cached_pass).operator.block)), sizes)
