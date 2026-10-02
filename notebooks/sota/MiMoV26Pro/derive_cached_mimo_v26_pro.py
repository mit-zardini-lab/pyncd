# Claude Opus 5.5 (1M context), effort 40.
'''One pass of MiMo-V2.6-Pro over its new tokens, derived from the uncached model, with
the caches of the reference.

`caching.algebra.derive_cached_pass` carries the read of the new tokens,
`i_x = |x_old| + i_new`, back from the logits of `whole_model.mimo`. Every wire of the
pass is a wire of the model read through a read of the token axis. The read of the new
tokens `x_new` stands on most wires. A causal read composes with it into a read of
earlier tokens, and where the operand of the causal read is produced from wires read at
the new tokens, the pass computes the operand at the new tokens and a `Caching` of it
loads the earlier tokens and appends the new ones.

The model is built with its causal reads where the reference applies its masks, on the
rotated keys and the scaled values of every layer, and the derivation is asked to
compute no operator over the cache. Every cache therefore holds the operand of a causal
read, which is the keys and the values the reference hands to its cache. A full
attention layer reads every earlier token, so its caches hold the axis `x_old + x_new`.
A sliding window layer reads at most `|w| - 1` tokens before the first new token, so its
caches hold the last `|w| - 1` earlier tokens followed by the new ones. The
`DynamicSlidingWindowLayer` of `transformers` 5.3.0 keeps the same tokens.

`notebooks/website/modern/validate_mimo_v26_pro.py` checks what each cache holds, and
`obsidian/08-caching/Deriving Caches by Dragging the New Tokens.md` states the
derivation and proves that the pass computes the model's logits for the new tokens.
'''
from __future__ import annotations

from dataclasses import dataclass

import caching.algebra.cache_contents as cache_contents
import caching.algebra.derive_cached_pass as derive_cached_pass
import caching.data_structure.Caching as Caching
import caching.registries.standard_expansions  # noqa: F401 - the expansion of a cache
import data_structure.Category as cat
import data_structure.Term as fd

import notebooks.sota.MiMoV26Pro.whole_model as whole_model
from notebooks.sota.MiMoV26Pro.declared_axes import x

OLD_TOKENS = cat.RawAxis.named(
    fd.DynamicName('x', fd.DynamicName('old'), code_form='earlier_tokens'))
NEW_TOKENS = cat.RawAxis.named(
    fd.DynamicName('x', fd.DynamicName('new'), code_form='new_tokens'))

DERIVED = derive_cached_pass.derive_cached_pass(
    whole_model.mimo, x, OLD_TOKENS, NEW_TOKENS, frozenset())

cached_mimo = DERIVED.expression


def cached_assigned_sizes(term: cat.Morphism = cached_mimo) -> dict[str, int]:
    '''The released size of every symbol of `term` the configuration of the checkpoint
    names. The earlier and the new tokens change from pass to pass and stay
    symbolic.'''
    return whole_model.released_assigned_sizes(term)


class LayersOfOneModeCacheDifferently(ValueError):
    '''Two layers of one attention mode whose caches differ in name or in width.'''


@dataclass(frozen=True)
class CachesOfAMode:
    '''The caches every layer of one attention mode runs, each with its name, the values
    it holds per token and the number of earlier tokens it keeps, `None` where it keeps
    every earlier token, and the number of layers running them in one pass.'''
    mode: str
    layers: int
    caches: fd.Prod[tuple[str, int, int | None]]

    def values_per_token_of_a_layer(self) -> int:
        return sum(values for _, values, _ in self.caches)

    def values_per_token(self) -> int:
        return self.layers * self.values_per_token_of_a_layer()


def earlier_tokens_kept(cache: cache_contents.CacheNode,
                        sizes: dict[str, int]) -> int | None:
    '''The number of earlier tokens `cache` keeps, or `None` where its cached axis is
    every earlier token followed by the new ones.'''
    cached_axis = cache.cod()[0].shape()[Caching.tokens_position(cache)]
    if cached_axis == DERIVED.cached_tokens:
        return None
    return cache_contents.size_of(cached_axis.parts[0].local_size(), sizes)


def caches_by_mode(term: cat.Morphism = cached_mimo) -> fd.Prod[CachesOfAMode]:
    '''The caches of each attention mode of `term`, in the order the modes first
    run.'''
    sizes = cached_assigned_sizes(term)
    found: dict[str, CachesOfAMode] = {}
    for mode, caches in cache_contents.caches_by_outermost_box(term):
        entries = tuple((cache_contents.cache_name(cache),
                         cache_contents.entries_per_token(cache, sizes),
                         earlier_tokens_kept(cache, sizes))
                        for cache in caches)
        earlier = found.get(mode, CachesOfAMode(mode=mode, layers=0, caches=entries))
        if earlier.caches != entries:
            raise LayersOfOneModeCacheDifferently(
                f'{mode} layers cache {earlier.caches} and {entries}')
        found[mode] = CachesOfAMode(
            mode=mode, layers=earlier.layers + 1, caches=entries)
    return tuple(found.values())


CACHE_TABLE_HEADER: tuple[str, str] = (
    '| attention mode | layers | caches, with the values each holds per token | '
    'earlier tokens kept | values per token of one layer |', '|---|---|---|---|---|')


def kept_in_words(kept: int | None) -> str:
    return 'every one' if kept is None else f'the last {kept}'


def cache_table(term: cat.Morphism = cached_mimo) -> str:
    '''The markdown table of `caches_by_mode`, for a notebook cell.'''
    return '\n'.join((*CACHE_TABLE_HEADER, *(
        f'| {mode.mode} | {mode.layers} | '
        + ', '.join(f'${name}$ {values:,}' for name, values, _ in mode.caches)
        + f' | {kept_in_words(mode.caches[0][2])} | '
        + f'{mode.values_per_token_of_a_layer():,} |'
        for mode in caches_by_mode(term))))
