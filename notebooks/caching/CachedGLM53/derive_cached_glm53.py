# Claude Opus 5.5 (1M context), effort 40.
'''One pass of GLM-5.3 over its new tokens, derived from the uncached model, with the
caches of the reference, in the reals and quantised as `transformers` runs the FP8
checkpoint.

`caching.algebra.derive_cached_pass` carries the read of the new tokens,
`i_x = |x_old| + i_new`, back from the logits of `whole_model.glm53`. Every wire of the
pass is a wire of the model read through one of two reads of the token axis. The read
of the new tokens `x_new` stands on most wires. The read of the cached axis
`x_old + x_new`, every earlier token followed by the new ones, stands on the wires a
causal read reaches. Where a wire read on the cached axis is produced from wires read
at the new tokens, the pass computes the wire at the new tokens and a `Caching` of it
loads the earlier tokens and appends the new ones.

The operators a wire read on the cached axis passes through are the placement of the
caches. `computed_over_the_cache` names the four the reference computes on every cached
token: the expansions `W^{Kb}` and `W^{Vb}` of the latent, the repeat of the turned key
over the heads and the join of the key. The read of the cached axis stops at the
operators before them, so the pass caches the normalised latent and the turned key of
every layer, and the indexer key of every Full layer, which are the three arrays
`modeling_glm_moe_dsa.py` hands to its cache. The selection travels on the tape slot
`sel` within one pass, and the derivation requires the grab of the slot to read the new
tokens its drop wrote, so it is not cached. Of the 104 placements
`cost_cache_placements.placements_by_sliding_caches_back` reaches, this one holds the
fewest values per token, and `notebooks/website/modern/validate_glm53.py` checks both
facts.

The quantised pass is the derived pass quantised by the policy of
`quantised_whole_model`. A `Caching` holds its values at the quantisation they arrive
with, because the `DynamicLayer` of `transformers` concatenates the states handed to it
in their own dtype, and those states are BF16.

`obsidian/08-caching/Deriving Caches by Dragging the New Tokens.md` states the
derivation and proves that the pass computes the model's logits for the new tokens.
'''
from __future__ import annotations

from dataclasses import dataclass

import advanced_axis_dynamics.data_structure.Operators as aops
import caching.algebra.cache_contents as cache_contents
import caching.algebra.derive_cached_pass as derive_cached_pass
import caching.registries.standard_expansions  # noqa: F401 - the expansion of a cache
import data_structure.Category as cat
import data_structure.Operators as ops
import data_structure.Term as fd
import quantization.processing.quantise_model as quantise_model
import term_utilities.term_utilities as tutil

import notebooks.sota.GLM53.multi_latent_attention as multi_latent_attention
import notebooks.sota.GLM53.quantised_whole_model as quantised_whole_model
import notebooks.sota.GLM53.whole_model as whole_model
from notebooks.sota.GLM53.block_titles_and_descriptions import TEXT as text
from notebooks.sota.GLM53.declared_axes import x

OLD_TOKENS = cat.RawAxis.named(
    fd.DynamicName('x', fd.DynamicName('old'), code_form='earlier_tokens'))
NEW_TOKENS = cat.RawAxis.named(
    fd.DynamicName('x', fd.DynamicName('new'), code_form='new_tokens'))

EXPANSION_WEIGHTS = ('W^{Kb}', 'W^{Vb}')


def expands_the_cached_keys_or_values(operator: cat.Operator) -> bool:
    '''Whether `operator`, an operator of the keys and values block, is one of the four
    the reference computes on every cached token: a projection of the latent into the
    keys or the values, the repeat of the turned key over the heads, or the join of the
    key.'''
    match operator:
        case ops.Linear(name=name) if name is not None:
            return name.to_bodies() in EXPANSION_WEIGHTS
        case ops.View(name=name) if name is not None:
            return name.to_bodies() == multi_latent_attention.REPEAT_VIEW_NAME
        case aops.ConcatenateAxes():
            return True
    return False


def computed_over_the_cache(model: cat.Morphism) -> frozenset[cat.Operator]:
    '''The operators of the keys and values block of `model` computed on every cached
    token.'''
    keys_and_values = whole_model.part_titled(text.KEYS_AND_VALUES_TITLE, model)
    return frozenset(
        node.operator for node in tutil.type_search(cat.Broadcasted, keys_and_values)
        if expands_the_cached_keys_or_values(node.operator))


DERIVED = derive_cached_pass.derive_cached_pass(
    whole_model.glm53, x, OLD_TOKENS, NEW_TOKENS,
    computed_over_the_cache(whole_model.glm53))

cached_glm53 = DERIVED.expression

QUANTISED_CACHED = quantise_model.quantise_model(
    cached_glm53, quantised_whole_model.RELEASED_POLICY)

cached_glm53_quantised = QUANTISED_CACHED.morphism


def cached_assigned_sizes(term: cat.Morphism = cached_glm53) -> dict[str, int]:
    '''The released size of every symbol of `term` the configuration of the checkpoint
    names. The earlier and the new tokens change from pass to pass and stay
    symbolic.'''
    return whole_model.released_assigned_sizes(term)


class LayersOfOneModeCacheDifferently(ValueError):
    '''Two layers of one attention mode whose caches differ in name or in width.'''


@dataclass(frozen=True)
class CachesOfAMode:
    '''The caches every layer of one attention mode runs, each with the values it holds
    per token, and the number of layers running them in one pass.'''
    mode: str
    layers: int
    caches: fd.Prod[tuple[str, int]]

    def values_per_token_of_a_layer(self) -> int:
        return sum(values for _, values in self.caches)

    def values_per_token(self) -> int:
        return self.layers * self.values_per_token_of_a_layer()


def caches_by_mode(term: cat.Morphism = cached_glm53) -> fd.Prod[CachesOfAMode]:
    '''The caches of each attention mode of `term`, in the order the modes first
    run.'''
    sizes = cached_assigned_sizes(term)
    found: dict[str, CachesOfAMode] = {}
    for mode, caches in cache_contents.caches_by_outermost_box(term):
        entries = tuple((cache_contents.cache_name(cache),
                         cache_contents.entries_per_token(cache, sizes))
                        for cache in caches)
        earlier = found.get(mode, CachesOfAMode(mode=mode, layers=0, caches=entries))
        if earlier.caches != entries:
            raise LayersOfOneModeCacheDifferently(
                f'{mode} layers cache {earlier.caches} and {entries}')
        found[mode] = CachesOfAMode(mode=mode, layers=earlier.layers + 1, caches=entries)
    return tuple(found.values())


CACHE_TABLE_HEADER: tuple[str, str] = (
    '| attention mode | layers | caches, with the values each holds per token | '
    'values per token of one layer |', '|---|---|---|---|')


def cache_table(term: cat.Morphism = cached_glm53) -> str:
    '''The markdown table of `caches_by_mode`, with a last row for every layer, for a
    notebook cell.'''
    modes = caches_by_mode(term)
    rows = tuple(
        f'| {mode.mode} | {mode.layers} | '
        + ', '.join(f'${name}$ {values}' for name, values in mode.caches)
        + f' | {mode.values_per_token_of_a_layer():,} |'
        for mode in modes)
    total = (f'| every layer | {sum(mode.layers for mode in modes)} | | '
             f'{sum(mode.values_per_token() for mode in modes):,} in all |')
    return '\n'.join((*CACHE_TABLE_HEADER, *rows, total))
