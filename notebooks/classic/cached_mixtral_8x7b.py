# Claude Opus 5.5 (1M context), effort 40.
'''The pass of Mixtral-8x7B over the new tokens, reading the keys and the values of
the earlier tokens from caches, derived from the model.

A pass of generation appends the new tokens `x_new` to the tokens `x_old` of the earlier
passes and needs the model's results at the new tokens alone.
`caching.algebra.derive_cached_pass` carries that read back through the model and
places a `Caching` wherever a causal read reaches an earlier token.
`cost_cache_placements.placements_by_sliding_caches_back` derives every placement of
the caches that moving one cache back past the operator it follows reaches.

The reference caches the keys after the rotary embedding and the values, one of each per
key-value head and per earlier token, in every layer: `Attention.forward` turns `xk` and
then hands it with `xv` to `CacheView.update`, and `BufferCache` holds one array of each
of shape `(batch, cache_size, n_kv_heads, head_dim)` per layer in the datatype of the
model. That is the placement with no operator computed over the cache, which stands on
the operands of the two causal reads of the model as it is built. The released
configuration sets no sliding window, so `get_cache_sizes` gives every layer a cache of
the whole sequence, and the derivation keeps every earlier token, because the reach of
the causal mask grows with the past.

The quantised pass is derived from the quantised model, so each cache holds its
values at the quantisation they arrive with, which is BF16 as in the reference. The
lines are pinned below and were read on 2026-09-27.
`obsidian/08-caching/Deriving Caches by Dragging the New Tokens.md` states the
derivation.
'''
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import caching.algebra.cache_contents as cache_contents
import caching.algebra.cost_cache_placements as cost_cache_placements
import caching.algebra.derive_cached_pass as derive_cached_pass
import data_structure.Category as cat
import data_structure.Term as fd
import performance_modeling.collective_cost as collective_cost
import performance_modeling.registries.machine_rates as machine_rates
import quantization.algebra.strip_quantisations as strip_quantisations
import quantization.processing.quantise_model as quantise_model

import notebooks.classic.mixtral_8x7b as mixtral_8x7b
import notebooks.classic.quantised_mixtral_8x7b as quantised_mixtral_8x7b
from notebooks.classic.reference_links import mistral_lines, mixtral_config_lines

EARLIER_TOKENS = cat.RawAxis.named(
    fd.DynamicName('x', fd.DynamicName('old'), code_form='cached_tokens'))
NEW_TOKENS = cat.RawAxis.named(
    fd.DynamicName('x', fd.DynamicName('new'), code_form='new_tokens'))

RELEASED_SIZES: dict[str, int] = {
    'm': 4096, "h'": 8, 'g': 4, 'd': 128, 'f': 14336, 'n': 8, 'k': 2, 'N': 32,
    'v': 32000}
'''The sizes of `config.json` of the released weights, keyed by the bodies of the names
of the symbols.'''

LAYER_COUNT = RELEASED_SIZES['N']
BYTES_PER_BF16 = 2

CACHE_REFERENCES: tuple[cat.CodeReference, ...] = (
    mistral_lines('transformer_layers.py', 66, 81),
    mistral_lines('cache.py', 83, 92),
    mistral_lines('cache.py', 13, 15),
    mistral_lines('cache.py', 160, 167),
    mistral_lines('generate.py', 67, 78),
    mixtral_config_lines(23))
'''The lines that state what the reference caches. `Attention.forward` projects the
keys and the values, turns the queries and the keys, and hands the turned keys and the
values to `CacheView.update`, which copies them into the cache of the layer. The cache
of each layer is as long as the sequence where no sliding window is set, holds
`(batch, cache_size, n_kv_heads, head_dim)` for the keys and for the values, and is
converted to the datatype of the model. The released configuration sets no sliding
window.'''


def placements_of(model: cat.BroadcastedCategory
                  ) -> tuple[derive_cached_pass.CachedPass, ...]:
    return cost_cache_placements.placements_by_sliding_caches_back(
        model, mixtral_8x7b.x, EARLIER_TOKENS, NEW_TOKENS)


def placement_on_the_operands_of_the_causal_reads(
    placements: tuple[derive_cached_pass.CachedPass, ...],
) -> derive_cached_pass.CachedPass:
    '''The placement with no operator computed over the cache, whose caches stand on
    the operands of the causal reads.'''
    return next(placement for placement in placements
                if not placement.computed_over_the_cache)


CACHED_PASS = derive_cached_pass.derive_cached_pass(
    quantised_mixtral_8x7b.MIXTRAL, mixtral_8x7b.x, EARLIER_TOKENS, NEW_TOKENS)
'''The pass with every cache on the operand of a causal read of the model as it is
built, which is the placement with no operator computed over the cache.'''

mixtral_cached = CACHED_PASS.expression

QUANTISED_CACHED_PASS = derive_cached_pass.derive_cached_pass(
    quantise_model.quantise_model(
        quantised_mixtral_8x7b.MIXTRAL,
        quantised_mixtral_8x7b.RELEASED_POLICY).morphism,
    mixtral_8x7b.x, EARLIER_TOKENS, NEW_TOKENS)
'''The pass derived from the quantised model as it is built, so that every wire of the
pass and every cache carries the quantisation of the value held there.'''

mixtral_cached_quantised = QUANTISED_CACHED_PASS.expression

mixtral_cached_without_quantisations = strip_quantisations.strip_quantisations(
    mixtral_cached_quantised)


# ==========================================================================
# What each placement caches, and what one pass costs.
# ==========================================================================
H100 = machine_rates.H100_SXM5
MACHINE = collective_cost.MachineRates(
    matrix_operations_per_second=H100.matrix_operations_per_second,
    scalar_operations_per_second=H100.scalar_operations_per_second,
    memory_bytes_per_second=H100.memory_bytes_per_second,
    network_bytes_per_second=H100.network_bytes_per_second,
    collective_latency_seconds=H100.collective_latency_seconds,
    memory_bytes_per_processor=int(H100.memory_bytes_per_processor),
    bytes_per_element=BYTES_PER_BF16)
'''An H100 SXM at its dense BF16 rate and its memory bandwidth, with two bytes for
every cached value.'''

EARLIER_TOKEN_COUNT = 32767
DECODE_SIZES: dict[str, int] = {
    **RELEASED_SIZES, 'xold': EARLIER_TOKEN_COUNT, 'xnew': 1}
'''One new token after 32,767 earlier ones, which fills the 32,768 positions the
Mixtral paper states as the context.'''

LAYER_PLACEMENTS = placements_of(mixtral_8x7b.decoder_layer().body)
'''The placements of one decoder layer, whose repetition is left out so that its
caches and its operations are counted once.'''


def values_per_token(placement: derive_cached_pass.CachedPass,
                     sizes: Mapping[str, int] = RELEASED_SIZES) -> int:
    '''The values the caches of one layer of `placement` keep for one token.'''
    return sum(cache_contents.entries_per_token(cache, sizes)
               for cache in placement.caches())


def cache_names(placement: derive_cached_pass.CachedPass) -> tuple[str, ...]:
    return tuple(sorted(cache_contents.cache_name(cache)
                        for cache in placement.caches()))


@dataclass(frozen=True)
class PlacementRow:
    '''One placement of the caches of a layer: the names of its caches, the values
    they keep per token in one layer, the operators computed over the cache, and the
    microseconds of one layer of a pass at `DECODE_SIZES` on `MACHINE`.'''
    caches: tuple[str, ...]
    values_per_token: int
    computed_over_the_cache: int
    microseconds: float


def placement_rows() -> tuple[PlacementRow, ...]:
    '''Every placement of one layer, from the one standing on the operands of the
    causal reads to the one caching the normalised state once.'''
    rows = tuple(
        PlacementRow(
            caches=cache_names(placement),
            values_per_token=values_per_token(placement),
            computed_over_the_cache=len(placement.computed_over_the_cache),
            microseconds=cost_cache_placements.cost_of_a_pass(
                placement, DECODE_SIZES, MACHINE).seconds * 1e6)
        for placement in LAYER_PLACEMENTS)
    return tuple(sorted(rows, key=lambda row: (row.computed_over_the_cache,
                                               row.caches)))


PLACEMENT_TABLE_HEADER: tuple[str, str] = (
    '| caches | values per token in one layer | operators computed over the cache '
    '| µs per layer for one new token |', '|---|---|---|---|')


def placement_table() -> str:
    '''The markdown table of `placement_rows`, for a notebook cell.'''
    rows = tuple(
        f'| {", ".join(f"${name}$" for name in row.caches)} | {row.values_per_token:,} '
        f'| {row.computed_over_the_cache} | {row.microseconds:,.1f} |'
        for row in placement_rows())
    return '\n'.join((*PLACEMENT_TABLE_HEADER, *rows))


def cached_bytes_per_token() -> int:
    '''The bytes the reference's caches keep for one token over every layer.'''
    return (values_per_token(CACHED_PASS) * LAYER_COUNT * BYTES_PER_BF16)
