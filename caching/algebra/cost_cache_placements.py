# Claude Opus 5.5 (1M context), effort 40.
'''Every placement of the caches of a pass, and what each placement costs.

`derive_cached_pass` places a cache on the operand of every causal read, which is the
last array on the path from a layer's input to the read. The operators on that path
after the copy feeding the queries are broadcast over the tokens, so a cache can stand
after any of them. The operators after the cache are then computed at every cached
token in every pass, and the array cached is the one they read.
`placements_by_sliding_caches_back` starts from the last placement, moves one cache
back past the operator it follows, and derives the pass again, until every set of
operators computed over the cache that the moves reach has been derived.

`cost_of_a_pass` reads three quantities off a derived pass at given sizes. The first
is the number of entries its caches hold per token, summed over every cache the pass
runs. The second is the operations of every linear map and contraction of the pass,
from `count_pass_operations`. The third is the bytes the caches move. A cache loads the
kept earlier tokens of its cached axis and stores the `|n|` new tokens, for every entry
it holds per token, so a cache of every earlier token moves `|P| + |n|` tokens and a
cache kept for a window moves the size of the window. The seconds are the operations at
the matrix rate plus the bytes at the memory bandwidth of a
`collective_cost.MachineRates`. Every placement reads the same weights and the same
arrays of the new tokens, so the cost leaves them out.

`obsidian/08-caching/Deriving Caches by Dragging the New Tokens.md` compares the
placements of multi-head latent attention.
'''
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import data_structure.Category as cat
import data_structure.Operators as ops
import performance_modeling.collective_cost as collective_cost

import caching.algebra.absorb_into_the_queries as absorb_into_the_queries
import caching.algebra.cache_contents as cache_contents
import caching.algebra.count_pass_operations as count_pass_operations
import caching.algebra.derive_cached_pass as derive_cached_pass
import caching.data_structure.Caching as Caching


@dataclass(frozen=True)
class PassCost:
    '''What one pass costs at given sizes: the entries its caches hold per token, the
    operations of its linear maps and contractions, the bytes its caches load and
    store, and the seconds of the two at the rates of a machine.'''
    entries_per_token: int
    operations: int
    cache_bytes: int
    seconds: float


def placements_by_sliding_caches_back[B: cat.Datatype, A: cat.Axis](
        model: cat.BroadcastedCategory[B, A], tokens: A, past_tokens: A, new_tokens: A,
        ) -> tuple[derive_cached_pass.CachedPass[B, A], ...]:
    '''The pass of `model` derived with every set of operators computed over the
    cache that moving one cache back past the operator it follows, repeatedly, reaches
    from the last placement, in the order the moves reach them.'''
    derived: dict[frozenset[cat.Operator], derive_cached_pass.CachedPass[B, A]] = {}
    frontier: list[frozenset[cat.Operator]] = [frozenset()]
    while frontier:
        computed_over_the_cache = frontier.pop(0)
        if computed_over_the_cache in derived:
            continue
        cached_pass = derive_cached_pass.derive_cached_pass(
            model, tokens, past_tokens, new_tokens, computed_over_the_cache)
        derived[computed_over_the_cache] = cached_pass
        frontier.extend(
            computed_over_the_cache | {operator}
            for operator in cached_pass.operators_before_caches
            if operator is not None and operator not in computed_over_the_cache)
    return tuple(derived.values())


def entries_per_token_of_the_caches(cached_pass: derive_cached_pass.CachedPass,
                                    sizes: Mapping[str, int]) -> int:
    return sum(cache_contents.entries_per_token(cache, sizes)
               for cache in cache_contents.caches_in_the_order_they_run(
                   cached_pass.expression))


def cost_of_a_pass(cached_pass: derive_cached_pass.CachedPass, sizes: Mapping[str, int],
                   machine: collective_cost.MachineRates) -> PassCost:
    '''The cost of `cached_pass` with every size bound by the bodies of its name,
    `|P|` and `|n|` among them.'''
    entries = entries_per_token_of_the_caches(cached_pass, sizes)
    operations = count_pass_operations.operations_at_sizes(
        tuple(count_pass_operations.operations_of_a_pass(cached_pass.expression)), sizes)
    cache_bytes = machine.bytes_per_element * sum(
        cache_contents.entries_per_token(cache, sizes)
        * cache_contents.size_of(Caching.cached_tokens_of(cache).local_size(), sizes)
        for cache in cache_contents.caches_in_the_order_they_run(cached_pass.expression))
    return PassCost(
        entries_per_token=entries, operations=operations, cache_bytes=cache_bytes,
        seconds=(operations / machine.matrix_operations_per_second
                 + cache_bytes / machine.memory_bytes_per_second))


def cheapest_placement(placements: tuple[derive_cached_pass.CachedPass, ...],
                       sizes: Mapping[str, int], machine: collective_cost.MachineRates,
                       ) -> derive_cached_pass.CachedPass:
    '''The placement whose pass takes the fewest seconds, and of those the one
    computing the fewest operators over the cache, since the cost counts the linear
    maps and the contractions alone.'''
    return min(placements, key=lambda cached_pass: (
        cost_of_a_pass(cached_pass, sizes, machine).seconds,
        len(cached_pass.computed_over_the_cache)))


def first_linear_placement(placements: tuple[derive_cached_pass.CachedPass, ...],
                           ) -> derive_cached_pass.CachedPass:
    '''The placement whose every cache follows an `ops.Linear`, with the most
    operators computed over the cache, which caches the result of the first linear
    map on every path from the copy feeding the queries to a causal read.'''
    return max((cached_pass for cached_pass in placements
                if cached_pass.operators_before_caches
                and all(isinstance(operator, ops.Linear)
                        for operator in cached_pass.operators_before_caches)),
               key=lambda cached_pass: len(cached_pass.computed_over_the_cache))


def narrowest_placement(placements: tuple[derive_cached_pass.CachedPass, ...],
                        sizes: Mapping[str, int]) -> derive_cached_pass.CachedPass:
    '''The placement whose caches hold the fewest entries per token, and of those the
    one computing the fewest operators over the cache.'''
    return min(placements, key=lambda cached_pass: (
        entries_per_token_of_the_caches(cached_pass, sizes),
        len(cached_pass.computed_over_the_cache)))


def cheapest_pass_in_either_order(placements: tuple[derive_cached_pass.CachedPass, ...],
                                  sizes: Mapping[str, int],
                                  machine: collective_cost.MachineRates,
                                  ) -> derive_cached_pass.CachedPass:
    '''The pass taking the fewest seconds among every placement, each in the order it
    is written and with its linear maps over the cache absorbed into the queries at
    `sizes`.'''
    passes = tuple(
        written_or_absorbed
        for cached_pass in placements
        for written_or_absorbed in (
            cached_pass,
            absorb_into_the_queries.with_linear_maps_absorbed(cached_pass, sizes)))
    return min(passes, key=lambda cached_pass: (
        cost_of_a_pass(cached_pass, sizes, machine).seconds,
        len(cached_pass.computed_over_the_cache)))
