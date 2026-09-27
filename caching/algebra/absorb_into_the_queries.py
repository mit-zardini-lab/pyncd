# Claude Opus 5.5 (1M context), effort 40.
'''A derived pass with every linear map computed over its caches contracted against the
queries of the pass instead, where that order costs fewer operations.

A placement of the caches that computes a linear map over the cache recomputes that map
for every earlier token in every pass. Multi-head latent attention is the case: with
the latent cached, `W^{UK}` and `W^{UV}` expand every cached latent into a key and a
value. The contraction that reads the expanded key is linear in it, so the same result
is reached by contracting the queries against `W^{UK}` and the result against the
cached latent, and likewise for the values after the weighted sum.
`advanced_axis_dynamics.algebra.absorb_linear_maps` chooses between the two orders by
their operation counts at bound sizes. A derived pass is the form in which that choice
can be made, because its queries stand on the new tokens and its keys on every cached
token, so the two counts differ in the ratio of `|n|` to `|P| + |n|`.

`with_linear_maps_absorbed` prepares the pass the way the rewrite needs it. It writes
out every block whose repetition is one, reads every strided view through one stride
morphism, splits every contraction over a concatenated axis into one contraction per
part with `concatenation_expansion`, and writes every `ops.Linear` computed over the
cache as its weight contracted against its operand. It then applies the rewrite at the
sizes of the pass, and writes every view as the product of the independent factors of
its reindexing again, so that a figure draws the axes a view leaves alone as straight
wires.
'''
from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import advanced_axis_dynamics.algebra.absorb_linear_maps as absorb_linear_maps
import advanced_axis_dynamics.algebra.concatenation_expansion as concatenation_expansion
import advanced_axis_dynamics.algebra.disentangle_reindexings as disentangle_reindexings
import algebra.discovering_broadcasts as discovering_broadcasts
import algebra.linear_expansion as linear_expansion
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import graphs.processing.hypergraph_functor as hypergraph_functor
import term_utilities.term_utilities as tutil

import caching.algebra.derive_cached_pass as derive_cached_pass


@dataclasses.dataclass
class ExpandLinearMaps(hypergraph_functor.Endofunctor[cat.Array, cat.Broadcasted]):
    '''Every `ops.Linear` among `linear_maps` written as its weight contracted against
    its operand.'''
    linear_maps: frozenset[cat.Operator] = frozenset()

    def apply_root(self, target: cat.Broadcasted) -> cat.BroadcastedCategory:
        if (isinstance(target, cat.Broadcasted) and isinstance(target.operator, ops.Linear)
                and target.operator in self.linear_maps):
            return linear_expansion.expand_linear_root(target)
        return target


@dataclasses.dataclass
class FactorViews(hypergraph_functor.Endofunctor[cat.Array, cat.Broadcasted]):
    '''Every view with its reindexing written as the product of its independent
    factors.'''

    def apply_root(self, target: cat.Broadcasted) -> cat.Broadcasted:
        if not (isinstance(target, cat.Broadcasted) and isinstance(target.operator, ops.View)):
            return target
        factored = tuple(disentangle_reindexings.disentangle_reindexing(reindexing)
                         for reindexing in target.reindexings)
        if all(before is after for before, after in zip(target.reindexings, factored)):
            return target
        return target.reconstruct(reindexings=factored)


def bindings_by_name(term: object, sizes: Mapping[str, int]) -> dict[nm.Numeric, int]:
    '''Every symbol of `term` that `sizes` names by the bodies of its name, bound to its
    size.'''
    return {symbol: sizes[symbol.uid._name.to_bodies()]
            for symbol in tutil.type_search(nm.FreeNumeric, term)
            if symbol.uid._name is not None and symbol.uid._name.to_bodies() in sizes}


def with_linear_maps_absorbed(cached_pass: derive_cached_pass.CachedPass,
                              sizes: Mapping[str, int]
                              ) -> derive_cached_pass.CachedPass:
    '''`cached_pass` with every linear map computed over its caches contracted in the
    order that costs the fewest operations at `sizes`.'''
    linear_maps = frozenset(operator for operator in cached_pass.computed_over_the_cache
                            if isinstance(operator, ops.Linear))
    if not linear_maps:
        return cached_pass
    written_out = discovering_broadcasts.remove_grouping_blocks(cached_pass.expression)
    split = concatenation_expansion.expand_concatenations(
        absorb_linear_maps.read_through_one_stride_morphism(written_out))
    contracted = ExpandLinearMaps(linear_maps=linear_maps)(split)
    absorbed = absorb_linear_maps.absorb_linear_maps(
        contracted, bindings_by_name(contracted, sizes))
    return dataclasses.replace(cached_pass, expression=FactorViews()(absorbed))

