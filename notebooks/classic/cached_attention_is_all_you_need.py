# Claude Opus 5.5 (1M context), effort 40.
'''The pass of the transformer of *Attention Is All You Need* that generates new target
tokens and reads the earlier target tokens from caches.

A translation is generated one target token at a time, and every step runs the decoder
again. The step needs the model's results at the new target tokens alone.
`caching.algebra.derive_cached_pass` derives that step from a model written in
`attention_is_all_you_need`. It splits the target axis `y` into the tokens of the
earlier steps, `EARLIER_TARGETS`, and the tokens of this step, `NEW_TARGETS`, carries
the read of the new tokens back through the model, and places a cache wherever the
masked self-attention reads an earlier token.

`placements_over_new_targets` returns every placement reached by the derivation when it
moves a cache back past the operator it follows. `reference_placement` is the one run
by tensor2tensor, with the caches directly after `W^{K}` and `W^{V}` in every decoder
layer, which is the placement that computes no operator over the cache.
`notebooks/website/classic/AttentionIsAllYouNeed.ipynb` cites the lines of
tensor2tensor that hold the cache.

The decoder grabs the encoded input `A` from the tape, and the derivation leaves that
grab and the cross-attention reading it as they are, because they read no target token.
Derived from the whole model, the pass therefore runs the encoder over the whole source
sentence in every step. tensor2tensor runs the encoder once per sentence, and once per
sentence projects the encoded input into the keys and the values of every
cross-attention. The derivation states no value computed once per sentence, and
`obsidian/06-practice/Open Gaps.md` records the gap.

`obsidian/08-caching/Deriving Caches by Dragging the New Tokens.md` states the
derivation and proves that the derived pass computes the model's results for the new
tokens.
'''
from __future__ import annotations

from collections.abc import Iterable

import caching.algebra.cost_cache_placements as cost_cache_placements
import caching.algebra.derive_cached_pass as derive_cached_pass
import caching.registries.standard_expansions  # noqa: F401 - the expansion of a cache
import data_structure.Category as cat
import data_structure.Term as fd

import notebooks.classic.attention_is_all_you_need as transformer

EARLIER_TARGETS = cat.RawAxis.named(fd.DynamicName(
    'y', fd.DynamicName('old'), code_form='earlier_target_positions'))
NEW_TARGETS = cat.RawAxis.named(fd.DynamicName(
    'y', fd.DynamicName('new'), code_form='new_target_positions'))

type PassOverNewTargets = derive_cached_pass.CachedPass[cat.Datatype, cat.Axis]


def derive_pass_over_new_targets(
    model: cat.BroadcastedCategory,
    computed_over_the_cache: Iterable[cat.Operator] = (),
) -> PassOverNewTargets:
    '''The pass of `model` that computes its results at the new target positions, with
    the operators of `computed_over_the_cache` computed at every cached position.'''
    return derive_cached_pass.derive_cached_pass(
        model, transformer.y, EARLIER_TARGETS, NEW_TARGETS, computed_over_the_cache)


def placements_over_new_targets(
    model: cat.BroadcastedCategory,
) -> tuple[PassOverNewTargets, ...]:
    '''The pass of `model` with its caches at every placement reached by moving a cache
    back past the operator it follows.'''
    return cost_cache_placements.placements_by_sliding_caches_back(
        model, transformer.y, EARLIER_TARGETS, NEW_TARGETS)


def reference_placement() -> PassOverNewTargets:
    '''The pass of the decoder with every cache on the operand of the causal read,
    directly after `W^{K}` and `W^{V}`, which is the cache of tensor2tensor.'''
    return derive_pass_over_new_targets(transformer.decode())
