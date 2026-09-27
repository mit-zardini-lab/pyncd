# Claude Opus 5.5 (1M context), effort 40.
'''The operator an array passes through to be kept for the passes that follow.

A model that generates text runs one pass per step. The first pass reads the prompt,
and every later pass reads the tokens appended since the pass before. The keys an
attention reads for the tokens of earlier passes were computed by those passes, and a
cache keeps them so that no pass computes them again.

`Caching` is that cache as one operator. Its operand is the array this pass computes
for its own tokens, which is what the pass saves. Its result is the same array over
every token the cache holds once the save is done, which is what the pass loads. The
tokens of this pass are an axis `x`, the tokens cached by earlier passes are an axis
`P`, and the result stands on the axis `P + x`, an
`AxisConcatenation.ConcatenatedAxis` whose positions are those of `P` followed by those
of `x`. Token `i_x` of this pass therefore stands at position `|P| + i_x` of the result.
Every other axis of the operand is broadcast, because the cache keeps each channel of a
token apart from the others.

The first part of the cached axis may also be the last `|K|` earlier tokens alone, the
kept tokens `K`, which a sliding window of `|K| + 1` slots needs. The result then stands
on `K + x`, and token `i_x` of this pass stands at position `|K| + i_x` of it. A
position of `K` before the first token of the sequence holds the universal unit.
`caching.algebra.derive_cached_pass` chooses between the two by the reach of the causal
read the cache serves.

The operator is the `update` of a `DynamicLayer` in the cache of `transformers`, which
concatenates the states it is handed onto the states it holds and returns the whole.
`caching/registries/standard_expansions.py` writes the operator out in those terms: the
array held for the tokens of `P` is read from the tape, the operand is concatenated
after it, and the concatenation is written back to the tape and handed on.

A `Caching` inside a repeated block stands for one cache per iteration, as a weight
inside a repeated block stands for one weight per iteration.

`obsidian/08-caching/Caching Between Passes.md` states the feature.
'''
from __future__ import annotations

from dataclasses import dataclass

import advanced_axis_dynamics.data_structure.AxisConcatenation as AxisConcatenation
import data_structure.Category as cat
import data_structure.Term as fd


class CachedAxisIsNotPastThenThisPass(ValueError):
    '''An axis handed to `Caching.template` as the axis of the cached tokens that is not
    the concatenation of the past tokens and the tokens of this pass, in that order.'''


def cached_token_axis[A: cat.Axis](
    past_tokens: A,
    tokens_of_this_pass: A,
) -> AxisConcatenation.ConcatenatedAxis:
    '''The axis of every token the cache holds after this pass, labelled `P + x`: the
    tokens cached by earlier passes followed by the tokens of this pass.'''
    return AxisConcatenation.ConcatenatedAxis(parts=(past_tokens, tokens_of_this_pass))


@dataclass(frozen=True)
class Caching(cat.Operator):
    '''The save of an array into a cache and the load of the whole cache, as one
    operator. The operand holds the tokens of this pass on its target position, and the
    result holds every cached token there, with the tokens of this pass last.

    `name` names the cache, and the figure draws it on the box.
    '''
    name: fd.DynamicName | None = None

    @classmethod
    def template[B: cat.Datatype, A: cat.Axis](
        cls,
        saved_shape: fd.Prod[A],
        cached_tokens: AxisConcatenation.ConcatenatedAxis,
        name: str | fd.DynamicName,
        tokens_position: int = 0,
        base: B = cat.Reals(),
    ) -> cat.Broadcasted[B, A, Caching]:
        '''The cache named `name` of an array of `saved_shape`, whose token axis stands
        at `tokens_position` and is the last part of `cached_tokens`. Every other axis
        of the shape is the degree.'''
        tokens_of_this_pass = saved_shape[tokens_position]
        if (not isinstance(cached_tokens, AxisConcatenation.ConcatenatedAxis)
                or len(cached_tokens.parts) != 2
                or cached_tokens.parts[1] != tokens_of_this_pass):
            raise CachedAxisIsNotPastThenThisPass(
                f'{cached_tokens} is not the past tokens followed by '
                f'{tokens_of_this_pass}')
        degree = (*saved_shape[:tokens_position], *saved_shape[tokens_position + 1:])
        leading = (cat.WeaveMode.TILED,) * tokens_position
        trailing = (cat.WeaveMode.TILED,) * (len(degree) - tokens_position)
        return cat.Broadcasted(
            operator=cls(name=fd.DynamicName.from_str(name)),
            input_weaves=(cat.Weave(base, (*leading, tokens_of_this_pass, *trailing)),),
            output_weaves=(cat.Weave(base, (*leading, cached_tokens, *trailing)),),
            reindexings=(cat.ProdObject(degree).identity(),))


def tokens_position(target: cat.Broadcasted[cat.Datatype, cat.Axis, Caching]) -> int:
    '''The position of the token axis in the operand and in the result of a cache.'''
    return next(position
                for position, entry in enumerate(target.input_weaves[0]._shape)
                if not isinstance(entry, cat.WeaveMode))


def cached_tokens_of(
    target: cat.Broadcasted[cat.Datatype, cat.Axis, Caching],
) -> AxisConcatenation.ConcatenatedAxis:
    '''The axis `P + x` of every token the cache holds after this pass.'''
    return target.output_weaves[0]._shape[tokens_position(target)]


def past_tokens_of[A: cat.Axis](
    target: cat.Broadcasted[cat.Datatype, A, Caching],
) -> A:
    '''The axis `P` of the tokens cached by earlier passes.'''
    return cached_tokens_of(target).parts[0]


def tokens_of_this_pass[A: cat.Axis](
    target: cat.Broadcasted[cat.Datatype, A, Caching],
) -> A:
    '''The axis `x` of the tokens this pass saves.'''
    return target.input_weaves[0]._shape[tokens_position(target)]


def past_array_of[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A, Caching],
) -> cat.Array[B, A]:
    '''The array the cache holds before this pass: the result with `P` in place of
    `P + x`.'''
    loaded = target.cod()[0]
    position = tokens_position(target)
    shape = tuple(loaded.shape())
    return cat.Array(loaded.datatype, (*shape[:position], past_tokens_of(target),
                                       *shape[position + 1:]))
