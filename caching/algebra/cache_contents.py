# Claude Opus 5.5 (1M context), effort 40.
'''The caches of a term, in the order they run, and what each keeps per token.

`caches_in_the_order_they_run` walks a term in the order its parts compose and writes a
repeated block out as many times as it repeats, so a cache inside a block of repetition
3 inside a block of repetition 18 appears 54 times. The result is one entry per cache
per pass, which is one array held by the cache of the reference for every layer.

`caches_by_outermost_box` groups the same sequence by the outermost `ops.BlockOperator`
holding each cache, so the caches of one attention sublayer stay together however the
recycling of its box ordered them.

`entries_per_token` is the number of values a cache keeps for one token: the product of
the sizes of every axis of its operand but the token axis. A size is a numeric over the
symbols of the term, and `sizes` binds each symbol by the bodies of its name, as
`term_utilities.generate_config.NumericConfig.assign_values` binds them.
'''
from __future__ import annotations

import math
from collections.abc import Iterator, Mapping

import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import para.data_structure.ParaWrap as para_wrap
import performance_modeling.morphism_work as morphism_work
import term_utilities.term_utilities as tutil

import caching.data_structure.Caching as Caching


type CacheNode = cat.Broadcasted[cat.Datatype, cat.Axis, Caching.Caching]


class RepetitionIsNotAnInteger(ValueError):
    '''A repeated block whose repetition is a symbol, so it cannot be written out.'''


class SizeIsNotBound(ValueError):
    '''A size holding a symbol that `sizes` does not name.'''


def caches_in_the_order_they_run(term: object) -> Iterator[CacheNode]:
    '''Every `Caching` of `term`, once for every time it runs in one pass.'''
    match term:
        case cat.Composed(content=parts) | cat.ProductOfMorphisms(content=parts):
            for part in parts:
                yield from caches_in_the_order_they_run(part)
        case cat.Block(body=body, block_tag=block_tag):
            if not isinstance(block_tag.repetition, nm.Integer):
                raise RepetitionIsNotAnInteger(
                    f'the repetition {block_tag.repetition.to_latex()} is no integer')
            for _ in range(block_tag.repetition._value):
                yield from caches_in_the_order_they_run(body)
        case para_wrap.ParaWrap(body=body):
            yield from caches_in_the_order_they_run(body)
        case cat.Broadcasted(operator=Caching.Caching()):
            yield term
        case cat.Broadcasted(operator=ops.BlockOperator(block=block)):
            yield from caches_in_the_order_they_run(block)


def caches_by_outermost_box(
    term: object,
) -> Iterator[tuple[str, tuple[CacheNode, ...]]]:
    '''The name of every outermost box of `term` holding a cache, with the caches it
    holds, once for every time the box runs in one pass.'''
    match term:
        case cat.Composed(content=parts) | cat.ProductOfMorphisms(content=parts):
            for part in parts:
                yield from caches_by_outermost_box(part)
        case cat.Block(body=body, block_tag=block_tag):
            if not isinstance(block_tag.repetition, nm.Integer):
                raise RepetitionIsNotAnInteger(
                    f'the repetition {block_tag.repetition.to_latex()} is no integer')
            for _ in range(block_tag.repetition._value):
                yield from caches_by_outermost_box(body)
        case para_wrap.ParaWrap(body=body):
            yield from caches_by_outermost_box(body)
        case cat.Broadcasted(operator=ops.BlockOperator(block=block, name=name)):
            held = tuple(caches_in_the_order_they_run(block))
            if held:
                yield (name.to_bodies() if name is not None else '', held)


def size_of(size: nm.Numeric, sizes: Mapping[str, int]) -> int:
    '''`size` with every free symbol bound by the bodies of its name.'''
    names = {symbol: symbol.uid._name.to_bodies() if symbol.uid._name else None
             for symbol in tutil.type_search(nm.FreeNumeric, size)}
    unbound = sorted(str(name) for name in names.values() if name not in sizes)
    if unbound:
        raise SizeIsNotBound(f'{size.to_latex()} holds the unbound symbols {unbound}')
    bindings = {symbol: sizes[name] for symbol, name in names.items()}
    return int(morphism_work.numeric_value(size, bindings))


def entries_per_token(cache: CacheNode, sizes: Mapping[str, int]) -> int:
    '''The number of values `cache` keeps for one token.'''
    position = Caching.tokens_position(cache)
    shape = tuple(cache.dom()[0].shape())
    return math.prod(size_of(axis.local_size(), sizes)
                     for index, axis in enumerate(shape) if index != position)


def cache_name(cache: CacheNode) -> str:
    return cache.operator.name.to_text()
