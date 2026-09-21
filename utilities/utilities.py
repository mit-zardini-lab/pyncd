'''The collection operations the rest of the package is written against.

`Prod[T]` is the tuple alias that every field held inside a `Term` is annotated with.
Every function here returns a new tuple or a generator, and none of them mutates its
argument.

`Multidict` is the one-to-many mapping the package uses. It holds the values for a key
in a `set`, so iteration reaches them in an order that is neither the insertion order
nor the same order in a later process. `UID._id` is drawn at random when the process
starts, which is what makes the order vary. Code that iterates over a set of terms must
sort them by `uid._id` before it acts on them. Without the sort, a rewrite produces a
different result on each run.
'''

from typing import (
    Callable,
    Iterator,
    Type,
    Iterable
)
from enum import Enum

type Prod[T] = tuple[T, ...]
class AllEqualsFallback(Enum):
    '''The sentinel `iallequals` reads as "no fallback was supplied".

    A plain default of `None` would be indistinguishable from a caller asking for
    `None` when the elements differ.
    '''
    RAISE = 'RAISE'

def iallequals[T](xs: Iterable[T], fallback: AllEqualsFallback | T = AllEqualsFallback.RAISE) -> T:
    '''The single value every element of `xs` equals.

    Returns `fallback` when two elements differ, and raises `ValueError` when
    `fallback` is `RAISE`. An empty `xs` raises `StopIteration`.
    '''
    iterator = iter(xs)
    _first = next(iterator)
    while True:
        try:
            x = next(iterator)
            if x != _first:
                if fallback != AllEqualsFallback.RAISE:
                    return fallback
                raise ValueError(f"Elements are not all equal: {_first} != {x}")
        except StopIteration:
            break
    return _first

def yielder[K, V](target: Iterable[K], generator: Callable[[K], V]) -> Iterator[V]:
    '''One value per key in `target`, with `generator` called once per distinct key.

    A key that appears twice yields the same value both times.
    '''
    previous: dict[K, V] = {}
    for key in target:
        if key not in previous:
            previous[key] = generator(key)
        yield previous[key]

def join_with_none[T](target: Iterable[T], separator: T | None) -> Iterator[T]:
    '''`target` with `separator` between consecutive items.

    A `separator` of `None` yields `target` unchanged. No separator is placed before
    the first item or after the last.
    '''
    iterator = iter(target)
    if separator is None:
        yield from iterator
        return
    try:
        next_one = next(iterator)
    except StopIteration:
        return ()
    while True:
        yield next_one
        try:
            next_one = next(iterator)
        except StopIteration:
            break
        yield separator

def unique_iterable[T, S=T](target: Iterable[T], unique_test: Callable[[T], S] = lambda x: x) -> Iterator[T]:
    '''The first item carrying each distinct `unique_test(item)`, in arrival order.'''
    seen: set[S] = set()
    for item in target:
        key = unique_test(item)
        if key not in seen:
            seen.add(key)
            yield item

def unique_tuple[T, S=T](target: Iterable[T], unique_test: Callable[[T], S] = lambda x: x) -> Prod[T]:
    return tuple(unique_iterable(target, unique_test=unique_test))

def difference[S](first: Iterable[S], second: Iterable[S]) -> Prod[S]:
    '''The items of `first` absent from `second`, in the order of `first`.

    A duplicate in `first` survives as a duplicate.
    '''
    set_second = set(second)
    return tuple(x for x in first if x not in set_second)

def deconcatenate(target: Prod[int], dom_length: int | None = None) -> Prod[tuple[int, int]]:
    '''Every way of cutting a rearrangement's mapping into two independent halves.

    `target[k]` is the domain position that codomain position `k` draws from. A pair
    `(L0, R0)` is returned when the first `L0` codomain positions are exactly the ones
    drawing from the first `R0` domain positions. The mapping is then the parallel
    product of the two halves.

    The pair `(0, 0)`, which cuts nothing off the left, is dropped. `dom_length`
    defaults to one past the largest entry of `target`.
    '''
    dom_length = dom_length or max(target) + 1
    return tuple(
        (L0, R0)
        for L0 in range(len(target))
        for R0 in range(dom_length)
        if all((k < L0) == (target[k] < R0) for k in range(len(target)))
    )[1:]

def concat[T, Y=T](xss: Iterable[Iterable[T]], func: Callable[[T], Y] = lambda x: x) -> Prod[Y]:
    return tuple(
        func(x)
        for xs in xss
        for x in xs
    )

def unique_concat[T, S=T](target: Iterable[Iterable[T]], unique_test: Callable[[T], S] = lambda x: x) -> Iterator[T]:
    return unique_iterable(
        concat(target),
        unique_test=unique_test
    )

def intersection[T, S=T](xs: Iterable[T], ys: Iterable[T], unique_test: Callable[[T], S] = lambda x: x) -> Prod[T]:
    '''The items of `xs` whose key appears among the keys of `ys`, in the order of `xs`.

    A duplicate in `xs` survives as a duplicate, so the result can be longer than
    either argument.
    '''
    set_ys = set(unique_test(y) for y in ys)
    return tuple(x for x in xs if unique_test(x) in set_ys)


def predicate_partition[T](xs: Iterable[T], predicate: Callable[[T], bool]) -> tuple[Prod[T], Prod[T]]:
    '''The items satisfying `predicate`, then the items that do not.'''
    true_part: list[T] = []
    false_part: list[T] = []
    for x in xs:
        (true_part if predicate(x) else false_part).append(x)
    return tuple(true_part), tuple(false_part)

class Multidict[K, V]:
    '''A mapping from a key to a set of values.

    A key absent from the mapping reads as the empty set, so `__getitem__` never
    raises and `append` never has to be preceded by a check.

    The values for a key are held in a `set`. Iterating one reaches the values in an
    order that varies between processes, because `UID._id` is drawn at random when the
    process starts. Sort a set of terms by `uid._id` before acting on it.
    '''

    def __init__(self, generator: Iterable[tuple[K, V]] | None = None) -> None:
        if generator is None:
            generator = ()
        self.mapping: dict[K, set[V]] = {}
        self.update(generator)

    def __getitem__(self, key: K) -> set[V]:
        return self.mapping.get(key, set())
        
    def __contains__(self, key: K) -> bool:
        return key in self.mapping
    
    def union(self, targets: Iterable[K]) -> set[V]:
        '''The union of the value sets at every key in `targets`.'''
        return set().union(*(self.mapping.get(k, set()) for k in targets))

    def __iter__(self) -> Iterator[K]:
        return iter(self.mapping)

    def find_key(self, value: V) -> K | None:
        '''The first key whose set contains `value`, or None when no key does.

        The keys are searched in the order they were first added.
        '''
        for key, values in self.mapping.items():
            if value in values:
                return key
        return None

    def items(self) -> Iterable[tuple[K, set[V]]]:
        return self.mapping.items()

    def keys(self) -> Iterable[K]:
        return self.mapping.keys()

    def append(self, key: K, *value: V) -> None:
        '''Add each `value` to the set at `key`, creating the set when the key is new.'''
        if key not in self.mapping:
            self.mapping[key] = set()
        self.mapping[key].update(value)

    def update(self, generator: Iterable[tuple[K, V]]) -> 'Multidict[K, V]':
        '''Add every `(key, value)` pair, and return this same mapping.'''
        for key, value in generator:
            if key not in self.mapping:
                self.mapping[key] = set()
            self.mapping[key].add(value)
        return self