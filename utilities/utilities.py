from typing import (
    Callable,
    Iterator,
    Type,
    Iterable
)
from enum import Enum

type Prod[T] = tuple[T, ...]
class AllEqualsFallback(Enum):
    RAISE = 'RAISE'

def iallequals[T](xs: Iterable[T], fallback: AllEqualsFallback | T = AllEqualsFallback.RAISE) -> T:
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

def yielder[K, V](target: Iterable[K], generator: Callable[[K], V]):
    previous: dict[K, V] = {}
    for key in target:
        if key not in previous:
            previous[key] = generator(key)
        yield previous[key]

def join_with_none[T](target: Iterable[T], separator: T | None) -> Iterator[T]:
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
    seen: set[S] = set()
    for item in target:
        key = unique_test(item)
        if key not in seen:
            seen.add(key)
            yield item

def unique_tuple[T, S=T](target: Iterable[T], unique_test: Callable[[T], S] = lambda x: x) -> Prod[T]:
    return tuple(unique_iterable(target, unique_test=unique_test))

def difference[S](first: Iterable[S], second: Iterable[S]) -> Prod[S]:
    set_second = set(second)
    return tuple(x for x in first if x not in set_second)

def deconcatenate(target: Prod[int], dom_length: int | None = None) -> Prod[tuple[int, int]]:
    dom_length = dom_length or max(target) + 1
    return tuple(
        (L0, R0)
        for L0 in range(len(target))
        for R0 in range(dom_length)
        if all((k < L0) == (target[k] < R0) for k in range(len(target)))
    )[1:]

def concat[T, Y=T](xss: Iterable[Iterable[T]], func: Callable[[T], Y] = lambda x: x):
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
    set_ys = set(unique_test(y) for y in ys)
    return tuple(x for x in xs if unique_test(x) in set_ys)



def predicate_partition[T](xs: Iterable[T], predicate: Callable[[T], bool]) -> tuple[Prod[T], Prod[T]]:
    true_part: list[T] = []
    false_part: list[T] = []
    for x in xs:
        (true_part if predicate(x) else false_part).append(x)
    return tuple(true_part), tuple(false_part)

class Multidict[K, V]:
    def __init__(self, generator: Iterable[tuple[K, V]] | None = None):
        if generator is None:
            generator = ()
        self.mapping: dict[K, set[V]] = {}
        self.update(generator)

    def __getitem__(self, key: K) -> set[V]:
        return self.mapping.get(key, set())
        
    def __contains__(self, key: K) -> bool:
        return key in self.mapping
    
    def union(self, targets: Iterable[K]) -> set[V]:
        return set().union(*(self.mapping.get(k, set()) for k in targets))
    
    def __iter__(self) -> Iterator[K]:
        return iter(self.mapping)

    def find_key(self, value: V) -> K | None:
        for k, vs in self.mapping.items():
            if value in vs:
                return k
        return None
    
    def items(self):
        return self.mapping.items()
    
    def keys(self):
        return self.mapping.keys()
    
    def append(self, key: K, *value: V):
        if key not in self.mapping:
            self.mapping[key] = set()
        self.mapping[key].update(value)

    def update(self, generator: Iterable[tuple[K, V]]):
        for k, v in generator:
            if k not in self.mapping:
                self.mapping[k] = set()
            self.mapping[k].add(v)
        return self