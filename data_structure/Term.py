from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    Any,
    Self,
    Type,
    TypeVar,
    Callable,
    Iterable,
    overload,
    Sequence,
    Iterator,
    TypedDict,
    Protocol,
)
import random
import math
from abc import ABC
from enum import Enum
import functools

import utilities.utilities as util
from functools import cached_property

T = TypeVar('T', covariant=True)

type Prod[T] = tuple[T, ...]
''' Prod[T] acts as a substitute for an immutable sequence T[] within Python '''

TermDirectory: dict[str, Type[Term]] = {}
EnumDirectory: dict[str, Type[Enum]] = {}

def register_enum(cls: Type[Enum]) -> Type[Enum]:
    assert cls.__qualname__ not in EnumDirectory, f"Enum class {cls.__qualname__} already registered."
    EnumDirectory[cls.__qualname__] = cls
    return cls

def _structural_hash(self) -> int:
    '''
    The hash ``@dataclass(frozen=True)`` would generate, computed once.

    Terms form a DAG with heavy sharing, but a structural hash walks it as a
    tree: a subterm reachable by n paths is hashed n times, so the cost is the
    number of PATHS rather than of nodes. Fusing attention made 8.3M hash
    calls - 114k of them primitive - over a result holding 35.6k distinct
    objects.

    Caching is sound because terms are frozen; nothing in the package mutates
    one after construction. The cache lives in the instance ``__dict__``,
    outside ``__dataclass_fields__``, so it takes no part in ``__eq__`` and
    does not change what a term is.
    '''
    cached = self.__dict__.get('_hash')
    if cached is None:
        cached = hash(tuple(
            getattr(self, name) for name in self.__dataclass_fields__))
        object.__setattr__(self, '_hash', cached)
    return cached

@dataclass(frozen=True)
class Term:
    '''
    A term is an element in our formal representational language. It provides a container for data
    structures that appear in expressions. Abstractly, a term is a construction rule;
      $\\gamma_k: \\Pi_ki x_i \\rightarrow T_k $
    So that, within our grammar, all properties of $T_k$ can be derived from $\\Pi_ki x_i$.
    '''
    # Subclasses opt out by setting this False in their own body - see Numeric,
    # whose equality goes through its own numeric_hash() instead. Deliberately
    # unannotated: an annotated ClassVar still lands in __dataclass_fields__,
    # which keys()/dict()/reconstruct() iterate.
    _memoize_hash = True

    def __init_subclass__(cls) -> None:
        assert cls.__qualname__ not in TermDirectory, f"Term class {cls.__qualname__} already registered."
        TermDirectory[cls.__qualname__] = cls
        # Installed HERE, before @dataclass runs on the subclass: the decorator
        # leaves __hash__ alone when the class already has an explicit one, so
        # every Term picks this up automatically - including those declared
        # later, and without touching any of the ~50 declaration sites.
        if cls._memoize_hash:
            cls.__hash__ = _structural_hash
        return super().__init_subclass__()
    def keys(self) -> Iterable[str]:
        return (f.name for f in self.__dataclass_fields__.values())
    def dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.keys()}
    def reconstruct(self, **kwargs: Any) -> Self:
        return type(self)(**{**self.dict(), **kwargs})

IDMAX = 2**31-1
type IDType = int
def fresh_id() -> IDType:
    return random.randint(1, IDMAX)
def hash_id(obj) -> IDType:
    return hash(obj) % IDMAX

''' UID Handler '''
@dataclass(frozen=True)
class DynamicNameSettings(Term):
    bold: bool = False
    overline: bool = False
    absolute: bool = False
@dataclass(frozen=True)
class DynamicName(Term):
    '''
    A term for names which provides the control needed for aesthetic purposes.
    When subscripts are chained together, we display them as {$x}_{{$y}{$z}}
    instead of $x_{y_z}$.
    '''
    body: None | str = None
    subscript: None | DynamicName = None
    settings: None | DynamicNameSettings = None

    def __lt__(self, other: DynamicName) -> bool:
        return (
            (self.body or '') < (other.body or '')
            or (
                other.subscript is not None
                and (self.subscript is None or self.subscript < other.subscript)
            )
        )

    def lineage(self) -> Prod[DynamicName]:
        if self.body is None:
            return ()
        if self.subscript is None:
            return (self,)
        return (self, *self.subscript.lineage())
            
    def capture[S:Term](self, target: S) -> S:
        return target.reconstruct(
            uid=target.uid.reconstruct( # type: ignore
                _name=self
            )
        )
    
    def to_bodies(self) -> str:
        return ''.join(
            b.body or ''
            for b in self.lineage()
        )
    
    def body_latex(self) -> str:
        body = self.body or ''
        if self.settings is None:
            return body
        if self.settings.bold:
            body = f'\\bold{{{body}}}'
        if self.settings.overline:
            body = f'\\overline{{{body}}}'
        return body
    
    def to_latex(self) -> str:
        latex = self.body_latex() + (
            ('_' + ''.join(
                b.body_latex() 
                for b in self.subscript.lineage()))
            if self.subscript
            else ''
        )
        if self.settings is not None and self.settings.absolute:
            latex = f'|{latex}|' 
        return latex
    
    def subscript_target(self, target: DynamicName) -> DynamicName:
        ''' {A}_{B} subscript_target {C}_{D} = {C}_{D}{A}{B}'''
        return DynamicName(
            body=target.body,
            subscript=(
                self 
                if target.subscript is None 
                else self.subscript_target(target.subscript)
                ),
            settings=target.settings
        )
    def subscript_capture[S:UTerm](self, target: S) -> S:
        if (old_name := target.uid._name) is None:
            return target
        new_name = self.subscript_target(old_name)
        return new_name.capture(target)
    
    def add_subscript(self, other: DynamicName | str | None) -> DynamicName:
        if not other:
            return self
        other = other if isinstance(other, DynamicName) else DynamicName(body=other)
        return DynamicName(
            body=self.body,
            subscript=(
                other if self.subscript is None
                else self.subscript.add_subscript(other)
            )
        )

    @overload
    @classmethod
    def from_str(cls, 
            name: None, 
            lineage: bool = True,
            settings: None | DynamicNameSettings = None) -> None: ...
    @overload
    @classmethod
    def from_str(cls,
                 name: DynamicName | str,
                 lineage: bool = True,
                 settings: None | DynamicNameSettings = None) -> DynamicName: ...

    @classmethod
    def from_str(cls,
            name: DynamicName | str | None,
            lineage: bool = True,
            settings: None | DynamicNameSettings = None) -> None | DynamicName:
        match name:
            case None | DynamicName():
                return name
            case str() if not lineage:
                return DynamicName(body=name, settings=settings)
            case str() if lineage:
                body, *subscript_parts = name.split('_', 1)
                subscript = cls.from_str(
                    subscript_parts[0]
                    if subscript_parts else None, 
                    lineage=True,
                    settings=None)
                return DynamicName(
                    body=body,
                    subscript=subscript,
                    settings=settings
                )

@dataclass(frozen=True)
class UID[T:Term](Term):
    '''
    Terms with the same ``uid`` property are the same in every way. Each UTerm with
    a ``uid`` acts as a "degree of freedom" in an expression. When we substitute
    UTerms for each other we can perform actions such as aligning axes. Furthermore,
    UIDs are used to track Blocks of morphisms across an expression.
    '''
    _type: Type[T]
    _id: IDType = field(default_factory=fresh_id)
    _name: DynamicName | None = None

    def __lt__(self, other: UID[T]) -> bool:
        match self._name, other._name:
            case None, None:
                return self._id < other._id
            case None, DynamicName():
                return True
            case DynamicName(), None:
                return False
            case DynamicName(), DynamicName():
                return self._name < other._name
    
    @classmethod
    def field(cls, _type: Type[T]):
        return field(default_factory=lambda: cls(_type))
    
    def to_latex(self) -> str:
        return self._name.to_latex() if self._name is not None else f'\\text{{{self._type.__qualname__}}}_{{{self._id}}}'

@dataclass(frozen=True)
class UTerm(Term):
    uid: UID[Self] = field(default_factory=lambda: UID(UTerm)) # type: ignore
    def __init_subclass__(cls) -> None:
        cls.__dataclass_fields__['uid'].default_factory = lambda: UID(cls)
        return super().__init_subclass__()

'''
Term Utilities
'''

type GeneralTerm = Term | Prod[GeneralTerm]
@overload
def deep_reconstruct[T](target: T, func: Callable[[T], T]) -> T: ...
@overload
def deep_reconstruct[T](target: Prod[T], func: Callable[[T], T]) -> Prod[T]: ...
def deep_reconstruct(target, func):
    '''
    Rebuild `target` with `func` applied to each of its parts.

    A part that comes back as the very same object leaves nothing to rebuild,
    and a term all of whose parts are unchanged is returned as-is. Most of a
    rewrite touches nothing, so this skips the bulk of the allocation - and it
    keeps sharing that rebuilding would destroy, which matters because the next
    pass over a term walks it once per path.
    '''
    match target:
        case Term():
            values: dict[str, Any] = {}
            changed = False
            for name in target.keys():
                old = getattr(target, name)
                new = func(old)
                changed = changed or new is not old
                values[name] = new
            return type(target)(**values) if changed else target
        case tuple():
            items = tuple(func(item) for item in target)
            return items if any(
                new is not old for new, old in zip(items, target)) else target
        case _:
            return target
            
        
'''
Term Equalization Mechanisms
'''

class UIDEquipped(Protocol):
    uid: UID
@dataclass
class EqualityClass[T:UTerm]:
    _type: Type[T]
    bucket: set[UID[T]]
    canonical: T
    priority: int = 0

    def apply[S: GeneralTerm](self, target: S, iterate: bool = True) -> S:
        match target:
            case Term(uid=UID() as uid) if uid in self.bucket:
                return self.canonical  # type: ignore
        return deep_reconstruct(target, self.apply) if iterate else target
    
    @classmethod
    def from_iter(cls, target: Iterable[T], priority: int = 0) -> EqualityClass[T]:
        target = tuple(target)
        #_type = util.iallequals(map(type, target))
        _type = type(target[0])
        return EqualityClass(
            _type=_type,
            bucket={t.uid for t in target},
            canonical=max(
                target,
                key=lambda uterm: uterm.uid
            ),
            priority=priority
        )

    def try_merge(self, other: EqualityClass[T]) -> None | EqualityClass[T]:
        # None represents the equality classes have no overlap, and cannot be merged.
        if self.bucket.isdisjoint(other.bucket):
            return None
        canonical = max(
            self, other,
            key=lambda eq_class: (
                eq_class.priority,
                eq_class.canonical.uid, 
            )
        ).canonical
        return EqualityClass(
            _type=self._type,
            bucket=self.bucket.union(other.bucket),
            canonical=canonical,
            priority=max(self.priority, other.priority)
        )
    
    def merge(self, other: EqualityClass[T]) -> EqualityClass[T]:
        canonical = max(
            self, other,
            key=lambda eq_class: (
                eq_class.priority,
                eq_class.canonical.uid, 
            )
        ).canonical
        return EqualityClass(
            _type=self._type, # type: ignore
            bucket=self.bucket.union(other.bucket), # type: ignore
            canonical=canonical,
            priority=max(self.priority, other.priority)
        )
    
    @classmethod
    def template(cls, *targets: T, priority: int = 0) -> EqualityClass[T]:
        return cls.from_iter(targets, priority=priority)
    
    @classmethod
    def set_canonical(cls, target: T, *original: T) -> EqualityClass[T]:
        # Map the equality class of original to target. This is used for morphism transfer.
        _type = type(target)
        bucket = {t.uid for t in (target, *original)}
        return EqualityClass(
            _type=_type,
            bucket=bucket,
            canonical=target
        )
    
@dataclass
class Context:
    equality_classes: list[EqualityClass] = field(default_factory=list)
    # Live only for the duration of one top-level apply(); see there.
    _apply_memo: dict[int, Any] | None = field(
        default=None, repr=False, compare=False)

    def buckets_set(self) -> set[UID]:
        return {uid for eq_class in self.equality_classes for uid in eq_class.bucket} or set()
    def __iter__(self) -> Iterator[EqualityClass]:
        return iter(self.equality_classes)
    def apply[T: GeneralTerm](self, target: T) -> T:
        '''
        Rewrite `target` under these equality classes.

        Memoized on object identity for the duration of one top-level call.
        Terms form a DAG, and without the memo a shared subterm is rebuilt once
        per path - and, worse, the result comes back with less sharing than the
        input, so every later pass has more paths to walk. Keying on identity is
        safe because everything keyed on stays reachable from `target` for the
        whole traversal, so no id can be recycled under us.
        '''
        memo, top = self._apply_memo, self._apply_memo is None
        if top:
            memo = self._apply_memo = {}
        try:
            key = id(target)  # the ORIGINAL object, before any rewriting
            if key in memo:
                return memo[key]
            rewritten = target
            match rewritten:
                case Term(uid=UID()):
                    for eq_class in self.equality_classes:
                        rewritten = eq_class.apply(rewritten, iterate=False)
            result = memo[key] = deep_reconstruct(rewritten, self.apply)
            return result
        finally:
            if top:
                self._apply_memo = None
    def __call__[T: GeneralTerm](self, target: T) -> T:
        return self.apply(target)
    
    def append_iter[T: UTerm](self, target: Iterable[T]) -> None:
        new_eq_class = EqualityClass.from_iter(target)
        self.append_bucket(new_eq_class)
        # to_del = []
        # for i, eq_class in enumerate(self.equality_classes):
        #     merged = eq_class.merge(new_eq_class)
        #     if merged is not None:
        #         new_eq_class = merged
        #         to_del.append(i)
        # for i in reversed(to_del):
        #     del self.equality_classes[i]
        # self.equality_classes.append(new_eq_class)

    def append_bucket[T: UTerm](self, bucket: EqualityClass[T]) -> Self:
        # Merge must ACCUMULATE: a bucket can overlap several existing classes
        # (a chain of identifications arriving out of order), and each merge
        # has to fold into the union so far - merging each against the original
        # bucket would keep only the last union and silently drop the rest.
        new_eq_class = bucket
        to_del = []
        for i, eq_class in enumerate(self.equality_classes):
            merged = eq_class.try_merge(new_eq_class)
            if merged is not None:
                new_eq_class = merged
                to_del.append(i)
        for i in reversed(to_del):
            del self.equality_classes[i]
        self.equality_classes.append(new_eq_class)
        return self

    def append_buckets[T: UTerm](self, buckets: Iterable[EqualityClass[T]]) -> Self:
        for bucket in buckets:
            self.append_bucket(bucket)
        return self

    def append_contexts[T: UTerm](self, contexts: Iterable[Context]) -> Self:
        for context in contexts:
            if self.buckets_set().isdisjoint(context.buckets_set()):
                self.equality_classes.extend(context.equality_classes)
            self.append_buckets(context.equality_classes)
        return self
