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

T = TypeVar('T', covariant=True)

type Prod[T] = tuple[T, ...]
''' Prod[T] acts as a substitute for an immutable sequence T[] within Python '''

TermDirectory: dict[str, Type[Term]] = {}
EnumDirectory: dict[str, Type[Enum]] = {}

def register_enum(cls: Type[Enum]) -> Type[Enum]:
    assert cls.__qualname__ not in EnumDirectory, f"Enum class {cls.__qualname__} already registered."
    EnumDirectory[cls.__qualname__] = cls
    return cls

@dataclass(frozen=True)
class Term(ABC):
    '''
    A term is an element in our formal representational language. It provides a container for data
    structures that appear in expressions. Abstractly, a term is a construction rule;
      $\\gamma_k: \\Pi_ki x_i \\rightarrow T_k $
    So that, within our grammar, all properties of $T_k$ can be derived from $\\Pi_ki x_i$.
    '''
    def __init_subclass__(cls) -> None:
        assert cls.__qualname__ not in TermDirectory, f"Term class {cls.__qualname__} already registered."
        TermDirectory[cls.__qualname__] = cls
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

@dataclass(frozen=True)
class UTerm(Term, ABC):
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
    match target:
        case Term():
            return type(target)(**{
                f: func(getattr(target, f))
                for f in target.keys()
            })
        case tuple():
            return tuple(func(item) for item in target)
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

    def apply[S: GeneralTerm](self, target: S, iterate: bool = True) -> S:
        match target:
            case Term(uid=UID() as uid) if uid in self.bucket:
                return self.canonical  # type: ignore
        return deep_reconstruct(target, self.apply) if iterate else target
    
    @classmethod
    def from_iter(cls, target: Iterable[T]) -> EqualityClass[T]:
        target = tuple(target)
        # Determine the most-general common type.  NormAxis and NatAxis are
        # subclasses of RawAxis; allow them to coexist in one equality class
        # by widening to the common base.
        types = [type(t) for t in target]
        _type: type = types[0]
        for t in types[1:]:
            if t == _type:
                continue
            if issubclass(_type, t):
                _type = t        # t is more general; widen
            elif not issubclass(t, _type):
                raise ValueError(
                    f"Elements are not all equal: {_type.__name__} != {t.__name__}"
                )
        return EqualityClass(
            _type=_type,
            bucket={t.uid for t in target},
            canonical=max(target, key=lambda uterm: uterm.uid),
        )

    def merge(self, other: EqualityClass[T]) -> None | EqualityClass[T]:
        # None represents the equality classes have no overlap, and cannot be merged.
        if self.bucket.isdisjoint(other.bucket):
            return None
        canonical = max(
            self.canonical, other.canonical,
            key=lambda uterm: uterm.uid
        )
        return EqualityClass(
            _type=self._type,
            bucket=self.bucket.union(other.bucket),
            canonical=canonical
        )
    
@dataclass
class Context:
    equality_classes: list[EqualityClass] = field(default_factory=list)
    
    def apply[T: GeneralTerm](self, target: T) -> T:
        match target:
            case Term(uid=UID()):
                for eq_class in self.equality_classes:
                    target = eq_class.apply(target, iterate=False)
        return deep_reconstruct(target, self.apply)
    
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

    def append_bucket[T: UTerm](self, bucket: EqualityClass[T]) -> None:
        new_eq_class = bucket
        to_del = []
        for i, eq_class in enumerate(self.equality_classes):
            merged = eq_class.merge(bucket)
            if merged is not None:
                new_eq_class = merged
                to_del.append(i)
        for i in reversed(to_del):
            del self.equality_classes[i]
        self.equality_classes.append(new_eq_class)

    def append_buckets[T: UTerm](self, buckets: Iterable[EqualityClass[T]]) -> None:
        for bucket in buckets:
            self.append_bucket(bucket)

    def without(self, uid: UID) -> Context:
        """Return a new Context with all equality classes that involve uid removed."""
        ctx = Context()
        ctx.equality_classes = [eq for eq in self.equality_classes if uid not in eq.bucket]
        return ctx
