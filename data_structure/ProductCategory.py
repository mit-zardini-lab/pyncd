from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    Any,
    Self,
    Type,
    TypeVar,
    Callable,
    Iterable,
    Iterator,
    overload,
    Sequence,
    Iterable,
)
import random
import math
from abc import ABC
from enum import Enum
import data_structure.Numeric as nm
from collections.abc import Sequence

import data_structure.Term as fd # for 'foundations'
import utilities.utilities as util

L = TypeVar('L', covariant=True)

type ProdCategory[L, M:Morphism] = \
    (M | Rearrangement[L]
     | Composed[L, ProdCategory[L, M]]
     | ProductOfMorphisms[L, ProdCategory[L, M]]
     | Block[L, ProdCategory[L, M]]
    )

@dataclass(frozen=True)
class ProdObject[L](fd.Term, Sequence[L]):
    content: fd.Prod[L] = ()

    def identity(self) -> Rearrangement[L]:
        return Rearrangement(
            mapping=tuple(range(len(self.content))), 
            _dom=self.content
        )

    # Conveniances
    @classmethod
    def from_iter(cls, xs: Iterable[L]) -> ProdObject[L]:
        return cls(content=tuple(xs))

    def __len__(self) -> int:
        return len(self.content)
    def __getitem__(self, index):
        return self.content[index]

    
@dataclass(frozen=True)
class Morphism[L](fd.Term):
    def dom(self) -> ProdObject[L]:
        raise NotImplementedError()
    def cod(self) -> ProdObject[L]:
        raise NotImplementedError()
    
    # Conveniances
    def __matmul__[M:Morphism](self,
            other: ProdCategory[L, M] | fd.Prod[int]) -> ProdCategory[L, M]:
        raise NotImplementedError()
    def __rmatmul__[M:Morphism](self, 
            other: ProdCategory[L, M] | fd.Prod[int]) -> ProdCategory[L, M]:
        raise NotImplementedError()
    def __mul__[M:Morphism](self, other: M | ProdObject[L] | L) -> ProdCategory[L, M]:
        raise NotImplementedError()
    def __rrshift__(self, other) -> Self:
        raise NotImplementedError()
    
@dataclass(frozen=True)
class BlockAesthetics(fd.Term):
    title:       str | None = None
    description: str | None = None
    fill_color:  str | None = None

@dataclass(frozen=True)
class BlockTag(fd.UTerm):
    repetition: nm.Numeric = nm.Integer(1)
    aesthetics: BlockAesthetics | None = None

@dataclass(frozen=True)
class Block[L, M: Morphism](Morphism[L]):
    body: M
    block_tag: BlockTag = BlockTag()
    def dom(self) -> ProdObject[L]:
        return self.body.dom()
    def cod(self) -> ProdObject[L]:
        return self.body.cod()
    @classmethod
    def template(cls, 
                 target: M,
                 title: str | None = None,
                 description: str | None = None,
                 fill_color: str | None = None,
                 repetition: int | nm.Numeric = 1) -> Block[L, M]:
        return Block(
            body=target,
            block_tag=BlockTag(
                repetition=nm.Integer(repetition) if isinstance(repetition, int) else repetition,
                aesthetics=BlockAesthetics(
                    title=title,
                    description=description,
                    fill_color=fill_color
                )
            )
        )
    @property
    def aesthetics(self) -> BlockAesthetics | None:
        return self.block_tag.aesthetics
    @property
    def repetition(self) -> nm.Numeric:
        return self.block_tag.repetition

@dataclass(frozen=True)
class Composed[L, M: Morphism](Morphism[L]):
    content: fd.Prod[M] = ()
    def dom(self) -> ProdObject[L]:
        return self.content[0].dom()
    def cod(self) -> ProdObject[L]:
        return self.content[-1].cod()
    
    # Conveniances
    @classmethod
    def from_iter(cls, xs: Iterable[M]) -> Composed[L, M]:
        return cls(content=tuple(xs))

@dataclass(frozen=True)
class ProductOfMorphisms[L, M: Morphism](Morphism[L]):
    content: fd.Prod[M] = ()
    def dom(self) -> ProdObject[L]:
        return ProdObject.from_iter(segment for m in self.content for segment in m.dom())
    def cod(self) -> ProdObject[L]:
        return ProdObject.from_iter(segment for m in self.content for segment in m.cod())


    def partition_object[T](self, target: ProdObject[T]) -> Iterable[tuple[M, ProdObject[T]]]:
        return ((m, ProdObject(xs)) for m, xs in self.partition(tuple(target)))

    def partition[T](self, target: Sequence[T]) -> Iterable[tuple[M, fd.Prod[T]]]: # type: ignore
        _target = tuple(target)
        start = 0
        for m in self.content:
            end = start + len(m.dom())
            yield (m, type(target)(_target[start:end])) # type: ignore
            start = end

    def partition_codomain[T](self, target: Sequence[T]) -> Iterable[tuple[M, Sequence[T]]]:
        start = 0
        for m in self.content:
            end = start + len(m.cod())
            yield (m, target[start:end])
            start = end

    # Conveniances
    @classmethod
    def from_iter(cls, xs: Iterable[M]) -> ProductOfMorphisms[L, M]:
        return cls(content=tuple(xs))
    def __iter__(self) -> Iterator[M]:
        return iter(self.content)
    
@dataclass(frozen=True)
class Rearrangement[L](Morphism[L]):
    mapping: fd.Prod[int] = ()
    _dom: fd.Prod[L] = ()

    def dom(self) -> ProdObject[L]:
        return ProdObject(self._dom)
    def cod(self) -> ProdObject[L]:
        return ProdObject(self.apply(self._dom))
    # from domain to codomain
    def apply[S](self, target: fd.Prod[S]) -> fd.Prod[S]:
        return tuple(target[i] for i in self.mapping)
    # from codomain to domain
    def invert[S](self, target: fd.Prod[S]) -> fd.Prod[S]:
        return tuple(
            util.iallequals(
                segment 
                for segment, muj in 
                zip(target, self.mapping) if i == muj)
            for i in range(len(self._dom))
        )
    
    def pairwise(self) -> fd.Prod[tuple[int, int]]:
        return tuple(
            (i, j)
            for i, _ in enumerate(self._dom)
            for j, m in enumerate(self.mapping) if m == i
        )