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
class CodeReference(fd.Term):
    '''A place in a codebase a block stands for, drawn as a link beside the block's
    title. `path` is relative to the root of the repository the reference names, and
    `line` and `end_line` bound the lines it points at. `url` is where a browser
    opens it, and a reference with no url is linked by the display from its path.'''
    label: str
    url: str | None = None
    path: str | None = None
    line: int | None = None
    end_line: int | None = None

class BlockDrawing(Enum):
    '''How the operator holding a block is drawn. Under `BOX` it is a titled box and
    the block's body is drawn beside the figure. Under `BODY_IN_PLACE` the body is
    drawn where the box would stand, with no box, no title and no room of its own,
    so the figure reads as it would without the block, and the block still answers
    the pointer with its title, its formula, its description and its references.
    `notebooks/display/explain_operators.py` wraps an operator that way to explain
    it, as the reviewer asked on 2026-09-16.'''
    BOX = 'BOX'
    BODY_IN_PLACE = 'BODY_IN_PLACE'
fd.register_enum(BlockDrawing)


@dataclass(frozen=True)
class BlockAesthetics(fd.Term):
    title:       str | None = None
    description: str | None = None
    fill_color:  str | None = None
    # The three fields below are declared last and in the order they were added,
    # because tsncd constructs a term positionally in field order, and an older
    # bundle receiving more fields than it declares ignores the rest.
    references:  fd.Prod[CodeReference] | None = None
    # LaTeX an inspection box shows under the title.
    formula:     str | None = None
    # `None` is read as `BlockDrawing.BOX`. An ordinary block carries `None`, so
    # a tsncd bundle built before the enum existed still reads the term.
    drawing:     BlockDrawing | None = None

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
                 # White, never None: an uncoloured block draws with no
                 # drop shadow and is invisible against the page, which
                 # makes a mis-nested block impossible to see. Pass a
                 # colour to mean something; leave it to mean "a block".
                 fill_color: str | None = 'white',
                 repetition: int | nm.Numeric = 1,
                 index_name: str | fd.DynamicName | None = None,
                 references: fd.Prod[CodeReference] | None = None,
                 formula: str | None = None,
                 drawing: BlockDrawing | None = None) -> Block[L, M]:
        '''A block over `target`. `index_name` names the tag, and a repeated
        block's tag name is the index of its loop, which
        `para.processing.tape_members` reads to index the tape members a
        plain grab or drop inside the loop touches on each iteration.
        `references` are the places in a codebase the block stands for,
        `formula` is LaTeX an inspection box shows under the title, and
        `drawing` says whether the operator holding the block is drawn as a
        box or as the block's body in place.'''
        block_tag = BlockTag(
            repetition=nm.Integer(repetition) if isinstance(repetition, int) else repetition,
            aesthetics=BlockAesthetics(
                title=title,
                description=description,
                fill_color=fill_color,
                references=references,
                formula=formula,
                drawing=drawing,
            )
        )
        if index_name is not None:
            block_tag = fd.DynamicName.from_str(index_name).capture(block_tag)
        return Block(body=target, block_tag=block_tag)
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


class SidesOfADefinitionDisagree(Exception):
    '''A `DefinedExpression` whose two sides differ in their domain or their
    codomain.'''


@dataclass(frozen=True)
class DefinedExpression[L, M: Morphism](fd.Term):
    '''The statement that `left_hand_side` is defined to be `right_hand_side`, which
    a diagram draws as the two morphisms with `:=` between them.

    The two sides have one domain and one codomain, and `template` raises where they do
    not. A definition is a term and is not a morphism, so it is drawn and is not
    composed. It states that an operator equals its expansion, as
    `algebra.define_by_expansion.define_by_standard_expansion` builds it, or that an
    operator whose value depends on an index equals a formula in that index, per
    `obsidian/06-practice/Representing Models.md`.
    '''
    left_hand_side: ProdCategory[L, M]
    right_hand_side: ProdCategory[L, M]

    @classmethod
    def template(
        cls,
        left_hand_side: ProdCategory[L, M],
        right_hand_side: ProdCategory[L, M],
    ) -> DefinedExpression[L, M]:
        for side_of_morphism in ('dom', 'cod'):
            left = getattr(left_hand_side, side_of_morphism)()
            right = getattr(right_hand_side, side_of_morphism)()
            if tuple(left) != tuple(right):
                raise SidesOfADefinitionDisagree(
                    f'the {side_of_morphism} of the left-hand side is {tuple(left)} '
                    f'and the {side_of_morphism} of the right-hand side is '
                    f'{tuple(right)}')
        return cls(left_hand_side=left_hand_side, right_hand_side=right_hand_side)
