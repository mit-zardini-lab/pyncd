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
    Iterable,
    Iterator,
)
import random
import math
from abc import ABC
from enum import Enum

import data_structure.Term as fd # for 'foundations'
import data_structure.Numeric as nm
import utilities.utilities as util
import data_structure.ProductCategory as pc
import data_structure.StrideCategory as sc

B = TypeVar('B', covariant=True)

'''
A note on __xxx__ methods.
    These are used to easily construct expressions, while avoiding the complexities of the data structure.
    We associate __mul__ to a monoidal product. Between 'object-like' (Objectoid) entities, it returns an object.
    Between 'morphism-like' entities, it returns a morphism. Objectoids include Axes, ProdObject[Axis], and tuples
    thereof (for the stride category) and Datatypes, Arrays, and tuples thereof (for the broadcasted
    category).
'''

type BroadcastedCategory[B:Datatype, A:sc.Axis = sc.Axis] = pc.ProdCategory[Array[B, A], Broadcasted[B, A]]

# Conveniences
type AxisObjectoid[A:sc.Axis] = A | pc.ProdObject[A] | fd.Prod[AxisObjectoid[A]]
type BroadcastedObjectoid[B:Datatype, A:sc.Axis] = 'B | AxisObjectoid[A] | fd.Prod[BroadcastedObjectoid[B, A]]'
@dataclass(frozen=True)
class Datatype(fd.Term):
    # Conveniences
    @overload
    def __rrshift__[A:sc.Axis](self, other: AxisObjectoid[A]) -> pc.ProdObject[Array[Self, A]]: ...
    @overload
    def __rrshift__[A:sc.Axis](self, other: sc.StrideCategory[A]) -> BroadcastedCategory[Self, A]: ...
    def __rrshift__(self, other): raise NotImplementedError()
    @overload
    def __mul__[B:Datatype, A:sc.Axis](self, other: BroadcastedObjectoid[B, A]) -> pc.ProdObject[Array[Self | B, A]]: ...
    @overload
    def __mul__[B:Datatype, A:sc.Axis](self, other: BroadcastedCategory[B, A]) -> BroadcastedCategory[Self | B, A]: ...
    def __mul__(self, other): raise NotImplementedError()    

@dataclass(frozen=True)
class Reals(Datatype): ...

@dataclass(frozen=True)
class Natural(Datatype):
    max_value: nm.Numeric = nm.FreeNumeric.field()
    @classmethod
    def template(cls, name: str | fd.DynamicName) -> Natural:
        return Natural(
            fd.DynamicName.from_str(name).capture(nm.FreeNumeric())
        )

@dataclass(frozen=True)
class Array[B:Datatype, A:sc.Axis](fd.Term):
    datatype: B
    _shape: fd.Prod[A] = ()
    def shape(self) -> pc.ProdObject[A]:
        return pc.ProdObject(self._shape)
    
    # Conveniences
    @overload
    def __rrshift__(self, other: AxisObjectoid[A]) -> Self: ...
    @overload
    def __rrshift__(self, other: sc.StrideCategory[A]) -> Broadcasted[B, A]: ...
    def __rrshift__(self, other): raise NotImplementedError()
    @overload
    def __mul__(self, other: BroadcastedObjectoid[B, A]) -> pc.ProdObject[Self]: ...
    @overload
    def __mul__(self, other: BroadcastedCategory[B, A]) -> BroadcastedCategory[B, A]: ...
    def __mul__(self, other): raise NotImplementedError()


class WeaveMode(Enum):
    TILED = 'TILED'
fd.register_enum(WeaveMode)
@dataclass(frozen=True)
class Weave[B: Datatype, A: sc.Axis](fd.Term):
    datatype: B
    _shape: fd.Prod[A | WeaveMode] = ()

    def target(self) -> Array[B, A]:
        return Array[B, A](
            datatype=self.datatype,
            _shape=tuple(
                axis for axis in self._shape
                if not isinstance(axis, WeaveMode)
            )
        )
    
    def select_degree[T](self, target: Iterable[T]) -> Iterable[T]:
        yield from (
            item for item, mode in zip(target, self._shape)
            if isinstance(mode, WeaveMode)
        )
    
    def select_target[T](self, target: Iterable[T]) -> Iterable[T]:
        yield from (
            item for item, axis in zip(target, self._shape)
            if isinstance(axis, sc.Axis)
        )

    def target_idx(self) -> fd.Prod[int]:
        return tuple(self.select_target(range(len(self._shape))))

    def imprint[T](
            self, 
            tiling_imprint: Iterable[T], 
        ) -> fd.Prod[A | T]:
        tilings = iter(tiling_imprint)
        return tuple(
            axis if isinstance(axis, sc.Axis)
            else next(tilings)
            for axis in self._shape
        )
    
    def imprint_target[B2: Datatype, A2:sc.Axis](
            self,
            new_target: Array[B2, A2],
    ) -> Weave[B2, A2]:
        new_target_shape = iter(new_target.shape())
        return Weave(
            datatype=new_target.datatype,
            _shape=tuple(
                next(new_target_shape) if isinstance(axis, sc.Axis) else axis
                for axis in self._shape
            )
        )
        
    def imprint_axes[T](
            self,
            tiling_imprint: Iterable[T],
            axes_imprint: Iterable[T]
    ) -> fd.Prod[T]:
        tilings = iter(tiling_imprint)
        axes = iter(axes_imprint)
        return tuple(
            next(axes if isinstance(axis, sc.Axis) else tilings)
            for axis in self._shape
        )

    def imprint_to_degree(self, other: Iterable[A]) -> Array[B, A]:
        other = iter(other)
        return Array[B, A](
            datatype=self.datatype,
            _shape=tuple(
                next(other) if isinstance(axis, WeaveMode) else axis
                for axis in self._shape
            )
        )
    
    def rearrangement(self, degree: Iterable[A]) -> pc.Rearrangement[A]:
        # goes from degree : target => weaved ie mixed
        degree_size = sum(isinstance(mode, WeaveMode) for mode in self._shape)
        degree_idx = iter(range(degree_size))
        target_idx = iter(range(len(self._shape) - degree_size))
        mapping = tuple(
            next(degree_idx) if isinstance(axis, WeaveMode)
            else degree_size + next(target_idx)
            for axis in self._shape
        )
        return pc.Rearrangement(
            mapping=mapping,
            _dom=(*degree, *self.target().shape())
        )
    
    def inverse_rearrangement(self, degree: Iterable[A]) -> pc.Rearrangement[A]:
        degree_spots = (i for i, axis in enumerate(self._shape) if isinstance(axis, WeaveMode))
        target_spots = (i for i, axis in enumerate(self._shape) if not isinstance(axis, WeaveMode))
        mapping = (*degree_spots, *target_spots)
        _dom = self.imprint_to_degree(degree).shape()
        return pc.Rearrangement(mapping, tuple(_dom))

    @classmethod
    def from_arrays(cls, arrays: Iterable[Array[B, A]]) -> fd.Prod[Weave[B, A]]:
        return tuple(Weave(
            datatype=array.datatype,
            _shape=tuple(array.shape())
        ) for array in arrays)
    
@dataclass(frozen=True)
class Operator(fd.Term):
    name: fd.DynamicName | None = None

    def bc_signature[B: Datatype](
        self,
        signature: str = '',
        datatype: B = Reals(),
    ) -> Broadcasted[B, sc.RawAxis]:
        raise NotImplementedError()

@dataclass(frozen=True)
class Broadcasted[B: Datatype, A: sc.Axis, O: Operator = Any](pc.Morphism[Array[B, A]]):
    operator: O
    input_weaves: fd.Prod[Weave[B, A]] = ()
    output_weaves: fd.Prod[Weave[B, A]] = ()
    reindexings: fd.Prod[sc.StrideCategory[A]] = ()

    def degree(self) -> pc.ProdObject[A]:
        if len(self.reindexings) == 0:
            return pc.ProdObject()
        try:
            return util.iallequals(
                    morphism.dom()
                    for morphism in self.reindexings
                )
        except Exception:
            # for morphism in self.reindexings:
            #     print(morphism.dom())
            print(self.operator)
            raise ValueError("Inconsistent reindexing morphisms in Broadcasted morphism.")
    
    def dom(self) -> pc.ProdObject[Array[B, A]]:
        return pc.ProdObject.from_iter(
            weave.imprint_to_degree(reindexing.cod())
            for weave, reindexing in zip(self.input_weaves, self.reindexings)
        )
    
    def cod(self) -> pc.ProdObject[Array[B, A]]:
        return pc.ProdObject.from_iter(
            weave.imprint_to_degree(self.degree())
            for weave in self.output_weaves
        )
    
    def target(self) -> Self:
        return Broadcasted(
            operator=self.operator,
            input_weaves=tuple(
                Weave(weave.datatype, weave._shape) for weave in self.input_weaves),
            output_weaves=tuple(
                Weave(weave.datatype, weave._shape) for weave in self.output_weaves),
            reindexings=tuple(pc.ProdObject().identity() for _ in self.reindexings)
        ) # type: ignore