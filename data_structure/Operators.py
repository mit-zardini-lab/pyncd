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
)
import random
import math
from abc import ABC
from enum import Enum

import data_structure.Term as fd # for 'foundations'
import data_structure.Numeric as nm
import utilities.utilities as util
import data_structure.Category as cat

import construction_helpers.product as chp
import construction_helpers.einops as che
import construction_helpers.signature as chs

def broadcast[B: cat.Datatype = cat.Reals](
    self: cat.Operator,
    signature: str = '',
    datatype: B = cat.Reals(),
    give_names: bool = True,
) -> cat.Broadcasted[B, cat.RawAxis]:
    input_indexes, output_indexes, indexes_names = che.signature_to_buckets(signature)
    _signature, input_weaves, output_weaves, reindexings = che.bucketed_to_broadcast(
        input_indexes,
        output_indexes,
        indexes_names if give_names else {},
        datatype
    )
    return cat.Broadcasted[B, cat.RawAxis](
        operator=self,
        input_weaves=input_weaves,
        output_weaves=output_weaves,
        reindexings=reindexings
    )
cat.Operator.bc_signature = broadcast

def sized[B: cat.Datatype = cat.Reals](
    self: cat.Operator,
    input_size: int | chp.ProductObjectTarget[cat.RawAxis, str | fd.DynamicName] = 1,
    output_size: None | int | chp.ProductObjectTarget[cat.RawAxis, str | fd.DynamicName] = None,
    datatype: B = cat.Reals(),
):
    input_shape = linear_size_to_shape(input_size)
    output_shape = linear_size_to_shape(output_size) if output_size is not None else input_shape
    return cat.Broadcasted[B, cat.RawAxis](
        operator=self,
        input_weaves=(cat.Weave(datatype, tuple(input_shape)),),
        output_weaves=(cat.Weave(datatype, tuple(output_shape)),),
        reindexings=(cat.ProdObject().identity(),)
    )

@dataclass(frozen=True)
class Elementwise(cat.Operator):
    name: fd.DynamicName | None = fd.DynamicName('\\sigma')
    operator: str | None = 'sigmoid'
    @classmethod
    def template[B:cat.Datatype = cat.Reals, A:cat.Axis = cat.RawAxis](
        cls, 
        base: chp.ProductObjectTarget[cat.Array[B, A], B] = cat.Reals(),
        reindexing: chp.ProductMorphismTarget[A, cat.StrideCategory[A]] = ()
        ):
        base = chp.object_product(base, conversion=chp.datatype_converter)[0]
        _reindexing: cat.StrideCategory[A] = chp.morphism_product((reindexing, base.shape())) # type: ignore
        return cat.Broadcasted(
            operator=cls(name=fd.DynamicName('\\sigma')),
            input_weaves=(cat.Weave(base.datatype, (cat.WeaveMode.TILED,) * len(_reindexing.cod())),),
            output_weaves=(cat.Weave(base.datatype, (cat.WeaveMode.TILED,) * len(_reindexing.dom())),),
            reindexings=(_reindexing,)
        )
    



@dataclass(frozen=True)
class Identity(Elementwise):
    name: fd.DynamicName | None = None
    operator: str | None = None

@dataclass(frozen=True)
class SoftMax(cat.Operator):
    name: fd.DynamicName | None = fd.DynamicName('SoftMax')
    contracted: bool = False
    @classmethod
    def template[B:cat.Datatype=cat.Reals](
        cls,
        base: B = cat.Reals(),
    ):
        axis = cat.RawAxis()
        return cat.Broadcasted[B, cat.RawAxis](
            operator=SoftMax(name=fd.DynamicName('SoftMax')),
            input_weaves=(cat.Weave(base, (axis,)),),
            output_weaves=(cat.Weave(base, (axis,)),),
            reindexings=(cat.ProdObject().identity(),)
        )

@dataclass(frozen=True)
class Einops(cat.Operator):
    name: fd.DynamicName | None = fd.DynamicName('einops')
    # Each integer corresponds to a contraction group.
    # First level corresponds to segments, second level
    # corresponds to axes.
    signature: chs.SignatureSegment = ()
    @classmethod
    def template[B: cat.Datatype = cat.Reals](
        cls,
        signature: str = '',
        datatype: B = cat.Reals(),
        give_names: bool = True
    ):
        input_indexes, output_indexes, indexes_names = che.signature_to_buckets(signature)
        _signature, input_weaves, output_weaves, reindexings = che.bucketed_to_broadcast(
            input_indexes,
            output_indexes,
            indexes_names if give_names else {},
            datatype
        )
        return cat.Broadcasted[B, cat.RawAxis](
            operator=Einops(name=fd.DynamicName(signature), signature=_signature),
            input_weaves=input_weaves,
            output_weaves=output_weaves,
            reindexings=reindexings
        )

def linear_size_to_shape(size: int | chp.ProductObjectTarget[cat.RawAxis, str | fd.DynamicName]) -> cat.ProdObject[cat.RawAxis]:
    if isinstance(size, int):
        return cat.ProdObject.from_iter(cat.RawAxis() for _ in range(size))
    return chp.object_product(size, chp.axis_converter)

@dataclass(frozen=True)
class Linear(cat.Operator):
    name: fd.DynamicName | None = fd.DynamicName('L')
    bias: bool = False
    @classmethod
    def template[B: cat.Datatype = cat.Reals](
        cls,
        input_size: int | chp.ProductObjectTarget[cat.RawAxis, str | fd.DynamicName] = 1,
        output_size: int | chp.ProductObjectTarget[cat.RawAxis, str | fd.DynamicName] = 1,
        name: str | None | fd.DynamicName = None,
        datatype: B = cat.Reals(),
    ):
        operator = Linear(
            name=fd.DynamicName(
                body='L',
                subscript=fd.DynamicName.from_str(name),
                settings=fd.DynamicNameSettings(bold=True)
            )
        )
        return sized(operator, input_size, output_size, datatype)
        # return cat.Broadcasted[B, cat.RawAxis](
        #     operator=Linear(name=fd.DynamicName(
        #         body='L',
        #         subscript=fd.DynamicName.from_str(name),
        #         settings =fd.DynamicNameSettings(bold=True)
        #     )),
        #     input_weaves=(cat.Weave(datatype, tuple(input_shape)),),
        #     output_weaves=(cat.Weave(datatype, tuple(output_shape)),),
        #     reindexings=(cat.ProdObject().identity(),)
        # )

@dataclass(frozen=True)
class Embedding(cat.Operator):
    name: fd.DynamicName | None = fd.DynamicName('E')
    @classmethod
    def template[B: cat.Datatype = cat.Reals](
        cls,
        embedding_size: str | fd.DynamicName | cat.Natural,
        output_size: int | chp.ProductObjectTarget[cat.RawAxis, str] = 1,
        name: str | None | fd.DynamicName = None,
        datatype: B = cat.Reals(),
    ):
        embedding_size = (
            embedding_size
            if isinstance(embedding_size, cat.Natural)
            else cat.Natural.template(embedding_size)
        )
        operator = cls(
            name=fd.DynamicName(
                body='E',
                subscript=fd.DynamicName.from_str(name),
                settings=fd.DynamicNameSettings(bold=True)
            )
        )
        return cat.Broadcasted[B | cat.Natural, cat.RawAxis](
            operator=operator,
            input_weaves=(cat.Weave(embedding_size, ()),),
            output_weaves=(cat.Weave(
                datatype, 
                linear_size_to_shape(output_size).content
            ),),
            reindexings=(cat.ProdObject().identity(),)
        )

@dataclass(frozen=True)
class AdditionOp(cat.Operator):
    name: fd.DynamicName | None = fd.DynamicName('+')
    @classmethod
    def template[B: cat.Datatype = cat.Reals](
        cls,
        signature: str = ',->',
        datatype: B = cat.Reals(),
    ):
        signature_segments, input_weaves, output_weaves, reindexings = chs.generic_signature(
            signature,
            datatype,
        )
        assert all(
            segment == ()
            for segment in signature_segments
        )
        assert len(output_weaves) == 1
        return cat.Broadcasted[B, cat.RawAxis](
            operator=AdditionOp(),
            input_weaves=input_weaves,
            output_weaves=output_weaves,
            reindexings=reindexings
        )
    
@dataclass(frozen=True)
class Normalize(cat.Operator):
    name: fd.DynamicName | None = fd.DynamicName('RMSNorm')
    @classmethod
    def template[B: cat.Datatype = cat.Reals](
        cls,
        input_size: int | chp.ProductObjectTarget[cat.RawAxis, str] = 1,
        datatype: B = cat.Reals(),
    ):
        return sized(
            cls(),
            input_size,
            None,
            datatype
        )
    
@dataclass(frozen=True)
class WeightedTriangularLower(cat.Operator):
    name: fd.DynamicName | None = fd.DynamicName('wtril')
    @classmethod
    def template[B: cat.Datatype = cat.Reals](
        cls,
        size: int | chp.ProductObjectTarget[cat.RawAxis, str] = 2,
        datatype: B = cat.Reals(),
    ):
        shape = linear_size_to_shape(size)
        return cat.Broadcasted[B, cat.RawAxis](
            operator=WeightedTriangularLower(),
            input_weaves=(cat.Weave(datatype, tuple(shape)),),
            output_weaves=(cat.Weave(datatype, tuple(shape)),),
            reindexings=(cat.ProdObject().identity(),)
        )
    
@dataclass(frozen=True)
class ReLU(Elementwise):
    name: fd.DynamicName | None = fd.DynamicName('R')

@dataclass(frozen=True)
class Dropout(Elementwise):
    name: fd.DynamicName | None = fd.DynamicName('\\lightning')