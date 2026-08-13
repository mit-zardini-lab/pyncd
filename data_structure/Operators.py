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

import construction_helpers.simple_helper as sh
import term_utilities.term_utilities as tutil
import construction_helpers.lift as lift

def broadcast[B: cat.Datatype = cat.Reals](
    self: cat.Operator,
    signature: str = '',
    datatype: B = cat.Reals(),
    support_unit_object: bool = False,
) -> cat.Broadcasted[B, cat.RawAxis]:
    return che.signature_to_broadcasted(self, signature, datatype, support_unit_object=support_unit_object) # type: ignore
cat.Operator.bc_signature = broadcast

def sized[B: cat.Datatype = cat.Reals](
    self: cat.Operator,
    input_size: int | chp.ProductObjectTarget[cat.RawAxis, str | fd.DynamicName] = 1,
    output_size: None | int | chp.ProductObjectTarget[cat.RawAxis, str | fd.DynamicName] = None,
    datatype: B = cat.Reals(),
    output_datatype: B | None = None,
):
    output_datatype = output_datatype if output_datatype is not None else datatype
    input_shape = linear_size_to_shape(input_size)
    output_shape = linear_size_to_shape(output_size) if output_size is not None else input_shape
    return cat.Broadcasted[B, cat.RawAxis](
        operator=self,
        input_weaves=(cat.Weave(datatype, tuple(input_shape)),),
        output_weaves=(cat.Weave(output_datatype, tuple(output_shape)),),
        reindexings=(cat.ProdObject().identity(),)
    )

@dataclass(frozen=True)
class GenericOperator(cat.Operator):
    @classmethod
    def template[B: cat.Datatype= cat.Reals](cls, name: str, signature: str, datatype: B = cat.Reals(), support_unit_object: bool = False, **kwargs):
        return broadcast(
            cls(name=fd.DynamicName(name)),
            signature,
            datatype=datatype,
            support_unit_object=support_unit_object
        )

type BBlock[B: cat.Datatype = cat.Reals, A: cat.Axis = cat.RawAxis] = cat.Block[cat.Array[B, A], cat.BroadcastedCategory[B, A]]
@dataclass(frozen=True)
class BlockOperator[B: cat.Datatype = cat.Reals, A: cat.Axis = cat.RawAxis](cat.Operator):
    name: fd.DynamicName | None = None # gives the short name
    block: BBlock[B, A] = None # type: ignore

    @classmethod
    def template(cls, block: BBlock[B, A], name: str | None | fd.DynamicName = None):
        name = fd.DynamicName.from_str(name) if name is not None else None
        name = name.reconstruct(settings=fd.DynamicNameSettings(bold=True)) if name is not None else None
        return cat.Broadcasted(
            operator=cls(name=name, block=block),
            input_weaves=tuple(
                cat.Weave(dom.datatype, tuple(dom.shape()))
                for dom in block.dom()
            ),
            output_weaves=tuple(
                cat.Weave(cod.datatype, tuple(cod.shape()))
                for cod in block.cod()
            ),
            reindexings=tuple(
                cat.ProdObject().identity()
                for _ in block.dom()
            )
        )

    def expand(self, parent: cat.Broadcasted[B, A, BlockOperator[B, A]]):
        assert self == parent.operator
        degree = parent.degree()
        main_body = lift.morphism_object_lift(self.block, degree)
        reindexings = tuple(
            sh.make_composed(
                sh.make_product(reindexing, weave.target().shape().identity()), 
                weave.rearrangement(reindexing.cod())
            )
            for reindexing, weave 
            in zip(parent.reindexings, parent.input_weaves)
        )
        reindexing_morphisms = sh.make_product(
            *(Identity.template(base=weave.target(), reindexing=reindexing) # type: ignore
            for reindexing, weave
            in zip(reindexings, parent.input_weaves))
        )
        return sh.make_composed(
            reindexing_morphisms,
            main_body,
            sh.make_product(
                *(weave.inverse_rearrangement(degree)
                  for weave in parent.output_weaves)
            )
        )
    
        

@dataclass(frozen=True)
class Elementwise(cat.Operator):
    name: fd.DynamicName | None = fd.DynamicName('\\sigma')
    operator: str | None = 'sigmoid'
    @classmethod
    def template[B:cat.Datatype = cat.Reals, A:cat.Axis = cat.RawAxis](
        cls, 
        base: chp.ProductObjectTarget[cat.Array[B, A], B] = cat.Reals(),
        reindexing: chp.ProductMorphismTarget[A, cat.StrideCategory[A]] = (),
        name: str | None | fd.DynamicName = None,
        ):
        base = chp.object_product(base, conversion=chp.datatype_converter)[0]
        _reindexing: cat.StrideCategory[A] = chp.morphism_product((reindexing, base.shape())) # type: ignore
        name = fd.DynamicName.from_str(name) if name is not None else fd.DynamicName('\\sigma')
        return cat.Broadcasted(
            operator=cls(name=name),
            input_weaves=(cat.Weave(base.datatype, (cat.WeaveMode.TILED,) * len(_reindexing.cod())),),
            output_weaves=(cat.Weave(base.datatype, (cat.WeaveMode.TILED,) * len(_reindexing.dom())),),
            reindexings=(_reindexing,)
        )
    



@dataclass(frozen=True)
class Identity(Elementwise):
    name: fd.DynamicName | None = None
    operator: str | None = None

    @classmethod
    def template[B:cat.Datatype = cat.Reals, A:cat.Axis = cat.RawAxis]( # type: ignore
        cls, 
        base: chp.ProductObjectTarget[cat.Array[B, A], B] = cat.Reals(),
        reindexing: chp.ProductMorphismTarget[A, cat.StrideCategory[A]] = ()
        ):
        bases = chp.object_product(base, conversion=chp.datatype_converter)
        if len(bases) > 1:
            return chp.morphism_product(tuple(cls.template(base, reindexing) for base in bases))
        base = bases[0]
        _reindexing: cat.StrideCategory[A] = chp.morphism_product((reindexing, base.shape())) # type: ignore
        if tutil.is_identity(_reindexing):
            base = chp.object_product(base, conversion=chp.datatype_converter)[0]
            shape = (*_reindexing.dom(), *base.shape())
            return cat.ProdObject((
                cat.Array(base.datatype, shape),
            )).identity()
        return super().template(base=base, reindexing=reindexing)
    
def is_identity[B: cat.Datatype, A: cat.Axis](morphism: cat.Broadcasted[B, A]) -> bool:
    match morphism:
        case cat.Composed(content=ms) | cat.ProductOfMorphisms(content=ms):
            return all(is_identity(m) for m in ms)
        case cat.Rearrangement(mapping=mapping, _dom=dom):
            return mapping == tuple(range(len(dom)))
        case cat.Block(body=body):
            return is_identity(body)
        case cat.Broadcasted(operator=op, reindexings=reindexings):
            return isinstance(op, Identity) and all(tutil.is_identity(r) for r in reindexings)


@dataclass(frozen=True)
class SoftMax(cat.Operator):
    name: fd.DynamicName | None = fd.DynamicName('SoftMax')
    contracted: bool = False
    @classmethod
    def template[B:cat.Datatype=cat.Reals](
        cls,
        base: B = cat.Reals(),
        contracted: bool = False
    ):
        axis = cat.RawAxis()
        return cat.Broadcasted[B, cat.RawAxis](
            operator=SoftMax(name=fd.DynamicName('SoftMax'), contracted=contracted),
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
    ):
        input_indexes, output_indexes, input_weaves, output_weaves, reindexings = che.signature_to_broadcast(
            signature, datatype, support_unit_object=False
        )
        assert not output_indexes or all(i < 0 for segment in output_indexes for i in segment)
        assert len(output_indexes) == 1
        input_signature = tuple(tuple(i for i in segment if i >= 0) for segment in input_indexes)
        # A single input with nothing contracted is not an einsum at all.
        operator = (
            Identity() if input_signature == ((),)
            else Einops(name=fd.DynamicName('einops'), signature=input_signature)
        )
        if operator == Identity() and tutil.is_identity(reindexings[0]):
            return cat.ProdObject((input_weaves[0].target(),)).identity()
        return cat.Broadcasted[B, cat.RawAxis](
            operator=operator,
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
        output_datatype: B | None = None
    ):
        output_datatype = output_datatype if output_datatype is not None else datatype
        name = fd.DynamicName.from_str(name) if name is not None else fd.DynamicName('L')
        name = name.reconstruct(settings=fd.DynamicNameSettings(bold=True))
        operator = Linear(
            name=name
        )
        return sized(operator, input_size, output_size, datatype, output_datatype)

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
    part_of_fma: bool = False
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
class Constant(cat.Operator):
    '''A nullary constant emitting `value` (default zero). Used as a streaming
    initializer: the identity a fold accumulates onto - e.g. 0 to seed a sum.'''
    name: fd.DynamicName | None = fd.DynamicName('Constant')
    value: nm.Numeric = nm.Integer(0)

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

@dataclass(frozen=True)
class Maximum(cat.Operator):
    name: fd.DynamicName | None = fd.DynamicName('\\max')