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
import solver.algebra.differentiate_numeric as differentiate_numeric
import solver.registries.numeric_derivative  # one derivative rule per Numeric class
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
    block: BBlock[B, A]

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
            *(View.template(base=weave.datatype, reindexing=reindexing) # type: ignore
            for reindexing, weave
            in zip(reindexings, parent.input_weaves))
        )
        return sh.make_composed(
            reindexing_morphisms,
            main_body,
            sh.make_product(
                *(View.template(base=weave.datatype,
                                reindexing=weave.inverse_rearrangement(degree))
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
        _reindexing: cat.StrideCategory[A] = chp.morphism_product(
            (reindexing, base.shape())) # type: ignore
        name = fd.DynamicName.from_str(name) if name is not None else fd.DynamicName('\\sigma')
        return cat.Broadcasted(
            operator=cls(name=name),
            input_weaves=(cat.Weave(base.datatype, (cat.WeaveMode.TILED,) * len(_reindexing.cod())),),
            output_weaves=(cat.Weave(base.datatype, (cat.WeaveMode.TILED,) * len(_reindexing.dom())),),
            reindexings=(_reindexing,)
        )
    



@dataclass(frozen=True)
class Arithmetic(Elementwise):
    '''
    An elementwise map given by a formula: `x -> formula`, with `x` the
    `nm.FreeInput`. Unlike a bare `Elementwise`, which is a name the algebra
    cannot read, an `Arithmetic` has a derivative the algebra can write out as
    `solver.algebra.differentiate_numeric.differentiate(formula)`, and a value
    `torch_compile` can evaluate, so a derived backward pass is checkable
    against autograd with no table of names.

    The `name` is the formula's latex with `x` at the input's position, so the
    listing and the diagram show `e^{x}` rather than a letter. It is filled in
    from the formula when not given. Give one where the published name of the
    function is shorter than its formula, as `\\sigma` is shorter than
    `(1 + e^{-x})^{-1}`. The algebra reads the formula either way.
    '''
    name: fd.DynamicName | None = None
    operator: str | None = 'arithmetic'
    formula: nm.Numeric = nm.FreeInput()

    def __post_init__(self) -> None:
        if self.name is None:
            object.__setattr__(
                self, 'name', fd.DynamicName(self.formula.to_latex()))

    def derivative(self) -> Arithmetic:
        return Arithmetic(
            formula=differentiate_numeric.differentiate(self.formula))

    @classmethod
    def template[B:cat.Datatype = cat.Reals, A:cat.Axis = cat.RawAxis]( # type: ignore
        cls,
        formula: nm.Numeric,
        base: chp.ProductObjectTarget[cat.Array[B, A], B] = cat.Reals(),
        reindexing: chp.ProductMorphismTarget[A, cat.StrideCategory[A]] = (),
        name: str | None | fd.DynamicName = None,
        output_datatype: cat.Datatype | None = None,
        ):
        '''`output_datatype` is the datatype of the result where it differs from the
        datatype of `base`, as the complex exponential of a real angle does and a
        formula applied to the `Natural` positions an `Arrange` returns does.'''
        base = chp.object_product(base, conversion=chp.datatype_converter)[0]
        _reindexing: cat.StrideCategory[A] = chp.morphism_product((reindexing, base.shape())) # type: ignore
        result_datatype = base.datatype if output_datatype is None else output_datatype
        return cat.Broadcasted(
            operator=cls(formula=formula, name=fd.DynamicName.from_str(name)
                         if isinstance(name, str) else name),
            input_weaves=(cat.Weave(base.datatype, (cat.WeaveMode.TILED,) * len(_reindexing.cod())),),
            output_weaves=(cat.Weave(result_datatype, (cat.WeaveMode.TILED,) * len(_reindexing.dom())),),
            reindexings=(_reindexing,)
        )


@dataclass(frozen=True)
class View(Elementwise):
    name: fd.DynamicName | None = None
    operator: str | None = None

    @classmethod
    def template[B:cat.Datatype = cat.Reals, A:cat.Axis = cat.RawAxis]( # type: ignore
        cls,
        base: chp.ProductObjectTarget[cat.Array[B, A], B] = cat.Reals(),
        reindexing: chp.ProductMorphismTarget[A, cat.StrideCategory[A]] = (),
        name: str | None | fd.DynamicName = None,
        ):
        bases = chp.object_product(base, conversion=chp.datatype_converter)
        if len(bases) > 1:
            return chp.morphism_product(tuple(cls.template(base, reindexing, name) for base in bases))
        base = bases[0]
        _reindexing: cat.StrideCategory[A] = chp.morphism_product((reindexing, base.shape())) # type: ignore
        if tutil.is_identity(_reindexing):
            base = chp.object_product(base, conversion=chp.datatype_converter)[0]
            shape = (*_reindexing.dom(), *base.shape())
            return cat.ProdObject((
                cat.Array(base.datatype, shape),
            )).identity()
        return super().template(base=base, reindexing=reindexing, name=name)
    
def is_identity[B: cat.Datatype, A: cat.Axis](morphism: cat.Broadcasted[B, A]) -> bool:
    match morphism:
        case cat.Composed(content=ms) | cat.ProductOfMorphisms(content=ms):
            return all(is_identity(m) for m in ms)
        case cat.Rearrangement(mapping=mapping, _dom=dom):
            return mapping == tuple(range(len(dom)))
        case cat.Block(body=body):
            return is_identity(body)
        case cat.Broadcasted(operator=op, reindexings=reindexings):
            return isinstance(op, View) and all(tutil.is_identity(r) for r in reindexings)


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

NO_EPSILON = nm.Integer(0)
'''The epsilon of a norm that adds none. It is the unit of addition, so
`nm.Addition.template` drops it and the divisor of the norm is the bare sum.'''

SYMBOLIC_EPSILON = nm.FreeNumeric.named(r'\epsilon')
'''The epsilon of a norm whose model does not say what the number is. A
floating-point implementation adds one under the root or to the divisor, and a model
that states the released value supplies it instead.'''


@dataclass(frozen=True)
class L1Norm(cat.Operator):
    '''`y = x / (sum(x) + epsilon)`, over the one axis the operator consumes.

    The divisor is the sum of the values over that axis and not their mean, so the
    values of the result sum to one where `epsilon` is zero. The normalisation a
    `SoftMax` performs after its exponential is this operator at `NO_EPSILON`, and the
    Sinkhorn iteration of DeepSeek-V4.1-Flash and the gate of its mixture of experts
    each supply the epsilon their released code adds to the divisor. An axis whose
    values have all underflowed to zero would otherwise be divided by zero.

    `contracted` carries the same meaning it carries on a `SoftMax`, because the two
    read every position of the axis to write every position and so divide a kernel the
    same way.
    '''
    name: fd.DynamicName | None = fd.DynamicName('L^{1}')
    contracted: bool = False
    epsilon: nm.Numeric = NO_EPSILON
    @classmethod
    def template[B: cat.Datatype = cat.Reals](
        cls,
        base: B = cat.Reals(),
        contracted: bool = False,
        epsilon: nm.Numeric = NO_EPSILON,
    ):
        axis = cat.RawAxis()
        return cat.Broadcasted[B, cat.RawAxis](
            operator=L1Norm(name=fd.DynamicName('L^{1}'), contracted=contracted,
                            epsilon=epsilon),
            input_weaves=(cat.Weave(base, (axis,)),),
            output_weaves=(cat.Weave(base, (axis,)),),
            reindexings=(cat.ProdObject().identity(),)
        )

@dataclass(frozen=True)
class L2Norm(cat.Operator):
    '''`y = x / \\sqrt{\\sum x^{2}}`, over the axes the operator consumes.

    The sum of the squares runs over every axis in the target, so one value of the
    result is one value of the input divided by the length of the whole array the
    input names, and the result has length one.

    A `Normalize` divides by the root of the mean of the squares rather than by the
    root of their sum, and multiplies by a gain of one learned weight per position.
    A `Normalize` is therefore this operator times the root of the number of values
    and times that gain, and a normalisation a model applies with no learned weight
    is written with this operator. `algebra.operator_expansion.expand_l2_norm`
    writes it out.
    '''
    name: fd.DynamicName | None = fd.DynamicName('L^{2}')
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
        # An axis in the output only is repeated along: it is degree, and no
        # operand's reindexing names it. `'q -> q v'` is the repeat of `q`
        # along `v`, a `View` whose reindexing reads `(q,)`.
        input_indexes, output_indexes, input_weaves, output_weaves, reindexings, degree = che.signature_to_broadcast(
            signature, datatype, support_unit_object=False, produced_as_degree=True
        )
        assert len(output_indexes) == 1
        input_signature = tuple(tuple(i for i in segment if i >= 0) for segment in input_indexes)
        # A single input with nothing contracted is not an einsum at all.
        operator = (
            View() if input_signature == ((),)
            else Einops(name=fd.DynamicName('einops'), signature=input_signature)
        )
        if operator == View() and tutil.is_identity(reindexings[0]):
            return cat.ProdObject((input_weaves[0].target(),)).identity()
        return cat.Broadcasted[B, cat.RawAxis](
            operator=operator,
            input_weaves=input_weaves,
            output_weaves=output_weaves,
            reindexings=reindexings,
            backup_degree=degree if not input_weaves else None,
        )

def linear_size_to_shape(size: int | chp.ProductObjectTarget[cat.RawAxis, str | fd.DynamicName]) -> cat.ProdObject[cat.RawAxis]:
    if isinstance(size, int):
        return cat.ProdObject.from_iter(cat.RawAxis() for _ in range(size))
    return chp.object_product(size, chp.axis_converter)

@dataclass(frozen=True)
class Linear(cat.Operator):
    '''A contraction against a weight the operator holds rather than names.

    An input reads one of two ways, and the datatype of its target is what
    says which.

    A **real** input is contracted. Its target axes are axes of the weight,
    and the operation sums over them, so `Linear : R[m] -> R[f]` holds a
    weight `R[m, f]` and computes `y_f = sum_m x_m W_{mf}`. With no inputs at
    all the sum is empty and the weight is the whole of the result, which is
    the parameter-array case `obsidian/06-practice/Representing Models.md`
    describes.

    A **`Natural(n)`** input is an index, and it selects. It names one of `n`
    weights rather than a value to be summed against, so the weight gains an
    axis of `n` entries and the index picks the slab at that position. The
    target of such an input is rank 0, because an index is a single number.
    `Linear : (Nat(n), R[m]) -> R[f]` therefore holds `R[n, m, f]` and
    computes `y_f = sum_m x_m W_{i m f}` at the index `i` the operand carries.

    A selection is not a contraction, and writing it as one would need the
    index expanded into a one-hot vector of length `n` and contracted away.
    Keeping it in the datatype is what lets the index stay runtime data while
    the rest of the expression stays affine, per
    `obsidian/02-categories/Sparse Expansion.md`. The expert `Linear` of a
    mixture of experts is the case the distinction was written for: the index
    comes from the router's `TopK`, and the weight holds every expert.

    `deepseek.IndexSelect` takes the same posture for a payload on a wire, so
    the two consumers of a selection's index read it identically.
    `para.processing.show_grabbed_parameters.weight_axes` is where the rule is
    applied.
    '''
    name: fd.DynamicName | None = fd.DynamicName('L')
    bias: bool = False
    @classmethod
    def template[B: cat.Datatype = cat.Reals](
        cls,
        input_size: int | chp.ProductObjectTarget[cat.RawAxis, str | fd.DynamicName] = 1,
        output_size: int | chp.ProductObjectTarget[cat.RawAxis, str | fd.DynamicName] = 1,
        name: str | None | fd.DynamicName = None,
        datatype: B = cat.Reals(),
        output_datatype: B | None = None,
        bias: bool = False,
    ):
        output_datatype = output_datatype if output_datatype is not None else datatype
        name = fd.DynamicName.from_str(name) if name is not None else fd.DynamicName('L')
        name = name.reconstruct(settings=fd.DynamicNameSettings(bold=True))
        operator = Linear(
            name=name,
            bias=bias
        )
        return sized(operator, input_size, output_size, datatype, output_datatype)

def selects_weights[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A],
) -> bool:
    '''True when one of a `Linear`'s inputs is an index rather than a value.

    A `Linear` with no such input is a contraction, and every rule that reads
    it as one applies to it: `algebra.linear_expansion` rewrites
    it into a weight array and an `Einops`, and the dependency analysis then
    sees its output axes arrive through an affine reindexing like any other.

    A `Linear` that selects is not a contraction in that sense. Its index
    operand names one of the weights and the map from index to weight is
    runtime data, so no reindexing can carry it and no `Einops` can state it.
    A pass that treats such a `Linear` as a contraction sums over the selection
    axis instead of reading one entry of it.

    False for any operator other than a `Linear`.
    '''
    return isinstance(target.operator, Linear) and any(
        isinstance(weave.target().datatype, cat.Natural)
        for weave in target.input_weaves
    )

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
        signature_segments, input_weaves, output_weaves, reindexings, degree = chs.generic_signature(
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
            reindexings=reindexings,
            backup_degree=degree if not input_weaves else None,
        )
    
@dataclass(frozen=True)
class ConstantOp(cat.Operator):
    '''A nullary constant emitting `value` (default zero). Used as a streaming
    initializer: the identity a fold accumulates onto - e.g. 0 to seed a sum.

    The `Op` suffix separates the operator from `nm.Constant`, which is the
    numeric naming a mathematical constant such as Euler's number.
    '''
    name: fd.DynamicName | None = fd.DynamicName('ConstantOp')
    value: nm.Numeric = nm.Integer(0)

class ArrangeNeedsOneAxis(ValueError):
    '''An `Arrange` asked to write the positions of anything other than one axis.'''


@dataclass(frozen=True)
class Arrange(cat.Operator):
    '''The positions of an axis as an array over that axis. The operator has no
    operands, and its result holds `i_x` at index `i_x` of the axis `x`, so the
    array is `[0, 1, ..., |x| - 1]`.

    The datatype of the result is `Natural(|x|)`, because every entry is an index
    of `x`. A product of the result with a real array therefore has operands of
    two datatypes, and `multiply_by_positions` writes it.

    The axis stands in the target of the output weave. A tiled operator is the
    same function at every index of its degree, and an `Arrange` writes a
    different value at every position.

    The default name is the index of the axis, `i_x`, which is the value the
    result holds at that index.
    '''
    name: fd.DynamicName | None = fd.DynamicName('\\mathrm{arrange}')
    @classmethod
    def template(
        cls,
        axis: chp.ProductObjectTarget[cat.RawAxis, str | fd.DynamicName],
        name: str | None | fd.DynamicName = None,
    ) -> cat.Broadcasted[cat.Natural, cat.RawAxis, Arrange]:
        axes = tuple(linear_size_to_shape(axis))
        if len(axes) != 1:
            raise ArrangeNeedsOneAxis(f'{len(axes)} axes given to Arrange: {axes}')
        arranged = axes[0]
        axis_name = arranged.uid._name
        if name is not None:
            operator = cls(name=fd.DynamicName.from_str(name))
        elif axis_name is not None:
            operator = cls(name=fd.DynamicName(
                body='i', subscript=axis_name.with_exponent(None)))
        else:
            operator = cls()
        return cat.Broadcasted(
            operator=operator,
            input_weaves=(),
            output_weaves=(cat.Weave(
                cat.Natural(arranged.local_size()), (arranged,)),),
            reindexings=())


def multiply_by_positions[B: cat.Datatype = cat.Reals, A: cat.Axis = cat.RawAxis](
    positions: cat.Array[cat.Natural, A],
    values: cat.Array[B, A],
) -> cat.Broadcasted[B | cat.Natural, A, Einops]:
    '''The product of every entry of `positions`, which an `Arrange` writes, with
    every entry of `values`, over the axes of `positions` followed by the axes of
    `values`. The entry at `(i_x, i_t)` is `i_x` times `values[i_t]`.

    `Einops.template` gives one datatype to every weave. The operands here carry
    two, a `Natural` and the datatype of `values`, and the result carries the
    datatype of `values`, so the weaves are written out. Nothing is contracted,
    so every position is tiled and the signature has an empty group per operand.
    '''
    position_axes = tuple(positions.shape())
    value_axes = tuple(values.shape())
    degree = cat.ProdObject((*position_axes, *value_axes))
    tiled = cat.WeaveMode.TILED
    return cat.Broadcasted(
        operator=Einops(name=fd.DynamicName('einops'), signature=((), ())),
        input_weaves=(
            cat.Weave(positions.datatype, (tiled,) * len(position_axes)),
            cat.Weave(values.datatype, (tiled,) * len(value_axes))),
        output_weaves=(cat.Weave(values.datatype, (tiled,) * len(degree)),),
        reindexings=(
            cat.Rearrangement(
                mapping=tuple(range(len(position_axes))), _dom=tuple(degree)),
            cat.Rearrangement(
                mapping=tuple(range(len(position_axes), len(degree))),
                _dom=tuple(degree))))


@dataclass(frozen=True)
class Normalize(cat.Operator):
    '''`y = gamma * x * (mean(x^2) + epsilon)^{-1/2} + beta`, over the axes the
    operator consumes.

    The mean of the squares runs over every axis in the target, so one value of the
    result is one value of the input divided by the root mean square of the whole
    array the input names. `L2Norm` divides by the root of the sum of the squares
    instead, takes no learned weight and adds no epsilon.

    `gain` says that the operator is multiplied by `gamma`, which is one learned
    number per position of the normalised axes. `bias` says that `beta` is added
    after, which is one learned number per position as well. A model whose
    normalisation has neither declares both as `False`, and
    `algebra.operator_expansion.expand_normalize` then writes the scaling alone.
    `para.processing.show_grabbed_parameters.parameter_arrays` reads the two fields
    to decide which learned arrays to grab, and the grabbed arrays reach the operator
    as its leading operands, the gain first.

    `epsilon` is the number added under the root. A model that states the value its
    released code adds supplies it, and `SYMBOLIC_EPSILON` stands for the number
    where the model does not say what it is.
    '''
    name: fd.DynamicName | None = fd.DynamicName('RMSNorm')
    gain: bool = True
    bias: bool = False
    epsilon: nm.Numeric = SYMBOLIC_EPSILON
    @classmethod
    def template[B: cat.Datatype = cat.Reals](
        cls,
        input_size: int | chp.ProductObjectTarget[cat.RawAxis, str] = 1,
        datatype: B = cat.Reals(),
        gain: bool = True,
        bias: bool = False,
        epsilon: nm.Numeric = SYMBOLIC_EPSILON,
    ):
        return sized(
            cls(gain=gain, bias=bias, epsilon=epsilon),
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


class NotAPowerOfTwo(ValueError):
    '''A bitwise operation asked for on a `Natural` whose bound is not a power of two.'''


def bit_width(datatype: cat.Datatype) -> nm.Numeric:
    '''The `N` of a `Natural(2^N)`, which is the number of bits its values occupy.
    A bitwise operation is defined on bit patterns, so its operands and its result
    are `Natural(2^N)`, and any other datatype raises.'''
    match datatype:
        case cat.Natural(max_value=nm.Power(base=nm.Integer(_value=2), exponent=width)):
            return width
        case cat.Natural(max_value=nm.Integer(_value=value)) if value > 0 and value & (value - 1) == 0:
            return nm.Integer(value.bit_length() - 1)
    raise NotAPowerOfTwo(f'{datatype} is not a Natural bounded by a power of two')


def with_datatypes[A: cat.Axis](
    target: cat.Broadcasted[cat.Datatype, A],
    inputs: tuple[cat.Datatype, ...],
    outputs: tuple[cat.Datatype, ...],
) -> cat.Broadcasted[cat.Datatype, A]:
    '''`target` with the datatype of each input weave and each output weave replaced,
    in order. A signature template gives one datatype to every weave, and an operation
    on naturals returns a different bound from the ones it reads.'''
    if len(inputs) != len(target.input_weaves) or len(outputs) != len(target.output_weaves):
        raise ValueError(
            f'{len(inputs)} input and {len(outputs)} output datatypes for a Broadcasted '
            f'with {len(target.input_weaves)} inputs and {len(target.output_weaves)} outputs')
    return target.reconstruct(
        input_weaves=tuple(weave.reconstruct(datatype=datatype)
                           for weave, datatype in zip(target.input_weaves, inputs)),
        output_weaves=tuple(weave.reconstruct(datatype=datatype)
                            for weave, datatype in zip(target.output_weaves, outputs)))


@dataclass(frozen=True)
class BitwiseXor(cat.Operator):
    '''The bitwise exclusive or of every entry along one axis, which is a reduction
    whose unit is zero. Its operand and its result are `Natural(2^N)`, because the
    exclusive or acts on the `N` bits of each value and the result of two values
    below `2^N` is again below `2^N`.'''
    name: fd.DynamicName | None = fd.DynamicName('\\oplus')

    @classmethod
    def template(
        cls,
        axis: chp.ProductObjectTarget[cat.RawAxis, str | fd.DynamicName],
        datatype: cat.Natural,
    ) -> cat.Broadcasted[cat.Natural, cat.RawAxis, BitwiseXor]:
        bit_width(datatype)
        reduced = tuple(linear_size_to_shape(axis))
        return cat.Broadcasted(
            operator=cls(),
            input_weaves=(cat.Weave(datatype, reduced),),
            output_weaves=(cat.Weave(datatype, ()),),
            reindexings=(cat.ProdObject().identity(),))


@dataclass(frozen=True)
class Modulo(cat.Operator):
    '''The remainder of the first operand divided by the second, entry by entry. The
    remainder is below the divisor, so the result carries the divisor's datatype.'''
    name: fd.DynamicName | None = fd.DynamicName('\\bmod')

    @classmethod
    def template(
        cls,
        signature: str,
        dividend: cat.Natural,
        divisor: cat.Natural,
    ) -> cat.Broadcasted[cat.Natural, cat.RawAxis, Modulo]:
        '''`signature` names two operands and one result with nothing contracted, as
        `'x G, G K -> x G K'` reads a hash per order against a prime per order and
        head.'''
        built = che.signature_to_broadcasted(cls(), signature, dividend)
        if len(built.input_weaves) != 2:
            raise ValueError(f'{len(built.input_weaves)} operands in {signature!r} for a Modulo')
        return with_datatypes(built, (dividend, divisor), (divisor,))


@dataclass(frozen=True)
class Cast(Elementwise):
    '''The identity on values, read into another datatype. A cast from `Natural(a)` to
    `Natural(b)` with `a <= b` widens and loses nothing. A cast that narrows is exact
    only where every value the operand can hold is below the new bound, and the
    description of the block that holds it states why.'''
    name: fd.DynamicName | None = fd.DynamicName('\\mathrm{cast}')
    operator: str | None = 'cast'

    @classmethod
    def template[B: cat.Datatype, A: cat.Axis]( # type: ignore
        cls,
        base: chp.ProductObjectTarget[cat.Array[B, A], B],
        to: cat.Datatype,
        name: str | None | fd.DynamicName = None,
    ) -> cat.Broadcasted[B, A, Cast]:
        source = chp.object_product(base, conversion=chp.datatype_converter)[0]
        tiled = (cat.WeaveMode.TILED,) * len(source.shape())
        return cat.Broadcasted(
            operator=cls(name=fd.DynamicName.from_str(name) if name is not None
                         else cls.__dataclass_fields__['name'].default),
            input_weaves=(cat.Weave(source.datatype, tiled),),
            output_weaves=(cat.Weave(to, tiled),),
            reindexings=(source.shape().identity(),))


@dataclass(frozen=True)
class FixedArray(cat.Operator):
    '''An array the model reads and never learns, such as the multipliers, the primes
    and the offsets of a hash. The operator has no operands, and the axes of the array
    stand in the target of its output weave, because the array holds a different value
    at every position. A `Linear` with no inputs is a learned array and draws as a
    weight, and an `Embedding` is a learned table read at an index. This operator is
    neither, and `show_grabbed_parameters` grabs nothing for it.'''
    name: fd.DynamicName | None = fd.DynamicName('\\mathrm{fixed}')

    @classmethod
    def template[B: cat.Datatype, A: cat.Axis](
        cls,
        name: str | fd.DynamicName,
        shape: tuple[A, ...],
        datatype: B,
    ) -> cat.Broadcasted[B, A, FixedArray]:
        return cat.Broadcasted(
            operator=cls(name=fd.DynamicName.from_str(name)),
            input_weaves=(),
            output_weaves=(cat.Weave(datatype, tuple(shape)),),
            reindexings=())
