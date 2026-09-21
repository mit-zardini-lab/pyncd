from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    Any,
    Type,
    Callable,
)
import math
from abc import ABC

import data_structure.Term as fd # for 'foundations'
import data_structure.Numeric as nm
import term_utilities.term_utilities as tutil
import data_structure.Category as cat
import data_structure.Operators as ops
import functools

import data_structure.BrTyping as Br

import torch
import torch.nn as nn
import einops

import torch_compile.bcast as bcast
import torch_compile.torch_utilities as torch_utilities
import para.data_structure.transpose as ptr

@dataclass(frozen=True)
class TorchFunctionInfo:
    explicit_dim: bool = False
    semantic: bool = False
    implicit_lower: bool = True

class ConstructedModule[M: cat.Morphism](nn.Module, ABC):
    operation_registry: dict[Type[cat.Operator], Type[ConstructedModule]] = {}
    functions_registry: dict[Type[cat.Operator], tuple[Callable, TorchFunctionInfo]] = {}

    def __init_subclass__(cls, operation_key: Type[cat.Operator] | None = None) -> None:
        super().__init_subclass__()
        if operation_key:
            ConstructedModule.operation_registry[operation_key] = cls

    @classmethod
    def construct(cls, target: cat.Morphism, dim: None | int = None) -> ConstructedModule | Lambda:
        match target:
            case cat.Rearrangement():
                return ConstructedRearrangement(target)
            case cat.ProductOfMorphisms():
                return ConstructedProduct(target)
            case cat.Composed():
                return ConstructedComposed(target)
            case cat.Block():
                return ConstructedBlock(target)
            case cat.Broadcasted():
                return ConstructedModule.construct_broadcasted(target)
        print(target)
        raise NotImplementedError()
    
    @classmethod
    def construct_broadcasted(cls, target: cat.Broadcasted) -> ConstructedModule | Lambda:
        operator_type = type(target.operator)
        if operator_type in cls.operation_registry:
            return cls.operation_registry[operator_type](target)
        elif operator_type in cls.functions_registry:
            return Lambda(target)
        else:
            raise NotImplementedError(f'No constructor found for operator type {operator_type}')

    @classmethod
    def add_function(cls, 
                     func_type: Type[cat.Operator], 
                     func: Callable, 
                     dim: bool = False,
                     semantic: bool = False) -> None:
        ConstructedModule.functions_registry[func_type] = (
            func,
            TorchFunctionInfo(
                explicit_dim = dim,
                semantic = semantic
            )
        )

    def __init__(self, target: M, func_info: TorchFunctionInfo = TorchFunctionInfo()) -> None:
        super().__init__()
        self.target = target
        self.func_info = func_info
    

class Lambda(nn.Module):
    def __init__(self, target: cat.Broadcasted) -> None:
        super().__init__()
        self.target = target
        self.func, self.func_info = ConstructedModule.functions_registry[
            type(target.operator)
        ]
        self.func = broadcast_func(target, self.func, self.func_info)

    def forward(self, *xs: torch.Tensor):
        return self.func(*xs)
    
    def __repr__(self):
        return f'{type(self).__qualname__}({self.target.operator})'


##################
#### CATEGORY ####
##################

class ConstructedBlock[B: cat.Datatype, A: cat.Axis](
    ConstructedModule[Br.Block[B, A]]
):
    def __init__(self, target: Br.Block[B, A]) -> None:
        super().__init__(target)
        if isinstance(repetition := target.block_tag.repetition, nm.Integer) and repetition._value > 1:
            self.module = nn.Sequential(
                *(ConstructedModule.construct(target.body) for _ in range(repetition._value))
            )
        else:
            self.module = ConstructedModule.construct(target.body)

    def forward(self, *xs: torch.Tensor):
        if isinstance(self.module, nn.Sequential):
            for module in self.module:
                xs = to_tuple(module(*xs))
            return xs
        else:
            return self.module(*xs)
    
class ConstructedProduct[B: cat.Datatype, A: cat.Axis](
    ConstructedModule[Br.ProductOfMorphisms[B, A]]
    ):
    def __init__(self, target: Br.ProductOfMorphisms[B, A]) -> None:
        super().__init__(target)
        fs = map(ConstructedModule.construct, target.content)
        self.content = nn.ModuleList(fs)
        
    def forward(self, *xs: torch.Tensor) -> fd.Prod[torch.Tensor]:
        return tuple(
            y
            for f, (_, x)
            in zip(self.content, self.target.partition(xs))
            for y in to_tuple(f(*x))
        )
    
def to_tuple[T](x: T | tuple[T, ...]) -> tuple[T, ...]:
    return x if isinstance(x, tuple) else (x,)

class ConstructedComposed[B: cat.Datatype, A: cat.Axis](
    ConstructedModule[Br.Composed[B, A]]
):
    def __init__(self, target: Br.Composed[B, A]) -> None:
        super().__init__(target)
        fs = map(ConstructedModule.construct, target.content)
        self.chain = nn.Sequential(*fs)

    def forward(self, *xs):
        for module in self.chain:
            xs = to_tuple(module(*xs))
        return xs
    
class ConstructedRearrangement(ConstructedModule):
    def __init__(self, target: cat.Rearrangement) -> None:
        super().__init__(target)

    def forward(self, *xs: torch.Tensor):
        return self.target.apply(xs)

##################
## BROADCASTING ##
##################

def broadcast_func(
        target: cat.Broadcasted, 
        func: Callable,
        broadcast_info: TorchFunctionInfo = TorchFunctionInfo()):
    
    assert tutil.is_mappable_broadcast(target)
    # Einops covers it for now.
    displacement = bcast.get_displacement(target)
    match broadcast_info:
        case TorchFunctionInfo(explicit_dim=True) if displacement is not None:
            def dim_func(*xs: torch.Tensor, **kwargs: Any):
                return func(*xs, **kwargs, dim=displacement)
            return dim_func
        case TorchFunctionInfo(implicit_lower=True) if displacement == -1:
            return func
        case TorchFunctionInfo(semantic=True) if bcast.is_semantically_broadcastable(target):
            degree_size = len(target.degree())
            if all(tutil.is_identity(eta) for eta in target.reindexings):
                return func
            def _func(*xs: torch.Tensor):
                xs = tuple(
                    bcast.broadcast_to_degree(x, degree_size, tutil.get_mapping(eta))
                    if not tutil.is_identity(eta) else x
                    for x, eta, weave in zip(xs, target.reindexings, target.input_weaves)
                )
                return func(*xs)
            return _func
        case _ if bcast.vmappable(target):
            vmap_guide = bcast.broadcast_vmap(target)
            for input_loc, output_loc in vmap_guide:
                func = torch.vmap(func, in_dims=input_loc, out_dims=output_loc)
            return func
        case _:
            raise NotImplementedError()

################
## OPERATIONS ##
################

############
## STATIC ##
############

def generate_einops_signature(target: cat.Broadcasted[Any, Any, ops.Einops]):
    assert tutil.is_mappable_broadcast(target)
    operator = target.operator
    degree_tags = tuple(
        f'y{i}' for i, _ in enumerate(target.degree()))
    input_tags = tuple(
        f'x{i}' for i in set[int]().union(*target.operator.signature)
    )
    input_segments = (
        input_weave.imprint_axes(
            (degree_tags[i] for i in tutil.get_mapping(eta)),
            (input_tags[j] for j in signature_segment),
        )
        for input_weave, signature_segment, eta in 
        zip(
            target.input_weaves, 
            operator.signature,
            target.reindexings
        )
    )
    input_signature = ', '.join(
        '... ' + ' '.join(segment)
        for segment in input_segments
    )
    output_signature = ' '.join(degree_tags)
    return f'{input_signature} -> ... {output_signature}'

class ConstructedEinops[B: cat.Datatype, A: cat.Axis](
    ConstructedModule[cat.Broadcasted[B, A, ops.Einops]],
    operation_key=ops.Einops
):
    def __init__(self, target: cat.Broadcasted[B, A, ops.Einops]) -> None:
        super().__init__(target)
        self.signature = generate_einops_signature(self.target)
        segments, _ = self.signature.split(' -> ')
        self.names = tuple(segment.split()[1:] for segment in segments.split(', '))

    def forward(self, *xs: torch.Tensor):
        expanded = expand_repeats(xs, self.names)
        return einops.einsum(*expanded, self.signature)  # type: ignore


def expand_repeats(xs: tuple[torch.Tensor, ...],
                   names: tuple[list[str], ...]) -> tuple[torch.Tensor, ...]:
    '''Expand a dim of size 1 to the size the same name has in another
    operand. A `View` node compiles to a broadcastable view whose repeated
    axes have size 1, and an einsum reading it needs them at full size.'''
    sizes: dict[str, int] = {}
    for x, dims in zip(xs, names):
        for name, size in zip(dims, x.shape[len(x.shape) - len(dims):]):
            if size != 1:
                sizes[name] = size
    expanded = []
    for x, dims in zip(xs, names):
        leading = len(x.shape) - len(dims)
        shape = (*x.shape[:leading],
                 *(sizes.get(name, size) if size == 1 else size
                   for name, size in zip(dims, x.shape[leading:])))
        expanded.append(x.expand(shape) if shape != tuple(x.shape) else x)
    return tuple(expanded)
    

ConstructedModule.add_function(ops.SoftMax, torch.softmax, dim=True)


def epsilon_value(epsilon: nm.Numeric, like: torch.Tensor) -> torch.Tensor:
    '''The number a norm adds, as a scalar tensor. A `FreeNumeric` epsilon names a
    number the model has not stated, and it compiles to zero, so the norm divides by
    the statistic itself.'''
    zero = torch.zeros((), dtype=like.dtype)
    if isinstance(epsilon, nm.FreeNumeric):
        return zero
    return numeric_torch(epsilon, zero)


def l1_normalise(tensor: torch.Tensor, dim: int,
                 epsilon: torch.Tensor) -> torch.Tensor:
    '''`tensor` divided by its sum along `dim` plus `epsilon`, keeping the shape.'''
    return tensor / (tensor.sum(dim=dim, keepdim=True) + epsilon)


class ConstructedL1Norm[B: cat.Datatype, A: cat.Axis](
        ConstructedModule, operation_key=ops.L1Norm):
    '''An `L1Norm` carries the epsilon it adds to the divisor, so it compiles from
    its own operator rather than from the table of plain functions.'''
    def __init__(self, target: cat.Broadcasted[B, A, ops.L1Norm]) -> None:
        super().__init__(target, TorchFunctionInfo(explicit_dim=True))
        epsilon = target.operator.epsilon
        self.func = broadcast_func(
            target,
            lambda tensor, dim: l1_normalise(
                tensor, dim, epsilon_value(epsilon, tensor)),
            self.func_info)

    def forward(self, *xs: torch.Tensor):
        return self.func(*xs)

def l2_normalise(tensor: torch.Tensor) -> torch.Tensor:
    '''`tensor` divided by the root of the sum of its squares, keeping the shape.

    The sum runs over every dimension of the tensor the operator is given, which is
    the array its target names. The axes a model does not normalise over are the
    degree, and `broadcast_func` maps this function over them.
    '''
    return tensor / tensor.pow(2).sum().sqrt()


ConstructedModule.add_function(ops.L2Norm, l2_normalise)
# A `Maximum` folds the one axis in its input target, which is the axis the
# softmax it shifts normalises over.
ConstructedModule.add_function(ops.Maximum, torch.amax, dim=True)
ConstructedModule.add_function(
    ops.AdditionOp, lambda *xs: sum(xs[1:], xs[0]), semantic=True)
# A node: a reindexing and nothing else. Under the semantic path a dropped
# degree axis becomes a size-1 dim, so a repeat (`q -> q v`) compiles to the
# broadcastable view rather than a materialised copy, because the size of `v` is in
# no tensor, and downstream broadcasting supplies it.
ConstructedModule.add_function(ops.View, lambda x: x, semantic=True)
ConstructedModule.add_function(ops.Elementwise, torch.relu)
ConstructedModule.add_function(ops.ReLU, torch.relu)


def numeric_torch(formula: nm.Numeric, x: torch.Tensor) -> torch.Tensor:
    '''Evaluate a formula at a tensor: the `FreeInput` is `x`, everything else
    is arithmetic. A `FreeNumeric` is a size symbol and has no value here.'''
    match formula:
        case nm.FreeInput():
            return x
        case nm.Integer(_value=value):
            return torch.full_like(x, float(value))
        case nm.Constant() | nm.UnitOfMeasure():
            return torch.full_like(x, formula.to_float())
        case nm.Addition(content=parts):
            return functools.reduce(torch.add, (numeric_torch(p, x) for p in parts))
        case nm.Multiplication(content=parts):
            return functools.reduce(torch.mul, (numeric_torch(p, x) for p in parts))
        case nm.Power(base=nm.E, exponent=exponent):
            return torch.exp(numeric_torch(exponent, x))
        case nm.Power(base=base, exponent=exponent):
            return torch.pow(numeric_torch(base, x), numeric_torch(exponent, x))
        case nm.Logarithm(base=nm.E, argument=argument):
            return torch.log(numeric_torch(argument, x))
        case nm.Logarithm(base=base, argument=argument):
            return torch.log(numeric_torch(argument, x)) / torch.log(numeric_torch(base, x))
        case nm.CumulativeGaussian(argument=argument):
            return torch.special.ndtr(numeric_torch(argument, x))
        case nm.Sigmoid(argument=argument):
            return torch.sigmoid(numeric_torch(argument, x))
        case nm.Conjugate(argument=argument):
            return torch.conj(numeric_torch(argument, x))
        case nm.IsPositive(argument=argument):
            return (numeric_torch(argument, x) > 0).to(x.dtype)
        case nm.RectifiedLinear(argument=argument):
            return torch.relu(numeric_torch(argument, x))
        case nm.Clamp(argument=argument, lower=lower, upper=upper):
            return torch.clamp(numeric_torch(argument, x),
                               numeric_torch(lower, x), numeric_torch(upper, x))
        case nm.Sign(argument=argument):
            return torch.sign(numeric_torch(argument, x))
        case nm.AbsoluteValue(argument=argument):
            return torch.abs(numeric_torch(argument, x))
        case nm.LargerOf(first=first, second=second):
            return torch.maximum(numeric_torch(first, x), numeric_torch(second, x))
        case nm.SquareRoot(argument=argument):
            return torch.sqrt(numeric_torch(argument, x))
    raise NotImplementedError(f'no torch value for {type(formula).__qualname__}')


class ConstructedArithmetic(ConstructedModule, operation_key=ops.Arithmetic):
    '''An `Arithmetic` is its formula, so it compiles without a table.'''
    def __init__(self, target: cat.Broadcasted) -> None:
        super().__init__(target)
        formula = target.operator.formula
        self.func = broadcast_func(target, lambda x: numeric_torch(formula, x))

    def forward(self, *xs: torch.Tensor):
        return self.func(*xs)

def weighted_triangular_lower(x: torch.Tensor) -> torch.Tensor:
    trilled = torch.tril(x)
    return trilled / (torch.sum(trilled, dim=-1, keepdim=True) + 1e-8)

ConstructedModule.add_function(ops.WeightedTriangularLower, weighted_triangular_lower)

#############
## LEARNED ##
#############
def axis_einsum_signature(
    target: cat.Broadcasted, operands: fd.Prod[int] | None = None,
) -> str:
    '''
    An einsum signature read off AXIS IDENTITY rather than off an `Einops`
    signature. The parametrised `Linear` and `Transpose`
    (`para.show_grabbed_parameters`, `para.derivative._parametrised_linear`)
    build their weight operand from the seed's own weave targets, so an axis
    the operands share is the same axis object, and which axes contract is
    determined by which coincide - `... q m, ... m d -> ... q d` for a
    linear, `... m d, ... q m -> ... q d` for a weight-first linear.
    `operands` selects the input weaves by position, so a bias, which is added
    rather than multiplied, can be left out of the einsum.
    '''
    assert tutil.is_mappable_broadcast(target)
    tags: dict[Any, str] = {}

    def tag(axis) -> str:
        return tags.setdefault(axis, f'x{len(tags)}')

    degree_tags = tuple(f'y{i}' for i, _ in enumerate(target.degree()))
    pairs = tuple(zip(target.input_weaves, target.reindexings))
    if operands is not None:
        pairs = tuple(pairs[i] for i in operands)
    segments = (
        weave.imprint_axes(
            (degree_tags[i] for i in tutil.get_mapping(eta)),
            (tag(axis) for axis in weave.target().shape()))
        for weave, eta in pairs)
    output = target.output_weaves[0].imprint_axes(
        degree_tags,
        (tag(axis) for axis in target.output_weaves[0].target().shape()))
    return (', '.join('... ' + ' '.join(segment) for segment in segments)
            + ' -> ... ' + ' '.join(output))


def parametrised_linear_func(target: cat.Broadcasted) -> Callable:
    '''`(W[, b], x) -> y` with the weight as the FIRST operand: one einsum by
    axis identity, the bias broadcast on afterwards (its target axes are the
    output's trailing dims, so plain torch broadcasting right-aligns it).

    `show_grabbed_parameters` puts the parameters in the first slots, so the
    bias sits between the weight and the data and the einsum names the two
    operands it contracts rather than taking a prefix of them.
    '''
    if len(target.input_weaves) == 2:
        signature = axis_einsum_signature(target)
        return lambda w, x: einops.einsum(w, x, signature)
    signature = axis_einsum_signature(target, operands=(0, 2))
    return lambda w, b, x: einops.einsum(w, x, signature) + b


class ConstructedLinear[
    B: cat.Datatype, A: cat.Axis
](ConstructedModule, operation_key=ops.Linear):
    def __init__(self, target: cat.Broadcasted[B, A, ops.Linear]) -> None:
        super().__init__(target)
        match target.input_weaves, target.output_weaves:
            case [
                [cat.Weave() as weave_in],
                [cat.Weave() as weave_out]
            ]:
                self.in_size: fd.Prod[int] = tuple(x.local_size()._value for x in weave_in.target().shape()) # type: ignore
                self.out_size: fd.Prod[int] = tuple(y.local_size()._value for y in weave_out.target().shape()) # type: ignore
                self.bias = target.operator.bias
                self.module = torch_utilities.Multilinear(
                    self.in_size, self.out_size, self.bias
                )
                self.func = broadcast_func(
                    target,
                    self.module.forward)
            case _:
                # The parametrised form. The weight, and the bias where the
                # operator has one, arrives as an operand, so the module
                # holds no state of its own.
                self.func = parametrised_linear_func(target)

    def forward(self, *xs):
        return self.func(*xs)


class ConstructedTranspose(ConstructedModule, operation_key=ptr.Transpose):
    '''
    The 2-operand transpose `para.derivative._parametrised_linear` writes:
    the cotangent against the same weight, over the axes they share, again
    one einsum by axis identity. A bare 1-operand `Transpose` cannot compile:
    its weight is inside the operator, and the *values* live in whatever
    `ConstructedLinear` its forward became, so parametrise first.
    '''
    def __init__(self, target: cat.Broadcasted) -> None:
        super().__init__(target)
        assert len(target.input_weaves) >= 2, (
            'a bare Transpose has its weight inside the operator - '
            'parametrise the expression before compiling it')
        self.func = parametrised_linear_func(target)

    def forward(self, *xs):
        return self.func(*xs)
    
class ConstructedEmbedding[
    A: cat.Axis
](ConstructedModule, operation_key=ops.Embedding):
    def __init__(self, target: cat.Broadcasted[cat.Datatype, A, ops.Embedding]):
        super().__init__(target)
        match target.input_weaves, target.output_weaves:
            case [
                [cat.Weave(cat.Natural(size)) as weave_in],
                [cat.Weave() as weave_out]
            ]:
                self.num_embeddings = size._value # type: ignore
                self.dims = tuple(y.local_size()._value for y in weave_out.target().shape()) # type: ignore
            case _:
                assert False
            
        self.module = torch.nn.Embedding(
            self.num_embeddings, math.prod(self.dims)
        )

        def func(*xs):
            x = xs[0]
            original_shape = x.shape
            x = x.view(-1)
            embedded = self.module(x)
            return embedded.view(*original_shape, *self.dims)
        
        self.func = broadcast_func(target, func)
    
    def forward(self, *xs):
        return self.func(*xs)
    
def rms_normalise(tensor: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
    '''`tensor` divided by the root of the mean of its squares plus `epsilon`, over
    every dimension of the tensor the operator is given, which is the array its target
    names.'''
    return tensor * torch.rsqrt(tensor.pow(2).mean() + epsilon)


class ConstructedNorm[B: cat.Datatype, A: cat.Axis](ConstructedModule, operation_key=ops.Normalize):
    '''A `Normalize` divides by the root mean square over its target and then applies
    whichever of the gain and the bias its operator declares. The data is the LAST
    operand, because a grabbed gain and a grabbed bias occupy the slots ahead of it.'''
    def __init__(self, target: cat.Broadcasted[B, A, ops.Normalize]) -> None:
        super().__init__(target)
        epsilon = target.operator.epsilon
        self.func = broadcast_func(target, lambda *xs: self.combine(
            *xs[:-1],
            scaled=rms_normalise(xs[-1], epsilon_value(epsilon, xs[-1]))))

    def combine(self, *parameters: torch.Tensor,
                scaled: torch.Tensor) -> torch.Tensor:
        operator = self.target.operator
        declared = [name for name, present
                    in (('gain', operator.gain), ('bias', operator.bias)) if present]
        if not parameters:
            return scaled
        combined = scaled
        for name, parameter in zip(declared, parameters):
            combined = (combined * parameter if name == 'gain'
                        else combined + parameter)
        return combined

    def forward(self, *xs: torch.Tensor):
        return self.func(*xs)
