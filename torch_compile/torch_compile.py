from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    Any,
    Type,
    Callable,
)
import math
from abc import ABC

import warnings

import data_structure.Term as fd # for 'foundations'
import data_structure.Numeric as nm
import term_utilities.term_utilities as tutil
import data_structure.Category as cat
import data_structure.Operators as ops
from data_structure.TensorExpr import TensorRef, IversonBinOp, IversonUnaryOp, _factor_axes
from torch_compile.materialise import materialise_iverson

import data_structure.BrTyping as Br
from data_structure.TensorDSL import Scan, Slice, Reindex, Scatter

import torch
import torch.nn as nn
import einops

# associative_scan is a top-level API from PyTorch 2.5; fall back to the
# higher-order-ops location for older builds.
try:
    _associative_scan = torch.associative_scan  # type: ignore[attr-defined]
except AttributeError:
    from torch._higher_order_ops import associative_scan as _associative_scan

import torch_compile.bcast as bcast
import torch_compile.torch_utilities as torch_utilities

@dataclass(frozen=True)
class TorchFunctionInfo:
    explicit_dim: bool = False
    semantic: bool = False
    implicit_lower: bool = True

class ConstructedModule[M: cat.Morphism](nn.Module, ABC):
    operation_registry: dict[Type[cat.Operator], Type[ConstructedModule]] = {}
    functions_registry: dict[Type[cat.Operator], tuple[Callable, TorchFunctionInfo]] = {}

    def __init_subclass__(cls, operation_key: Type[cat.Operator] | None = None):
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
            case cat.ThreadedComposed():
                return ConstructedThreadedComposed(target)
            case cat.Composed():
                return ConstructedComposed(target)
            case cat.Block():
                return ConstructedBlock(target)
            case cat.Broadcasted():
                return ConstructedModule.construct_broadcasted(target)
            case Scan():
                return ConstructedScan(target)
            case Slice():
                return ConstructedSlice(target)
            case Reindex():
                return ConstructedReindex(target)
            case Scatter():
                return ConstructedScatter(target)
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
                     semantic: bool = False):
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
    def __init__(self, target: cat.Broadcasted):
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
    def __init__(self, target: Br.Block[B, A]):
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
    def __init__(self, target: Br.ProductOfMorphisms[B, A]):
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
    def __init__(self, target: Br.Composed[B, A]):
        super().__init__(target)
        fs = map(ConstructedModule.construct, target.content)
        self.chain = nn.Sequential(*fs)

    def forward(self, *xs):
        for module in self.chain:
            xs = to_tuple(module(*xs))
        return xs
    
class ConstructedThreadedComposed(ConstructedModule):
    def __init__(self, target: cat.ThreadedComposed):
        super().__init__(target)
        self.routing = target.routing
        self.chain = nn.ModuleList(
            [ConstructedModule.construct(m) for m in target.content]
        )

    def forward(self, *xs: torch.Tensor):
        live = list(xs)
        last: tuple[torch.Tensor, ...] = xs
        for module, route in zip(self.chain, self.routing):
            last = to_tuple(module(*(live[i] for i in route)))
            live.extend(last)
        return last


class ConstructedRearrangement(ConstructedModule):
    def __init__(self, target: cat.Rearrangement):
        super().__init__(target)

    def forward(self, *xs: torch.Tensor):
        return self.target.apply(xs)


class ConstructedSlice(ConstructedModule):
    """Select fixed integer coordinates from the (single) input tensor.

    Realises a `Slice` morphism (a constant index read, e.g. X[..., 3]) as
    repeated torch.select.  Positions are selected high-to-low so earlier axis
    positions stay valid as later axes are dropped.
    """
    def __init__(self, target: Slice):
        super().__init__(target)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        x = xs[0]
        for pos, idx in sorted(
            zip(self.target.positions, self.target.indices), reverse=True
        ):
            x = x.select(pos, idx)
        return x


class ConstructedReindex(ConstructedModule):
    """Affine gather: build per-input-axis index grids from the out-axis ranges and
    advanced-index the input.  Realises a `Reindex` (shift/stride/dilation/conv/
    diagonal).  Out-of-range affine coordinates raise (P2 requires in-range)."""
    def __init__(self, target: Reindex):
        super().__init__(target)
        sizes = []
        for ax in target.out_axes:
            if not isinstance(ax._size, nm.Integer):
                raise ValueError(
                    f"affine gather needs a concrete size for fan-out axis {ax}")
            sizes.append(ax._size._value)
        self.out_sizes = tuple(sizes)
        self.rows = target.rows

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        x = xs[0]
        grids = torch.meshgrid(
            *[torch.arange(s, device=x.device) for s in self.out_sizes],
            indexing='ij',
        )
        index = []
        for const, terms in self.rows:
            idx = torch.full(self.out_sizes, int(const), dtype=torch.long, device=x.device)
            for k, coeff in terms:
                idx = idx + int(coeff) * grids[k]
            index.append(idx)
        return x[tuple(index)]


class ConstructedScatter(ConstructedModule):
    """Affine scatter: write the value into a zero tensor at affine output positions
    (`out[const+Σcₖaₖ] = value`).  Realises a `Scatter` (upsampling/padding/strided
    placement).  Injective + zero-fill (P2)."""
    def __init__(self, target: Scatter):
        super().__init__(target)
        self.in_sizes = tuple(
            ax._size._value for ax in target.in_axes  # type: ignore[attr-defined]
        )
        self.out_shape = tuple(int(s) for s in target.out_shape)
        self.rows = target.rows
        self.fill = float(getattr(target, 'fill', 0.0))
        self.reduce = getattr(target, 'reduce', None)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        value = xs[0]
        grids = torch.meshgrid(
            *[torch.arange(s, device=value.device) for s in self.in_sizes],
            indexing='ij',
        )
        index = []
        for const, terms in self.rows:
            idx = torch.full(self.in_sizes, int(const), dtype=torch.long, device=value.device)
            for k, coeff in terms:
                idx = idx + int(coeff) * grids[k]
            index.append(idx)
        out = torch.full(
            self.out_shape, self.fill, dtype=value.dtype, device=value.device)
        if self.reduce == 'sum':
            # Overlapping writes accumulate (declared combine).  index_put_ with
            # accumulate=True is the only path that sums duplicate destinations.
            out.index_put_(tuple(index), value, accumulate=True)
        else:
            out[tuple(index)] = value
        return out

##################
## BROADCASTING ##
##################

def broadcast_func(
        target: cat.Broadcasted, 
        func: Callable,
        broadcast_info: TorchFunctionInfo = TorchFunctionInfo()):
    
    assert tutil.is_mappable_broadcast(target)
    # For now, we just use Einops
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
                    x.reshape(bcast.unsqueeze_guide(
                        degree_size,
                        tutil.get_mapping(eta),
                        x.shape
                    ))
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

def generate_tensor_equation_signature(target: cat.Broadcasted) -> str:
    """Build an einops contraction string from a TensorEquation-operator Broadcasted.

    Reads contracted axes from weave _shape directly: concrete RawAxis slots are
    contracted, WeaveMode.TILED slots are retained (degree). A UID→tag dict
    ensures axes shared across multiple input weaves receive the same tag, which
    is what causes einops to contract over them.

    Sized-Iverson factors are excluded from input_weaves (they are auto-buffered
    and do not appear in the morphism domain).  Their segments are appended after
    the domain segments here, matching the [caller-inputs, buffer-inputs] order
    that ConstructedTensorEquation.forward() uses when calling einops.einsum.
    """
    from data_structure.TensorLogic import TensorEquation, _iverson_is_materializable
    assert tutil.is_mappable_broadcast(target)
    degree_tags = tuple(f'y{i}' for i, _ in enumerate(target.degree()))
    contracted_tag: dict = {}
    tag_counter = 0
    for weave in target.input_weaves:
        for slot in weave._shape:
            if not isinstance(slot, cat.WeaveMode):
                uid = slot.uid
                if uid not in contracted_tag:
                    contracted_tag[uid] = f'x{tag_counter}'
                    tag_counter += 1
    input_segments = list(
        weave.imprint_axes(
            (degree_tags[i] for i in tutil.get_mapping(eta)),
            (contracted_tag[slot.uid] for slot in weave._shape
             if not isinstance(slot, cat.WeaveMode)),
        )
        for weave, eta in zip(target.input_weaves, target.reindexings)
    )
    # Append segments for sized-Iverson buffer factors excluded from input_weaves.
    # Their ordering here must match the buffer_inputs ordering in forward().
    #
    # An axis repeated within one Iverson predicate emits the SAME tag twice
    # here, so einops collapses the buffer's diagonal during the contraction.
    # The buffer itself is materialised in the expanded (one-dim-per-leaf) form;
    # see materialise_iverson()'s "PERFORMANCE TRADEOFF" note for why the collapse
    # lives here (uniformity with caller tensors) and how to move it earlier.
    if isinstance(target.operator, TensorEquation):
        lhs_uid_to_pos = {ax.uid: i for i, ax in enumerate(target.operator.lhs_indices)}
        for factor in target.operator.rhs:
            if isinstance(factor, (IversonBinOp, IversonUnaryOp)) and _iverson_is_materializable(factor):
                segment = []
                for ax in _factor_axes(factor):
                    if ax.uid in lhs_uid_to_pos:
                        segment.append(degree_tags[lhs_uid_to_pos[ax.uid]])
                    else:
                        if ax.uid not in contracted_tag:
                            contracted_tag[ax.uid] = f'x{tag_counter}'
                            tag_counter += 1
                        segment.append(contracted_tag[ax.uid])
                input_segments.append(segment)
    input_signature = ', '.join(
        '... ' + ' '.join(segment)
        for segment in input_segments
    )
    output_signature = ' '.join(degree_tags)
    return f'{input_signature} -> ... {output_signature}'


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
    def __init__(self, target: cat.Broadcasted[B, A, ops.Einops]):
        super().__init__(target)
        self.signature = generate_einops_signature(self.target)

    def forward(self, *xs: torch.Tensor):
        return einops.einsum(*xs, self.signature) # type: ignore


class ConstructedTensorEquation[B: cat.Datatype, A: cat.Axis](
    ConstructedModule[cat.Broadcasted[B, A, cat.TensorEquation]],
    operation_key=cat.TensorEquation,
):
    """Compile a TensorEquation-operator Broadcasted to an einsum.

    Expects one tensor per RHS factor, in rhs order. Iverson factors must be
    pre-materialised by the caller as concrete float tensors; this class treats
    them identically to TensorRef factors. Iverson materialisation is a
    separate upstream step so it can be implemented independently.

    If the output weave carries Bool(), applies H(x) = (x > 0) after the
    contraction to demote the real-valued result to {0, 1}.
    """

    def __init__(self, target: cat.Broadcasted[B, A, cat.TensorEquation]):
        super().__init__(target)
        assert len(target.output_weaves) == 1
        nonlinearity = target.operator.operator
        if nonlinearity is not None and not isinstance(nonlinearity, ops.Identity):
            raise NotImplementedError(
                f'ConstructedTensorEquation does not compile nonlinearity '
                f'{nonlinearity!r}; compose it as a separate Broadcasted step.'
            )
        self.signature = generate_tensor_equation_signature(target)
        self.demote = isinstance(target.output_weaves[0].datatype, cat.Bool)

        # _factor_slots[i] is a buffer name if the factor is a fixed Iverson
        # tensor (all axes sized), or None if the caller must supply it.
        self._factor_slots: list[str | None] = []
        for i, factor in enumerate(target.operator.rhs):
            if isinstance(factor, TensorRef):
                self._factor_slots.append(None)
            else:
                try:
                    buf = materialise_iverson(factor)
                    self.register_buffer(f'_mask_{i}', buf)
                    self._factor_slots.append(f'_mask_{i}')
                except ValueError as e:
                    warnings.warn(
                        f"RHS factor {i} is an Iverson predicate with unsized "
                        f"axes — it will not be auto-materialised and must be "
                        f"passed as a caller input. ({e})",
                        stacklevel=2,
                    )
                    self._factor_slots.append(None)
        # Precomputed for torch.compile: positions in _factor_slots that must
        # be filled from the caller's *xs tuple, in order.
        self._caller_positions: list[int] = [
            i for i, s in enumerate(self._factor_slots) if s is None
        ]

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        # Callers (*xs) first, then buffer tensors — matches the segment ordering
        # in generate_tensor_equation_signature (domain factors, then buffer factors).
        buffer_inputs = [getattr(self, s) for s in self._factor_slots if s is not None]
        result = einops.einsum(*xs, *buffer_inputs, self.signature)  # type: ignore
        if self.demote:
            return (result > 0).to(result.dtype)
        return result


ConstructedModule.add_function(ops.SoftMax, torch.softmax, dim=True)
ConstructedModule.add_function(ops.AdditionOp, lambda x, y: x + y, semantic=True)
ConstructedModule.add_function(ops.Elementwise, torch.relu)
ConstructedModule.add_function(ops.ReLU, torch.relu)

def weighted_triangular_lower(x: torch.Tensor) -> torch.Tensor:
    trilled = torch.tril(x)
    return trilled / (torch.sum(trilled, dim=-1, keepdim=True) + 1e-8)

ConstructedModule.add_function(ops.WeightedTriangularLower, weighted_triangular_lower)

#############
## LEARNED ##
#############
class ConstructedLinear[
    B: cat.Datatype, A: cat.Axis
](ConstructedModule, operation_key=ops.Linear):
    def __init__(self, target: cat.Broadcasted[B, A, ops.Linear]):
        super().__init__(target)
        match target.input_weaves, target.output_weaves:
            case [
                [cat.Weave() as weave_in],
                [cat.Weave() as weave_out]
            ]:
                self.in_size: fd.Prod[int] = tuple(x.local_size()._value for x in weave_in.target().shape()) # type: ignore
                self.out_size: fd.Prod[int] = tuple(y.local_size()._value for y in weave_out.target().shape()) # type: ignore
            case _:
                assert False
        self.bias = target.operator.bias
        self.module = torch_utilities.Multilinear(
            self.in_size, self.out_size, self.bias
        )
        self.func = broadcast_func(
            target, 
            self.module.forward)

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
    
class ConstructedNorm[B: cat.Datatype, A: cat.Axis](ConstructedModule, operation_key=ops.Normalize):
    def __init__(self, target: cat.Broadcasted[B, A, ops.Normalize]):
        super().__init__(target)
        displacement = bcast.get_displacement(target)
        self._dim = displacement if displacement is not None else -1

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        x = xs[0]
        return x / x.sum(dim=self._dim, keepdim=True).clamp(min=1e-8)


class ConstructedMaskedSoftMax(
    ConstructedModule, operation_key=ops.MaskedSoftMax
):
    """Softmax with Iverson predicates applied as -inf masks before exponentiation.

    Each Iverson factor is pre-materialised as a registered buffer.  At
    forward time the buffer is permuted and unsqueezed (using the compile-time
    alignment metadata) to match the score tensor shape, then masked_fill
    sets excluded positions to -inf before torch.softmax.
    """

    def __init__(self, target: cat.Broadcasted):
        super().__init__(target)
        op = target.operator
        displacement = bcast.get_displacement(target)
        self._dim = displacement if displacement is not None else -1
        from data_structure.TensorLogic import _iverson_diagonal
        self._n_masks = len(op.iverson_factors)
        self._alignments: list[tuple[list[int], int]] = [
            (list(perm), n_broadcast)
            for perm, n_broadcast in op.mask_alignments
        ]
        for i, factor in enumerate(op.iverson_factors):
            try:
                buf = materialise_iverson(factor)
                # Collapse axes repeated within one predicate to a diagonal so
                # the buffer carries one dim per distinct UID (matching the
                # alignment perm computed by _compute_mask_alignment).
                diag_eqn, _ = _iverson_diagonal(factor)
                if diag_eqn:
                    buf = torch.einsum(diag_eqn, buf)
                self.register_buffer(f'_mask_{i}', buf)
            except ValueError as e:
                warnings.warn(
                    f"MaskedSoftMax factor {i} has unsized axes and cannot "
                    f"be auto-materialised; it will be skipped. ({e})",
                    stacklevel=2,
                )

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        score = xs[0]
        for i, (perm, n_broadcast) in enumerate(self._alignments):
            raw = getattr(self, f'_mask_{i}', None)
            if raw is None:
                continue
            if perm:
                raw = raw.permute(perm)
            for _ in range(n_broadcast):
                raw = raw.unsqueeze(0)
            score = score.masked_fill(raw == 0, float('-inf'))
        return torch.softmax(score, dim=self._dim)


class ConstructedMaskedNormalize(
    ConstructedModule, operation_key=ops.MaskedNormalize
):
    """Normalize with Iverson predicates applied as zero-masks before summing.

    Each Iverson factor is pre-materialised as a registered buffer.  At
    forward time the buffer is permuted and unsqueezed (using the compile-time
    alignment metadata) to match the input tensor shape, then positions where
    the mask is 0 are zeroed before the sum-normalization denominator is
    computed.  Masked output positions are therefore 0.
    """

    def __init__(self, target: cat.Broadcasted):
        super().__init__(target)
        op = target.operator
        displacement = bcast.get_displacement(target)
        self._dim = displacement if displacement is not None else -1
        from data_structure.TensorLogic import _iverson_diagonal
        self._n_masks = len(op.iverson_factors)
        self._alignments: list[tuple[list[int], int]] = [
            (list(perm), n_broadcast)
            for perm, n_broadcast in op.mask_alignments
        ]
        for i, factor in enumerate(op.iverson_factors):
            try:
                buf = materialise_iverson(factor)
                # Collapse axes repeated within one predicate to a diagonal so
                # the buffer carries one dim per distinct UID (matching the
                # alignment perm computed by _compute_mask_alignment).
                diag_eqn, _ = _iverson_diagonal(factor)
                if diag_eqn:
                    buf = torch.einsum(diag_eqn, buf)
                self.register_buffer(f'_mask_{i}', buf)
            except ValueError as e:
                warnings.warn(
                    f"MaskedNormalize factor {i} has unsized axes and cannot "
                    f"be auto-materialised; it will be skipped. ({e})",
                    stacklevel=2,
                )

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        x = xs[0]
        for i, (perm, n_broadcast) in enumerate(self._alignments):
            raw = getattr(self, f'_mask_{i}', None)
            if raw is None:
                continue
            if perm:
                raw = raw.permute(perm)
            for _ in range(n_broadcast):
                raw = raw.unsqueeze(0)
            x = x * raw.float()
        return x / x.sum(dim=self._dim, keepdim=True).clamp(min=1e-8)


##############
##   SCAN   ##
##############

class ConstructedScan(ConstructedModule):
    """Iterative scan: sequential loop (always) with optional associative_scan fast path.

    Uncoupled (n_states == 1):
      forward(*xs) expects:
        xs[:n_base]  — inputs to the base-case morphism (no l dimension)
        xs[n_base:]  — per-step inputs, each with N as the LAST dimension
      Returns a single Tensor of shape (*state, N+1).

    Coupled (n_states > 1, Jacobi-style):
      forward(*xs) expects:
        xs[:n_bases[0]]                — base inputs for state 0
        xs[n_bases[0]:n_bases[0]+n_bases[1]]  — base inputs for state 1
        ...followed by per-step inputs for state 0, then state 1, etc.
      Each step module receives (*all_current_states, *own_sliced_step_xs).
      Returns tuple[Tensor, ...] with one (*state_k, N+1) tensor per state,
      in the same canonical (sorted-by-name) order used during compilation.
      ConstructedComposed handles this tuple automatically via to_tuple().
    """

    def __init__(self, target: Scan):
        super().__init__(target)

        if not isinstance(target.N, nm.Integer):
            raise ValueError(
                "Iterative axis has no concrete size; "
                "use real_axis('name', N) to supply a step count."
            )
        self.N: int = target.N._value  # type: ignore[attr-defined]
        self.n_states: int = target.n_states

        if target.n_states == 1:
            self._init_uncoupled(target)
        else:
            self._init_coupled(target)

    def _init_uncoupled(self, target: Scan) -> None:
        self.step_module = ConstructedModule.construct(target.step)
        self.base_module = ConstructedModule.construct(target.base)

        # Use _caller_positions so pre-materialised Iverson buffers are excluded.
        if hasattr(self.base_module, '_caller_positions'):
            self.n_base: int = len(self.base_module._caller_positions)
        else:
            self.n_base = len(target.base.dom())

        affine = target.affine
        self._has_affine = affine is not None
        if affine is not None:
            self.A_module = (
                ConstructedModule.construct(affine.A_morphism)
                if affine.A_morphism is not None else None
            )
            self.b_module = (
                ConstructedModule.construct(affine.b_morphism)
                if affine.b_morphism is not None else None
            )
            self.a_positions: tuple[int, ...] = affine.a_positions
            self.b_positions: tuple[int, ...] = affine.b_positions
            self._state_matrix = len(affine.state_in_axes) > 0

        # Position of l within each per-step input as declared (e.g. W[l,..]->0);
        # forward() moves it to last before slicing.  Empty => already l-last.
        self.step_x_l_positions: tuple[int, ...] = target.step_x_l_positions

        self._loop = torch._dynamo.disable(self._run_loop)

    def _init_coupled(self, target: Scan) -> None:
        self._has_affine = False  # no affine fast path for coupled groups
        self.step_modules = nn.ModuleList(
            [ConstructedModule.construct(s) for s in target.step]
        )
        self.base_modules = nn.ModuleList(
            [ConstructedModule.construct(b) for b in target.base]
        )
        # Use _caller_positions to correctly exclude Iverson buffers.
        self.n_bases: list[int] = [
            len(m._caller_positions) if hasattr(m, '_caller_positions')
            else len(target.base[k].dom())
            for k, m in enumerate(self.base_modules)
        ]
        # Which states (by canonical index) each step morphism reads, in domain order.
        # Falls back to "all states in order" if the Scan predates this field.
        n = target.n_states
        self.step_state_deps: list[tuple[int, ...]] = (
            list(target.step_state_deps)
            if target.step_state_deps
            else [tuple(range(n)) for _ in range(n)]
        )
        # Per-step inputs count: total caller inputs minus the state inputs for that morphism.
        self.n_step_xs: list[int] = [
            (len(m._caller_positions) if hasattr(m, '_caller_positions')
             else len(target.step[k].dom())) - len(self.step_state_deps[k])
            for k, m in enumerate(self.step_modules)
        ]
        self._loop = torch._dynamo.disable(self._run_loop_coupled)

    # ------------------------------------------------------------------
    # Sequential path (always correct)
    # ------------------------------------------------------------------

    def _run_loop(
        self,
        H: torch.Tensor,
        step_xs: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        outputs = [H]
        for l_idx in range(self.N):
            # l is the LAST dimension of each step tensor.
            sliced = tuple(x[..., l_idx] for x in step_xs)
            H = to_tuple(self.step_module(H, *sliced))[0]
            outputs.append(H)
        # Stack along last dimension to produce shape (*state, N+1).
        return torch.stack(outputs, dim=-1)

    # ------------------------------------------------------------------
    # Coupled sequential path (Jacobi-style: all states updated simultaneously)
    # ------------------------------------------------------------------

    def _run_loop_coupled(
        self,
        states: tuple[torch.Tensor, ...],
        step_xs_per_state: tuple[tuple[torch.Tensor, ...], ...],
    ) -> tuple[torch.Tensor, ...]:
        histories: list[list[torch.Tensor]] = [[s] for s in states]
        for l_idx in range(self.N):
            sliced = tuple(
                tuple(x[..., l_idx] for x in xs)
                for xs in step_xs_per_state
            )
            # Jacobi: each module sees OLD states. Only the states it actually
            # depends on are passed (in the order recorded in step_state_deps).
            new_states = tuple(
                to_tuple(mod(*[states[j] for j in self.step_state_deps[k]], *sliced[k]))[0]
                for k, mod in enumerate(self.step_modules)
            )
            states = new_states
            for k, s in enumerate(states):
                histories[k].append(s)
        return tuple(torch.stack(hist, dim=-1) for hist in histories)

    # ------------------------------------------------------------------
    # Associative scan fast path
    # ------------------------------------------------------------------

    def _batch_module(
        self,
        module: ConstructedModule,
        positions: tuple[int, ...],
        step_xs: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Apply module to each l-step via vmap, returning shape (N, *out)."""
        inputs_full = tuple(step_xs[j] for j in positions)  # each (*feat, N)
        inputs_lf = tuple(x.movedim(-1, 0) for x in inputs_full)   # (N, *feat)
        return torch.vmap(module)(*inputs_lf)                        # (N, *out)

    def _assoc_scan_forward(
        self,
        H0: torch.Tensor,
        step_xs: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        # Compute A_l and b_l for all N steps via batched vmap; shape (N, *out).
        if self.A_module is not None:
            A_all = self._batch_module(self.A_module, self.a_positions, step_xs)
        else:
            A_all = torch.ones(self.N, *H0.shape, device=H0.device, dtype=H0.dtype)

        if self.b_module is not None:
            b_all = self._batch_module(self.b_module, self.b_positions, step_xs)
        else:
            b_all = torch.zeros(self.N, *H0.shape, device=H0.device, dtype=H0.dtype)

        is_matrix = self._state_matrix

        def combine(s1: tuple, s2: tuple) -> tuple:
            A1, b1 = s1
            A2, b2 = s2
            if is_matrix:
                A_new = torch.einsum('ij,jk->ik', A2, A1)
                b_new = torch.einsum('ij,j->i', A2, b1) + b2
            else:
                A_new = A2 * A1
                b_new = A2 * b1 + b2
            return A_new, b_new

        # Scan along dim=0 (l-first layout); both A_all and b_all start with N.
        A_prefix, b_prefix = _associative_scan(
            combine, (A_all, b_all),
            dim=0,
            combine_mode='generic',
        )

        # Apply cumulative affine maps to H0; A_prefix/b_prefix have shape (N, ...).
        if is_matrix:
            H_seq = torch.einsum('nij,j->ni', A_prefix, H0) + b_prefix  # (N, out)
        else:
            H_seq = A_prefix * H0 + b_prefix  # (N, *state)

        # Prepend H0 as the step-0 state, then convert to l-last (*state, N+1).
        H_lf = torch.cat([H0.unsqueeze(0), H_seq], dim=0)  # (N+1, *state)
        return H_lf.movedim(0, -1)                          # (*state, N+1)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def forward(self, *xs: torch.Tensor):
        if self.n_states == 1:
            base_xs = xs[: self.n_base]
            step_xs = xs[self.n_base :]
            # Move l to the last axis of each per-step input (l-last contract).
            if self.step_x_l_positions:
                step_xs = tuple(
                    x.movedim(p, -1) if p not in (-1, x.ndim - 1) else x
                    for x, p in zip(step_xs, self.step_x_l_positions)
                )
            H0 = to_tuple(self.base_module(*base_xs))[0]
            if self._has_affine:
                return self._assoc_scan_forward(H0, step_xs)
            return self._loop(H0, step_xs)

        # Coupled path: split base inputs then per-step inputs.
        pos = 0
        states: list[torch.Tensor] = []
        for k, nb in enumerate(self.n_bases):
            s = to_tuple(self.base_modules[k](*xs[pos:pos + nb]))[0]
            states.append(s)
            pos += nb
        step_xs_per_state: list[tuple[torch.Tensor, ...]] = []
        for ns in self.n_step_xs:
            step_xs_per_state.append(xs[pos:pos + ns])
            pos += ns
        return self._loop(tuple(states), tuple(step_xs_per_state))
