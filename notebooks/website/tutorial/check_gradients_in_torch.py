# Claude Opus 5.5 (1M context), reasoning effort 40.
'''Running a derived training step in PyTorch and comparing it with `torch.autograd`.

`compile_training_step` writes each pass of a training step as a pure function, with
every grab a further operand and every drop a further result, binds every size to an
integer and compiles both passes with `torch_compile`. The blocks are removed first,
because `para.algebra.detape` reaches the tape operations of the top level alone, and a
block whose repetition is one groups operations without changing what they compute.
`compare_with_autograd` draws random inputs and weights, runs the forward pass, carries
every saved value to the backward pass by the name of its slot, and runs the backward
pass on a random cotangent. A reference written directly in PyTorch runs on the same
inputs and weights, its value is compared with the result of the forward pass, and the
gradients `torch.autograd.grad` returns for it are compared with the results of the
backward pass. `para/validate_backward.py` makes the same comparison for its fixtures.

`torch_compile` has no rule for a view through a stride morphism, for the transpose of
such a view, or for a contraction over an axis some of whose positions hold no value.
The causal mask needs all three, and a softmax over such an axis as well, and the rules
registered here state them. A view reads zero at a position outside its operand. The
transpose adds every entry of its cotangent into the position the view read it from, and
drops an entry read from outside. A contraction multiplies every operand carrying an
`AffineGuards.AffineSparseAxis` by the indicator of its live positions, so a position
that holds the universal unit adds nothing to a sum, and a softmax gives such a position
no weight. Both are the readings the softmax and the weighted sum give the unit.
`forward_agrees_with` compares a model with no training step with a reference in the
same way.
'''
from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch

import advanced_axis_dynamics.algebra.mark_sparse_domains as mark_sparse_domains
import advanced_axis_dynamics.algebra.move_reads_backwards as move_reads_backwards
import advanced_axis_dynamics.data_structure.AffineGuards as AffineGuards
import algebra.discovering_broadcasts as discovering_broadcasts
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import para.algebra.detape as detape
import para.data_structure.transpose as transpose
import para.processing.backprop as backprop
import para.processing.show_grabbed_parameters as show_grabbed_parameters
import term_utilities.generate_config as generate_config
import term_utilities.term_utilities as tutil
import torch_compile.torch_compile as torch_compile

DTYPE = torch.float64
TOLERANCE = 1e-9

type Tensors = Mapping[str, torch.Tensor]
type Reference = Callable[[Sequence[torch.Tensor], Tensors], torch.Tensor]


class GuideNotInShape(ValueError):
    '''A sparse axis whose guide the array carrying it does not carry exactly once.'''


def integer_of(size: nm.Numeric) -> int:
    return nm.evaluate_integer(size, {})


def shape_of(array: cat.Array) -> tuple[int, ...]:
    return tuple(integer_of(axis.local_size()) for axis in array.shape())


def with_sizes_bound[T: cat.BroadcastedCategory](expression: T,
                                                 sizes: Mapping[str, int]) -> T:
    '''`expression` with every size symbol named in `sizes` replaced by its integer.'''
    config = generate_config.NumericConfig.template(expression)
    config.assign_values(**sizes)
    return config.apply_context(expression)


def position_of_guide(shape: tuple[cat.Axis, ...], guide: cat.Axis) -> int:
    positions = tuple(position for position, axis in enumerate(shape) if axis == guide)
    if len(positions) != 1:
        raise GuideNotInShape(
            f'the guide {guide.uid._name.to_bodies()} stands at {len(positions)} '
            f'positions of a shape of {len(shape)} axes')
    return positions[0]


def live_indicator(array: cat.Array) -> torch.Tensor | None:
    '''One at every index of `array` where each of its sparse axes holds a value and
    zero elsewhere, or `None` where it carries no sparse axis.'''
    shape = tuple(array.shape())
    sparse = tuple((position, axis) for position, axis in enumerate(shape)
                   if isinstance(axis, AffineGuards.AffineSparseAxis))
    if not sparse:
        return None
    sizes = tuple(integer_of(axis.local_size()) for axis in shape)
    indicator = torch.ones(sizes, dtype=DTYPE)
    for position, axis in sparse:
        guides = tuple(position_of_guide(shape, guide) for guide in axis.guides)
        live = {guide_index: set(mark_sparse_domains.live_positions(axis, guide_index, {}))
                for guide_index in itertools.product(*(range(sizes[g]) for g in guides))}
        for index in itertools.product(*(range(size) for size in sizes)):
            if index[position] not in live[tuple(index[g] for g in guides)]:
                indicator[index] = 0
    return indicator


class ConstructedContractionOverLivePositions(
        torch_compile.ConstructedEinops, operation_key=ops.Einops):
    '''A contraction whose operands are first multiplied by the indicators of their live
    positions.'''
    def __init__(self, target: cat.Broadcasted) -> None:
        super().__init__(target)
        self.indicators = tuple(live_indicator(array) for array in target.dom())

    def forward(self, *operands: torch.Tensor) -> torch.Tensor:
        return super().forward(*(
            operand if indicator is None else operand * indicator
            for operand, indicator in zip(operands, self.indicators)))


class ConstructedSoftMaxOverLivePositions(torch_compile.ConstructedModule,
                                          operation_key=ops.SoftMax):
    '''A softmax whose operand holds minus infinity at every position that holds no
    value, so the position takes no weight.'''
    def __init__(self, target: cat.Broadcasted) -> None:
        super().__init__(target)
        self.indicator = live_indicator(target.dom()[0])
        self.softmax = torch_compile.Lambda(target)

    def forward(self, operand: torch.Tensor) -> torch.Tensor:
        if self.indicator is None:
            return self.softmax(operand)
        return self.softmax(operand.masked_fill(self.indicator == 0, -math.inf))


def positions_read(read: cat.StrideMorphism) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    '''The position every row of `read` names at every index of its domain, and whether
    that index names a position inside every axis the rows name.'''
    grids = torch.meshgrid(*(torch.arange(integer_of(axis.local_size()))
                             for axis in read._dom), indexing='ij')
    positions = tuple(
        sum((integer_of(stride) * grid for stride, grid in zip(strides, grids)),
            torch.full_like(grids[0], integer_of(shift)))
        for _, strides, shift in read._cod_stride_shift)
    inside = torch.ones_like(grids[0], dtype=torch.bool)
    for position, (axis, _, _) in zip(positions, read._cod_stride_shift):
        inside &= (position >= 0) & (position < integer_of(axis.local_size()))
    return positions, inside


class ConstructedStridedRead(torch_compile.ConstructedModule, operation_key=ops.View):
    '''A view through a rearrangement as `torch_compile` compiles it, and a view through
    a stride morphism as the entries it reads, with zero outside the operand.'''
    def __init__(self, target: cat.Broadcasted) -> None:
        super().__init__(target)
        self.through_rearrangement = (
            torch_compile.Lambda(target)
            if tutil.is_mappable(target.reindexings[0]) else None)
        self.read = move_reads_backwards.as_stride_morphism(target.reindexings[0])
        self.operand_shape = shape_of(target.dom()[0])

    def forward(self, operand: torch.Tensor) -> torch.Tensor:
        if self.through_rearrangement is not None:
            return self.through_rearrangement(operand)
        operand = operand.expand(*self.operand_shape)
        positions, inside = positions_read(self.read)
        clamped = tuple(position.clamp(0, length - 1)
                        for position, length in zip(positions, operand.shape))
        return operand[clamped] * inside


class ConstructedTransposeOfRead(torch_compile.ConstructedModule,
                                 operation_key=transpose.ReindexTranspose):
    '''The transpose of a strided view: every entry of the cotangent added into the
    position the view read it from, and an entry read from outside the operand
    dropped.'''
    def __init__(self, target: cat.Broadcasted) -> None:
        super().__init__(target)
        self.read = move_reads_backwards.as_stride_morphism(target.operator.reindexing)
        self.result_shape = shape_of(target.cod()[0])

    def forward(self, cotangent: torch.Tensor) -> torch.Tensor:
        positions, inside = positions_read(self.read)
        result = torch.zeros(self.result_shape, dtype=cotangent.dtype)
        kept = tuple(position[inside] for position in positions)
        return result.index_put(kept, cotangent.expand(inside.shape)[inside],
                                accumulate=True)


def run(module: torch.nn.Module, operands: Sequence[torch.Tensor],
        ) -> tuple[torch.Tensor, ...]:
    produced = module(*operands)
    return produced if isinstance(produced, tuple) else (produced,)


@dataclass(frozen=True)
class CompiledPass:
    '''A pass as a module whose operands are the operands of the pass followed by the
    slots named in `grabbed`, and whose results are the results of the pass followed by
    the slots named in `dropped`.'''
    module: torch.nn.Module
    operand_shapes: tuple[tuple[int, ...], ...]
    grabbed: tuple[str, ...]
    dropped: tuple[str, ...]

    def own_operand_shapes(self) -> tuple[tuple[int, ...], ...]:
        return self.operand_shapes[:len(self.operand_shapes) - len(self.grabbed)]

    def grabbed_shapes(self) -> dict[str, tuple[int, ...]]:
        return dict(zip(
            self.grabbed,
            self.operand_shapes[len(self.operand_shapes) - len(self.grabbed):]))


def compile_pass(expression: cat.BroadcastedCategory,
                 sizes: Mapping[str, int]) -> CompiledPass:
    pure, tape = detape.detape(discovering_broadcasts.remove_grouping_blocks(
        with_sizes_bound(expression, sizes)))
    return CompiledPass(
        module=torch_compile.ConstructedModule.construct(pure),
        operand_shapes=tuple(shape_of(array) for array in pure.dom()),
        grabbed=tuple(name for name, is_grab in tape if is_grab),
        dropped=tuple(name for name, is_grab in tape if not is_grab))


@dataclass(frozen=True)
class GradientComparison:
    '''Whether the forward pass returns the value of the reference, and, for every
    input by its position and every weight by the name of its slot, whether the
    backward pass returns the gradient `torch.autograd` gives the reference.'''
    forward_value_agrees: bool
    input_gradients_agree: tuple[bool, ...]
    weight_gradients_agree: dict[str, bool]

    def all_agree(self) -> bool:
        return (self.forward_value_agrees and all(self.input_gradients_agree)
                and all(self.weight_gradients_agree.values()))


def gradient_slot_name(weight_slot_name: str) -> str:
    '''The name `backprop.gradient_slot` gives the slot of the gradient of a weight.'''
    return 'd' + weight_slot_name


def compare_with_autograd(
    taped: backprop.Taped, reference: Reference, sizes: Mapping[str, int],
    generator: torch.Generator,
) -> GradientComparison:
    '''`taped` run on random inputs, weights and cotangent, against `reference` and the
    gradients `torch.autograd.grad` gives it.'''
    forward = compile_pass(taped.forward, sizes)
    backward = compile_pass(taped.backward, sizes)
    inputs = [torch.randn(*shape, generator=generator, dtype=DTYPE, requires_grad=True)
              for shape in forward.own_operand_shapes()]
    weights = {name: torch.randn(*shape, generator=generator, dtype=DTYPE,
                                 requires_grad=True)
               for name, shape in forward.grabbed_shapes().items()}
    results = run(forward.module, [*(value.detach() for value in inputs),
                                   *(weights[name].detach() for name in forward.grabbed)])
    output, saved = results[0], dict(zip(forward.dropped, results[1:]))

    expected = reference(inputs, weights)
    cotangent = torch.randn(*expected.shape, generator=generator, dtype=DTYPE)
    gradients = torch.autograd.grad(
        expected, [*inputs, *weights.values()], grad_outputs=cotangent)
    expected_input_gradients = gradients[:len(inputs)]
    expected_weight_gradients = dict(zip(weights, gradients[len(inputs):]))

    tape = {**saved, **{name: value.detach() for name, value in weights.items()}}
    returned = run(backward.module, [cotangent, *(tape[name] for name in backward.grabbed)])
    returned_input_gradients = returned[:len(returned) - len(backward.dropped)]
    returned_weight_gradients = dict(zip(
        backward.dropped, returned[len(returned) - len(backward.dropped):]))
    return GradientComparison(
        forward_value_agrees=torch.allclose(output, expected.detach(), atol=TOLERANCE),
        input_gradients_agree=tuple(
            torch.allclose(returned_gradient, expected_gradient, atol=TOLERANCE)
            for returned_gradient, expected_gradient
            in zip(returned_input_gradients, expected_input_gradients, strict=True)),
        weight_gradients_agree={
            name: torch.allclose(
                returned_weight_gradients[gradient_slot_name(name)],
                expected_weight_gradients[name], atol=TOLERANCE)
            for name in weights})


def forward_agrees_with(expression: cat.BroadcastedCategory, reference: Reference,
                        sizes: Mapping[str, int], generator: torch.Generator) -> bool:
    '''Whether `expression`, with its weights grabbed and compiled, returns the result
    of `reference` on random inputs and weights.'''
    compiled = compile_pass(show_grabbed_parameters.grab_parameters(expression), sizes)
    inputs = [torch.randn(*shape, generator=generator, dtype=DTYPE)
              for shape in compiled.own_operand_shapes()]
    weights = {name: torch.randn(*shape, generator=generator, dtype=DTYPE)
               for name, shape in compiled.grabbed_shapes().items()}
    result = run(compiled.module, [*inputs, *(weights[name] for name in compiled.grabbed)])
    return torch.allclose(result[0], reference(inputs, weights), atol=TOLERANCE)
