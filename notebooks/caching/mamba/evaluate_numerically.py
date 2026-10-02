# Claude Opus 5.5 (1M context), effort 40.
'''Evaluating the Mamba layer and its pass over the new tokens on numbers, at small
sizes.

`torch_compile` cannot evaluate either expression. It maps every `ops.View` to the
identity, so a read at a shift or at the loop counter is not performed, it compiles a
repeated block only when the repetition is an integer and never binds a counter, and it
has no rule for a tape seed, an `ops.ConstantOp`, an `aops.CovariantView` or a
`Caching`. This module evaluates the operators the two expressions hold and nothing
else, in `torch.float64`.

A `cat.Broadcasted` is evaluated at every index of its degree at once. Each operand is
read through its reindexing, written as one `sc.StrideMorphism`, at the positions the
rows give for every index of the degree, and a position outside its axis reads the
universal unit. The unit is read as zero, which is the unit of every sum this layer
takes over a guarded axis, the convolution's sum over the taps. The operator then acts
on the targets.

A repeated block that holds a `Para.StreamGrab` is a loop over the tape. Each iteration
binds the counter the block's tag names to the iteration, reads every domain wire, and
reads each loop variable at the value the slot held when the iteration started. The
values its `Para.StreamDrop`s write are the slot's values from the next iteration on,
and the loop's result is the result of its last iteration.

A cache persists between passes in `CachesBetweenPasses`. A `Caching` and a
`Para.CacheGrab` load the last entries appended to their cache, as many as the kept
axis holds, with zeros before the first entry. A `Caching` and a `Para.CacheDrop`
append their entries once the pass is over, so no read in a pass sees an entry the same
pass appends. The token axis of an entry stands first in a `Para.CacheGrab` and a
`Para.CacheDrop`.
'''
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

import advanced_axis_dynamics.algebra.move_reads_backwards as move_reads_backwards
import advanced_axis_dynamics.data_structure.Operators as aops
import caching.data_structure.Caching as Caching
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd
import para.data_structure.Para as Para
import term_utilities.generate_config as generate_config
import torch_compile.torch_compile as torch_compile

type Tensors = tuple[torch.Tensor, ...]
DTYPE = torch.float64


class OperatorNotEvaluated(NotImplementedError):
    '''An operator this module has no evaluation for.'''


class WeightNotGiven(KeyError):
    '''A `Linear` whose name has no entry among the weights handed to the
    evaluation.'''


@dataclass(frozen=True)
class LinearValues:
    '''The weight of a `Linear`, over the axes of its input targets followed by the axes
    of its output target, and its bias over the output target where it has one.'''
    weight: torch.Tensor
    bias: torch.Tensor | None = None


@dataclass
class CachesBetweenPasses:
    '''The entries appended to every cache, keyed by the name of the cache, each with
    its token axis first.'''
    entries: dict[str, torch.Tensor] = field(default_factory=dict)
    appended_this_pass: dict[str, torch.Tensor] = field(default_factory=dict)

    def last_entries(self, name: str, count: int, entry_shape: tuple[int, ...]
                     ) -> torch.Tensor:
        '''The last `count` entries appended to the cache `name`, with zeros before the
        first entry.'''
        held = self.entries.get(name, torch.zeros((0, *entry_shape), dtype=DTYPE))
        missing = max(count - held.shape[0], 0)
        padded = torch.cat([torch.zeros((missing, *entry_shape), dtype=DTYPE), held])
        return padded[padded.shape[0] - count:]

    def entries_of(self, name: fd.DynamicName) -> torch.Tensor:
        '''Every entry appended to the cache named `name`, oldest first.'''
        return self.entries[name_key(name)]

    def append(self, name: str, new_entries: torch.Tensor) -> None:
        if name in self.appended_this_pass:
            raise ValueError(f'the cache {name} is appended to twice in one pass')
        self.appended_this_pass[name] = new_entries

    def end_the_pass(self) -> None:
        for name, new_entries in self.appended_this_pass.items():
            held = self.entries.get(name)
            self.entries[name] = (new_entries if held is None
                                  else torch.cat([held, new_entries]))
        self.appended_this_pass = {}


def name_key(name: fd.DynamicName | str) -> str:
    '''The text a weight is looked up by, which is the same for the name a template was
    given and for the name the operator carries.'''
    if isinstance(name, str):
        name = fd.DynamicName.from_str(name)
    return name.to_bodies()


def sizes_by_symbol(term: fd.GeneralTerm, sizes: Mapping[str, int]
                    ) -> dict[nm.Numeric, int]:
    '''Every free size of `term` whose name is a key of `sizes`, bound to the value
    there.'''
    return {symbol: sizes[symbol.uid._name.to_bodies()]
            for symbol in generate_config.NumericConfig.template(term).terms
            if symbol.uid._name is not None
            and symbol.uid._name.to_bodies() in sizes}


@dataclass
class Evaluation:
    '''One pass of an expression evaluated on numbers.'''
    weights: Mapping[str, LinearValues]
    bindings: dict[nm.Numeric, int]
    caches: CachesBetweenPasses
    tape: dict[fd.UID, torch.Tensor] = field(default_factory=dict)
    written_for_the_next_iteration: dict[fd.UID, torch.Tensor] = field(default_factory=dict)

    def integer(self, numeric: nm.Numeric) -> int:
        return nm.evaluate_integer(numeric, self.bindings)

    def size(self, axis: cat.Axis) -> int:
        return self.integer(axis.local_size())

    def shape(self, array: cat.Array) -> tuple[int, ...]:
        return tuple(self.size(axis) for axis in array.shape())

    def morphism(self, target: cat.Morphism, inputs: Tensors) -> Tensors:
        match target:
            case cat.Composed(content=parts):
                values = inputs
                for part in parts:
                    values = self.morphism(part, values)
                return values
            case cat.ProductOfMorphisms(content=parts):
                results: list[torch.Tensor] = []
                start = 0
                for part in parts:
                    wires_read = len(part.dom())
                    results.extend(self.morphism(part, inputs[start:start + wires_read]))
                    start += wires_read
                return tuple(results)
            case cat.Rearrangement():
                return target.apply(inputs)
            case cat.Block():
                return self.block(target, inputs)
            case Para.StreamGrab() | Para.Grab() if not isinstance(target, Para.CacheGrab):
                return (self.tape[target.tape.uid],)
            case Para.StreamDrop():
                self.written_for_the_next_iteration[target.tape.uid] = inputs[0]
                return ()
            case Para.CacheGrab():
                return (self.cache_grab(target),)
            case Para.CacheDrop():
                self.caches.append(name_key(target.tape.uid._name), inputs[0])
                return ()
            case Para.Drop():
                self.tape[target.tape.uid] = inputs[0]
                return ()
            case cat.Broadcasted():
                return self.broadcasted(target, inputs)
        raise OperatorNotEvaluated(f'no evaluation for {type(target).__name__}')

    def cache_grab(self, target: Para.CacheGrab) -> torch.Tensor:
        count, *entry_shape = self.shape(target.size)
        return self.caches.last_entries(
            name_key(target.tape.uid._name), count, tuple(entry_shape))

    def block(self, target: cat.Block, inputs: Tensors) -> Tensors:
        repetition = target.block_tag.repetition
        if repetition == nm.Integer(1):
            return self.morphism(target.body, inputs)
        count = self.integer(repetition)
        counter = nm.FreeNumeric.named(target.block_tag.uid._name)
        taped = any(True for _ in _stream_grabs(target.body))
        values = inputs
        for iteration in range(count):
            self.bindings[counter] = iteration
            values = self.morphism(target.body, inputs if taped else values)
            self.tape.update(self.written_for_the_next_iteration)
            self.written_for_the_next_iteration = {}
        del self.bindings[counter]
        return values

    def read_through(self, operand: torch.Tensor, weave: cat.Weave,
                     reindexing: cat.Morphism, grids: Tensors,
                     degree_sizes: tuple[int, ...]) -> torch.Tensor:
        '''`operand` read at every index of the degree through `reindexing`, with the
        degree first and the target after it.'''
        tiled = [position for position, entry in enumerate(weave._shape)
                 if isinstance(entry, cat.WeaveMode)]
        targets = [position for position, entry in enumerate(weave._shape)
                   if not isinstance(entry, cat.WeaveMode)]
        moved = operand.permute(*tiled, *targets)
        if not tiled:
            return moved.expand(*degree_sizes, *moved.shape)
        rows = move_reads_backwards.as_stride_morphism(reindexing)._cod_stride_shift
        valid = torch.ones(degree_sizes, dtype=torch.bool)
        indices: list[torch.Tensor] = []
        for (_, strides, shift), position in zip(rows, tiled):
            index = torch.full(degree_sizes, self.integer(shift), dtype=torch.long)
            for grid, stride in zip(grids, strides):
                index = index + self.integer(stride) * grid
            extent = operand.shape[position]
            valid = valid & (index >= 0) & (index < extent)
            indices.append(index.clamp(0, extent - 1))
        read = moved[tuple(indices)]
        return read * valid.reshape(*degree_sizes, *(1,) * len(targets)).to(DTYPE)

    def broadcasted(self, target: cat.Broadcasted, inputs: Tensors) -> Tensors:
        degree_sizes = tuple(self.size(axis) for axis in target.degree())
        grids = tuple(torch.meshgrid(
            *(torch.arange(size) for size in degree_sizes), indexing='ij')) \
            if degree_sizes else ()
        read = tuple(
            self.read_through(operand, weave, reindexing, grids, degree_sizes)
            for operand, weave, reindexing
            in zip(inputs, target.input_weaves, target.reindexings))
        result = self.operator(target, read, degree_sizes)
        return (self.placed(result, target.output_weaves[0]),)

    def placed(self, result: torch.Tensor, weave: cat.Weave) -> torch.Tensor:
        '''`result`, with the degree first and the target after it, in the order of
        the positions of `weave`.'''
        tiled = [position for position, entry in enumerate(weave._shape)
                 if isinstance(entry, cat.WeaveMode)]
        targets = [position for position, entry in enumerate(weave._shape)
                   if not isinstance(entry, cat.WeaveMode)]
        order = [*tiled, *targets]
        return result.permute(*(order.index(position) for position in range(len(order))))

    def target_shape(self, weave: cat.Weave) -> tuple[int, ...]:
        return tuple(self.size(axis) for axis in weave._shape
                     if not isinstance(axis, cat.WeaveMode))

    def operator(self, target: cat.Broadcasted, read: Tensors,
                 degree_sizes: tuple[int, ...]) -> torch.Tensor:
        operator = target.operator
        match operator:
            case Caching.Caching():
                return self.caching(target, read[0], degree_sizes)
            case aops.CovariantView():
                return self.covariant_view(operator, read[0], degree_sizes,
                                           self.target_shape(target.output_weaves[0]))
            case ops.Arithmetic():
                return torch_compile.numeric_torch(operator.formula, read[0])
            case ops.View():
                return read[0]
            case ops.Einops():
                return einsum_of_groups(operator.signature, read)
            case ops.AdditionOp():
                return sum(read[1:], read[0])
            case ops.ConstantOp():
                shape = (*degree_sizes, *self.target_shape(target.output_weaves[0]))
                return torch_compile.numeric_torch(
                    operator.value, torch.zeros(shape, dtype=DTYPE))
            case ops.Linear():
                return self.linear(target, read, degree_sizes)
        raise OperatorNotEvaluated(f'no evaluation for {type(operator).__name__}')

    def linear(self, target: cat.Broadcasted, read: Tensors,
               degree_sizes: tuple[int, ...]) -> torch.Tensor:
        key = name_key(target.operator.name)
        if key not in self.weights:
            raise WeightNotGiven(f'no weight for the Linear {key}')
        values = self.weights[key]
        if not read:
            return values.weight.expand(*degree_sizes, *values.weight.shape)
        contracted = len(target.input_weaves[0].target().shape())
        result = torch.tensordot(read[0], values.weight, dims=contracted)
        return result if values.bias is None else result + values.bias

    def covariant_view(self, operator: aops.CovariantView, operand: torch.Tensor,
                       degree_sizes: tuple[int, ...], output_target: tuple[int, ...]
                       ) -> torch.Tensor:
        '''`operand` written at the codomain positions of the reindexing, with zero at
        every position nothing writes.'''
        reindexing = operator.reindexing
        written = torch.zeros((*degree_sizes, *output_target), dtype=DTYPE)
        domain_sizes = tuple(self.size(axis) for axis in reindexing._dom)
        for domain_index in _every_index(domain_sizes):
            codomain_index = tuple(
                self.integer(shift) + sum(self.integer(stride) * position
                                          for stride, position in zip(strides, domain_index))
                for _, strides, shift in reindexing._cod_stride_shift)
            if all(0 <= position < extent
                   for position, extent in zip(codomain_index, output_target)):
                written[(..., *codomain_index)] += operand[(..., *domain_index)]
        return written

    def caching(self, target: cat.Broadcasted, operand: torch.Tensor,
                degree_sizes: tuple[int, ...]) -> torch.Tensor:
        '''The earlier entries the cache keeps, followed by `operand`, along the token
        axis, which the read puts last. The operand is appended once the pass is
        over.'''
        name = name_key(target.operator.name)
        kept = self.size(Caching.past_tokens_of(target))
        entries = operand.movedim(-1, 0)
        earlier = self.caches.last_entries(name, kept, tuple(entries.shape[1:]))
        self.caches.append(name, entries)
        return torch.cat([earlier, entries]).movedim(0, -1)


def einsum_of_groups(signature: tuple[tuple[int, ...], ...], read: Tensors) -> torch.Tensor:
    '''The product of the operands, summed over every contraction group, with the degree
    leading every operand and the result.'''
    letters = 'abcdefghijklmnopqrstuvw'
    operands = ','.join('...' + ''.join(letters[group] for group in groups)
                        for groups in signature)
    return torch.einsum(f'{operands}->...', *read)


def _every_index(sizes: tuple[int, ...]):
    if not sizes:
        yield ()
        return
    for first in range(sizes[0]):
        for rest in _every_index(sizes[1:]):
            yield (first, *rest)


def _stream_grabs(term: cat.Morphism):
    match term:
        case Para.StreamGrab():
            yield term
        case cat.Composed(content=parts) | cat.ProductOfMorphisms(content=parts):
            for part in parts:
                yield from _stream_grabs(part)
        case cat.Block(body=body):
            yield from _stream_grabs(body)


def evaluate_a_pass(expression: cat.Morphism, inputs: Tensors,
                    weights: Mapping[str, LinearValues], sizes: Mapping[str, int],
                    caches: CachesBetweenPasses) -> Tensors:
    '''`expression` evaluated on `inputs`, with every size named in `sizes`, reading
    and appending to `caches`.'''
    evaluation = Evaluation(weights={name_key(name): values for name, values in weights.items()},
                            bindings=sizes_by_symbol(expression, sizes),
                            caches=caches)
    results = evaluation.morphism(expression, inputs)
    caches.end_the_pass()
    return results
