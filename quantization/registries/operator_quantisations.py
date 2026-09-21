# Claude Opus 5 (1M context), effort high. Rewritten by Claude Fable 5.1, effort 80,
# on 2026-09-20, to the pattern followed by the released DeepSeek-V4.1-Flash.
'''The quantisation carried by each result of an operator, and the quantisation
required of each of its operands, with one rule per operator class.

A quantisation is the number format a value is held in, BF16, FP32, E4M3 or E2M1,
together with the size of that format in bits. A released model is eager PyTorch
around a few kernels, and its quantisations follow four facts. A tensor passed from one
module or kernel to the next is held in the activation quantisation. Arithmetic inside a
module upcasting on entry, and inside every kernel, runs at the scalar quantisation. An
activation is rounded to a quantisation with fewer bits in one place, in front of a
projection whose weight has fewer bits than an activation, by the rounding kernel
called by that projection. Everything else computes at the quantisation carried by its
operands, promoted to the one with the most bits where they differ, as PyTorch does.
The rules here state those four facts, and `QuantizationPolicy` names the quantisations
and the few boxes departing from them.

`quantization.processing.quantise_model` reads a rule for every operation of a model
and writes the result quantisations onto the wires. Where a rule requires of an operand
a quantisation not carried by the wire, that pass inserts a `TypeConvert` on the wire,
which is the cast shown by the figure of a quantised model. A cast into a quantisation
with more bits changes no value and stands where the released code writes `.float()`.

An index, which is a `cat.Natural`, carries an integer quantisation. Every operation
producing one returns the integer quantisation of the policy, `int64` for the
released code, which holds its token identifiers, its hash and the positions picked
by `torch.topk` in that format. A box returns another where the policy names one,
because the released indexer and candidate pool convert the positions handed to the
attention kernel to `int32`, and a box requires one of its index operands where the
policy names one, because that kernel declares the datatype of its positions. An
index passes through a reindexing at the quantisation it arrived with. The user asked
for the quantisation of an index on 2026-09-20. A rule returns `None` at a position
holding neither a real number nor an index, which the pass reads as the instruction
to keep the datatype already carried by the wire.

`obsidian/04-quantization/Quantization.md` states the package and the rules.
'''
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import advanced_axis_dynamics.data_structure.Operators as aops
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd
import deepseek.data_structure as dst
import quantization.data_structure.Quantization as Quantization

if TYPE_CHECKING:
    # The policy is the set of quantisations read by every rule, and it belongs beside
    # the pass assigning them. Importing it at runtime would close a cycle, because
    # that pass reads this registry.
    from quantization.processing.quantise_model import QuantizationPolicy

type Quantisation = Quantization.Quantified[Any]


class ArithmeticQuantisation(Enum):
    '''The quantisation at which the elementwise arithmetic of a box computes.
    `SCALAR` is a module upcasting on entry, or a kernel with a compute quantisation,
    and is the default. `CARRIED` is eager code never upcasting, so each map returns
    the quantisation of its operand.'''
    SCALAR = 'SCALAR'
    CARRIED = 'CARRIED'


class ContractionQuantisation(Enum):
    '''The quantisation returned by a contraction inside a box. `PROMOTED` is an eager
    einsum, which returns the quantisation of its operands, and is the default.
    `ACCUMULATED` is a fused kernel, whose contractions read their operands at one
    quantisation and keep their accumulators at the scalar quantisation.'''
    PROMOTED = 'PROMOTED'
    ACCUMULATED = 'ACCUMULATED'


fd.register_enum(ArithmeticQuantisation)
fd.register_enum(ContractionQuantisation)


@dataclass(frozen=True)
class BoxPolicy:
    '''How the body of one box departs from the policy. `results` is the quantisation
    returned by the box for a real number, and `None` is the activation
    quantisation. `integer_results` is the quantisation returned by the box for an
    index, and `None` is the integer quantisation of the policy. `integer_operands`
    is the quantisation required of every index read by the box, for a kernel
    declaring the datatype of its positions, and `None` requires nothing.'''
    results: Quantisation | None = None
    arithmetic: ArithmeticQuantisation = ArithmeticQuantisation.SCALAR
    contractions: ContractionQuantisation = ContractionQuantisation.PROMOTED
    integer_results: Quantisation | None = None
    integer_operands: Quantisation | None = None


DEFAULT_BOX_POLICY = BoxPolicy()


@dataclass(frozen=True)
class OperatorQuantisation:
    '''The quantisation carried by each result of an operation, and the quantisation
    required of each of its operands.

    `None` at a result position says that the wire keeps the datatype already carried
    by it, and `None` at an operand position says that any quantisation is accepted by
    the operation.
    '''
    results: fd.Prod[Quantisation | None]
    operands: fd.Prod[Quantisation | None]


@dataclass(frozen=True)
class QuantisationQuestion:
    '''One operation asked about its quantisations: the operation itself, the policy,
    the quantisation carried by each of its operands, and the name of the box holding
    the operation in its body, which is `None` at the top level of a model.'''
    morphism: cat.Broadcasted
    policy: QuantizationPolicy
    operand_quantisations: fd.Prod[Quantisation | None]
    enclosing_box: str | None = None

    def operand_datatypes(self) -> fd.Prod[cat.Datatype]:
        return tuple(array.datatype for array in self.morphism.dom())

    def result_datatypes(self) -> fd.Prod[cat.Datatype]:
        return tuple(array.datatype for array in self.morphism.cod())

    def box_policy(self) -> BoxPolicy:
        return self.policy.box_policy(self.enclosing_box)

    def accepts_any_operand(self) -> fd.Prod[None]:
        return (None,) * len(self.operand_datatypes())

    def operands_of(self, real: Quantisation | None, natural: Quantisation | None,
                    ) -> fd.Prod[Quantisation | None]:
        '''`real` required at every operand holding a real number and `natural` at
        every operand holding an index, with `None` requiring nothing.'''
        return tuple(_by_kind(datatype, real, natural)
                     for datatype in self.operand_datatypes())

    def results_of(self, real: Quantisation | None, natural: Quantisation | None,
                   ) -> fd.Prod[Quantisation | None]:
        '''`real` at every result holding a real number and `natural` at every
        result holding an index.'''
        return tuple(_by_kind(datatype, real, natural)
                     for datatype in self.result_datatypes())

    def real_operands(self, quantisation: Quantisation | None
                      ) -> fd.Prod[Quantisation | None]:
        '''`quantisation` required at every operand holding a real number.'''
        return self.operands_of(quantisation, None)

    def real_results(self, quantisation: Quantisation | None
                     ) -> fd.Prod[Quantisation | None]:
        '''`quantisation` at every result holding a real number.'''
        return self.results_of(quantisation, None)

    def quantisations_of_real_operands(self) -> fd.Prod[Quantisation | None]:
        return self._quantisations_of_operands(Quantization.holds_real_numbers)

    def quantisations_of_natural_operands(self) -> fd.Prod[Quantisation | None]:
        return self._quantisations_of_operands(Quantization.holds_natural_numbers)

    def _quantisations_of_operands(
        self, holds: Callable[[cat.Datatype], bool],
    ) -> fd.Prod[Quantisation | None]:
        return tuple(
            carried for datatype, carried
            in zip(self.operand_datatypes(), self.operand_quantisations)
            if holds(datatype))

    def known_operand_quantisations(self) -> fd.Prod[Quantisation]:
        '''The quantisations of the real operands already assigned one.'''
        return tuple(carried for carried in self.quantisations_of_real_operands()
                     if carried is not None)

    def known_natural_operand_quantisations(self) -> fd.Prod[Quantisation]:
        '''The quantisations of the index operands already assigned one.'''
        return tuple(carried for carried in self.quantisations_of_natural_operands()
                     if carried is not None)


def _by_kind(datatype: cat.Datatype, real: Quantisation | None,
             natural: Quantisation | None) -> Quantisation | None:
    if Quantization.holds_real_numbers(datatype):
        return real
    if Quantization.holds_natural_numbers(datatype):
        return natural
    return None


type Rule = Callable[[QuantisationQuestion], OperatorQuantisation]

RULES: dict[type[cat.Operator], Rule] = {}


def register(*kinds: type[cat.Operator]) -> Callable[[Rule], Rule]:
    '''Register `rule` as the quantisation rule of every class in `kinds`.'''
    def decorate(rule: Rule) -> Rule:
        for kind in kinds:
            RULES[kind] = rule
        return rule
    return decorate


def rule_for(operator: cat.Operator) -> Rule:
    '''The rule of the most specific registered class of `operator`.

    The lookup walks the type's own method resolution order, as
    `para.registries.derivative.rule_for` does, so `ops.View` takes the movement rule
    declared for it rather than the elementwise rule of `ops.Elementwise`, and
    `para.data_structure.ParaBlockOperator.ParaBlockOperator` takes the rule of
    `ops.BlockOperator`.
    '''
    for kind in type(operator).__mro__:
        if kind in RULES:
            return RULES[kind]
    return computed_at_the_scalar_quantisation


def has_a_rule(operator: cat.Operator) -> bool:
    '''Whether a class of `operator` declares a rule, rather than falling back on the
    scalar quantisation.'''
    return any(kind in RULES for kind in type(operator).__mro__)


# ==========================================================================
# Comparing quantisations by their size in bits.
# ==========================================================================
class SizeIsNotAWholeNumber(ValueError):
    '''A quantisation whose size is a symbol rather than a number of bits, so two
    quantisations cannot be ordered by it.'''


def bits(quantisation: Quantisation) -> int:
    size = quantisation.size
    if not isinstance(size, nm.Integer):
        raise SizeIsNotAWholeNumber(f'{quantisation} is {size} bits in size')
    return size._value


def with_the_most_bits(quantisations: Iterable[Quantisation]) -> Quantisation:
    '''The quantisation of a mixed expression under PyTorch promotion, which is the
    operand quantisation with the most bits.'''
    return max(quantisations, key=bits)


def with_the_fewest_bits(quantisations: Iterable[Quantisation]) -> Quantisation:
    return min(quantisations, key=bits)


def has_fewer_bits_than_the_activations(quantisation: Quantisation,
                                        policy: QuantizationPolicy) -> bool:
    return bits(quantisation) < bits(policy.activations)


# ==========================================================================
# The rules.
# ==========================================================================
@register(ops.Linear)
def projected(question: QuantisationQuestion) -> OperatorQuantisation:
    '''A projection whose weight has fewer bits than an activation reads its operand
    through a rounding kernel, and every other projection reads its operand at the
    read quantisation of its weight.

    The released `linear()` rounds an activation to the quantisation named for rounded
    operands when, and only when, the weight given to it is FP8 or FP4. That
    quantisation is E4M3 with one UE8M0 scale per 32 channels, and the rounding kernel
    writes the E4M3 elements and the scales, which the GEMM multiplies into its FP32
    accumulator with the scales of the weight. The rounding kernel reads the
    activation quantisation, so an operand arriving with more bits is first brought to
    the activation quantisation and then rounded, and an operand already carrying the
    rounded quantisation, a row of a block-scaled table, needs no cast. The matrix
    multiply writes its result at the activation quantisation. A projection with a
    weight at the activation quantisation or with more bits computes at the
    quantisation of the weight, and an operand arriving with fewer bits is read into
    it, which is the `.float()` written by the released code in front of such a
    projection.
    '''
    policy = question.policy
    operator = question.morphism.operator
    if not isinstance(operator, ops.Linear):
        raise QuantisationRuleMismatch(
            f'{type(operator).__qualname__} is not a Linear')
    weight = policy.weight_quantisation(operator.name)
    if has_fewer_bits_than_the_activations(weight, policy):
        return OperatorQuantisation(
            results=question.real_results(policy.activations),
            operands=tuple(
                _rounding_step(carried, policy)
                if Quantization.holds_real_numbers(datatype) else None
                for datatype, carried
                in zip(question.operand_datatypes(), question.operand_quantisations)))
    return OperatorQuantisation(
        results=question.real_results(weight),
        operands=question.real_operands(weight))


def _rounding_step(carried: Quantisation | None,
                   policy: QuantizationPolicy) -> Quantisation:
    '''The quantisation asked by the rounding kernel in front of a rounded projection
    of an operand carrying `carried`: the activation quantisation where the operand
    arrives with more bits than it, because the kernel reads that quantisation, and
    the quantisation named for rounded operands otherwise.'''
    if carried is not None and bits(carried) > bits(policy.activations):
        return policy.activations
    return policy.rounded_operands


def has_a_contraction(operator: ops.Einops) -> bool:
    '''Whether the signature sums over any axis. An `Einops` whose signature names no
    group is a broadcast product.'''
    return any(len(groups) > 0 for groups in operator.signature)


@register(ops.Einops)
def contracted(question: QuantisationQuestion) -> OperatorQuantisation:
    '''A contraction or a broadcast product written with an `Einops` computes at the
    quantisation of its operands, promoted to the one with the most bits where they
    differ, and returns that quantisation, which is what eager PyTorch does with an
    einsum or a product.

    Inside a box named by the policy as one fused kernel, a contraction reads its
    operands at the quantisation with the fewest bits among them and accumulates at
    the scalar quantisation, which is what a tensor-core instruction does. The released
    attention kernel casts the probabilities down to the quantisation of the values
    before the second matrix multiply and keeps both accumulators in FP32. A broadcast
    product inside such a kernel still promotes, because it is elementwise arithmetic.
    '''
    policy = question.policy
    operator = question.morphism.operator
    if not isinstance(operator, ops.Einops):
        raise QuantisationRuleMismatch(
            f'{type(operator).__qualname__} is not an Einops')
    known = question.known_operand_quantisations()
    if not known:
        return OperatorQuantisation(
            results=question.results_of(policy.scalars, policy.integers),
            operands=question.accepts_any_operand())
    fused = (question.box_policy().contractions
             is ContractionQuantisation.ACCUMULATED)
    if fused and has_a_contraction(operator):
        return OperatorQuantisation(
            results=question.results_of(policy.scalars, policy.integers),
            operands=question.real_operands(with_the_fewest_bits(known)))
    promoted = with_the_most_bits(known)
    return OperatorQuantisation(
        results=question.results_of(promoted, policy.integers),
        operands=question.real_operands(promoted))


@register(ops.Elementwise, ops.Arithmetic, ops.SoftMax, ops.L1Norm, ops.L2Norm,
          ops.Maximum, ops.ConstantOp, ops.FixedArray, ops.GenericOperator,
          dst.Rotary, dst.YarnRotary)
def computed_at_the_scalar_quantisation(
    question: QuantisationQuestion,
) -> OperatorQuantisation:
    '''An elementwise map, a softmax, a maximum and a table of constants compute at
    the scalar quantisation and read their operands at it, because a released module
    upcasts on entry and a kernel computes at its compute quantisation.

    Inside a box named by the policy as computing at the carried quantisation, which
    is eager code never upcasting, the operation returns the quantisation of its
    operand and requires nothing of it, and a constant written into such code takes
    the activation quantisation of the tensor receiving it. The released indexer
    scores its queries against its keys in BF16 from end to end.
    '''
    policy = question.policy
    if question.box_policy().arithmetic is ArithmeticQuantisation.CARRIED:
        known = question.known_operand_quantisations()
        carried = with_the_most_bits(known) if known else policy.activations
        return OperatorQuantisation(
            results=question.results_of(carried, policy.integers),
            operands=question.accepts_any_operand())
    return OperatorQuantisation(
        results=question.results_of(policy.scalars, policy.integers),
        operands=question.real_operands(policy.scalars))


@register(ops.Normalize)
def normalised(question: QuantisationQuestion) -> OperatorQuantisation:
    '''A normalisation with a learned gain is the released RMSNorm module, which reads
    a tensor at the activation quantisation, computes in FP32 and returns the
    quantisation given to it. One without a gain is arithmetic written inline in a
    module already upcast, so it reads and returns the scalar quantisation.'''
    policy = question.policy
    operator = question.morphism.operator
    if not isinstance(operator, ops.Normalize):
        raise QuantisationRuleMismatch(
            f'{type(operator).__qualname__} is not a Normalize')
    quantisation = policy.activations if operator.gain else policy.scalars
    return OperatorQuantisation(
        results=question.real_results(quantisation),
        operands=question.real_operands(quantisation))


@register(ops.AdditionOp)
def added(question: QuantisationQuestion) -> OperatorQuantisation:
    '''An addition computes at the quantisation with the most bits among its real
    operands and returns it, with the other operand read into it, which is what
    PyTorch does when a BF16 tensor is added into an FP32 one. A sum of indices, as
    the Engram hash adds an offset to a remainder, is an integer.'''
    policy = question.policy
    known = question.known_operand_quantisations()
    if not known:
        return OperatorQuantisation(
            results=question.results_of(policy.scalars, policy.integers),
            operands=question.accepts_any_operand())
    promoted = with_the_most_bits(known)
    return OperatorQuantisation(
        results=question.results_of(promoted, policy.integers),
        operands=question.real_operands(promoted))


@register(ops.View, aops.CovariantView, aops.ConcatenateAxes, aops.DeconcatenateAxes,
          dst.MergedPositions, dst.PairsAsComplex, dst.Decomplex)
def moved(question: QuantisationQuestion) -> OperatorQuantisation:
    '''An operation reindexing, concatenating or pairing values performs no
    arithmetic on them, so it returns the quantisation of the values and requires
    every real operand at that one quantisation.

    Where the operands carry two quantisations, which a concatenation can, the result
    takes the one with the fewest bits, because the released rotary embedding writes
    its rotated half back into the array given to it and that array has the fewer
    bits. Indices pass through the same way, at the quantisation with the fewest bits
    among them, and every index operand is required at it, so the released `.int()`
    on the positions picked by the indexer stands where those positions meet the
    positions of the candidate pool. An operation reading no real operand is a
    constant array, and returns the scalar quantisation, or the integer quantisation
    where it holds indices.
    '''
    known = question.known_operand_quantisations()
    quantisation = with_the_fewest_bits(known) if known else question.policy.scalars
    index = _carried_index(question)
    return OperatorQuantisation(
        results=question.results_of(quantisation, index),
        operands=question.operands_of(quantisation, index))


def _carried_index(question: QuantisationQuestion) -> Quantisation:
    '''The quantisation of the indices passing through an operation: the one with
    the fewest bits among its index operands, and the integer quantisation of the
    policy where none carries one.'''
    known = question.known_natural_operand_quantisations()
    return with_the_fewest_bits(known) if known else question.policy.integers


@register(dst.TopK)
def ranked(question: QuantisationQuestion) -> OperatorQuantisation:
    '''A top-k selection returns the values it picks at the quantisation of the
    values, and the positions it picks as integers at the integer quantisation of the
    policy, which is the `int64` returned by `torch.topk`, whatever quantisation the
    candidate positions read by it carry.'''
    known = question.known_operand_quantisations()
    quantisation = with_the_fewest_bits(known) if known else question.policy.scalars
    return OperatorQuantisation(
        results=question.results_of(quantisation, question.policy.integers),
        operands=question.real_operands(quantisation))


@register(dst.Select, dst.IndexSelect)
def selected(question: QuantisationQuestion) -> OperatorQuantisation:
    '''A selection returns the payload at the positions named by a selection, so its
    result carries the quantisation of the payload, which is the last of its real
    operands.'''
    carried = question.quantisations_of_real_operands()
    payload = carried[-1] if carried else None
    return OperatorQuantisation(
        results=question.results_of(payload or question.policy.scalars,
                                    _carried_index(question)),
        operands=question.accepts_any_operand())


@register(ops.Embedding)
def embedded(question: QuantisationQuestion) -> OperatorQuantisation:
    '''An embedding reads a table at an index and returns the row at the quantisation
    of the table, which the policy gives by name. The table itself has no wire, and
    `quantise_model` records that quantisation beside the model.

    The released n-gram table holds its rows in E4M3 with one UE8M0 scale per 32
    channels, multiplies a row by its scales into BF16 when the row is read, and hands
    it to a projection whose rounding kernel rounds it to E4M3 with one UE8M0 scale
    per 32 channels again. An E4M3 element times a power of two is an E4M3 element,
    and the two groups of 32 begin at the same channels, so the second rounding
    changes no value, and the user ruled on 2026-09-20 that the row carries the
    quantisation of the table and reaches the projection with no cast.
    '''
    operator = question.morphism.operator
    if not isinstance(operator, ops.Embedding):
        raise QuantisationRuleMismatch(
            f'{type(operator).__qualname__} is not an Embedding')
    policy = question.policy
    return OperatorQuantisation(
        results=question.results_of(
            policy.weight_quantisation(operator.name),
            policy.integer_weight_quantisation(operator.name)),
        operands=question.accepts_any_operand())


@register(ops.BitwiseXor, ops.Modulo)
def integer_arithmetic(question: QuantisationQuestion) -> OperatorQuantisation:
    '''An integer operation returns integers at the integer quantisation of the
    policy and requires nothing of its operands. The exclusive-or and the remainder
    of the Engram hash are the two in this package, and the released hash computes
    both in `int64`.'''
    return OperatorQuantisation(
        results=question.results_of(None, question.policy.integers),
        operands=question.accepts_any_operand())


@register(ops.Cast)
def cast_to_a_declared_datatype(
    question: QuantisationQuestion,
) -> OperatorQuantisation:
    '''An `ops.Cast` reads a value into the datatype declared by it. The result
    carries the scalar quantisation where that datatype holds a real number, and
    the integer quantisation of the policy where it holds an index, because the
    cast narrows the bound of the index and not the format holding it.'''
    policy = question.policy
    return OperatorQuantisation(
        results=question.results_of(policy.scalars, policy.integers),
        operands=question.accepts_any_operand())


@register(Quantization.TypeConvert)
def converted(question: QuantisationQuestion) -> OperatorQuantisation:
    '''A `TypeConvert` returns the quantisation named by its target. A target naming
    none is a read of a rounded value back into the reals, as written by the round
    trips of the quantised caches, and the released kernels read such a value back
    into their compute quantisation, so it returns the scalar quantisation.'''
    operator = question.morphism.operator
    if not isinstance(operator, Quantization.TypeConvert):
        raise QuantisationRuleMismatch(
            f'{type(operator).__qualname__} is not a TypeConvert')
    policy = question.policy
    target = Quantization.quantisation_of(operator.target)
    return OperatorQuantisation(
        results=question.results_of(target or policy.scalars, target or policy.integers),
        operands=question.accepts_any_operand())


@register(ops.BlockOperator)
def boxed(question: QuantisationQuestion) -> OperatorQuantisation:
    '''A boxed block returns the quantisation named by the policy for a box of its
    name, and the activation quantisation where the policy names none, because a
    released module returns to the quantisation given to it at its end unless it is
    one of the few handing FP32 on. The body is a morphism of its own, and
    `quantise_model` descends into it and casts its last operation to the required
    quantisation where the body computes something else.'''
    operator = question.morphism.operator
    if not isinstance(operator, ops.BlockOperator):
        raise QuantisationRuleMismatch(
            f'{type(operator).__qualname__} is not a BlockOperator')
    policy = question.policy
    name = None if operator.name is None else operator.name.to_bodies()
    box = policy.box_policy(name)
    return OperatorQuantisation(
        results=question.results_of(
            policy.box_results(name), box.integer_results or policy.integers),
        operands=question.operands_of(None, box.integer_operands))


class QuantisationRuleMismatch(ValueError):
    '''A rule was asked about an operation of another class.'''
