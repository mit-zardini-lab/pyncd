'''Checks values, ports and nested block occurrences across graph conversion.

The evaluator interprets the public product-category constructors directly. It
supports scalar addition and ReLU leaves, and integer repetitions on endomorphisms.
It does not call the graph converter, compilation backend or construction helpers.
Each case is checked through three consecutive conversions. Every exception or
mismatch is a failure. Torch also checks swap values and input gradients.
'''
from __future__ import annotations

from collections.abc import Iterator
from collections import Counter
from dataclasses import dataclass
from enum import Enum
import itertools
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import graphs.data_structure.Hypergraph as hg
import graphs.processing.Hypergraph2Morphism as Hypergraph2Morphism
import torch_compile.torch_compile as torch_compile


type Values = tuple[int, ...]
type BlockOccurrence = tuple[cat.BlockTag, frozenset[tuple[BlockOccurrence, int]]]
type Occurrences = Counter[BlockOccurrence]
SCALAR = cat.Array(cat.Reals(), ())


class InputMode(Enum):
    ORDINARY = 'ordinary'
    ALIASED_EXACT = 'aliased exact'
    ALIASED_REDUCED = 'aliased reduced'


@dataclass(frozen=True)
class Case:
    label: str
    term: cat.Morphism
    input_mode: InputMode = InputMode.ORDINARY
    known_output: Values | None = None


def evaluate(term: cat.Morphism, inputs: Values) -> Values:
    if len(inputs) != len(term.dom()):
        raise ValueError(f'{len(inputs)} values for {len(term.dom())} input ports')
    match term:
        case cat.Rearrangement(mapping=mapping):
            result = tuple(inputs[position] for position in mapping)
        case cat.Composed(content=content):
            result = inputs
            for child in content:
                result = evaluate(child, result)
        case cat.ProductOfMorphisms(content=content):
            result = ()
            offset = 0
            for child in content:
                length = len(child.dom())
                result += evaluate(child, inputs[offset:offset + length])
                offset += length
        case cat.Block(body=body, block_tag=tag):
            repetitions = nm.evaluate_integer(tag.repetition, {})
            if repetitions < 0:
                raise ValueError(f'negative repetition {repetitions}')
            if repetitions != 1 and body.dom() != body.cod():
                raise ValueError('repeated body is not an endomorphism')
            result = inputs
            for _ in range(repetitions):
                result = evaluate(body, result)
        case cat.Broadcasted(operator=ops.AdditionOp()):
            result = (sum(inputs),)
        case cat.Broadcasted(operator=ops.ReLU()):
            result = (max(0, inputs[0]),)
        case _:
            raise TypeError(f'unsupported evaluator leaf {type(term).__name__}')
    if len(result) != len(term.cod()):
        raise ValueError(f'{len(result)} results for {len(term.cod())} output ports')
    return result


def occurrences(term: cat.Morphism) -> Occurrences:
    '''Counts exact BlockTags together with each occurrence's nested children.

    Independent sibling order can change during graph scheduling. Multiplicity,
    tag identity and enclosing-block order must survive that scheduling.
    '''
    match term:
        case cat.Block(body=body, block_tag=tag):
            return Counter({(tag, frozenset(occurrences(body).items())): 1})
        case cat.Composed(content=content) | cat.ProductOfMorphisms(content=content):
            return sum((occurrences(child) for child in content), Counter())
        case _:
            return Counter()


def wiring(arity: int, mapping: tuple[int, ...]) -> cat.Rearrangement:
    return cat.Rearrangement(mapping=mapping, _dom=(SCALAR,) * arity)


def block(term: cat.Morphism, title: str, repetition: int = 1) -> cat.Block:
    return cat.Block.template(term, title=title, repetition=repetition)


def compose(*terms: cat.Morphism) -> cat.Composed:
    return cat.Composed(content=terms)


def product(*terms: cat.Morphism) -> cat.ProductOfMorphisms:
    return cat.ProductOfMorphisms(content=terms)


def arithmetic(operator: cat.Operator, arity: int) -> cat.Broadcasted:
    weave = cat.Weave(cat.Reals(), ())
    return cat.Broadcasted(
        operator=operator, input_weaves=(weave,) * arity,
        output_weaves=(weave,), reindexings=(cat.ProdObject().identity(),) * arity)


def cases() -> Iterator[Case]:
    swap = block(wiring(2, (1, 0)), 'swap')
    identity = wiring(1, (0,))
    unit = wiring(0, ())
    yield Case('single swap', swap)
    yield Case('same swap used twice', compose(swap, swap))
    yield Case('distinct swaps', compose(swap, block(wiring(2, (1, 0)), 'other')))
    yield Case('nested swap', block(block(swap, 'middle'), 'outer'))
    yield Case('parallel same swap', product(swap, swap))
    yield Case('discard after block', compose(swap, wiring(2, (0,))))
    yield Case('discard every output', compose(swap, wiring(2, ())))
    yield Case('block discards every output', block(wiring(2, ()), 'discard'))
    yield Case('unit block', block(unit, 'unit'))
    yield Case('consecutive unit blocks', compose(block(unit, 'u1'), block(unit, 'u2')))
    yield Case('parallel unit and identity', product(block(unit, 'unit'), identity))
    yield Case('zero-wire product', product())
    yield Case('explicit unit composition', compose(unit, block(unit, 'unit'), unit))
    for mode in (InputMode.ALIASED_EXACT, InputMode.ALIASED_REDUCED):
        yield Case(f'{mode.value} graph input swap', swap, input_mode=mode)
        yield Case(f'{mode.value} graph input copy',
                   block(wiring(2, (1, 0, 1)), 'copy'), input_mode=mode)
        yield Case(f'{mode.value} graph input discard',
                   block(wiring(2, ()), 'drop'), input_mode=mode)
    addition = arithmetic(ops.AdditionOp(), 2)
    relu = arithmetic(ops.ReLU(), 1)
    mixed = compose(swap, product(relu, identity), addition)
    yield Case('swap then computation', mixed)
    yield Case('nested mixed computation', block(mixed, 'mixed'))
    yield Case('copy then addition',
               compose(block(wiring(1, (0, 0)), 'copy'), addition))
    yield Case('computation then wiring block', compose(product(relu, relu), swap))
    yield Case('computation discarded', compose(block(mixed, 'unused'), wiring(1, ())))
    step = compose(wiring(2, (0, 1, 1)), product(addition, identity))
    for repetitions in range(4):
        repeated_swap = block(wiring(2, (1, 0)), f'swap-{repetitions}', repetitions)
        yield Case(f'swap repeated {repetitions}', repeated_swap)
        yield Case(f'nested repeated swap {repetitions}', block(repeated_swap, 'outer'))
        yield Case(f'reused repeated swap {repetitions}',
                   compose(repeated_swap, repeated_swap))
        yield Case(f'parallel repeated swap {repetitions}',
                   product(repeated_swap, repeated_swap))
        yield Case(f'repeated mixed step {repetitions}',
                   block(step, 'step', repetitions))
        yield Case(f'zero-wire repeated {repetitions}',
                   block(unit, 'unit', repetitions))
    loop = block(step, 'accumulation', 2)
    for mode in (InputMode.ALIASED_EXACT, InputMode.ALIASED_REDUCED):
        yield Case(f'aliased accumulation {mode.value}', loop, input_mode=mode)
        yield Case(f'nested aliased accumulation {mode.value}',
                   block(loop, 'outer accumulation'), input_mode=mode)
    yield Case('copied initial accumulation', compose(wiring(1, (0, 0)), loop),
               known_output=(30, 10))
    first = block(wiring(3, (1, 0, 2)), 'A', 2)
    second = block(wiring(3, (0, 2, 1)), 'B', 2)
    yield Case('collision sequential', compose(first, second),
               known_output=(10, 20, 30))
    yield Case('collision parallel', compose(
        wiring(3, (0, 1, 2, 1, 0, 2)), product(first, second), wiring(6, (3, 4, 5))),
        known_output=(20, 10, 30))
    for arity in range(1, 4):
        for permutation in itertools.permutations(range(arity)):
            term = block(wiring(arity, permutation), 'permutation')
            label = f'permutation {arity}:{permutation}'
            yield Case(label, term)
            yield Case(label + ' twice', compose(term, term))
    for input_arity in range(1, 3):
        for output_arity in range(4):
            for mapping in itertools.product(range(input_arity), repeat=output_arity):
                term = block(wiring(input_arity, mapping), 'mapping')
                label = f'mapping {input_arity}:{mapping}'
                yield Case(label, term)
                yield Case(label + ' nested', block(term, 'outer'))
                reversed_positions = tuple(reversed(range(output_arity)))
                yield Case(label + ' then reverse', compose(
                    term, block(wiring(output_arity, reversed_positions), 'reverse')))


def check_case(case: Case) -> bool:
    generator = random.Random(137)
    arity = len(case.term.dom())
    inputs = [tuple(10 * (position + 1) for position in range(arity))]
    inputs += [tuple(generator.randrange(-19, 20) for _ in range(arity))
               for _ in range(3)]
    if case.input_mode != InputMode.ORDINARY:
        inputs = [tuple(values[0] for _ in values) for values in inputs]
    supplied = ([values[:1] for values in inputs]
                if case.input_mode == InputMode.ALIASED_REDUCED else inputs)
    expected_dom = (
        cat.ProdObject((SCALAR,))
        if case.input_mode == InputMode.ALIASED_REDUCED else case.term.dom())
    try:
        expected = tuple(evaluate(case.term, values) for values in inputs)
        if case.known_output is not None and expected[0] != case.known_output:
            raise ValueError(f'fixture expected {case.known_output}, got {expected[0]}')
        tags = occurrences(case.term)
        current = case.term
        for iteration in range(3):
            if case.input_mode != InputMode.ORDINARY and iteration == 0:
                shared = hg.HypergraphObject.template((SCALAR,))[0]
                graph = hg.Multigraph.from_morphism(
                    current, dom=(shared,) * arity,
                    reduce=case.input_mode == InputMode.ALIASED_REDUCED)
                current = Hypergraph2Morphism.hypergraph_to_morphism(graph)
            else:
                current = Hypergraph2Morphism.recycle(current)
            if current.dom() != expected_dom or current.cod() != case.term.cod():
                raise ValueError(f'round {iteration + 1}: ports '
                                 f'{len(expected_dom)}/{len(case.term.cod())} -> '
                                 f'{len(current.dom())}/{len(current.cod())}')
            actual = tuple(evaluate(current, values) for values in supplied)
            if actual != expected:
                raise ValueError(
                    f'round {iteration + 1}: values {expected} -> {actual}')
            if occurrences(current) != tags:
                raise ValueError(
                    f'round {iteration + 1}: nested BlockTag occurrences changed')
        print(f'PASS {case.label}', flush=True)
        return True
    except Exception as error:
        print(f'FAIL {case.label}: {type(error).__name__}: '
              f'{str(error).splitlines()[0][:240]}', flush=True)
        return False


def check_torch_swaps(count: int) -> bool:
    '''Compares compiled swap values and reverse gradients with explicit tensors.'''
    swap = block(wiring(2, (1, 0)), 'tensor swap')
    original = swap if count == 1 else compose(swap, swap)
    operands = (torch.tensor([1., -2., 3.], dtype=torch.float64),
                torch.tensor([7., 4., -5.], dtype=torch.float64))
    expected = operands[::-1] if count % 2 else operands
    gradient_weights = (5., 2.) if count % 2 else (2., 5.)
    try:
        for term in (original, Hypergraph2Morphism.recycle(original)):
            inputs = tuple(value.clone().requires_grad_() for value in operands)
            results = torch_compile.ConstructedModule.construct(term)(*inputs)
            if len(results) != len(expected) or any(
                    not torch.equal(actual, wanted)
                    for actual, wanted in zip(results, expected)):
                raise ValueError('compiled swap returned incorrect values')
            loss = 2 * results[0].sum() + 5 * results[1].sum()
            gradients = torch.autograd.grad(loss, inputs)
            if any(not torch.equal(gradient, torch.full_like(gradient, weight))
                   for gradient, weight in zip(gradients, gradient_weights)):
                raise ValueError('compiled swap returned incorrect input gradients')
        print(f'PASS torch {count} swaps: values and gradients', flush=True)
        return True
    except Exception as error:
        print(f'FAIL torch {count} swaps: {type(error).__name__}: {error}', flush=True)
        return False


def main() -> int:
    results = tuple(check_case(case) for case in cases()) + tuple(
        check_torch_swaps(count) for count in (1, 2))
    passed = sum(results)
    print(f'TOTAL {len(results)} PASS {passed} FAIL {len(results) - passed}',
          flush=True)
    return int(not all(results))


if __name__ == '__main__':
    raise SystemExit(main())
