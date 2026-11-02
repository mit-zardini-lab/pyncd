'''Compares linear values and reverse derivatives with positional torch einsums.'''
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

import construction_helpers as ch
import data_structure.Category as cat
import data_structure.Operators as ops
import para.data_structure.transpose as transpose
import para.processing.backprop as backprop
import para.processing.show_grabbed_parameters as show_grabbed_parameters
import term_utilities.term_utilities as tutil
import notebooks.website.tutorial.check_gradients_in_torch as check_gradients_in_torch


@dataclass(frozen=True)
class LinearCase:
    label: str
    term: cat.Broadcasted
    signature: str


def cases() -> tuple[LinearCase, ...]:
    width = cat.RawAxis.named('m')
    output = cat.RawAxis.named('n')
    batch = cat.RawAxis.named('q')
    tiled = cat.WeaveMode.TILED
    interleaved = cat.Broadcasted(
        operator=ops.Linear(),
        input_weaves=(cat.Weave(cat.Reals(), (width, tiled)),),
        output_weaves=(cat.Weave(cat.Reals(), (output, tiled)),),
        reindexings=(cat.ProdObject((batch,)).identity(),))
    return (
        LinearCase('square', ops.Linear.template((width,), (width,), 'W'), 'i,ij->j'),
        LinearCase('square with bias', ops.Linear.template(
            (width,), (width,), 'W', bias=True), 'i,ij->j'),
        LinearCase('rectangular', ops.Linear.template(
            (width,), (output,), 'W'), 'i,ij->j'),
        LinearCase('repeated input', ops.Linear.template(
            (width, width), (output,), 'W'),
             'ij,ijk->k'),
        LinearCase('repeated output', ops.Linear.template(
            (width,), (output, output), 'W'),
             'i,ijk->jk'),
        LinearCase('all repeated', ops.Linear.template(
            (width, width), (width, width), 'W'),
             'ij,ijkl->kl'),
        LinearCase('overlapping input/result',
             ops.Linear.template((width, output), (width,), 'W'), 'ij,ijk->k'),
        LinearCase('interleaved rectangular', interleaved, 'iq,ij->jq'),
        LinearCase('interleaved square', interleaved.reconstruct(
            output_weaves=(cat.Weave(cat.Reals(), (width, tiled)),)), 'iq,ij->jq'),
    )


def run_case(case: LinearCase) -> bool:
    '''Reports each path separately so one failure cannot hide another.'''
    grabbed = show_grabbed_parameters.grab_parameters(case.term)
    sizes = {'m': 3, 'n': 2, 'q': 4}
    generator = torch.Generator().manual_seed(137)

    def reference(inputs: Sequence[torch.Tensor],
                  weights: Mapping[str, torch.Tensor]) -> torch.Tensor:
        values = tuple(weights.values())
        result = torch.einsum(case.signature, inputs[0], values[0])
        return result + values[1] if len(values) == 2 else result

    passed = True
    try:
        compiled = check_gradients_in_torch.compile_pass(grabbed, sizes)
        inputs = [torch.randn(shape, generator=generator, dtype=torch.float64)
                  for shape in compiled.own_operand_shapes()]
        weights = {name: torch.randn(shape, generator=generator, dtype=torch.float64)
                   for name, shape in compiled.grabbed_shapes().items()}
        actual = check_gradients_in_torch.run(
            compiled.module,
            [*inputs, *(weights[name] for name in compiled.grabbed)])[0]
        expected = reference(inputs, weights)
        error = (actual - expected).abs().max().item()
        agrees = torch.allclose(actual, expected, atol=1e-9)
        passed &= agrees
        print(f'{case.label}: forward={agrees} max_error={error:.9g}', flush=True)
    except Exception as error:
        passed = False
        print(f'{case.label}: forward {type(error).__name__}: {str(error)[:180]}',
              flush=True)
    try:
        result = check_gradients_in_torch.compare_with_autograd(
            backprop.forward_backward(grabbed), reference, sizes, generator)
        passed &= result.all_agree()
        print(f'{case.label}: backward {result}', flush=True)
    except Exception as error:
        passed = False
        print(f'{case.label}: backward {type(error).__name__}: {str(error)[:180]}',
              flush=True)
    return passed


def check_transpose(case: LinearCase) -> bool:
    '''Differentiates the transposed map with its weight as an ordinary input.'''
    if case.term.operator.bias:
        return True
    grabbed = show_grabbed_parameters.grab_parameters(case.term)
    seed = next(node for node in tutil.type_search(cat.Broadcasted, grabbed)
                if isinstance(node.operator, ops.Linear))
    transposed = seed.reconstruct(
        operator=transpose.Transpose(name=seed.operator.name, operator=seed.operator),
        input_weaves=(seed.input_weaves[0], seed.output_weaves[0]),
        output_weaves=(seed.input_weaves[-1],),
        reindexings=(seed.reindexings[0], seed.degree().identity()))
    data_indices, weight_and_output = case.signature.split(',')
    weight_indices, output_indices = weight_and_output.split('->')
    signature = f'{output_indices},{weight_indices}->{data_indices}'

    def reference(inputs: Sequence[torch.Tensor],
                  weights: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.einsum(signature, inputs[1], inputs[0])

    try:
        result = check_gradients_in_torch.compare_with_autograd(
            backprop.forward_backward(transposed), reference,
            {'m': 3, 'n': 2, 'q': 4}, torch.Generator().manual_seed(137))
        print(f'{case.label}: transposed {result}', flush=True)
        return result.all_agree()
    except Exception as error:
        print(f'{case.label}: transposed {type(error).__name__}: {str(error)[:180]}',
              flush=True)
        return False


def check_explicit_square() -> bool:
    '''Checks a non-diagonal weight and reports the complete numerical result.'''
    term = cases()[0].term
    taped = backprop.forward_backward(show_grabbed_parameters.grab_parameters(term))
    forward = check_gradients_in_torch.compile_pass(taped.forward, {'m': 3})
    backward = check_gradients_in_torch.compile_pass(taped.backward, {'m': 3})
    operand = torch.tensor([1., 2., 3.], dtype=torch.float64)
    weight = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 10.]],
                          dtype=torch.float64)
    cotangent = torch.tensor([2., -1., 3.], dtype=torch.float64)
    results = check_gradients_in_torch.run(forward.module, [operand, weight])
    saved = dict(zip(forward.dropped, results[1:])) | {'W': weight}
    gradients = check_gradients_in_torch.run(
        backward.module, [cotangent, *(saved[name] for name in backward.grabbed)])
    print(f'explicit square: y={results[0].tolist()} '
          f'dx={gradients[0].tolist()} dW={gradients[1].tolist()}', flush=True)
    return (torch.equal(results[0], operand @ weight)
            and torch.equal(gradients[0], weight @ cotangent)
            and torch.equal(gradients[1], torch.outer(operand, cotangent)))


def main() -> None:
    results = tuple(run_case(case) for case in cases())
    transposes = tuple(check_transpose(case) for case in cases())
    try:
        explicit = check_explicit_square()
    except Exception as error:
        explicit = False
        print(f'explicit square: {type(error).__name__}: {str(error)[:180]}',
              flush=True)
    raise SystemExit(0 if all(results + transposes) and explicit else 1)


if __name__ == '__main__':
    main()
