'''
Checking derived backward passes against `torch.autograd`.

`validate_simplification` checks that rewrites preserve a morphism's value.
This checks something stronger: that the *derivation* is the derivative. Each
expression is put through `forward_backward`, simplified, and, where there is
anything to collapse, through `pathway_collapse`. Both passes are then compiled with
`torch_compile` and run, and the backward's outputs are compared against
`torch.autograd.grad` of the forward on the same random inputs.

    python para/validate_backward.py

It needs `torch`. Without `torch` the script reports the fact and exits 0.

## detape

`para.algebra.detape.detape` turns each pass into a pure function, with a
`Drop` an extra output and a `Grab` an extra input, in slot order. The harness
carries the residuals from the forward's extra outputs to the backward's extra
inputs by hand, which is what a training loop does with its saved activations.

## Elementwise semantics

`torch_compile` maps every `ops.Elementwise` to `torch.relu`, which is a
placeholder. The expressions here write their maps as `ops.Arithmetic` -
`e^{x}`, `x^{-1}` - whose formula `torch_compile` evaluates directly, and
whose derivative the rule writes as a formula too, so nothing needs naming.
The name table below remains for a bare `Elementwise` that still appears,
and an unknown name raises rather than silently reluing.
'''
from __future__ import annotations
import sys

import construction_helpers as ch  # noqa: F401 - operator overloads
import data_structure.Term as fd
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import graphs.data_structure.Hypergraph as hg
import graphs.processing.Hypergraph2Morphism as h2m
import term_utilities.term_utilities as tutil

import agent_display as ad
import algebra.einops_rearrange as er
import algebra.operator_expansion as oe
import para.algebra.detape as detape_module
import para.data_structure.Para as para
import para.processing.backprop as bp
import para.data_structure.contraction as pcon
import para.algebra.pathway_collapse as pc


SIZES = {'q': 3, 'd': 4, 'v': 5, 'x': 6, 'w': 2, 'm': 7}


detape = detape_module.detape


def _named_elementwise(torch):
    import torch_compile.torch_compile as tc

    table = {
        'e': torch.exp,
        "e'": torch.exp,
        '/z': torch.reciprocal,
        "/z'": lambda z: -z.pow(-2),
        '-': torch.neg,
        'ReLU': torch.relu,
        "ReLU'": lambda x: (x > 0).to(x.dtype),
    }

    class NamedElementwise(tc.ConstructedModule,
                           operation_key=ops.Elementwise):
        def __init__(self, target: cat.Broadcasted) -> None:
            super().__init__(target)
            name = (target.operator.name.to_bodies()
                    if target.operator.name is not None else None)
            if isinstance(target.operator, ops.View):
                func = lambda x: x
            elif name in table:
                func = table[name]
            else:
                raise NotImplementedError(
                    f'no torch semantics declared a-priori for Elementwise<{name}>')
            self.func = tc.broadcast_func(target, func)

        def forward(self, *xs):
            return self.func(*xs)

    # The registry keys on the exact operator type, which is the same rule as
    # tsncd's box registry, so the Elementwise subclass the ffn writes needs
    # its own row.
    tc.ConstructedModule.operation_registry[ops.ReLU] = NamedElementwise
    return NamedElementwise


def _sizes(array: cat.Array) -> list[int]:
    return [SIZES[axis.uid._name.to_bodies()] for axis in array.shape()]


def _run(module, *xs):
    result = module(*xs)
    return result if isinstance(result, tuple) else (result,)


def check(name, target, taped, torch) -> dict[str, bool]:
    '''One Taped pair against autograd. The reference is the *expression*
    compiled directly, and the pair is compiled through `detape`.'''
    import torch_compile.torch_compile as tc
    results = {}

    inputs = [torch.randn(*_sizes(array), requires_grad=True)
              for array in target.dom()]
    reference = _run(tc.ConstructedModule.construct(target), *inputs)
    assert len(reference) == 1, 'one-output expressions only'
    dy = torch.randn_like(reference[0])
    grads = torch.autograd.grad(reference[0], inputs, grad_outputs=dy)

    forward, forward_tape = detape(taped.forward)
    backward, backward_tape = detape(taped.backward)
    results['slots agree'] = ([name for name, _ in forward_tape]
                              == [name for name, _ in backward_tape])

    outs = _run(tc.ConstructedModule.construct(forward),
                *(x.detach() for x in inputs))
    primary, residuals = outs[:1], outs[1:]
    results['forward value'] = torch.allclose(
        primary[0], reference[0], atol=1e-4)

    cotangents = _run(tc.ConstructedModule.construct(backward),
                      dy, *residuals)
    results['backward = autograd'] = (
        len(cotangents) == len(grads)
        and all(torch.allclose(ours, theirs, atol=1e-4)
                for ours, theirs in zip(cotangents, grads)))
    print(f'  {name:34}',
          '  '.join(f'{"ok" if v else "FAIL"} {k}' for k, v in results.items()))
    return results


def named_softmax(over='x', *degree):
    base = ops.SoftMax.template()
    for name in reversed(degree):
        base = cat.RawAxis.named(name) >> base
    target = cat.Array(cat.Reals(), (cat.RawAxis.named(over),))
    return base.reconstruct(
        input_weaves=tuple(w.imprint_target(target) for w in base.input_weaves),
        output_weaves=tuple(w.imprint_target(target) for w in base.output_weaves))


def named_l1_norm(over='x', *degree):
    base = ops.L1Norm.template()
    for name in reversed(degree):
        base = cat.RawAxis.named(name) >> base
    target = cat.Array(cat.Reals(), (cat.RawAxis.named(over),))
    return base.reconstruct(
        input_weaves=tuple(w.imprint_target(target) for w in base.input_weaves),
        output_weaves=tuple(w.imprint_target(target) for w in base.output_weaves))


def expanded_l1_norm(batch='q'):
    '''`L1Norm` written out by `algebra.operator_expansion`.'''
    return oe.expand_l1_norms(named_l1_norm('x', batch))


def softmax_as_l1_norm(batch='q'):
    '''`SoftMax` in one step, as the exponential followed by the L1 norm.'''
    return oe.rewrite_softmaxes_as_l1_norms(named_softmax('x', batch))


def expanded_softmax(batch='q'):
    '''`SoftMax` written out by `algebra.operator_expansion`.'''
    return oe.expand_softmaxes(named_softmax('x', batch))


def shifted_softmax(batch='q'):
    '''`SoftMax` written out with its scores shifted by their maximum.'''
    return oe.expand_shifted_softmaxes(named_softmax('x', batch))


def duplicated_exponential():
    '''One input copied into two equal exponentials that are added. The forward
    pass holds two roots computing one value, and `dedup_roots` keeps one.'''
    exponential = cat.RawAxis.named('q') >> ops.Arithmetic.template(nm.E ** nm.x)
    return (0, 0) @ (exponential * exponential) @ ops.AdditionOp.template()


def cases(torch) -> dict[str, bool]:
    _named_elementwise(torch)
    attention = (ops.Einops.template('q d, x d -> q x')
                 @ ops.SoftMax.template()
                 @ ops.Einops.template('q x, x v -> q v'))
    attention_projected = (
        ops.Einops.template('q d, x d -> q x') @ ops.SoftMax.template()
        @ ops.Einops.template('q x, x v -> q v')
        @ ops.Einops.template('q v, v w -> q w'))

    def simplified(target):
        taped = pc.dedup_roots(pc.dedup_slots(bp.forward_backward(target)))
        return bp.Taped.from_passes(
            er.merge_reindexings_and_einops(taped.forward),
            er.merge_reindexings_and_einops(taped.backward))

    everything = {}
    for name, target in (
        ('matmul', ops.Einops.template('q d, d v -> q v')),
        ('duplicated exponential', duplicated_exponential()),
        ('softmax', named_softmax('x', 'q')),
        ('expanded softmax', expanded_softmax()),
        ('l1 norm', named_l1_norm('x', 'q')),
        ('expanded l1 norm', expanded_l1_norm()),
        ('softmax as exp then l1 norm', softmax_as_l1_norm()),
        ('attention', attention),
        ('attention, softmax expanded', oe.expand_softmaxes(attention)),
    ):
        everything.update({
            f'{name}: {k}': v
            for k, v in check(name, target, simplified(target), torch).items()})
    for name, target in (('attention, collapsed', attention),
                         ('attention+proj, collapsed', attention_projected)):
        collapsed = pc.pathway_collapse(simplified(target))
        everything.update({
            f'{name}: {k}': v
            for k, v in check(name, target, collapsed, torch).items()})
    # All three rewrites, meaning drops migrated through the exponential, additions
    # distributed, chains collapsed. On the expanded softmax this tapes y and
    # frees x and 1/z. On expanded attention it also fires, at a cost in
    # einsums the distribution does not yet pay back.
    for name, target in (
            ('expanded softmax, all rewrites', expanded_softmax()),
            ('attention, expanded, all rewrites', oe.expand_softmaxes(attention))):
        collapsed = pc.collapse(simplified(target))
        everything.update({
            f'{name}: {k}': v
            for k, v in check(name, target, collapsed, torch).items()})
    # The shifted expansion. The maximum's cotangent is the zero map, which
    # `prune_zero_cotangents` removes with the chain that fed it, so the
    # collapsed pair has the unshifted pair's form and reads the shifted
    # exponent and reciprocal off the tape.
    for name, target, reference in (
            ('shifted softmax, derived', shifted_softmax(), named_softmax('x', 'q')),
            ('attention, shifted, derived',
             er.merge_reindexings_and_einops(oe.expand_shifted_softmaxes(attention)),
             attention)):
        everything.update({
            f'{name}: {k}': v
            for k, v in check(name, reference, simplified(target), torch).items()})
        collapsed = pc.dedup_and_collapse(simplified(target))
        listing = ad.listing(h2m.recycle(collapsed.backward))
        everything[f'{name}, collapsed: no zero map'] = 'Arithmetic<0>' not in listing
        everything.update({
            f'{name}, collapsed: {k}': v
            for k, v in check(f'{name}, collapsed', reference, collapsed,
                              torch).items()})
    shifted_forward = ad.listing(h2m.recycle(
        pc.dedup_and_collapse(simplified(shifted_softmax())).forward))
    everything['shifted softmax keeps its maximum'] = 'Maximum' in shifted_forward
    print(f'  {"shifted softmax keeps its maximum":34}',
          'ok' if everything['shifted softmax keeps its maximum'] else 'FAIL')
    # Self-attention from one input copied three ways, so the token axis
    # stands at both positions of the scores. Every rule reads its operator by
    # position, so the pair derives and collapses as it does for two axes.
    copied = (
        (0, 1, 0, 2, 0, 3)
        @ (ops.Einops.template('x m, m d -> x d')
           * ops.Einops.template('x m, m d -> x d')
           * ops.Einops.template('x m, m v -> x v'))
        @ (ops.Einops.template('q d, x d -> q x') @ ops.SoftMax.template()
           @ ops.Einops.template('q x, x v -> q v'))
        @ ops.Einops.template('x v, v m -> x m'))
    for name, target in (
            ('self-attention, copied axis', copied),
            ('self-attention, copied axis, expanded',
             er.merge_reindexings_and_einops(oe.expand_softmaxes(copied)))):
        everything.update({
            f'{name}: {k}': v
            for k, v in check(name, target, simplified(target), torch).items()})
        collapsed = pc.dedup_and_collapse(simplified(target))
        everything.update({
            f'{name}, all rewrites: {k}': v
            for k, v in check(f'{name}, all rewrites', target, collapsed,
                              torch).items()})
    # And the collapse itself did something: the attention forward tapes one
    # more value (O), and the backward's statistic reads it.
    collapsed = pc.pathway_collapse(simplified(attention))
    listing = ad.listing(h2m.recycle(collapsed.backward))
    everything['collapse fired'] = 'Grab<s5>' in listing
    print(f'  {"collapse fired":34}',
          'ok' if everything['collapse fired'] else 'FAIL')
    # And on the expanded softmax: the output taped (s4), the input's tape
    # (s0) and the reciprocal's (s2) no longer read, nothing recomputed.
    collapsed = pc.collapse(simplified(expanded_softmax()))
    listing = ad.listing(h2m.recycle(collapsed.backward))
    everything['softmax collapse fired'] = (
        'Grab<s4>' in listing and 'Grab<s0>' not in listing
        and 'Grab<s2>' not in listing and '1x Arithmetic' in listing)
    print(f'  {"softmax collapse fired":34}',
          'ok' if everything['softmax collapse fired'] else 'FAIL')
    everything.update(check_pointwise_formulas(torch, simplified))
    everything.update(check_parametrised(torch))
    return everything


def pointwise(formula: nm.Numeric) -> cat.Broadcasted:
    return cat.RawAxis.named('x') >> ops.Arithmetic.template(formula)


def check_pointwise_formulas(torch, simplified) -> dict[str, bool]:
    '''Formulas over `nm.IsPositive` and the `nm.Expandable` numerics, and the
    split of a backward formula at a forward one.

    The clamp's bound is one half, so the random inputs fall on both sides of
    it, and so is the floor under the magnitude of the signed root, which is
    Engram's gate. The sigmoid's derivative becomes `x * (1 - x)` on the taped output.
    The SiLU's derivative is not a function of the SiLU alone, so it keeps its
    `\\sigma`. A log-sigmoid beside a sigmoid spelled in primitives matches
    only through the expansion, and reads the spelled sigmoid off the tape.
    '''
    half = nm.Integer(1) / nm.Integer(2)
    everything = {}
    for name, target in (
            ('rectified linear', pointwise(nm.RectifiedLinear(nm.x))),
            ('clamp on both sides', pointwise(
                nm.x - nm.RectifiedLinear(nm.x - half)
                + nm.RectifiedLinear(-half - nm.x))),
            ('clamp to the unit interval', pointwise(nm.Clamp(nm.x))),
            ('clamp between bounds', pointwise(nm.Clamp(nm.x, -half, half))),
            ('absolute value', pointwise(nm.AbsoluteValue(nm.x))),
            ('larger of the input and one half', pointwise(nm.LargerOf(nm.x, half))),
            ('square root of the magnitude', pointwise(
                nm.SquareRoot(nm.AbsoluteValue(nm.x) + half))),
            ('signed root with a floor', pointwise(nm.Sigmoid(
                nm.Sign(nm.x) * nm.SquareRoot(nm.LargerOf(nm.AbsoluteValue(nm.x), half)))))):
        everything.update({
            f'{name}: {k}': v
            for k, v in check(name, target, simplified(target), torch).items()})
    spelled_sigmoid = pointwise(1 / (1 + nm.E ** -nm.x))
    log_sigmoid = pointwise(nm.Logarithm.template(nm.E, nm.Sigmoid(nm.x)))
    beside = ((0, 0) @ (spelled_sigmoid * log_sigmoid)
              @ ops.Einops.template('x, x -> x'))
    for name, target, holds in (
            ('sigmoid, all rewrites', pointwise(nm.Sigmoid(nm.x)),
             lambda listing: ('\\sigma' not in listing
                              and 'Arithmetic<x (1 - x)>' in listing)),
            ('silu, all rewrites', pointwise(nm.x * nm.Sigmoid(nm.x)),
             lambda listing: '\\sigma(x)' in listing and 'e^{' not in listing),
            ('log-sigmoid, all rewrites', beside,
             lambda listing: '\\sigma' not in listing)):
        collapsed = pc.collapse(simplified(target))
        everything.update({
            f'{name}: {k}': v
            for k, v in check(name, target, collapsed, torch).items()})
        listing = ad.listing(h2m.recycle(collapsed.backward))
        everything[f'{name}: split as stated'] = holds(listing)
        print(f'  {name + ", split as stated":34}',
              'ok' if everything[f'{name}: split as stated'] else 'FAIL')
    return everything


def check_parametrised(torch) -> dict[str, bool]:
    '''
    The parametrised FFN: the weights grabbed as operands, the backward
    pass's transposes grabbing the same slots, and the gradients dropped to
    `dW{n}`.

    The generic `check` cannot run it, because the forward and backward tapes
    are no longer equal. The backward pass reads the parameters the forward
    pass reads and writes gradients that never appear in the forward pass. The residuals
    and the parameters are therefore both carried to the backward pass by slot
    name, and every gradient is compared against `torch.autograd.grad` of the
    compiled forward pass with respect to (x, W1, W2). The forward pass itself
    is compared against the expression written directly in torch.
    '''
    import torch_compile.torch_compile as tc
    import para.processing.show_grabbed_parameters as sgp
    ffn = cat.RawAxis.named('q') >> (
        ops.Linear.template('m', 'f', name='1')
        @ ops.ReLU.template(name='ReLU')
        @ ops.Linear.template('f', 'm', name='2'))
    base = pc.dedup_roots(pc.dedup_slots(sgp.collapse_grabbed_residuals(
        bp.forward_backward(sgp.grab_parameters(ffn)))))
    # The recompute variant saves the pre-activation alone and re-runs the
    # ReLU in the backward pass, giving one residual fewer and the same
    # gradients.
    variants = {'parametrised ffn': base,
                'ffn, relu recomputed': pc.recompute_elementwise_slots(base)}
    everything = {}
    for label, taped in variants.items():
        taped = bp.Taped.from_passes(
            er.merge_reindexings_and_einops(taped.forward),
            er.merge_reindexings_and_einops(taped.backward))
        results = _check_parametrised_one(label, taped, torch, tc)
        everything.update({f'{label}: {k}': v for k, v in results.items()})
    saved = {label: sum(not is_grab for _, is_grab in
                        detape(bp.Taped.from_passes(
                            er.merge_reindexings_and_einops(t.forward),
                            er.merge_reindexings_and_einops(t.backward)).forward)[1])
             for label, t in variants.items()}
    everything['recompute saves a slot'] = (
        saved['ffn, relu recomputed'] == saved['parametrised ffn'] - 1)
    print(f'  {"recompute saves a slot":34}',
          'ok' if everything['recompute saves a slot'] else 'FAIL')
    return everything


def _check_parametrised_one(label, taped, torch, tc) -> dict[str, bool]:
    sizes = dict(SIZES, m=4, f=7)
    def tensor(array, grad=False):
        return torch.randn(
            *[sizes[axis.uid._name.to_bodies()] for axis in array.shape()],
            requires_grad=grad)

    results = {}
    forward, forward_tape = detape(taped.forward)
    backward, backward_tape = detape(taped.backward)

    # Forward: dom = (x, *parameter grabs), cod = (y, *residual drops).
    inputs = [tensor(array, grad=True) for array in forward.dom()]
    grab_names = [name for name, is_grab in forward_tape if is_grab]
    drop_names = [name for name, is_grab in forward_tape if not is_grab]
    outs = _run(tc.ConstructedModule.construct(forward), *inputs)
    primary, residuals = outs[0], dict(zip(drop_names, outs[1:]))
    x, w1, w2 = inputs
    results['forward value'] = torch.allclose(
        primary, torch.relu(x @ w1) @ w2, atol=1e-4)

    dy = torch.randn_like(primary)
    grads = dict(zip(('x', *grab_names),
                     torch.autograd.grad(primary, inputs, grad_outputs=dy)))

    # Backward: dom = (dy, *grabs) - residuals and parameters, fed by slot
    # name, and cod = (dx, *gradient drops), each `dW` against its `W`.
    by_name = residuals | {name: inputs[1 + grab_names.index(name)]
                           for name in grab_names}
    fed = [dy.detach(), *(by_name[name].detach()
                          for name, is_grab in backward_tape if is_grab)]
    outs = _run(tc.ConstructedModule.construct(backward), *fed)
    gradient_names = [name for name, is_grab in backward_tape if not is_grab]
    cotangents = {'x': outs[0],
                  **dict(zip(gradient_names, outs[1:]))}
    results['backward reads the parameters'] = (
        set(grab_names) < {name for name, is_grab in backward_tape if is_grab})
    results['gradients = autograd'] = all(
        torch.allclose(cotangents['d' + name if name != 'x' else 'x'],
                       grads[name], atol=1e-4)
        for name in ('x', *grab_names))
    print(f'  {label:34}',
          '  '.join(f'{"ok" if v else "FAIL"} {k}' for k, v in results.items()))
    return results


if __name__ == '__main__':
    try:
        import torch
    except ImportError:
        print('torch is not installed. Nothing was checked.')
        sys.exit(0)
    torch.manual_seed(0)
    results = cases(torch)
    failed = [k for k, v in results.items() if not v]
    print('all ok' if not failed else f'FAILED: {failed}')
    sys.exit(1 if failed else 0)
