'''
Checking the two rewrites against arithmetic.

A rewrite that preserves the domain and codomain and reads well can still be
wrong, and these two are exactly the kind that would be. `absorb` permutes a
weave by a mapping, and `compose` rebuilds an einsum out of nothing but
shapes. The check is therefore numeric: compile both forms with
`torch_compile` and run them on the same random inputs.

    python algebra/validate_simplification.py

It needs `torch`, which `requirements.txt` does not list. Without `torch` the
script reports the fact and exits 0, because nothing else in the package
depends on it.

It covers both halves of the pass the `para` notebooks run, `absorb` and
`einops_rearrange.merge_rule`, and stands in the package that holds both rewrites.

The cases are the ones where the answer is known independently:

    absorb(expand_to_nodes(m)) == m       node expansion undone
    repeat                      == the same repeat, written as a string
    repeat ; multiply           == the same product, broadcast by a reindexing
    diagonal ; sum              == contraction
    contraction ; contraction   == the unified einsum, which is the house rule
                                  of always unifying and then disentangling,
                                  per `Einops Rearrangement` in the vault. It
                                  is not free, and the note below the case
                                  states why
'''
from __future__ import annotations
import sys

import construction_helpers as ch  # noqa: F401 - operator overloads
from construction_helpers import simple_helper as chsh
import data_structure.Category as cat
import data_structure.Operators as ops

import algebra.einops_rearrange as er
import algebra.merge_into_consumer as merge_into_consumer
import algebra.node_expansion as ne
import algebra.reindexing_absorption as ra
import para.data_structure.contraction as pcon


SIZES = {'q': 3, 'd': 4, 'v': 5, 'x': 6, 'w': 2}


def _sizes(array: cat.Array) -> list[int]:
    return [SIZES[axis.uid._name.to_bodies()] for axis in array.shape()]


def agree(left, right, torch) -> bool:
    '''Do two morphisms with the same domain compute the same thing?'''
    import torch_compile.torch_compile as tc
    inputs = [torch.randn(*_sizes(array)) for array in left.dom()]
    outputs = []
    for morphism in (left, right):
        result = tc.ConstructedModule.construct(morphism)(*inputs)
        outputs.append(result if isinstance(result, tuple) else (result,))
    return (len(outputs[0]) == len(outputs[1])
            and all(torch.allclose(a, b, atol=1e-4)
                    for a, b in zip(*outputs)))


def cases(torch) -> dict[str, bool]:
    matmul = ops.Einops.template('q d, d v -> q v')
    nodes, core = ne.expand_to_nodes(matmul)
    expanded = chsh.make_composed(nodes, core)
    absorbed = merge_into_consumer.merge_producers_into_consumers(expanded, ra.absorb)

    # A repeat feeding a product. A repeat is a `View` whose reindexing drops
    # the repeated axis, which makes it a node, so it compiles on its own and
    # absorbs by the node rule. The reference is the same product written with
    # the broadcast in the reindexing, which is what the merge produces.
    q, v = cat.RawAxis.named('q'), cat.RawAxis.named('v')
    repeated = pcon.contract(((q,),), (q, v))
    product = pcon.contract(((q, v), (q, v)), (q, v))
    unmerged_repeat = chsh.make_composed(
        chsh.make_product(repeated, cat.ProdObject(
            (cat.Array(cat.Reals(), (q, v)),)).identity()),
        product)
    absorbed_repeat = merge_into_consumer.merge_producers_into_consumers(
        unmerged_repeat, ra.absorb)
    reference = ops.Einops.template('q, q v -> q v')
    # The same repeat written as a string: `Einops.template` accepts a
    # produced axis now, and reads it the same way.
    templated = ops.Einops.template('q -> q v')

    # A repeat read by two products. The node is copied over the fan-out and
    # each product absorbs its own copy, so no node remains and the repeated
    # axis is broadcast inside each reader.
    two_operands = cat.ProdObject(
        (cat.Array(cat.Reals(), (q, v)), cat.Array(cat.Reals(), (q, v)))).identity()
    fanned_out = (chsh.make_product(repeated, two_operands)
                  @ ((0, 1, 0, 2) @ chsh.make_product(product, product)))
    absorbed_fan_out = ra.absorb_nodes(fanned_out)
    fan_out_reference = (0, 1, 0, 2) @ chsh.make_product(reference, reference)
    # A repeat read by one product and returned as an output. The product
    # absorbs its copy and the node stays for the output.
    one_operand = cat.ProdObject((cat.Array(cat.Reals(), (q, v)),)).identity()
    fanned_to_output = (chsh.make_product(repeated, one_operand)
                        @ ((0, 0, 1) @ chsh.make_product(one_operand, product)))
    absorbed_to_output = ra.absorb_nodes(fanned_to_output)
    to_output_reference = (0, 0, 1) @ chsh.make_product(repeated, reference)

    diagonal = ops.Einops.template('q v, d v -> q v d')
    then_summed = diagonal @ ops.Einops.template('q v d -> q d')
    merged = er.rearrange_local(then_summed)

    # Two contractions in a row unify into one three-operand einsum, and
    # `disentangle_einops` cannot split it back: Q shares `d` with K and K
    # shares `x` with V, so they are one connected component. The value is
    # unchanged, which is what is checked here. Whether the unified form is
    # the preferable one belongs to the cost model in `Einops Rearrangement`,
    # and it is open, per [[Open Gaps]]. Attention as `para` derives it never
    # reaches this case, because a `SoftMax` sits between the two
    # contractions.
    chain = (ops.Einops.template('q d, x d -> q x')
             @ ops.Einops.template('q x, x v -> q v'))
    unified = er.merge_reindexings_and_einops(chain)

    return {
        'absorb undoes node expansion': agree(matmul, absorbed, torch),
        'absorbed form has no node': not any(
            ra.is_node(root) for root in _roots(absorbed)),
        'a repeat is a node': ra.is_node(repeated),
        'a repeat compiles unmerged': agree(reference, unmerged_repeat, torch),
        'templated repeat agrees': agree(repeated, templated, torch),
        'absorb undoes a repeat': agree(reference, absorbed_repeat, torch),
        'absorbed repeat is one operation': len(_roots(absorbed_repeat)) == 1,
        'a repeat copied over a fan-out absorbs into both readers':
            agree(fan_out_reference, absorbed_fan_out, torch)
            and len(_roots(absorbed_fan_out)) == 2
            and not any(ra.is_node(root) for root in _roots(absorbed_fan_out)),
        'a repeat that is also an output stays for the output':
            agree(to_output_reference, absorbed_to_output, torch)
            and len(_roots(absorbed_to_output)) == 2
            and sum(ra.is_node(root) for root in _roots(absorbed_to_output)) == 1,
        'diagonal ; sum == contraction': agree(then_summed, merged, torch),
        'merged form is one einsum': len(_roots(merged)) == 1,
        'contraction ; contraction unifies, same value':
            len(_roots(unified)) == 1 and agree(chain, unified, torch),
    }


def _roots(target: cat.Morphism) -> list[cat.Broadcasted]:
    match target:
        case cat.Broadcasted():
            return [target]
        case cat.Composed(content=parts) | cat.ProductOfMorphisms(content=parts):
            return [root for part in parts for root in _roots(part)]
        case cat.Block(body=body):
            return _roots(body)
    return []


if __name__ == '__main__':
    try:
        import torch
    except ImportError:
        print('torch is not installed. Nothing was checked.')
        sys.exit(0)
    torch.manual_seed(0)
    results = cases(torch)
    for name, passed in results.items():
        print(f'  {"ok  " if passed else "FAIL"}  {name}')
    sys.exit(0 if all(results.values()) else 1)
