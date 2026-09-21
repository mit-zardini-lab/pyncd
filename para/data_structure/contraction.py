'''
Building a `Broadcasted` from axis *objects* rather than from a signature
string.

`ops.Einops.template('q d, x d -> q x')` is the ordinary way to write a
contraction, and it calls `cat.RawAxis.named` for every symbol, so every call
mints fresh axes. Fresh axes are right for writing an expression
down, and exactly wrong for deriving one, because a reverse pass
has to contract the incoming cotangent against axes that already exist in the
forward pass, and two axes that merely print as `|d|` are different axes
(CLAUDE.md, on an axis being identified by its UID rather than by its name).

So this module is `construction_helpers.einops.signature_to_broadcast` with the
name lookup taken out: hand it the shapes as tuples of `Axis`, and identity is
the axis itself. The construction lives in
`algebra.einops_simplification.einsum`; `contract` is that function, kept here
so the reverse pass has one name for "the contraction dual to this".

An axis is either absorbed, meaning it is in an operand and not in the result, or degree, meaning it is in
the result). There is no "produced" case: an axis in the result that no
operand has is a degree axis that no operand's reindexing names, which is
exactly a repeat along it, which is the dual of a sum. `contract(((q,),), (q, x))`
is a `View` whose reindexing for its one operand does not mention `x`,
which is the same thing `expand_to_nodes` writes for a broadcast node, and it
is absorbed by `reindexing_absorption.absorb` like any other node.
'''
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

import data_structure.Term as fd
import data_structure.Category as cat
import algebra.einops_simplification as es


@dataclass(frozen=True)
class Zero(cat.Operator):
    '''
    The nullary constant emitting zeros of a given shape, which is the dual of deleting
    a wire. `ops.ConstantOp` is the same idea but is a streaming initialiser that
    the kernel layer reads. Keeping the cotangent zero separate keeps a reverse
    pass distinguishable from a fold seed.
    '''
    name: fd.DynamicName | None = fd.DynamicName('0')


def contract[B: cat.Datatype, A: cat.Axis](
    inputs: Iterable[Iterable[A]],
    output: Iterable[A],
    datatype: B = cat.Reals(),
    operator: cat.Operator | None = None,
) -> cat.Broadcasted[B, A]:
    '''
    `inputs` are the shapes of the operands and `output` the shape of the
    result, as tuples of `einops_simplification.IndexVariable`, which
    `index_shapes` reads off a forward morphism, or of axes. The operator is
    inferred unless given:

        anything absorbed              `Einops`, with the contraction groups
        nothing absorbed, one operand  `View` - a pure reindexing, which
                                       is also how a repeat is written
        nothing absorbed, several      `Einops` with an empty signature, which
                                       is the pointwise product

    Passing `operator` overrides the inference, which is how an `AdditionOp` or
    an elementwise map gets built at a degree that is not written anywhere as a
    string.
    '''
    return es.einsum(inputs, output, datatype, operator)


def zeros[B: cat.Datatype, A: cat.Axis](
    array: cat.Array[B, A]
) -> cat.Broadcasted[B, A]:
    '''The zero cotangent of an array: a nullary morphism onto its shape.'''
    return cat.Broadcasted[B, A](
        operator=Zero(),
        input_weaves=(),
        output_weaves=(cat.Weave(array.datatype, tuple(array.shape())),),
        reindexings=(),
    )
