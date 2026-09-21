'''The operations of DeepSeek-V4.1-Flash that are not one call to a standard template.

Written by Claude Opus 5, effort high.

`weights` is the parameter array every learned tensor that rides no `ops.Linear` is
written as. The rest are the pointwise maps, each an `ops.Arithmetic` over a formula, so
a derived backward pass carries the written derivative rather than a primed name. There
is no opaque operator here. The model held one until 2026-09-15, for the initial
collapse vector, and that vector is now the covariant reading of the row that selects
the first stream.

The table in the notebook's *Custom operations* cell says what each operation is and
which of them the standard operator set cannot state.
'''
from __future__ import annotations

import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd

from notebooks.sota.DeepSeekV41Flash.construction_idioms import axis_name
from notebooks.sota.DeepSeekV41Flash.declared_axes import R


def weights[B: cat.Datatype, A: cat.Axis](
    name: str,
    shape: tuple[A, ...],
    datatype: B = R,
) -> cat.Broadcasted[B, A]:
    '''A parameter array, which is an `ops.Linear` with no inputs. A linear map out of the
    empty product is a constant tensor, so the Linear's implicit weight is the parameter.
    A consumer of such an array broadcasts it by giving its own reindexing a degree the
    array ignores, because `morphism_object_lift` adds a tiled slot to the weave and leaves
    no reindexing to fill it.'''
    name = fd.DynamicName.from_str(name).reconstruct(
        settings=fd.DynamicNameSettings(bold=True))
    return cat.Broadcasted(
        operator=ops.Linear(name=name),
        input_weaves=(),
        output_weaves=(cat.Weave(datatype, tuple(shape)),),
        reindexings=())


def multiply_along[A: cat.Axis](axis: A) -> cat.Broadcasted:
    '''The elementwise product of two wires, along the declared axis whose name supplies
    the einops letter. A letter mints a fresh axis, and the fresh axis can win canonicality
    on a merge, so the letter is read off the axis that is already there.'''
    letter = axis_name(axis)
    return ops.Einops.template(f'{letter}, {letter} -> {letter}')


def indicator() -> cat.Broadcasted:
    return ops.Arithmetic.template(nm.IsPositive(nm.x))


def exponential() -> cat.Broadcasted:
    return ops.Arithmetic.template(nm.E ** nm.x)


def reciprocal() -> cat.Broadcasted:
    return ops.Arithmetic.template(nm.Integer(1) / nm.x, name='z^{-1}')


def sigmoid() -> cat.Broadcasted:
    return ops.Arithmetic.template(nm.Sigmoid(nm.x), name='\\sigma')


def doubled_sigmoid() -> cat.Broadcasted:
    return ops.Arithmetic.template(nm.Integer(2) * nm.Sigmoid(nm.x), name='2\\sigma')


def sigmoid_weighted_input() -> cat.Broadcasted:
    return ops.Arithmetic.template(nm.Sigmoid(nm.x) * nm.x)


def sqrt_softplus() -> cat.Broadcasted:
    return ops.Arithmetic.template(
        nm.Power.template(nm.Logarithm.template(nm.E, nm.E ** nm.x + nm.Integer(1)),
                          nm.Integer(1) / nm.Integer(2)),
        name='\\sqrt{s^{+}}')
