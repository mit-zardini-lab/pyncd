"""Converters from pyncd data-structure types to acset instance dataclasses."""
from __future__ import annotations

from dataclasses import dataclass

import data_structure.Term as fd
import data_structure.Numeric as nm
import data_structure.BroadcastedCategory as bc
from data_structure.StrideCategory import StrideMorphism
from data_structure.TensorLogic import TensorEquation, TensorProgram, topological_sort
from data_structure.TensorExpr import TensorRef, IversonBinOp, IversonUnaryOp, _factor_axes, _serialize_iverson
from data_structure.TensorDSL import NormAxis
import data_structure.Operators as ops

from acset.instances import (
    OpTag, DataTag, EntryRow, SStInstance,
    EquationRow, ArrayRow, ArrayAxisRow, SampleRow, SBrInstance,
)


def _dt_fields(dt: bc.Datatype | None) -> tuple[DataTag, nm.Numeric | None]:
    """Return (DataTag, max_value) for a Datatype, handling None as Reals."""
    if isinstance(dt, bc.Natural):
        return DataTag.NATURAL, dt.max_value
    if isinstance(dt, bc.Bool):
        return DataTag.BOOL, None
    return DataTag.REALS, None


@dataclass
class _OpFields:
    tag: OpTag
    bias: bool | None
    elementwise_fn: str | None
    iverson_expr: str | None = None


_TAG_FROM_TYPE: dict[type, OpTag] = {
    ops.Normalize:               OpTag.NORMALIZE,
    ops.Embedding:               OpTag.EMBEDDING,
    ops.AdditionOp:              OpTag.ADDITION_OP,
    ops.WeightedTriangularLower: OpTag.WEIGHTED_TRIANGULAR_LOWER,
}


def _operator_fields(op) -> _OpFields:
    """Extract all operator-derived ArrayRow fields in one place.

    Identity is checked before Elementwise because Identity is a subclass of
    Elementwise; all other operator types are disjoint.  SoftMax is checked
    before the generic tag lookup so that where_predicate (masked softmax)
    is handled separately from plain softmax.
    """
    if op is None or isinstance(op, ops.Identity):
        return _OpFields(OpTag.IDENTITY, None, None)
    if isinstance(op, ops.Elementwise):
        # op.operator is None for Identity (caught above), name string otherwise
        return _OpFields(OpTag.ELEMENTWISE, None, op.operator)
    if isinstance(op, ops.Linear):
        return _OpFields(OpTag.LINEAR, op.bias, None)
    if isinstance(op, ops.SoftMax):
        if op.where_predicate:
            preds = op.where_predicate
            if len(preds) == 1:
                expr = _serialize_iverson(preds[0])
            else:
                parts = ' '.join(_serialize_iverson(p) for p in preds)
                expr = f'(and {parts})'
            return _OpFields(OpTag.MASKED_SOFTMAX, None, None, iverson_expr=expr)
        return _OpFields(OpTag.SOFTMAX, None, None)
    tag = _TAG_FROM_TYPE.get(type(op))
    if tag is None:
        raise ValueError(f"unrecognised operator type: {type(op)}")
    return _OpFields(tag, None, None)


def from_stride_morphism(m: StrideMorphism) -> SStInstance:
    """Convert a StrideMorphism to an SStInstance.

    Each coefficient in the matrix becomes one EntryRow.
    Domain and codomain axes live together in axis_sizes, keyed by UID.
    """
    inst = SStInstance()
    dom_axes = m.dom()
    for ax in dom_axes:
        inst.axis_sizes[ax.uid] = ax.local_size()
    for cod_ax, coeffs in m.cod_strides:
        inst.axis_sizes[cod_ax.uid] = cod_ax.local_size()
        for dom_ax, coeff in zip(dom_axes, coeffs):
            inst.entries.append(EntryRow(
                src=dom_ax.uid, tgt=cod_ax.uid, coeff=coeff
            ))
    return inst


def _add_equation(
    inst: SBrInstance,
    eq_idx: int,
    eq: TensorEquation,
    datatypes: dict[fd.DynamicName, bc.Datatype],
) -> None:
    """Append all rows for one equation into inst, tagged with eq_idx."""
    retained = eq.retained_uids()
    op = _operator_fields(eq.operator)
    out_datatype_tag, out_max_value = _dt_fields(datatypes.get(eq.lhs_name))

    inst.equations.append(EquationRow(equation_idx=eq_idx, lhs_name=eq.lhs_name))
    inst.arrays.append(ArrayRow(
        equation_idx=eq_idx,
        slot=0,
        name=eq.lhs_name,
        is_input=False,
        operator_tag=op.tag,
        norm_axis=next((ax.uid for ax in eq.lhs_indices if isinstance(ax, NormAxis)), None),
        datatype_tag=out_datatype_tag,
        max_value=out_max_value,
        bias=op.bias,
        elementwise_fn=op.elementwise_fn,
        iverson_expr=op.iverson_expr,
    ))
    for pos, ax in enumerate(eq.lhs_indices):
        inst.axis_sizes[ax.uid] = ax.local_size()
        inst.array_axes.append(ArrayAxisRow(
            equation_idx=eq_idx, array_slot=0, axis_uid=ax.uid, is_target=False, position=pos
        ))

    for rhs_slot, factor in enumerate(eq.rhs, start=1):
        if isinstance(factor, TensorRef):
            tensor_name = factor.name
            input_axes = factor.axes
            in_datatype_tag, in_max_value = _dt_fields(datatypes.get(tensor_name))
            inst.arrays.append(ArrayRow(
                equation_idx=eq_idx,
                slot=rhs_slot,
                name=tensor_name,
                is_input=True,
                datatype_tag=in_datatype_tag,
                max_value=in_max_value,
            ))
        else:
            # Iverson factor: Bool-typed, no tensor name
            input_axes = _factor_axes(factor)
            inst.arrays.append(ArrayRow(
                equation_idx=eq_idx,
                slot=rhs_slot,
                name=None,
                is_input=True,
                datatype_tag=DataTag.BOOL,
                iverson_expr=_serialize_iverson(factor),
            ))
        for pos, ax in enumerate(input_axes):
            inst.axis_sizes[ax.uid] = ax.local_size()
            inst.array_axes.append(ArrayAxisRow(
                equation_idx=eq_idx,
                array_slot=rhs_slot,
                axis_uid=ax.uid,
                is_target=ax.uid not in retained,
                position=pos,
            ))
        for ax in input_axes:
            if ax.uid in retained:
                # src_uid == tgt_uid: TensorEquation retained axes are the same
                # objects on lhs and rhs, so sampling is always an identity map.
                inst.samples.append(SampleRow(
                    equation_idx=eq_idx,
                    reindexing_slot=rhs_slot,
                    src_uid=ax.uid,
                    tgt_uid=ax.uid,
                    coeff=nm.Integer(1),
                ))


def from_tensor_equation(
    eq: TensorEquation,
    array_datatypes: dict[fd.DynamicName, bc.Datatype] | None = None,
) -> SBrInstance:
    """Convert one TensorEquation to an SBrInstance.

    Retained indices (lhs_indices) become degree/TILED axes (is_target=False).
    Contracted indices (in rhs but not lhs_indices) become target axes (is_target=True).
    One SampleRow per (input_slot, retained_axis) pair, all with coeff=Integer(1).

    array_datatypes: optional mapping from tensor name to bc.Datatype; when supplied,
    populates datatype_tag and max_value on ArrayRow. Keyed by name rather than slot,
    so a self-join (the same tensor appearing in multiple rhs slots) will receive the
    same datatype at every slot — which is correct, since a tensor has one datatype.

    This is the intended path for equations with mixed datatypes (e.g. Natural-valued
    embedding indices alongside Reals-valued weights). Each ArrayRow carries its own
    datatype_tag independently. For homogeneously-typed structural reasoning, use
    TensorEquation.bc_signature() instead, which produces a Broadcasted parameterised
    by a single datatype.
    """
    inst = SBrInstance()
    _add_equation(inst, 0, eq, array_datatypes or {})
    return inst


def from_tensor_program(
    prog: TensorProgram,
    array_datatypes: dict[fd.DynamicName, bc.Datatype] | None = None,
) -> SBrInstance:
    """Convert a TensorProgram to a single SBrInstance.

    Equations are processed in topological order and assigned equation_idx values
    matching their position in that order. Shared axes are identified by UID across
    all equations in the single axis_sizes dict.
    array_datatypes is applied to every equation.
    """
    inst = SBrInstance()
    datatypes = array_datatypes or {}
    for eq_idx, eq in enumerate(topological_sort(prog.equations)):
        _add_equation(inst, eq_idx, eq, datatypes)
    return inst
