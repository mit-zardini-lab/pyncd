"""Tests for acset.convert: from_stride_morphism, from_tensor_equation, from_tensor_program."""

import data_structure.Term as fd
import data_structure.BroadcastedCategory as bc
from data_structure.StrideCategory import RawAxis, StrideMorphism
from data_structure.Numeric import Integer
from data_structure.TensorLogic import TensorEquation, TensorProgram
from data_structure.TensorExpr import TensorRef, IversonBinOp, _factor_axes
import data_structure.Operators as ops
from data_structure.Operators import (
    Identity, SoftMax, Linear, Elementwise,
    Normalize, Embedding, AdditionOp, WeightedTriangularLower,
)
from data_structure.TensorDSL import NormAxis
from acset.convert import from_stride_morphism, from_tensor_equation, from_tensor_program
from acset.instances import SStInstance, SBrInstance, EquationRow, OpTag, DataTag


def _identity_morphism():
    """Identity on one axis: (p,) → (p,) with coeff=1."""
    return StrideMorphism.from_matrix(
        (1,), dom_names=('p',), cod_names=('p',)
    )

def _duplication_morphism():
    """Duplication: (p,) → (p0, p1) each coeff=1."""
    return StrideMorphism.from_matrix(
        (1,), (1,), dom_names=('p',), cod_names=('p0', 'p1')
    )

def _conv_shift_morphism():
    """Convolution shift: (x', w) → (x,) coeffs=(1,1)."""
    return StrideMorphism.from_matrix(
        (1, 1), dom_names=("x'", 'w'), cod_names=('x',), name='+'
    )


def test_from_stride_morphism_returns_sstinstance():
    """from_stride_morphism returns an SStInstance."""
    result = from_stride_morphism(_identity_morphism())
    assert isinstance(result, SStInstance)


def test_identity_morphism():
    """from_matrix creates fresh dom/cod axes; axis_sizes has 2 entries, entries has 1 row."""
    m = _identity_morphism()
    inst = from_stride_morphism(m)
    assert len(inst.axis_sizes) == 2
    assert len(inst.entries) == 1
    assert inst.entries[0].coeff == Integer(1)


def test_duplication_has_three_axes():
    """dom: p; cod: p0, p1 — three distinct axes, two entry rows."""
    inst = from_stride_morphism(_duplication_morphism())
    assert len(inst.axis_sizes) == 3
    assert len(inst.entries) == 2
    for e in inst.entries:
        assert e.coeff == Integer(1)


def test_conv_shift_entries():
    """(x', w) → (x,): two entries, both coeff=1."""
    inst = from_stride_morphism(_conv_shift_morphism())
    assert len(inst.entries) == 2
    assert all(e.coeff == Integer(1) for e in inst.entries)


def test_stride_morphism_non_unit_coeff():
    """Stride-2 mapping p → n produces an EntryRow with coeff=Integer(2)."""
    m = StrideMorphism.from_matrix((2,), dom_names=('p',), cod_names=('n',))
    inst = from_stride_morphism(m)
    assert len(inst.entries) == 1
    assert inst.entries[0].coeff == Integer(2)


# ── from_tensor_equation ────────────────────────────────────────────────────

def _matmul_eq():
    """Y[i,j] = W[i,k] X[k,j]"""
    i = RawAxis.named('i')
    j = RawAxis.named('j')
    k = RawAxis.named('k')
    eq = TensorEquation(
        lhs_name=fd.DynamicName('Y'),
        lhs_indices=(i, j),
        rhs=(
            TensorRef(fd.DynamicName('W'), (i, k)),
            TensorRef(fd.DynamicName('X'), (k, j)),
        ),
        operator=Identity(),
    )
    return eq, i, j, k


def test_from_tensor_equation_returns_sbr_instance():
    """from_tensor_equation returns an SBrInstance."""
    eq, *_ = _matmul_eq()
    assert isinstance(from_tensor_equation(eq), SBrInstance)


def test_matmul_has_three_arrays():
    """Matmul equation produces three Array rows (Y, W, X)."""
    eq, *_ = _matmul_eq()
    inst = from_tensor_equation(eq)
    assert len(inst.arrays) == 3


def test_matmul_output_array_not_input():
    """Output array Y is is_input=False with Identity operator_tag."""
    eq, *_ = _matmul_eq()
    inst = from_tensor_equation(eq)
    output = next(a for a in inst.arrays if not a.is_input)
    assert output.name == fd.DynamicName('Y')
    assert output.operator_tag == OpTag.IDENTITY


def test_matmul_contracted_axis_is_target():
    """Contracted axis k appears in two ArrayAxis rows, both is_target=True."""
    eq, _i, _j, k = _matmul_eq()
    inst = from_tensor_equation(eq)
    k_rows = [r for r in inst.array_axes if r.axis_uid == k.uid]
    assert len(k_rows) == 2
    assert all(r.is_target for r in k_rows)


def test_matmul_retained_axes_not_target():
    """Retained axis i appears only in is_target=False rows."""
    eq, i, _j, _k = _matmul_eq()
    inst = from_tensor_equation(eq)
    i_rows = [r for r in inst.array_axes if r.axis_uid == i.uid]
    assert all(not r.is_target for r in i_rows)


def test_matmul_samples_count():
    """W contributes i → one sample; X contributes j → one sample."""
    eq, *_ = _matmul_eq()
    inst = from_tensor_equation(eq)
    assert len(inst.samples) == 2


def test_matmul_samples_have_unit_coeff():
    """All TensorEquation samples have coeff=1 (identity reindexing)."""
    eq, *_ = _matmul_eq()
    inst = from_tensor_equation(eq)
    assert all(s.coeff == Integer(1) for s in inst.samples)


def test_matmul_axis_sizes_contains_all_axes():
    """axis_sizes contains UIDs for all three axes i, j, k."""
    eq, i, j, k = _matmul_eq()
    inst = from_tensor_equation(eq)
    assert i.uid in inst.axis_sizes
    assert j.uid in inst.axis_sizes
    assert k.uid in inst.axis_sizes


def test_matmul_output_positions():
    """Y[i,j]: lhs_indices=(i,j), so ya_i→position 0, ya_j→position 1."""
    eq, i, j, _k = _matmul_eq()
    inst = from_tensor_equation(eq)
    output_rows = {r.axis_uid: r.position for r in inst.array_axes if r.array_slot == 0}
    assert output_rows[i.uid] == 0
    assert output_rows[j.uid] == 1


def test_matmul_input_positions():
    """W[i,k]: i→0, k→1 (slot 1); X[k,j]: k→0, j→1 (slot 2)."""
    eq, i, j, k = _matmul_eq()
    inst = from_tensor_equation(eq)
    w_rows = {r.axis_uid: r.position for r in inst.array_axes if r.array_slot == 1}
    x_rows = {r.axis_uid: r.position for r in inst.array_axes if r.array_slot == 2}
    assert w_rows[i.uid] == 0
    assert w_rows[k.uid] == 1
    assert x_rows[k.uid] == 0
    assert x_rows[j.uid] == 1


def test_softmax_operator_tag():
    """SoftMax operator maps to OpTag.SOFTMAX; norm_axis points to the NormAxis UID."""
    i = NormAxis.named('i')
    eq = TensorEquation(
        lhs_name=fd.DynamicName('Y'),
        lhs_indices=(i,),
        rhs=(TensorRef(fd.DynamicName('X'), (i,)),),
        operator=SoftMax(),
    )
    inst = from_tensor_equation(eq)
    output = next(a for a in inst.arrays if not a.is_input)
    assert output.operator_tag == OpTag.SOFTMAX
    assert output.norm_axis == i.uid


def test_norm_axis_is_none_for_regular_output():
    """Output arrays with no NormAxis in lhs_indices have norm_axis=None."""
    eq, *_ = _matmul_eq()
    inst = from_tensor_equation(eq)
    output = next(a for a in inst.arrays if not a.is_input)
    assert output.norm_axis is None


def test_normalize_operator_tag():
    """Normalize operator maps to OpTag.NORMALIZE."""
    i = RawAxis.named('i')
    eq = TensorEquation(
        lhs_name=fd.DynamicName('Y'),
        lhs_indices=(i,),
        rhs=(TensorRef(fd.DynamicName('X'), (i,)),),
        operator=Normalize(),
    )
    inst = from_tensor_equation(eq)
    output = next(a for a in inst.arrays if not a.is_input)
    assert output.operator_tag == OpTag.NORMALIZE


def test_embedding_operator_tag():
    """Embedding operator maps to OpTag.EMBEDDING."""
    i = RawAxis.named('i')
    eq = TensorEquation(
        lhs_name=fd.DynamicName('Y'),
        lhs_indices=(i,),
        rhs=(TensorRef(fd.DynamicName('E'), (i,)),),
        operator=Embedding(),
    )
    inst = from_tensor_equation(eq)
    output = next(a for a in inst.arrays if not a.is_input)
    assert output.operator_tag == OpTag.EMBEDDING


def test_addition_op_operator_tag():
    """AdditionOp operator maps to OpTag.ADDITION_OP."""
    i = RawAxis.named('i')
    eq = TensorEquation(
        lhs_name=fd.DynamicName('Y'),
        lhs_indices=(i,),
        rhs=(
            TensorRef(fd.DynamicName('X'), (i,)),
            TensorRef(fd.DynamicName('Z'), (i,)),
        ),
        operator=AdditionOp(),
    )
    inst = from_tensor_equation(eq)
    output = next(a for a in inst.arrays if not a.is_input)
    assert output.operator_tag == OpTag.ADDITION_OP


def test_weighted_triangular_lower_operator_tag():
    """WeightedTriangularLower operator maps to OpTag.WEIGHTED_TRIANGULAR_LOWER."""
    i = RawAxis.named('i')
    eq = TensorEquation(
        lhs_name=fd.DynamicName('Y'),
        lhs_indices=(i,),
        rhs=(TensorRef(fd.DynamicName('X'), (i,)),),
        operator=WeightedTriangularLower(),
    )
    inst = from_tensor_equation(eq)
    output = next(a for a in inst.arrays if not a.is_input)
    assert output.operator_tag == OpTag.WEIGHTED_TRIANGULAR_LOWER


# ── from_tensor_program ─────────────────────────────────────────────────────

def _two_equation_program():
    """
    H[i, k] = W1[i, j] X[j, k]
    Y[i, k] = W2[i, m] H[m, k]
    """
    i  = RawAxis.named('i')
    j  = RawAxis.named('j')
    k  = RawAxis.named('k')
    m  = RawAxis.named('m')
    eq1 = TensorEquation(
        lhs_name=fd.DynamicName('H'),
        lhs_indices=(i, k),
        rhs=(
            TensorRef(fd.DynamicName('W1'), (i, j)),
            TensorRef(fd.DynamicName('X'),  (j, k)),
        ),
        operator=Identity(),
    )
    eq2 = TensorEquation(
        lhs_name=fd.DynamicName('Y'),
        lhs_indices=(i, k),
        rhs=(
            TensorRef(fd.DynamicName('W2'), (i, m)),
            TensorRef(fd.DynamicName('H'),  (m, k)),
        ),
        operator=Identity(),
    )
    return TensorProgram(equations=(eq1, eq2)), i, j, k, m


def test_from_tensor_program_returns_sbr_instance():
    """from_tensor_program returns a single SBrInstance."""
    prog, *_ = _two_equation_program()
    result = from_tensor_program(prog)
    assert isinstance(result, SBrInstance)


def test_from_tensor_program_has_two_equations():
    """Two-equation program produces an SBrInstance with two EquationRows."""
    prog, *_ = _two_equation_program()
    inst = from_tensor_program(prog)
    assert len(inst.equations) == 2
    assert all(isinstance(r, EquationRow) for r in inst.equations)


def test_from_tensor_program_each_has_three_arrays():
    """Each equation in the two-equation program has three Array rows."""
    prog, *_ = _two_equation_program()
    inst = from_tensor_program(prog)
    for eq_row in inst.equations:
        arrays = [a for a in inst.arrays if a.equation_idx == eq_row.equation_idx]
        assert len(arrays) == 3


def test_from_tensor_program_shared_axis_same_uid():
    """Axis i appears in the single inst.axis_sizes under the same UID."""
    prog, i, *_ = _two_equation_program()
    inst = from_tensor_program(prog)
    assert i.uid in inst.axis_sizes


def test_from_tensor_program_does_not_unify_fresh_axes():
    """from_tensor_program preserves distinct UIDs; fresh axes in eq2 are not merged with eq1's."""
    i1 = RawAxis.named('i')
    j1 = RawAxis.named('j')
    k1 = RawAxis.named('k')
    eq1 = TensorEquation(
        lhs_name=fd.DynamicName('H'),
        lhs_indices=(i1, k1),
        rhs=(
            TensorRef(fd.DynamicName('W1'), (i1, j1)),
            TensorRef(fd.DynamicName('X'), (j1, k1)),
        ),
        operator=Identity(),
    )
    i2 = RawAxis.named('i')
    k2 = RawAxis.named('k')
    m2 = RawAxis.named('m')
    eq2 = TensorEquation(
        lhs_name=fd.DynamicName('Y'),
        lhs_indices=(i2, k2),
        rhs=(
            TensorRef(fd.DynamicName('W2'), (i2, m2)),
            TensorRef(fd.DynamicName('H'), (m2, k2)),
        ),
        operator=Identity(),
    )
    inst = from_tensor_program(TensorProgram(equations=(eq1, eq2)))
    # i2 and i1 are distinct objects — from_tensor_program never calls ctx.apply(),
    # so their UIDs are not merged (contrast with to_morphism() which does merge them).
    assert i1.uid != i2.uid
    assert i1.uid in inst.axis_sizes
    assert i2.uid in inst.axis_sizes


# ── array_datatypes parameter ────────────────────────────────────────────────

def _embedding_eq():
    """Y[i] = E[i] where E has a Natural (vocabulary) input."""
    i = RawAxis.named('i')
    return TensorEquation(
        lhs_name=fd.DynamicName('Y'),
        lhs_indices=(i,),
        rhs=(TensorRef(fd.DynamicName('E'), (i,)),),
        operator=Identity(),
    ), i


def test_default_datatype_tag_is_reals():
    """Without array_datatypes, all arrays default to DataTag.REALS."""
    eq, *_ = _matmul_eq()
    inst = from_tensor_equation(eq)
    assert all(a.datatype_tag == DataTag.REALS for a in inst.arrays)


def test_natural_datatype_tag():
    """Supplying a Natural datatype sets DataTag.NATURAL and max_value on that array."""
    eq, _i = _embedding_eq()
    vocab_size = Integer(32000)
    datatypes = {fd.DynamicName('E'): bc.Natural(max_value=vocab_size)}
    inst = from_tensor_equation(eq, array_datatypes=datatypes)
    e_row = next(a for a in inst.arrays if a.name == fd.DynamicName('E'))
    assert e_row.datatype_tag == DataTag.NATURAL
    assert e_row.max_value == vocab_size


def test_output_datatype_tag_from_caller():
    """Output array datatype_tag is set when output name is in array_datatypes."""
    eq, _i = _embedding_eq()
    datatypes = {fd.DynamicName('Y'): bc.Reals()}
    inst = from_tensor_equation(eq, array_datatypes=datatypes)
    y_row = next(a for a in inst.arrays if a.name == fd.DynamicName('Y'))
    assert y_row.datatype_tag == DataTag.REALS


def test_linear_bias_true():
    """Linear(bias=True) operator sets bias=True on output ArrayRow."""
    i = RawAxis.named('i')
    j = RawAxis.named('j')
    eq = TensorEquation(
        lhs_name=fd.DynamicName('Y'),
        lhs_indices=(i,),
        rhs=(TensorRef(fd.DynamicName('W'), (i, j)),),
        operator=Linear(bias=True),
    )
    inst = from_tensor_equation(eq)
    output = next(a for a in inst.arrays if not a.is_input)
    assert output.bias is True


def test_linear_bias_false():
    """Linear(bias=False) operator sets bias=False on output ArrayRow."""
    i = RawAxis.named('i')
    j = RawAxis.named('j')
    eq = TensorEquation(
        lhs_name=fd.DynamicName('Y'),
        lhs_indices=(i,),
        rhs=(TensorRef(fd.DynamicName('W'), (i, j)),),
        operator=Linear(bias=False),
    )
    inst = from_tensor_equation(eq)
    output = next(a for a in inst.arrays if not a.is_input)
    assert output.bias is False


def test_elementwise_fn_stored():
    """Elementwise operator stores the function name in elementwise_fn."""
    i = RawAxis.named('i')
    eq = TensorEquation(
        lhs_name=fd.DynamicName('Y'),
        lhs_indices=(i,),
        rhs=(TensorRef(fd.DynamicName('X'), (i,)),),
        operator=Elementwise(operator='relu'),
    )
    inst = from_tensor_equation(eq)
    output = next(a for a in inst.arrays if not a.is_input)
    assert output.elementwise_fn == 'relu'


def test_identity_has_no_elementwise_fn():
    """Identity (subclass of Elementwise) does not set elementwise_fn."""
    eq, *_ = _matmul_eq()
    inst = from_tensor_equation(eq)
    output = next(a for a in inst.arrays if not a.is_input)
    assert output.elementwise_fn is None


def test_non_natural_arrays_have_no_max_value():
    """Arrays without a Natural datatype have max_value=None."""
    eq, *_ = _matmul_eq()
    inst = from_tensor_equation(eq)
    assert all(a.max_value is None for a in inst.arrays)


# ── slot indexing ────────────────────────────────────────────────────────────

def test_output_array_has_slot_zero():
    """Output array is always assigned slot=0."""
    eq, *_ = _matmul_eq()
    inst = from_tensor_equation(eq)
    output = next(a for a in inst.arrays if not a.is_input)
    assert output.slot == 0


def test_input_arrays_have_sequential_slots():
    """rhs inputs are assigned slots 1, 2, ... in rhs order."""
    eq, *_ = _matmul_eq()
    inst = from_tensor_equation(eq)
    inputs = [a for a in inst.arrays if a.is_input]
    assert sorted(a.slot for a in inputs) == [1, 2]


def test_array_axes_reference_by_slot():
    """ArrayAxisRow.array_slot is an integer, not a name."""
    eq, *_ = _matmul_eq()
    inst = from_tensor_equation(eq)
    assert all(isinstance(r.array_slot, int) for r in inst.array_axes)


def test_samples_reference_by_slot():
    """SampleRow.reindexing_slot is an integer matching the input array's slot."""
    eq, *_ = _matmul_eq()
    inst = from_tensor_equation(eq)
    input_slots = {a.slot for a in inst.arrays if a.is_input}
    assert all(s.reindexing_slot in input_slots for s in inst.samples)


# ── self-join ────────────────────────────────────────────────────────────────

def _gram_eq():
    """Y[i,j] = H[i,k] H[j,k] — H appears twice in the rhs."""
    i = RawAxis.named('i')
    j = RawAxis.named('j')
    k = RawAxis.named('k')
    h_name = fd.DynamicName('H')
    eq = TensorEquation(
        lhs_name=fd.DynamicName('Y'),
        lhs_indices=(i, j),
        rhs=(
            TensorRef(h_name, (i, k)),
            TensorRef(h_name, (j, k)),
        ),
        operator=Identity(),
    )
    return eq, i, j, k


def test_self_join_produces_three_arrays():
    """Y + two H references = three Array rows, all with distinct slots."""
    eq, *_ = _gram_eq()
    inst = from_tensor_equation(eq)
    assert len(inst.arrays) == 3
    assert len({a.slot for a in inst.arrays}) == 3


def test_self_join_both_inputs_share_name():
    """Both input Array rows carry name='H' but have different slots."""
    eq, *_ = _gram_eq()
    inst = from_tensor_equation(eq)
    inputs = [a for a in inst.arrays if a.is_input]
    assert all(a.name == fd.DynamicName('H') for a in inputs)
    assert inputs[0].slot != inputs[1].slot


def test_self_join_degree_axes_are_in_separate_slots():
    """The two retained axes i and j are assigned to different array_slot buckets."""
    eq, i, j, _k = _gram_eq()
    inst = from_tensor_equation(eq)
    inputs = [a for a in inst.arrays if a.is_input]
    slot1, slot2 = inputs[0].slot, inputs[1].slot
    uids_slot1 = {r.axis_uid for r in inst.array_axes if r.array_slot == slot1}
    uids_slot2 = {r.axis_uid for r in inst.array_axes if r.array_slot == slot2}
    assert i.uid in uids_slot1 and j.uid not in uids_slot1
    assert j.uid in uids_slot2 and i.uid not in uids_slot2


def test_self_join_each_h_reference_gets_own_slot_in_array_axes():
    """ArrayAxis rows for the two H inputs reference different array_slots."""
    eq, i, j, k = _gram_eq()
    inst = from_tensor_equation(eq)
    inputs = [a for a in inst.arrays if a.is_input]
    slot1, slot2 = inputs[0].slot, inputs[1].slot
    axes_slot1 = {r.axis_uid for r in inst.array_axes if r.array_slot == slot1}
    axes_slot2 = {r.axis_uid for r in inst.array_axes if r.array_slot == slot2}
    # slot1 references i and k; slot2 references j and k
    assert i.uid in axes_slot1 and k.uid in axes_slot1
    assert j.uid in axes_slot2 and k.uid in axes_slot2


def test_self_join_samples_point_to_distinct_slots():
    """The two Sample rows point to different reindexing_slots (one per H reference)."""
    eq, *_ = _gram_eq()
    inst = from_tensor_equation(eq)
    assert len(inst.samples) == 2
    assert inst.samples[0].reindexing_slot != inst.samples[1].reindexing_slot


# ── Bool semiring ────────────────────────────────────────────────────────────

def test_bool_datatype_tag():
    """Supplying bc.Bool() sets DataTag.BOOL and no max_value on that array."""
    eq, _i = _embedding_eq()
    datatypes = {fd.DynamicName('E'): bc.Bool()}
    inst = from_tensor_equation(eq, array_datatypes=datatypes)
    e_row = next(a for a in inst.arrays if a.name == fd.DynamicName('E'))
    assert e_row.datatype_tag == DataTag.BOOL
    assert e_row.max_value is None


# ── Iverson factors ──────────────────────────────────────────────────────────

def test_iverson_factor_produces_bool_array_row():
    """An IversonBinOp in the rhs produces an ArrayRow with DataTag.BOOL."""
    q = RawAxis.named('q')
    x = RawAxis.named('x')
    pred = q < x  # IversonBinOp('<', q, x) via monkey-patched __lt__
    eq = TensorEquation(
        lhs_name=fd.DynamicName('Out'),
        lhs_indices=(q, x),
        rhs=(
            TensorRef(fd.DynamicName('Score'), (q, x)),
            pred,
        ),
        operator=Identity(),
    )
    inst = from_tensor_equation(eq)
    # slot 2 is the Iverson factor
    iverson_array = next(a for a in inst.arrays if a.slot == 2)
    assert iverson_array.datatype_tag == DataTag.BOOL
    assert iverson_array.name is None


def test_iverson_factor_serializes_expr():
    """An Iverson factor's ArrayRow stores a non-empty iverson_expr string."""
    q = RawAxis.named('q')
    x = RawAxis.named('x')
    pred = q < x
    eq = TensorEquation(
        lhs_name=fd.DynamicName('Out'),
        lhs_indices=(q, x),
        rhs=(
            TensorRef(fd.DynamicName('Score'), (q, x)),
            pred,
        ),
        operator=Identity(),
    )
    inst = from_tensor_equation(eq)
    iverson_array = next(a for a in inst.arrays if a.slot == 2)
    assert iverson_array.iverson_expr is not None
    assert len(iverson_array.iverson_expr) > 0


def test_iverson_factor_axes_are_registered():
    """Axes from an Iverson factor appear in axis_sizes."""
    q = RawAxis.named('q')
    x = RawAxis.named('x')
    pred = q < x
    eq = TensorEquation(
        lhs_name=fd.DynamicName('Out'),
        lhs_indices=(q, x),
        rhs=(pred,),
        operator=Identity(),
    )
    inst = from_tensor_equation(eq)
    assert q.uid in inst.axis_sizes
    assert x.uid in inst.axis_sizes


def test_masked_softmax_operator_tag():
    """softmax(where=pred) must produce MASKED_SOFTMAX operator_tag on the output array."""
    from data_structure.TensorDSL import norm_axis, real_axis, softmax
    q = real_axis('q', 4)
    x = norm_axis('x', 4)
    eq = TensorEquation(
        lhs_name=fd.DynamicName('S'),
        lhs_indices=(x, q),
        rhs=(
            TensorRef(fd.DynamicName('Q'), (q,)),
            TensorRef(fd.DynamicName('K'), (x,)),
        ),
        operator=ops.SoftMax(where_predicate=(x <= q,)),
    )
    inst = from_tensor_equation(eq)
    out = next(a for a in inst.arrays if not a.is_input)
    assert out.operator_tag == OpTag.MASKED_SOFTMAX


def test_masked_softmax_iverson_expr_populated():
    """The output array for MASKED_SOFTMAX must have iverson_expr set to the serialized predicate."""
    from data_structure.TensorDSL import norm_axis, real_axis
    q = real_axis('q', 4)
    x = norm_axis('x', 4)
    pred = x <= q   # IversonBinOp('<=', x, q)
    eq = TensorEquation(
        lhs_name=fd.DynamicName('S'),
        lhs_indices=(x, q),
        rhs=(TensorRef(fd.DynamicName('Score'), (x, q)),),
        operator=ops.SoftMax(where_predicate=(pred,)),
    )
    inst = from_tensor_equation(eq)
    out = next(a for a in inst.arrays if not a.is_input)
    assert out.iverson_expr is not None
    assert '<=' in out.iverson_expr


def test_plain_softmax_is_still_softmax():
    """softmax() without where= must still produce SOFTMAX (not MASKED_SOFTMAX)."""
    q = RawAxis.named('q')
    x = RawAxis.named('x')
    eq = TensorEquation(
        lhs_name=fd.DynamicName('S'),
        lhs_indices=(x, q),
        rhs=(TensorRef(fd.DynamicName('Score'), (x, q)),),
        operator=ops.SoftMax(),
    )
    inst = from_tensor_equation(eq)
    out = next(a for a in inst.arrays if not a.is_input)
    assert out.operator_tag == OpTag.SOFTMAX
    assert out.iverson_expr is None


def test_masked_softmax_roundtrips_through_csv(tmp_path):
    """MASKED_SOFTMAX with iverson_expr round-trips through write_sbr/read_sbr."""
    from acset.csv_io import write_sbr, read_sbr
    from data_structure.TensorDSL import norm_axis, real_axis
    q = real_axis('q', 4)
    x = norm_axis('x', 4)
    pred = x <= q
    eq = TensorEquation(
        lhs_name=fd.DynamicName('S'),
        lhs_indices=(x, q),
        rhs=(TensorRef(fd.DynamicName('Score'), (x, q)),),
        operator=ops.SoftMax(where_predicate=(pred,)),
    )
    inst = from_tensor_equation(eq)
    write_sbr(inst, tmp_path)
    inst2 = read_sbr(tmp_path)
    out2 = next(a for a in inst2.arrays if not a.is_input)
    assert out2.operator_tag == OpTag.MASKED_SOFTMAX
    assert out2.iverson_expr is not None
    assert '<=' in out2.iverson_expr


def test_masked_normalize_serialises_to_masked_normalize_tag():
    """from_tensor_equation with normalize(where=...) produces OpTag.MASKED_NORMALIZE."""
    from data_structure.TensorDSL import TL, real_axis, norm_axis, normalize

    SEQ = 4
    p = real_axis('p', SEQ)
    m = norm_axis('m', SEQ)
    tl = TL()
    tl.Out[p, m] = normalize(tl.X[p, m], where=(m <= p))
    eq = tl._equations[0]
    inst = from_tensor_equation(eq)
    out_row = next(r for r in inst.arrays if not r.is_input)
    assert out_row.operator_tag == OpTag.MASKED_NORMALIZE
    assert out_row.iverson_expr is not None
