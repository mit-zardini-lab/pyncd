"""Round-trip CSV tests for acset.csv_io: write_sst/read_sst and write_sbr/read_sbr."""
import csv

import data_structure.Term as fd
import data_structure.Numeric as nm
from data_structure.StrideCategory import RawAxis, StrideMorphism
from data_structure.TensorLogic import TensorEquation, TensorProgram
from data_structure.TensorExpr import TensorRef
from data_structure.Operators import Linear, SoftMax
from data_structure.TensorDSL import NormAxis

from acset.convert import from_stride_morphism, from_tensor_equation, from_tensor_program
from acset.instances import (
    SStInstance, SBrInstance, EntryRow, EquationRow,
)
from acset.csv_io import write_sst, read_sst, write_sbr, read_sbr, _UID_NAME_BY_TYPE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numeric_stable_key(n: nm.Numeric):
    """Stable identity for round-trip comparison: Integer by value, FreeNumeric by _id."""
    if isinstance(n, nm.Integer):
        return ('int', n._value)
    if isinstance(n, nm.FreeNumeric):
        return ('free', n.uid._id)
    raise TypeError(type(n))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _identity_sst():
    """Identity stride morphism (p→q): one entry, two named axes (FreeNumeric sizes)."""
    return from_stride_morphism(
        StrideMorphism.from_matrix((1,), dom_names=('p',), cod_names=('q',))
    )


def _duplication_sst():
    """Duplication morphism (p→p0,p1): two entries, three axes."""
    return from_stride_morphism(
        StrideMorphism.from_matrix(
            (1,), (1,), dom_names=('p',), cod_names=('p0', 'p1')
        )
    )


def _identity_sbr():
    """Y[p,d] = X[p,d] — simplest SBr: one equation, two arrays, two retained axes."""
    p = RawAxis(_size=nm.Integer(4))
    d = RawAxis(_size=nm.Integer(8))
    eq = TensorEquation(
        lhs_name=fd.DynamicName.from_str('Y'),
        lhs_indices=[p, d],
        rhs=(TensorRef(fd.DynamicName.from_str('X'), (p, d)),),
    )
    return from_tensor_equation(eq)


def _linear_sbr():
    """H[p,d2] = Linear(X[p,d], W[d2,d]) — one equation with bias=False."""
    p  = RawAxis(_size=nm.Integer(4))
    d  = RawAxis(_size=nm.Integer(8))
    d2 = RawAxis(_size=nm.Integer(16))
    eq = TensorEquation(
        lhs_name=fd.DynamicName.from_str('H'),
        lhs_indices=[p, d2],
        rhs=(
            TensorRef(fd.DynamicName.from_str('X'), (p, d)),
            TensorRef(fd.DynamicName.from_str('W'), (d2, d)),
        ),
        operator=Linear(bias=False),
    )
    return from_tensor_equation(eq)


def _two_equation_sbr():
    """Two-layer MLP: H = Linear(X, W1), Y = Linear(H, W2, bias=True)."""
    p  = RawAxis(_size=nm.Integer(4))
    d  = RawAxis(_size=nm.Integer(8))
    d2 = RawAxis(_size=nm.Integer(16))
    eq1 = TensorEquation(
        lhs_name=fd.DynamicName.from_str('H'),
        lhs_indices=[p, d2],
        rhs=(
            TensorRef(fd.DynamicName.from_str('X'),  (p, d)),
            TensorRef(fd.DynamicName.from_str('W1'), (d2, d)),
        ),
        operator=Linear(bias=False),
    )
    eq2 = TensorEquation(
        lhs_name=fd.DynamicName.from_str('Y'),
        lhs_indices=[p, d],
        rhs=(
            TensorRef(fd.DynamicName.from_str('H'),  (p, d2)),
            TensorRef(fd.DynamicName.from_str('W2'), (d, d2)),
        ),
        operator=Linear(bias=True),
    )
    return from_tensor_program(TensorProgram(equations=(eq1, eq2)))


# ---------------------------------------------------------------------------
# SSt write tests
# ---------------------------------------------------------------------------

def test_write_sst_creates_axis_sizes_csv(tmp_path):
    write_sst(_identity_sst(), tmp_path)
    assert (tmp_path / 'axis_sizes.csv').exists()


def test_write_sst_creates_entries_csv(tmp_path):
    write_sst(_identity_sst(), tmp_path)
    assert (tmp_path / 'entries.csv').exists()


def test_write_sst_axis_sizes_headers(tmp_path):
    write_sst(_identity_sst(), tmp_path)
    with open(tmp_path / 'axis_sizes.csv') as f:
        assert set(csv.DictReader(f).fieldnames) == {'axis_uid', 'size'}


def test_write_sst_entries_headers(tmp_path):
    write_sst(_identity_sst(), tmp_path)
    with open(tmp_path / 'entries.csv') as f:
        assert set(csv.DictReader(f).fieldnames) == {'src_uid', 'tgt_uid', 'coeff'}


def test_write_sst_axis_sizes_row_count(tmp_path):
    inst = _identity_sst()
    write_sst(inst, tmp_path)
    rows = list(csv.DictReader(open(tmp_path / 'axis_sizes.csv')))
    assert len(rows) == len(inst.axis_sizes)


def test_write_sst_entries_row_count(tmp_path):
    inst = _duplication_sst()
    write_sst(inst, tmp_path)
    rows = list(csv.DictReader(open(tmp_path / 'entries.csv')))
    assert len(rows) == len(inst.entries)


# ---------------------------------------------------------------------------
# SSt round-trip tests
# ---------------------------------------------------------------------------

def test_sst_roundtrip_axis_uid_set(tmp_path):
    inst = _identity_sst()
    write_sst(inst, tmp_path)
    rt = read_sst(tmp_path)
    assert {u._id for u in inst.axis_sizes} == {u._id for u in rt.axis_sizes}


def test_sst_roundtrip_axis_sizes_by_id(tmp_path):
    """Axis sizes round-trip correctly by _id (FreeNumeric _name is intentionally lost)."""
    inst = _identity_sst()
    write_sst(inst, tmp_path)
    rt = read_sst(tmp_path)
    orig = {u._id: _numeric_stable_key(v) for u, v in inst.axis_sizes.items()}
    got  = {u._id: _numeric_stable_key(v) for u, v in rt.axis_sizes.items()}
    assert orig == got


def test_sst_roundtrip_entries_count(tmp_path):
    inst = _duplication_sst()
    write_sst(inst, tmp_path)
    rt = read_sst(tmp_path)
    assert len(rt.entries) == len(inst.entries)


def test_sst_roundtrip_entry_src_uids(tmp_path):
    inst = _identity_sst()
    write_sst(inst, tmp_path)
    rt = read_sst(tmp_path)
    assert {e.src._id for e in inst.entries} == {e.src._id for e in rt.entries}


def test_sst_roundtrip_entry_tgt_uids(tmp_path):
    inst = _identity_sst()
    write_sst(inst, tmp_path)
    rt = read_sst(tmp_path)
    assert {e.tgt._id for e in inst.entries} == {e.tgt._id for e in rt.entries}


def test_sst_roundtrip_entry_coeffs(tmp_path):
    inst = _duplication_sst()
    write_sst(inst, tmp_path)
    rt = read_sst(tmp_path)
    orig = sorted(_numeric_stable_key(e.coeff) for e in inst.entries)
    got  = sorted(_numeric_stable_key(e.coeff) for e in rt.entries)
    assert orig == got


# ---------------------------------------------------------------------------
# SBr write tests
# ---------------------------------------------------------------------------

def test_write_sbr_creates_five_files(tmp_path):
    write_sbr(_identity_sbr(), tmp_path)
    for name in ['axis_sizes.csv', 'equations.csv', 'arrays.csv',
                 'array_axes.csv', 'samples.csv']:
        assert (tmp_path / name).exists(), f'{name} not created'


def test_write_sbr_equations_headers(tmp_path):
    write_sbr(_identity_sbr(), tmp_path)
    with open(tmp_path / 'equations.csv') as f:
        assert set(csv.DictReader(f).fieldnames) == {'equation_idx', 'lhs_name'}


def test_write_sbr_arrays_headers(tmp_path):
    write_sbr(_identity_sbr(), tmp_path)
    with open(tmp_path / 'arrays.csv') as f:
        assert set(csv.DictReader(f).fieldnames) == {
            'equation_idx', 'slot', 'name', 'is_input', 'operator_tag',
            'norm_axis', 'datatype_tag', 'max_value', 'bias', 'elementwise_fn',
            'op_predicate', 'wire_label',
        }


def test_write_sbr_array_axes_headers(tmp_path):
    write_sbr(_identity_sbr(), tmp_path)
    with open(tmp_path / 'array_axes.csv') as f:
        expected = {'equation_idx', 'array_slot', 'axis_uid', 'is_target', 'position'}
        assert set(csv.DictReader(f).fieldnames) == expected


def test_write_sbr_samples_headers(tmp_path):
    write_sbr(_identity_sbr(), tmp_path)
    with open(tmp_path / 'samples.csv') as f:
        assert set(csv.DictReader(f).fieldnames) == {
            'equation_idx', 'reindexing_slot', 'src_uid', 'tgt_uid', 'coeff', 'offset',
        }


def test_write_sbr_two_equations_row_count(tmp_path):
    inst = _two_equation_sbr()
    write_sbr(inst, tmp_path)
    rows = list(csv.DictReader(open(tmp_path / 'equations.csv')))
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# SBr round-trip tests
# ---------------------------------------------------------------------------

def test_sbr_roundtrip_equations_count(tmp_path):
    inst = _identity_sbr()
    write_sbr(inst, tmp_path)
    rt = read_sbr(tmp_path)
    assert len(rt.equations) == len(inst.equations)


def test_sbr_roundtrip_arrays_count(tmp_path):
    inst = _identity_sbr()
    write_sbr(inst, tmp_path)
    rt = read_sbr(tmp_path)
    assert len(rt.arrays) == len(inst.arrays)


def test_sbr_roundtrip_array_axes_count(tmp_path):
    inst = _identity_sbr()
    write_sbr(inst, tmp_path)
    rt = read_sbr(tmp_path)
    assert len(rt.array_axes) == len(inst.array_axes)


def test_sbr_roundtrip_samples_count(tmp_path):
    inst = _identity_sbr()
    write_sbr(inst, tmp_path)
    rt = read_sbr(tmp_path)
    assert len(rt.samples) == len(inst.samples)


def test_sbr_roundtrip_axis_uids_preserved(tmp_path):
    inst = _identity_sbr()
    write_sbr(inst, tmp_path)
    rt = read_sbr(tmp_path)
    assert {u._id for u in inst.axis_sizes} == {u._id for u in rt.axis_sizes}


def test_sbr_roundtrip_axis_sizes_integer(tmp_path):
    """Integer axis sizes round-trip exactly (no FreeNumeric naming issue)."""
    inst = _identity_sbr()
    write_sbr(inst, tmp_path)
    rt = read_sbr(tmp_path)
    orig = {u._id: _numeric_stable_key(v) for u, v in inst.axis_sizes.items()}
    got  = {u._id: _numeric_stable_key(v) for u, v in rt.axis_sizes.items()}
    assert orig == got


def test_sbr_roundtrip_equation_indices(tmp_path):
    inst = _two_equation_sbr()
    write_sbr(inst, tmp_path)
    rt = read_sbr(tmp_path)
    assert {r.equation_idx for r in rt.equations} == {r.equation_idx for r in inst.equations}


def test_sbr_roundtrip_is_input_flags(tmp_path):
    inst = _identity_sbr()
    write_sbr(inst, tmp_path)
    rt = read_sbr(tmp_path)
    assert sorted(a.is_input for a in rt.arrays) == sorted(a.is_input for a in inst.arrays)


def test_sbr_roundtrip_operator_tags(tmp_path):
    inst = _linear_sbr()
    write_sbr(inst, tmp_path)
    rt = read_sbr(tmp_path)
    orig = sorted((a.slot, str(a.operator_tag)) for a in inst.arrays if not a.is_input)
    got  = sorted((a.slot, str(a.operator_tag)) for a in rt.arrays  if not a.is_input)
    assert orig == got


def test_sbr_roundtrip_is_target_flags(tmp_path):
    inst = _identity_sbr()
    write_sbr(inst, tmp_path)
    rt = read_sbr(tmp_path)
    assert sorted(aa.is_target for aa in rt.array_axes) == sorted(aa.is_target for aa in inst.array_axes)


def test_sbr_roundtrip_sample_src_tgt(tmp_path):
    inst = _identity_sbr()
    write_sbr(inst, tmp_path)
    rt = read_sbr(tmp_path)
    orig = sorted((s.src_uid._id, s.tgt_uid._id) for s in inst.samples)
    got  = sorted((s.src_uid._id, s.tgt_uid._id) for s in rt.samples)
    assert orig == got


def test_sbr_roundtrip_sample_coeffs(tmp_path):
    inst = _identity_sbr()
    write_sbr(inst, tmp_path)
    rt = read_sbr(tmp_path)
    orig = sorted(_numeric_stable_key(s.coeff) for s in inst.samples)
    got  = sorted(_numeric_stable_key(s.coeff) for s in rt.samples)
    assert orig == got


def test_sbr_roundtrip_sample_offset_zero(tmp_path):
    """offset must survive a CSV round-trip:
    a lost or corrupted offset would silently shift reindexing into the wrong window."""
    inst = _identity_sbr()
    write_sbr(inst, tmp_path)
    rt = read_sbr(tmp_path)
    assert all(s.offset == nm.Integer(0) for s in rt.samples)


def test_sbr_roundtrip_linear_bias_false(tmp_path):
    inst = _linear_sbr()
    write_sbr(inst, tmp_path)
    rt = read_sbr(tmp_path)
    orig_biases = sorted(a.bias for a in inst.arrays if a.bias is not None)
    rt_biases   = sorted(a.bias for a in rt.arrays   if a.bias is not None)
    assert orig_biases == rt_biases


def test_sbr_roundtrip_linear_bias_true(tmp_path):
    inst = _two_equation_sbr()
    write_sbr(inst, tmp_path)
    rt = read_sbr(tmp_path)
    orig_biases = sorted(a.bias for a in inst.arrays if a.bias is not None)
    rt_biases   = sorted(a.bias for a in rt.arrays   if a.bias is not None)
    assert orig_biases == rt_biases


def test_sbr_roundtrip_datatype_tags(tmp_path):
    inst = _identity_sbr()
    write_sbr(inst, tmp_path)
    rt = read_sbr(tmp_path)
    orig = sorted(str(a.datatype_tag) for a in inst.arrays)
    got  = sorted(str(a.datatype_tag) for a in rt.arrays)
    assert orig == got


def test_sbr_roundtrip_two_equations_array_counts(tmp_path):
    inst = _two_equation_sbr()
    write_sbr(inst, tmp_path)
    rt = read_sbr(tmp_path)
    assert len(rt.arrays) == len(inst.arrays)
    assert len(rt.array_axes) == len(inst.array_axes)


def test_sbr_roundtrip_positions(tmp_path):
    """ArrayAxisRow.position values survive the round-trip."""
    inst = _linear_sbr()
    write_sbr(inst, tmp_path)
    rt = read_sbr(tmp_path)
    orig = sorted((aa.equation_idx, aa.array_slot, aa.position) for aa in inst.array_axes)
    got  = sorted((aa.equation_idx, aa.array_slot, aa.position) for aa in rt.array_axes)
    assert orig == got


def _softmax_sbr() -> SBrInstance:
    """Y[i] = SoftMax(X[i]) — contains a NormAxis so uid _type round-trip can be tested."""
    i = NormAxis(_size=nm.Integer(8))
    eq = TensorEquation(
        lhs_name=fd.DynamicName.from_str('Y'),
        lhs_indices=[i],
        rhs=(TensorRef(fd.DynamicName.from_str('X'), (i,)),),
        operator=SoftMax(),
    )
    return from_tensor_equation(eq)


def test_sbr_roundtrip_normaxis_type_in_axis_sizes(tmp_path):
    """NormAxis uid _type is preserved in axis_sizes keys after round-trip."""
    inst = _softmax_sbr()
    write_sbr(inst, tmp_path)
    rt = read_sbr(tmp_path)
    orig_types = {u._id: u._type for u in inst.axis_sizes}
    rt_types   = {u._id: u._type for u in rt.axis_sizes}
    assert orig_types == rt_types


def test_sbr_roundtrip_normaxis_type_in_norm_axis_field(tmp_path):
    """ArrayRow.norm_axis uid _type is NormAxis after round-trip."""
    inst = _softmax_sbr()
    write_sbr(inst, tmp_path)
    rt = read_sbr(tmp_path)
    output = next(a for a in rt.arrays if not a.is_input)
    assert output.norm_axis is not None
    assert output.norm_axis._type is NormAxis


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------

def _all_subclasses(cls: type) -> set[type]:
    result = set()
    for sub in cls.__subclasses__():
        result.add(sub)
        result.update(_all_subclasses(sub))
    return result


def test_free_numeric_roundtrip_equality(tmp_path):
    """FreeNumeric sizes compare equal before and after CSV round-trip.

    RawAxis.named() creates axes whose _size is a FreeNumeric with uid._name set.
    After CSV write/read the _name is lost (not serialized), so the deserialized
    FreeNumeric has uid._name=None. Equality must still hold because identity
    is uid._id, not the display name.
    """
    from data_structure.StrideCategory import RawAxis
    m = StrideMorphism.from_matrix((1,), dom_names=('p',), cod_names=('q',))
    inst = from_stride_morphism(m)
    write_sst(inst, tmp_path)
    rt = read_sst(tmp_path)

    original_sizes = {uid._id: size for uid, size in inst.axis_sizes.items()}
    for rt_uid, rt_size in rt.axis_sizes.items():
        orig_size = original_sizes[rt_uid._id]
        assert orig_size == rt_size, (
            f"Size mismatch after round-trip for axis {rt_uid._id}: "
            f"{orig_size!r} != {rt_size!r}"
        )


def test_uid_registry_covers_all_axis_subclasses():
    """Every concrete RawAxis subclass must appear in the UID type registry.

    Importing csv_io already pulls in AxisAnnotations, so all known subclasses
    are registered with Python's type system before this test runs.
    """
    all_axis_types = {RawAxis} | _all_subclasses(RawAxis)
    missing = [cls.__name__ for cls in all_axis_types if cls not in _UID_NAME_BY_TYPE]
    assert not missing, f'axis types missing from UID registry: {missing}'
