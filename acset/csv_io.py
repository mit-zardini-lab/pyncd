"""CSV serialization and deserialization for SStInstance and SBrInstance."""
from __future__ import annotations
import csv
from pathlib import Path

import data_structure.Term as fd
import data_structure.Numeric as nm
import data_structure.StrideCategory as sc
from data_structure.AxisAnnotations import NormAxis, NatAxis

from acset.instances import (
    OpTag, DataTag, EntryRow, SStInstance,
    EquationRow, ArrayRow, ArrayAxisRow, SampleRow, SBrInstance,
)


# ---------------------------------------------------------------------------
# UID type registry
# ---------------------------------------------------------------------------

_UID_TYPE_BY_NAME: dict[str, type] = {
    'RawAxis':  sc.RawAxis,
    'NormAxis': NormAxis,
    'NatAxis':  NatAxis,
}
_UID_NAME_BY_TYPE: dict[type, str] = {v: k for k, v in _UID_TYPE_BY_NAME.items()}


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _uid_str(uid: fd.UID) -> str:
    return f'{_UID_NAME_BY_TYPE[uid._type]}:{uid._id}'


def _name_str(name: fd.DynamicName | None) -> str:
    # DynamicName.settings (bold, overline, absolute) are not serialised;
    # names serve as labels for Lean interop and display metadata is not needed.
    if name is None:
        return ''
    if name.subscript is None:
        return name.body or ''
    # body=None and body='' both serialize to '' here; DynamicName(body=None, subscript=...)
    # would deserialize with body='' instead. This does not occur in practice.
    return f"{name.body or ''}_{_name_str(name.subscript)}"


def _numeric_str(n: nm.Numeric) -> str:
    # Compound expressions (Addition, Multiplication, Power) are not reachable
    # here: axis sizes are always Integer or FreeNumeric (from RawAxis._size or
    # explicit construction), and StrideMorphism/SampleRow coefficients are always
    # Integer. The ValueError below is an invariant guard, not a user-facing case.
    if isinstance(n, nm.Integer):
        return str(n._value)
    if isinstance(n, nm.FreeNumeric):
        # Only _id is stored; uid._name (set by RawAxis.named()) is dropped.
        # FreeNumeric.__eq__ hashes uid including _name, so original != deserialized.
        # Fix would require either serialising _name+settings or changing
        # FreeNumeric.numeric_hash() to use _id only.
        return f'?{n.uid._id}'
    raise ValueError(f'cannot serialize Numeric of type {type(n).__name__}')


def _bool_str(b: bool | None) -> str:
    if b is None:
        return ''
    return 'true' if b else 'false'


# ---------------------------------------------------------------------------
# Deserialization helpers
# ---------------------------------------------------------------------------

def _parse_uid(s: str) -> fd.UID:
    if ':' in s:
        tag, id_str = s.split(':', 1)
        uid_type = _UID_TYPE_BY_NAME.get(tag, sc.RawAxis)
    else:
        uid_type, id_str = sc.RawAxis, s  # backwards compat with untagged files
    return fd.UID(_type=uid_type, _id=int(id_str), _name=None)


def _parse_numeric(s: str) -> nm.Numeric:
    if s.startswith('?'):
        uid = fd.UID(_type=nm.FreeNumeric, _id=int(s[1:]), _name=None)
        return nm.FreeNumeric(uid=uid)
    return nm.Integer(int(s))


def _parse_name(s: str) -> fd.DynamicName | None:
    if not s:
        return None
    return fd.DynamicName.from_str(s)


def _parse_bool(s: str) -> bool | None:
    if not s:
        return None
    return s == 'true'


def _parse_required_bool(s: str) -> bool:
    return s == 'true'


def _parse_optag(s: str) -> OpTag | None:
    return OpTag(s) if s else None


def _parse_datatag(s: str) -> DataTag:
    return DataTag(s)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def _write_axis_sizes(axis_sizes: dict[fd.UID, nm.Numeric], path: Path) -> None:
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['axis_uid', 'size'])
        w.writeheader()
        for uid, size in axis_sizes.items():
            w.writerow({'axis_uid': _uid_str(uid), 'size': _numeric_str(size)})


def write_sst(inst: SStInstance, directory: Path) -> None:
    """Write an SStInstance to two CSV files inside directory."""
    directory.mkdir(parents=True, exist_ok=True)
    _write_axis_sizes(inst.axis_sizes, directory / 'axis_sizes.csv')
    with open(directory / 'entries.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['src_uid', 'tgt_uid', 'coeff'])
        w.writeheader()
        for e in inst.entries:
            w.writerow({
                'src_uid': _uid_str(e.src),
                'tgt_uid': _uid_str(e.tgt),
                'coeff':   _numeric_str(e.coeff),
            })


def write_sbr(inst: SBrInstance, directory: Path) -> None:
    """Write an SBrInstance to five CSV files inside directory."""
    directory.mkdir(parents=True, exist_ok=True)
    _write_axis_sizes(inst.axis_sizes, directory / 'axis_sizes.csv')

    with open(directory / 'equations.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['equation_idx', 'lhs_name'])
        w.writeheader()
        for eq in inst.equations:
            w.writerow({'equation_idx': eq.equation_idx, 'lhs_name': _name_str(eq.lhs_name)})

    with open(directory / 'arrays.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=[
            'equation_idx', 'slot', 'name', 'is_input', 'operator_tag',
            'norm_axis', 'datatype_tag', 'max_value', 'bias', 'elementwise_fn',
            'iverson_expr',
        ])
        w.writeheader()
        for a in inst.arrays:
            w.writerow({
                'equation_idx':  a.equation_idx,
                'slot':          a.slot,
                'name':          _name_str(a.name),
                'is_input':      _bool_str(a.is_input),
                'operator_tag':  a.operator_tag.value if a.operator_tag else '',
                'norm_axis':     _uid_str(a.norm_axis) if a.norm_axis is not None else '',
                'datatype_tag':  a.datatype_tag.value,
                'max_value':     _numeric_str(a.max_value) if a.max_value is not None else '',
                'bias':          _bool_str(a.bias),
                'elementwise_fn': a.elementwise_fn or '',
                'iverson_expr':  a.iverson_expr or '',
            })

    with open(directory / 'array_axes.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=[
            'equation_idx', 'array_slot', 'axis_uid', 'is_target', 'position',
        ])
        w.writeheader()
        for aa in inst.array_axes:
            w.writerow({
                'equation_idx': aa.equation_idx,
                'array_slot':   aa.array_slot,
                'axis_uid':     _uid_str(aa.axis_uid),
                'is_target':    _bool_str(aa.is_target),
                'position':     aa.position,
            })

    with open(directory / 'samples.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=[
            'equation_idx', 'reindexing_slot', 'src_uid', 'tgt_uid', 'coeff', 'offset',
        ])
        w.writeheader()
        for s in inst.samples:
            w.writerow({
                'equation_idx':    s.equation_idx,
                'reindexing_slot': s.reindexing_slot,
                'src_uid':         _uid_str(s.src_uid),
                'tgt_uid':         _uid_str(s.tgt_uid),
                'coeff':           _numeric_str(s.coeff),
                'offset':          _numeric_str(s.offset),
            })


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def _read_axis_sizes(path: Path) -> dict[fd.UID, nm.Numeric]:
    sizes: dict[fd.UID, nm.Numeric] = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            sizes[_parse_uid(row['axis_uid'])] = _parse_numeric(row['size'])
    return sizes


def read_sst(directory: Path) -> SStInstance:
    """Read an SStInstance from two CSV files inside directory."""
    inst = SStInstance()
    inst.axis_sizes = _read_axis_sizes(directory / 'axis_sizes.csv')
    with open(directory / 'entries.csv', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            inst.entries.append(EntryRow(
                src=_parse_uid(row['src_uid']),
                tgt=_parse_uid(row['tgt_uid']),
                coeff=_parse_numeric(row['coeff']),
            ))
    return inst


def read_sbr(directory: Path) -> SBrInstance:
    """Read an SBrInstance from five CSV files inside directory."""
    inst = SBrInstance()
    inst.axis_sizes = _read_axis_sizes(directory / 'axis_sizes.csv')

    with open(directory / 'equations.csv', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            inst.equations.append(EquationRow(
                equation_idx=int(row['equation_idx']),
                lhs_name=_parse_name(row['lhs_name']),
            ))

    with open(directory / 'arrays.csv', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            inst.arrays.append(ArrayRow(
                equation_idx=int(row['equation_idx']),
                slot=int(row['slot']),
                name=_parse_name(row['name']),
                is_input=_parse_required_bool(row['is_input']),
                operator_tag=_parse_optag(row['operator_tag']),
                norm_axis=_parse_uid(row['norm_axis']) if row['norm_axis'] else None,
                datatype_tag=_parse_datatag(row['datatype_tag']),
                max_value=_parse_numeric(row['max_value']) if row['max_value'] else None,
                bias=_parse_bool(row['bias']),
                elementwise_fn=row['elementwise_fn'] or None,
                iverson_expr=row.get('iverson_expr') or None,
            ))

    with open(directory / 'array_axes.csv', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            inst.array_axes.append(ArrayAxisRow(
                equation_idx=int(row['equation_idx']),
                array_slot=int(row['array_slot']),
                axis_uid=_parse_uid(row['axis_uid']),
                is_target=_parse_required_bool(row['is_target']),
                position=int(row['position']),
            ))

    with open(directory / 'samples.csv', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            inst.samples.append(SampleRow(
                equation_idx=int(row['equation_idx']),
                reindexing_slot=int(row['reindexing_slot']),
                src_uid=_parse_uid(row['src_uid']),
                tgt_uid=_parse_uid(row['tgt_uid']),
                coeff=_parse_numeric(row['coeff']),
                offset=_parse_numeric(row['offset']) if 'offset' in row else nm.Integer(0),
            ))

    return inst
