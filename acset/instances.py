"""Acset instance dataclasses for St (stride) and Br (broadcasted) morphisms."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum

import data_structure.Term as fd
import data_structure.Numeric as nm


class OpTag(Enum):
    """Base operation type for an output Array."""
    IDENTITY                  = 'identity'
    SOFTMAX                   = 'softmax'
    MASKED_SOFTMAX            = 'masked_softmax'
    MASKED_NORMALIZE          = 'masked_normalize'
    ELEMENTWISE               = 'elementwise'
    NORMALIZE                 = 'normalize'
    EMBEDDING                 = 'embedding'
    ADDITION_OP               = 'addition'
    WEIGHTED_TRIANGULAR_LOWER = 'weighted_triangular_lower'
    LINEAR                    = 'linear'


class DataTag(Enum):
    """Scalar datatype of an Array: continuous reals, discrete naturals, or booleans."""
    REALS   = 'reals'
    NATURAL = 'natural'
    BOOL    = 'bool'


@dataclass
class EntryRow:
    """One nonzero coefficient in a stride morphism's coefficient matrix."""
    src:   fd.UID
    tgt:   fd.UID
    coeff: nm.Numeric


@dataclass
class SStInstance:
    """Acset instance for one StrideMorphism."""
    axis_sizes: dict[fd.UID, nm.Numeric] = field(default_factory=dict)
    entries:    list[EntryRow]           = field(default_factory=list)


@dataclass
class EquationRow:
    """One equation in a TensorProgram, in topological order."""
    equation_idx: int                    # position in topological sort; primary key
    lhs_name:     fd.DynamicName | None  # output tensor name


@dataclass
class ArrayRow:
    """One tensor (input or output) in a Broadcasted morphism."""
    equation_idx:    int                    # foreign key → EquationRow.equation_idx
    slot:            int                    # 0 = output; 1..N = inputs; key within equation
    name:            fd.DynamicName | None  # metadata only
    is_input:        bool
    operator_tag:    OpTag | None = None
    norm_axis:       fd.UID | None     = None
    datatype_tag:    DataTag           = DataTag.REALS
    max_value:       nm.Numeric | None = None
    bias:            bool | None       = None
    elementwise_fn:  str | None        = None
    iverson_expr:    str | None        = None


@dataclass
class ArrayAxisRow:
    """One axis belonging to one Array, with its role and physical position."""
    equation_idx: int    # \  composite foreign key
    array_slot:   int    # /  → ArrayRow.(equation_idx, slot)
    axis_uid:     fd.UID
    is_target:    bool
    position:     int = 0


@dataclass
class SampleRow:
    """One component of the reindexing rule for one input Array."""
    equation_idx:    int    # \  composite foreign key
    reindexing_slot: int    # /  → ArrayRow.(equation_idx, slot)
    src_uid:         fd.UID
    tgt_uid:         fd.UID
    coeff:           nm.Numeric


@dataclass
class SBrInstance:
    """Acset instance for a TensorProgram (one or more TensorEquations)."""
    axis_sizes:  dict[fd.UID, nm.Numeric] = field(default_factory=dict)
    equations:   list[EquationRow]        = field(default_factory=list)
    arrays:      list[ArrayRow]           = field(default_factory=list)
    array_axes:  list[ArrayAxisRow]       = field(default_factory=list)
    samples:     list[SampleRow]          = field(default_factory=list)
