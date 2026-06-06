"""First-class RHS expression types for TensorEquation.

TensorRef replaces the raw (DynamicName, axes) tuple previously stored in
TensorEquation.rhs. The Iverson types form a Term-based expression tree for
Boolean predicates; being Terms ensures that Context.apply() / deep_reconstruct
correctly unifies axis UIDs across the whole equation.

Monkey-patching RawAxis at module load time adds comparison and arithmetic
operators that produce Iverson nodes. Import this module before using operator
syntax on RawAxis. __mul__ is excluded to avoid colliding with tensor-product
semantics in TensorDSL.py.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import data_structure.Term as fd
import data_structure.Numeric as nm
import data_structure.StrideCategory as sc


# ---------------------------------------------------------------------------
# TensorRef — Term wrapper for a named tensor reference on the RHS
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TensorRef(fd.Term):
    name: fd.DynamicName
    axes: fd.Prod[sc.RawAxis] = ()


# ---------------------------------------------------------------------------
# Iverson expression tree — all nodes are Terms so deep_reconstruct traverses
# axis UIDs inside predicates just as it does inside TensorRef.axes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IversonConst(fd.Term):
    """A numeric literal leaf in an Iverson predicate (e.g. nm.Integer(5))."""
    value: nm.Numeric


@dataclass(frozen=True)
class IversonBinOp(fd.Term):
    """Binary operation in an Iverson predicate: lhs op rhs."""
    op: str
    lhs: sc.RawAxis | IversonConst | IversonBinOp | IversonUnaryOp
    rhs: sc.RawAxis | IversonConst | IversonBinOp | IversonUnaryOp

    # Chaining: (x < y) & (y < z)
    def __and__(self, other: IversonBinOp | IversonUnaryOp) -> IversonBinOp:
        return IversonBinOp('&', self, other)

    def __or__(self, other: IversonBinOp | IversonUnaryOp) -> IversonBinOp:
        return IversonBinOp('|', self, other)

    def __invert__(self) -> IversonUnaryOp:
        return IversonUnaryOp('~', self)

    # Arithmetic with another axis/expr — produces a new IversonBinOp for use
    # inside compound predicates such as iabs(x - y) < z.
    def __add__(self, other: Any) -> IversonBinOp:
        return IversonBinOp('+', self, other)

    def __sub__(self, other: Any) -> IversonBinOp:
        return IversonBinOp('-', self, other)

    def __lt__(self, other: Any) -> IversonBinOp:
        return IversonBinOp('<', self, other)

    def __le__(self, other: Any) -> IversonBinOp:
        return IversonBinOp('<=', self, other)

    def __gt__(self, other: Any) -> IversonBinOp:
        return IversonBinOp('>', self, other)

    def __ge__(self, other: Any) -> IversonBinOp:
        return IversonBinOp('>=', self, other)


@dataclass(frozen=True)
class IversonUnaryOp(fd.Term):
    """Unary operation in an Iverson predicate: op(operand)."""
    op: str
    operand: sc.RawAxis | IversonConst | IversonBinOp | IversonUnaryOp

    def __lt__(self, other: Any) -> IversonBinOp:
        return IversonBinOp('<', self, other)

    def __le__(self, other: Any) -> IversonBinOp:
        return IversonBinOp('<=', self, other)

    def __gt__(self, other: Any) -> IversonBinOp:
        return IversonBinOp('>', self, other)

    def __ge__(self, other: Any) -> IversonBinOp:
        return IversonBinOp('>=', self, other)

    def __add__(self, other: Any) -> IversonBinOp:
        return IversonBinOp('+', self, other)

    def __sub__(self, other: Any) -> IversonBinOp:
        return IversonBinOp('-', self, other)

    def __invert__(self) -> IversonUnaryOp:
        return IversonUnaryOp('~', self)


# Unified type for one factor on the RHS of a TensorEquation
type RHSFactor = TensorRef | IversonBinOp | IversonUnaryOp

# Full Iverson sub-expression type (used internally and in helpers)
type IversonExpr = sc.RawAxis | IversonConst | IversonBinOp | IversonUnaryOp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iverson_axes(expr: IversonExpr) -> tuple[sc.RawAxis, ...]:
    """Return all RawAxis leaves in an Iverson expression (DFS order)."""
    if isinstance(expr, sc.RawAxis):
        return (expr,)
    if isinstance(expr, IversonConst):
        return ()
    if isinstance(expr, IversonBinOp):
        return _iverson_axes(expr.lhs) + _iverson_axes(expr.rhs)
    if isinstance(expr, IversonUnaryOp):
        return _iverson_axes(expr.operand)
    return ()


def _factor_axes(factor: RHSFactor) -> fd.Prod[sc.RawAxis]:
    """Return the axes of any RHSFactor (tensor axes or Iverson leaf axes)."""
    if isinstance(factor, TensorRef):
        return factor.axes
    return _iverson_axes(factor)


def _axis_label(ax: sc.RawAxis) -> str:
    name = ax.uid._name
    if name is None:
        return f'ax_{ax.uid._id}'
    return name.to_bodies()


def _serialize_iverson(expr: IversonExpr) -> str:
    """Serialize an Iverson expression to a human-readable string."""
    if isinstance(expr, sc.RawAxis):
        return _axis_label(expr)
    if isinstance(expr, IversonConst):
        return str(expr.value)
    if isinstance(expr, IversonBinOp):
        return f'({_serialize_iverson(expr.lhs)} {expr.op} {_serialize_iverson(expr.rhs)})'
    if isinstance(expr, IversonUnaryOp):
        return f'{expr.op}({_serialize_iverson(expr.operand)})'
    return repr(expr)


# ---------------------------------------------------------------------------
# Convenience constructors (avoid __mul__ ambiguity and __eq__ override issues)
# ---------------------------------------------------------------------------

def ieq(lhs: IversonExpr, rhs: IversonExpr) -> IversonBinOp:
    """Create an equality predicate [lhs == rhs]."""
    return IversonBinOp('==', lhs, rhs)


def imul(lhs: IversonExpr, rhs: IversonExpr) -> IversonBinOp:
    """Create an arithmetic multiplication [lhs * rhs]."""
    return IversonBinOp('*', lhs, rhs)


def iabs(operand: IversonExpr) -> IversonUnaryOp:
    """Create an absolute-value node."""
    return IversonUnaryOp('abs', operand)


def inot(operand: IversonExpr) -> IversonUnaryOp:
    """Create a logical-negation node: ~[operand] = 1 - [operand]."""
    return IversonUnaryOp('~', operand)


# ---------------------------------------------------------------------------
# Monkey-patch RawAxis with Iverson-producing operators
# __mul__ excluded: collides with tensor-product semantics in TensorDSL
# ---------------------------------------------------------------------------

def _rawaxis_lt(self: sc.RawAxis, other: Any) -> IversonBinOp:
    return IversonBinOp('<', self, other)

def _rawaxis_le(self: sc.RawAxis, other: Any) -> IversonBinOp:
    return IversonBinOp('<=', self, other)

def _rawaxis_gt(self: sc.RawAxis, other: Any) -> IversonBinOp:
    return IversonBinOp('>', self, other)

def _rawaxis_ge(self: sc.RawAxis, other: Any) -> IversonBinOp:
    return IversonBinOp('>=', self, other)

def _rawaxis_add(self: sc.RawAxis, other: Any) -> IversonBinOp:
    return IversonBinOp('+', self, other)

def _rawaxis_sub(self: sc.RawAxis, other: Any) -> IversonBinOp:
    return IversonBinOp('-', self, other)

def _rawaxis_radd(self: sc.RawAxis, other: Any) -> IversonBinOp:
    return IversonBinOp('+', other, self)

def _rawaxis_rsub(self: sc.RawAxis, other: Any) -> IversonBinOp:
    return IversonBinOp('-', other, self)

def _rawaxis_invert(self: sc.RawAxis) -> IversonUnaryOp:
    return IversonUnaryOp('~', self)

def _rawaxis_rmul(self: sc.RawAxis, other: Any) -> IversonBinOp:
    # Integer coefficient on an axis: `c * axis` (coefficient first), for affine
    # index expressions like `2*i + 1`.  __mul__ stays excluded (tensor product).
    return IversonBinOp('*', other, self)

sc.RawAxis.__lt__ = _rawaxis_lt  # type: ignore[method-assign]
sc.RawAxis.__le__ = _rawaxis_le  # type: ignore[method-assign]
sc.RawAxis.__gt__ = _rawaxis_gt  # type: ignore[method-assign]
sc.RawAxis.__ge__ = _rawaxis_ge  # type: ignore[method-assign]
sc.RawAxis.__add__ = _rawaxis_add  # type: ignore[method-assign]
sc.RawAxis.__sub__ = _rawaxis_sub  # type: ignore[method-assign]
sc.RawAxis.__radd__ = _rawaxis_radd  # type: ignore[method-assign]
sc.RawAxis.__rsub__ = _rawaxis_rsub  # type: ignore[method-assign]
sc.RawAxis.__rmul__ = _rawaxis_rmul  # type: ignore[method-assign]
sc.RawAxis.__invert__ = _rawaxis_invert  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Affine index expressions (P2): normalise an index-slot value to b + Σ cₖ·aₖ
# ---------------------------------------------------------------------------

def affine_normal_form(
    expr: Any,
) -> tuple[int, dict[sc.RawAxis, int]]:
    """Flatten an index-slot value to affine normal form ``(const, {axis: coeff})``.

    Accepts a bare ``RawAxis`` (-> ``(0, {axis: 1})``), an ``int`` constant
    (-> ``(c, {})``), and ``IversonBinOp`` trees built from ``+ - *`` where ``*``
    has an integer coefficient on one side.  Raises ``ValueError`` on a non-affine
    index (axis·axis products, ``//``, ``mod``, comparisons, etc.).
    """
    if isinstance(expr, sc.RawAxis):
        return 0, {expr: 1}
    if isinstance(expr, bool):
        raise ValueError(f"non-affine index expression: {expr!r}")
    if isinstance(expr, int):
        return expr, {}
    if isinstance(expr, IversonBinOp):
        if expr.op == '+':
            cl, dl = affine_normal_form(expr.lhs)
            cr, dr = affine_normal_form(expr.rhs)
            return cl + cr, _merge_coeffs(dl, dr, 1)
        if expr.op == '-':
            cl, dl = affine_normal_form(expr.lhs)
            cr, dr = affine_normal_form(expr.rhs)
            return cl - cr, _merge_coeffs(dl, dr, -1)
        if expr.op == '*':
            lc, ld = affine_normal_form(expr.lhs)
            rc, rd = affine_normal_form(expr.rhs)
            # exactly one operand must be a pure integer (no axes)
            if not ld and rd:
                return lc * rc, {a: lc * c for a, c in rd.items()}
            if not rd and ld:
                return rc * lc, {a: rc * c for a, c in ld.items()}
            raise ValueError(
                "non-affine index expression: product of two axis terms"
            )
    raise ValueError(f"non-affine index expression: {expr!r}")


def _merge_coeffs(
    a: dict[sc.RawAxis, int], b: dict[sc.RawAxis, int], sign: int,
) -> dict[sc.RawAxis, int]:
    out = dict(a)
    for axis, coeff in b.items():
        out[axis] = out.get(axis, 0) + sign * coeff
        if out[axis] == 0:
            del out[axis]
    return out


def is_plain_axis(expr: Any) -> bool:
    """True iff the slot is a bare axis (coeff 1, no constant) — no reindex needed."""
    return isinstance(expr, sc.RawAxis)
