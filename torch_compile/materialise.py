from __future__ import annotations
from typing import Iterator
import torch
import data_structure.Numeric as nm
import data_structure.StrideCategory as sc
from data_structure.TensorExpr import (
    IversonExpr, IversonConst, IversonBinOp, IversonUnaryOp,
    _iverson_axes, _axis_label,
)


def materialise_iverson(factor: IversonExpr) -> torch.Tensor:
    """Evaluate an Iverson expression tree to a float {0,1} tensor.

    Each RawAxis leaf in DFS order becomes one independent positional dimension.
    Repeated UIDs get independent dimensions so einops can trace the diagonal
    (e.g. `x0 x0` contracts T[..., x, x, ...] over x).

    Raises ValueError if any axis has no concrete integer size.

    ──────────────────────────────────────────────────────────────────────────
    PERFORMANCE TRADEOFF — deliberate "expanded" (one-dim-per-leaf) form
    ──────────────────────────────────────────────────────────────────────────
    When one axis of size N appears k times in a single predicate (e.g. x in
    (x>3)|(x==3), k=2), this returns an N**k tensor even though only its
    N-element diagonal carries information. For the einsum-buffer path that
    tensor is register_buffer'd, so the waste is *persistent* module memory:
    N**k floats stored to use N. einops fusion cannot shrink an already-stored
    buffer.

    We keep the expanded form on purpose, for uniformity. The einops layer must
    already resolve index identity by UID for two unavoidable reasons:
      1. cross-factor contraction (a UID shared between tensors → summed), and
      2. caller-supplied tensors with repeated axes (T[x,x], and caller-supplied
         unsized Iverson masks), where collapsing is IMPOSSIBLE because the data
         is the caller's and the diagonal is computed at runtime.
    Reason 2 forces the one-dim-per-leaf convention on the whole Broadcasted /
    weave / einops machinery, so sized Iverson buffers ride the same rails: this
    function stays a dumb evaluator and a single UID->tag mechanism in
    generate_tensor_equation_signature() handles every diagonal/contraction in
    one place. The repeated-axis-within-one-predicate case is uncommon (causal
    q<=x, banded |q-x|<w, diagonal q==x all use DISTINCT axes -> zero waste), so
    the bound on the waste is small.

    NOTE the masked softmax/normalize path differs: it has no einsum to ride, so
    it collapses the diagonal explicitly (see TensorLogic._iverson_diagonal,
    applied in torch_compile's ConstructedMasked{SoftMax,Normalize}.__init__).

    TO FIX (shrink the buffer) if repeated-axis predicates with large N become
    common: collapse here -- assign one grid dimension per DISTINCT UID
    (first-occurrence DFS order) instead of per leaf -- AND emit one tag per
    distinct axis for Iverson segments in generate_tensor_equation_signature().
    That also lets _iverson_diagonal be deleted (both paths unify on the
    collapsed form). Costs: the materialise contract changes (test_materialise_
    compound's (3,4,4,5) becomes (3,4,5)), and Iverson factors then use a
    distinct-UID convention while genuine TensorRefs must stay leaf-based (caller
    data), so the signature generator must branch on factor type.
    """
    axes = _iverson_axes(factor)
    for ax in axes:
        if not isinstance(ax._size, nm.Integer):
            raise ValueError(
                f"Axis {_axis_label(ax)!r} has no concrete size; "
                "pre-materialise this Iverson tensor and pass it as a caller input."
            )
    n = len(axes)
    grids = [
        torch.arange(ax._size._value, dtype=torch.float32)
              .reshape(*(1,) * i, ax._size._value, *(1,) * (n - i - 1))
        for i, ax in enumerate(axes)
    ]
    return _eval(factor, iter(grids))


def _eval(expr: IversonExpr, grid_iter: Iterator[torch.Tensor]) -> torch.Tensor:
    """Recursively evaluate an Iverson expression tree.

    Consumes grid_iter in left-before-right DFS order (matching _iverson_axes),
    so the i-th RawAxis leaf receives grids[i]. PyTorch broadcasting expands
    the per-leaf positional tensors to the full shape automatically.
    """
    if isinstance(expr, sc.RawAxis):
        return next(grid_iter)
    if isinstance(expr, IversonConst):
        if not isinstance(expr.value, nm.Integer):
            raise ValueError(f"IversonConst contains non-integer numeric: {expr.value!r}")
        return torch.tensor(float(expr.value._value))
    if isinstance(expr, IversonBinOp):
        l = _eval(expr.lhs, grid_iter)
        r = _eval(expr.rhs, grid_iter)
        match expr.op:
            case '<':  return (l < r).float()
            case '<=': return (l <= r).float()
            case '>':  return (l > r).float()
            case '>=': return (l >= r).float()
            case '==': return (l == r).float()
            case '+':  return l + r
            case '-':  return l - r
            case '*':  return l * r
            case '&':  return (l.bool() & r.bool()).float()
            case '|':  return (l.bool() | r.bool()).float()
            case _:    raise ValueError(f"Unknown IversonBinOp operator: {expr.op!r}")
    if isinstance(expr, IversonUnaryOp):
        v = _eval(expr.operand, grid_iter)
        match expr.op:
            case 'abs': return v.abs()
            case '-':   return -v
            case '~':   return 1.0 - v
            case _:     raise ValueError(f"Unknown IversonUnaryOp operator: {expr.op!r}")
    raise ValueError(f"Unknown Iverson node type: {type(expr)!r}")
