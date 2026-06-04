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
