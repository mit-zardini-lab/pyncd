'''Axes and affine index maps: the category St.

An axis carries a symbolic size. A `StrideMorphism` sends a tuple of domain indices to
a tuple of codomain indices, each an affine form of the domain indices. A codomain index
may fall outside `[0, size)` of its axis. The position read there holds the universal
unit, which a fold over an axis ignores and a pointwise operation preserves. That rule
is the whole of the category's account of an empty position. Which positions of a read
array are empty is an affine form of the position, and `advanced_axis_dynamics/` derives
it, carries it through further reads and restores it, so nothing here mentions it again.

`obsidian/02-categories/Stride Category.md` states the category, and
`obsidian/02-categories/Advanced Axis Dynamics.md` the axis a shifted or negatively
strided read produces.
'''
from __future__ import annotations
from dataclasses import dataclass
from typing import Self

import data_structure.Term as fd # for 'foundations'
import data_structure.Numeric as nm
import data_structure.ProductCategory as pc

type StrideCategory[A:Axis] = pc.ProdCategory[A, StrideMorphism[A]]

type StrideRow[A: Axis] = tuple[A, fd.Prod[nm.Numeric], nm.Numeric]

@dataclass(frozen=True)
class Axis(fd.UTerm):
    _size: nm.Numeric = nm.FreeNumeric.field()
    def local_size(self) -> nm.Numeric:
        return self._size
    @classmethod
    def named(cls, name: str | None | fd.DynamicName = None,
              code_form: str | None = None, **kwargs) -> Self:
        '''An axis named `name`, its size the same name between absolute bars.
        With `code_form`, the axis carries it and the size carries it with
        `_size` appended, so `named('q', code_form='queries')` has the size
        `queries_size`.'''
        if name is None:
            return cls(**kwargs)
        axis_name = fd.DynamicName.from_str(name, code_form=code_form)
        _size = axis_name.reconstruct(
            settings=fd.DynamicNameSettings(absolute=True)
        ).code_form_suffixed('size').capture(nm.FreeNumeric())
        return axis_name.capture(
            cls(_size=_size, **kwargs)
        )

@dataclass(frozen=True)
class RawAxis(Axis): ...


@dataclass(frozen=True)
class StrideMorphism[A:Axis](pc.Morphism[A]):
    '''An affine map from the indices of `_dom` to the indices of the codomain.

    Each member of `_cod_stride_shift` is one codomain axis with the stride it takes
    along each domain axis and the shift added afterwards, so that codomain index `i`
    is `sum(stride_i * dom) + shift_i`. A codomain index outside `[0, size)` of its
    axis names no position, and a read there yields the universal unit. Causality is
    written as such a read: the keys of a compressed cache read relative to the query
    stand at a negative slot where the query has not passed them. Which positions of
    the array a row leaves empty is derived in
    `advanced_axis_dynamics/algebra/mark_sparse_domains.py`.
    '''
    _dom: fd.Prod[A]
    _cod_stride_shift: fd.Prod[tuple[A, fd.Prod[nm.Numeric], nm.Numeric]]
    name: fd.DynamicName | None = None

    def dom(self) -> pc.ProdObject[A]:
        return pc.ProdObject(self._dom)
    def cod(self) -> pc.ProdObject[A]:
        return pc.ProdObject.from_iter(
            axis for axis, _, _ in self._cod_stride_shift)

    def strides(self) -> fd.Prod[fd.Prod[nm.Numeric]]:
        return tuple(stride for _, stride, _ in self._cod_stride_shift)

    @classmethod
    def from_matrix(cls,
                    *matrix: fd.Prod[int],
                    dom_names: None | fd.Prod[str] = None,
                    cod_names: None | fd.Prod[str] = None,
                    name: fd.DynamicName | str | None = None,
                    shifts: fd.Prod[int] = ()):
        matrix = tuple(tuple(row) for row in matrix)
        shifts = shifts or (0,) * len(matrix)
        _cod_stride_shift = tuple(
            (RawAxis.named(cod_names[i] if cod_names is not None else None),
             tuple(nm.Integer(value) for value in row),
             nm.Integer(shift))
            for (i, row), shift in zip(enumerate(matrix), shifts)
        )
        _dom = tuple(
            RawAxis.named(dom_names[i] if dom_names is not None else None)
            for i in range(len(matrix[0])))
        _name = fd.DynamicName.from_str(name)
        return StrideMorphism(_dom=_dom, _cod_stride_shift=_cod_stride_shift, name=_name)

