import construction_helpers as ch # Needed for @ auto-alignment
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd
from dataclasses import dataclass, field
from typing import List, Union

@dataclass(frozen=True)
class Complex[B:cat.Datatype = cat.Reals](cat.Datatype):
    base: B = field(default_factory=cat.Reals) # type: ignore

@dataclass(frozen=True)
class ComplexRotary[B:cat.Datatype](cat.Operator):
    name: fd.DynamicName | None = fd.DynamicName('Rotary')
    frequency: nm.Numeric = nm.Integer(10_000)

    @classmethod
    def template(cls, base: B = cat.Reals()):
        return cat.Broadcasted[Complex[B], cat.RawAxis](
            operator=cls(),
            input_weaves=(),
            output_weaves=(cat.Weave(Complex(base), (cat.RawAxis(), cat.RawAxis())),),
            reindexings=()
        )
    
def rotary_linear[B:cat.Datatype](
    base: B = cat.Reals(),
    frequency: nm.Numeric = nm.Integer(10_000)
):
    x_axis = cat.RawAxis()
    m_axis = cat.RawAxis()
    h_axis = cat.RawAxis()
    d_axis = cat.RawAxis()
    linear_part = cat.Broadcasted[B | Complex[B], cat.RawAxis](
        operator=ops.Linear(fd.DynamicName('R')),
        input_weaves=(cat.Weave(base, (cat.WeaveMode.TILED, m_axis)),),
        output_weaves=(cat.Weave(Complex(base), (cat.WeaveMode.TILED, h_axis, d_axis)),),
        reindexings=(cat.ProdObject((x_axis,)).identity(),)
    )
    rotary_part = cat.Broadcasted[Complex[B], cat.RawAxis](
        operator=ComplexRotary(),
        input_weaves=(),
        output_weaves=(cat.Weave(Complex(base), (x_axis, d_axis)),),
        reindexings=()
    )
    einops_part = ops.Einops.template(
        'x d/2, x h d/2 -> x h d/2',
        datatype=Complex(base),
    )
    return linear_part, rotary_part, einops_part

@dataclass(frozen=True)
class RotaryLinear[B:cat.Datatype](cat.Operator):
    name: fd.DynamicName | None = fd.DynamicName('RotaryLinear')
    frequency: nm.Numeric = nm.Integer(10_000)

@dataclass(frozen=True)
class Decomplex(cat.Operator):
    name: fd.DynamicName = fd.DynamicName('Decomplex')

    @classmethod
    def template[B: cat.Datatype](
        cls,
        base: B = cat.Reals(),
    ) -> cat.Broadcasted[B | Complex[B], cat.RawAxis]:
        return cat.Broadcasted(
            cls(),
            (cat.Weave(Complex(base), (cat.RawAxis(),)),),
            (cat.Weave(base, (cat.RawAxis(),)),),
            (cat.ProdObject().identity(),)
        )

    def sizing_rules[B: cat.Datatype, A: cat.Axis](self, broadcasted: cat.Broadcasted[B, A]) -> nm.Equality:
        input_dim = broadcasted.input_weaves[0].target().shape()[0]
        output_dim = broadcasted.output_weaves[0].target().shape()[0]
        return nm.Equality(
            2 * input_dim.local_size(), output_dim.local_size() 
        )

@dataclass(frozen=True)
class SparseAxis(cat.Axis):
    activity: nm.Numeric = nm.FreeNumeric.field()

    @classmethod
    def template(cls, parent: nm.Numeric):
        activity = fd.DynamicName('k').capture(nm.FreeNumeric())
        name = fd.DynamicName('k/n')
        return name.capture(cls(
            _size=parent,
            activity=activity
        ))

@dataclass(frozen=True)
class TopK(cat.Operator):
    name: fd.DynamicName = fd.DynamicName('TopK')
    k: nm.Numeric = field(default_factory=nm.FreeNumeric)

    @classmethod
    def template[B: cat.Datatype](
        cls,
        base: B = cat.Reals(),
        k: nm.Numeric | None = None
    ) -> cat.Broadcasted[B, cat.RawAxis | SparseAxis]:
        # if k is None:
        #     k = fd.DynamicName('k').capture(nm.FreeNumeric())
        parent = fd.DynamicName('n').capture(nm.FreeNumeric())
        n_axis = fd.DynamicName('n').capture(cat.RawAxis(_size=parent))
        k_axis = SparseAxis.template(parent)
        return cat.Broadcasted(
            cls(k=k_axis.activity),
            (cat.Weave(base, (n_axis,)),),
            (cat.Weave(base, (k_axis,)),),
            (cat.ProdObject().identity(),)
        )