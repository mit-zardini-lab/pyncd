from data_structure.ProductCategory import (
    ProdCategory,
    ProdObject,
    Morphism,
    Block,
    Composed,
    ThreadedComposed,
    ProductOfMorphisms,
    Rearrangement,
    BlockTag,
    BlockAesthetics
)
from data_structure.StrideCategory import (
    StrideCategory,
    Axis,
    RawAxis,
    StrideMorphism,
)
from data_structure.BroadcastedCategory import (
    BroadcastedCategory,
    AxisObjectoid,
    BroadcastedObjectoid,
    Datatype,
    Reals,
    Bool,
    Natural,
    Array,
    WeaveMode,
    Weave,
    Operator,
    Broadcasted,
)
# TensorEquation / TensorProgram are re-exported lazily (see __getattr__ below)
# to break the import cycle TensorLogic -> Operators -> Category -> TensorLogic.
# An eager `from data_structure.TensorLogic import ...` here fails whenever
# TensorLogic is the cycle entry point, because Category then runs against a
# half-initialised TensorLogic.  These names are only ever accessed at runtime,
# never at Category import time, so deferring the import is safe.
from data_structure.AxisAnnotations import NormAxis
__all__ = [
    'ProdCategory',
    'ProdObject',
    'Morphism',
    'Composed',
    'ThreadedComposed',
    'ProductOfMorphisms',
    'Rearrangement',
    'BlockTag',
    'BlockAesthetics',
    'Block',
    'StrideCategory',
    'Axis',
    'RawAxis',
    'StrideMorphism',
    'BroadcastedCategory',
    'AxisObjectoid',
    'BroadcastedObjectoid',
    'Datatype',
    'Reals',
    'Natural',
    'Array',
    'WeaveMode',
    'Weave',
    'Operator',
    'Broadcasted',
    'NormAxis',
    'TensorEquation',
    'TensorProgram',
]


def __getattr__(name: str):
    # PEP 562 lazy re-export: resolve TensorEquation / TensorProgram on first
    # access (always at runtime, after TensorLogic has finished loading) rather
    # than at Category import time.  This is what breaks the import cycle.
    if name in ('TensorEquation', 'TensorProgram'):
        from data_structure.TensorLogic import TensorEquation, TensorProgram
        globals()['TensorEquation'] = TensorEquation
        globals()['TensorProgram'] = TensorProgram
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
