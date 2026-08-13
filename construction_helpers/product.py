from __future__ import annotations
from typing import Callable, Iterable, Iterator, overload
import data_structure.Term as fd
import data_structure.Category as cat
import data_structure.Numeric as nm

'''
Multiplication looks like:
    - Object * Object     -> Object
    - Object * Morphism   -> Morphism
    - Morphism * Morphism -> Morphism
Multiplication also supports flattening a tuple of targets.
'''

type ProductObjectTarget[L, T=L] = (
    L | T
    | cat.ProdObject[L]
    | fd.Prod[ProductObjectTarget[L, T]]
)

type ProductMorphismTarget[L, M:cat.Morphism, T=L] = (
    L | T
    | cat.ProdObject[L]
    | cat.ProdCategory[L, M]
    | fd.Prod[ProductMorphismTarget[L, M, T]]
)

def contains_morphism[L,M:cat.Morphism,T=L](target: ProductMorphismTarget[L, M, T]):
    match target:
        case tuple():
            return any(contains_morphism(segment) for segment in target)
        case cat.Morphism():
            return True
        case _:
            return False

def target_expand[L,M:cat.Morphism,T=L](
        target: ProductMorphismTarget[L, M, T],
        conversion: Callable[[L | T], L] = lambda x: x) -> fd.Prod[L | cat.ProdCategory[L, M]]:
    match target:
        case tuple() | cat.ProdObject() | cat.ProductOfMorphisms():
            return tuple(
                segment 
                for member in target 
                for segment in target_expand(member, conversion)) # type: ignore
        case cat.Rearrangement(mapping=(), _dom=()):
            return ()
        case cat.Morphism():
            return (target,) # type: ignore
        case _:
            return (conversion(target),)
        
def object_product[L, T=L](
    target: ProductObjectTarget[L, T],
    conversion: Callable[[L | T], L] = lambda x: x
) -> cat.ProdObject[L]:
    content_expanded = target_expand(target, conversion)
    return cat.ProdObject(content_expanded) # type: ignore

def to_morphism[L, M: cat.Morphism](target: Iterable[L | cat.ProdCategory[L, M]]) -> Iterator[cat.ProdCategory[L, M]]:
    accumulated: list[L] = []
    for member in target:
        match member:
            case cat.Morphism():
                if accumulated:
                    yield cat.ProdObject.from_iter(accumulated).identity()
                    accumulated = []
                yield member  # type: ignore
            case _:
                accumulated.append(member)
    if accumulated:
        yield cat.ProdObject.from_iter(accumulated).identity()

def morphism_product[L, M:cat.Morphism, T=L](
    target: ProductMorphismTarget[L, M, T],
    conversion: Callable[[L | T], L] = lambda x: x
) -> cat.ProdCategory[L, M]:
    content_expanded = tuple(to_morphism(target_expand(target, conversion)))
    if len(content_expanded) == 1:
        return content_expanded[0]
    return cat.ProductOfMorphisms.from_iter(content_expanded)

def general_product[L, M:cat.Morphism, T=L](
    target: ProductMorphismTarget[L, M, T],
    conversion: Callable[[L | T], L] = lambda x: x
) -> cat.ProdCategory[L, M] | cat.ProdObject[L]:
    if contains_morphism(target):
        return morphism_product(target, conversion)
    return object_product(target, conversion) # type: ignore

def datatype_converter[B: cat.Datatype, A: cat.Axis](target: B | cat.Array[B, A]) -> cat.Array[B, A]:
    if isinstance(target, cat.Datatype):
        target = cat.Array[B, A](datatype=target)
    return target

def axis_converter[A:cat.Axis](target: A | str | fd.DynamicName | nm.Numeric) -> A | cat.RawAxis:
    match target:
        case str() | fd.DynamicName():
            return cat.RawAxis.named(target)
        case nm.Numeric():
            sized = cat.RawAxis(_size=target)
            return fd.DynamicName.from_str(target.to_latex()).capture(sized)
        case _:
            return target
    #return cat.RawAxis.named(target) if isinstance(target, (str, fd.DynamicName)) else target

def full_converter(target):
    if isinstance(target, cat.Datatype):
        return cat.Array(datatype=target)
    elif isinstance(target, (str, fd.DynamicName)):
        return cat.RawAxis.named(target)
    return target

def datatype_product(*target):
    return general_product(target, full_converter) # type: ignore

cat.Datatype.__mul__   = datatype_product # type: ignore
cat.Array.__mul__      = datatype_product # type: ignore
cat.ProdObject.__mul__ = datatype_product # type: ignore
cat.Morphism.__mul__   = datatype_product # type: ignore
cat.ProdObject.__mul__ = datatype_product # type: ignore