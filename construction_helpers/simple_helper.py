from __future__ import annotations
from typing import Callable, Iterable, Iterator, overload
import data_structure.Term as fd
import data_structure.Category as cat
import construction_helpers.product as chp
import construction_helpers.lift as chl
import utilities.utilities as util
import term_utilities.term_utilities as tutil
from enum import Enum



def expand_content[L, M: cat.Morphism](
    _type: type[cat.Composed | cat.ProductOfMorphisms],
    *target: cat.ProdCategory[L, M]
) -> fd.Prod[cat.ProdCategory[L, M]]:
    match target:
        case (morphism,) if isinstance(morphism, _type):
            return expand_content(_type, *morphism.content)
        case (morphism,):
            return (morphism,)
        case ():
            return ()
        case tuple():
            return util.concat(expand_content(_type, morphism) for morphism in target)


def make_composed[L, M: cat.Morphism](
    *content: cat.ProdCategory[L, M]
) -> cat.ProdCategory[L, M]:
    content = expand_content(cat.Composed, *content)
    if all(tutil.is_identity(morphism) for morphism in content):
        return content[0]
    content = tuple(morphism for morphism in content 
                    if not tutil.is_identity(morphism))
    if len(content) == 1:
        return content[0]
    return cat.Composed(content=content)

def make_product[L, M: cat.Morphism](
    *content: cat.ProdCategory[L, M]
) -> cat.ProdCategory[L, M]:
    content = expand_content(cat.ProductOfMorphisms, *content)
    if len(content) == 1:
        return content[0]
    if all(tutil.is_identity(morphism) for morphism in content):
        return cat.ProdObject(
            util.concat(morphism.dom() for morphism in content)
        ).identity()
    return cat.ProductOfMorphisms(content=content)