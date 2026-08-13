from __future__ import annotations
from enum import Enum
from typing import Any, Iterable, Callable
import utilities.utilities as util
import data_structure.Category as cat
import data_structure.Term as fd

class WhichCategory(Enum):
    UNCLEAR = 'UNCLEAR'
    BROADCASTED = 'BROADCASTED'
    STRIDE = 'STRIDE'

type CategoricalConstruct = (
    cat.StrideCategory | cat.BroadcastedCategory | cat.ProdObject
    | cat.Array | cat.Datatype | cat.Axis
)

def identify_category(
        target: CategoricalConstruct) -> WhichCategory:
    if isinstance(target, (cat.Datatype, cat.Array, cat.Broadcasted)):
        return WhichCategory.BROADCASTED
    if isinstance(target, cat.StrideMorphism):
        return WhichCategory.STRIDE
    match target:
        case cat.Block(body=body):
            return identify_category(body)
        case ((cat.ProdObject() as parts)
              | cat.ProductOfMorphisms(content=parts) 
              | cat.Composed(content=parts) 
              | cat.Rearrangement(_dom=parts)):
            possibilites = (
                category
                for segment in parts
                if (category := identify_category(segment)))
            return util.iallequals(possibilites, WhichCategory.UNCLEAR)
        case _:
            raise ValueError(f"Cannot identify category of {target}")
        
type Mappable[L=Any] = (cat.Rearrangement[L] 
                 | cat.ProductOfMorphisms[L, Mappable[L]] 
                 | cat.Composed[L, Mappable[L]]
                 | cat.Block[L, Mappable[L]])

def is_mappable_broadcast(
    target: cat.Broadcasted
):
    return all(is_mappable(eta) for eta in target.reindexings)

def is_mappable(
    target: cat.ProdCategory[Any, Any]
):
    match target:
        case cat.Rearrangement():
            return True
        case cat.Composed(content=ms) | cat.ProductOfMorphisms(content=ms):
            return all(is_mappable(segment) for segment in ms)
        case cat.Block(body=body):
            return is_mappable(body)
        case _:
            return False
        
def get_mapping(
    target: cat.ProdCategory[Any, Any]
) -> fd.Prod[int]:
    if not is_mappable(target):
        raise ValueError(f"Target {target} is not mappable")
    match target:
        case cat.Rearrangement():
            return target.mapping
        case cat.ProductOfMorphisms():
            offset = 0
            mapping = ()
            for segment in target.content:
                new_mapping = get_mapping(segment)
                mapping = (*mapping, *(i + offset for i in new_mapping))
                offset += len(segment.dom())
            return mapping
        case cat.Composed():
            mapping = get_mapping(target.content[0])
            for segment in target.content[1:]:
                mapping = tuple(mapping[i] for i in get_mapping(segment))
            return mapping
        case cat.Block(body=body):
            return get_mapping(body)
        case _:
            raise ValueError(f"Target {target} is not mappable")

def is_identity(
    target: cat.ProdCategory[Any, Any]
) -> bool:
    match target:
        case cat.Rearrangement():
            return target.mapping == tuple(range(len(target.dom())))
        case cat.ProductOfMorphisms(content=ms) | cat.Composed(content=ms):
            return all(is_identity(segment) for segment in ms)
        case cat.Block(body=body):
            return is_identity(body)
        case _:
            return False
        
def flat_axes[B:cat.Datatype, A:cat.Axis](
    target: Iterable[cat.Array[B, A]]
):
    return tuple(
        axis
        for array in target
        for axis in array.shape()
    )

def type_search[T: fd.Term](
    _type: type[T],
    target: fd.GeneralTerm,
    predicate: Callable[[T], bool] = lambda term: True,
    deep: bool = True
) -> Iterable[T]:
    _predicate = lambda term: isinstance(term, _type) and predicate(term)
    return search(target, _predicate, deep) # type: ignore

def search[T: fd.Term](
        target: fd.GeneralTerm,
        predicate: Callable[[fd.Term], bool],
        deep: bool = True
) -> Iterable[T]:
    '''
    Every subterm of `target` satisfying `predicate`, each yielded once.

    Terms form a DAG, so `search_repeat` reaches a shared subterm once per path
    - it is named for yielding those repeats. Since the results are deduplicated
    anyway, a subterm already visited can be skipped outright, which turns the
    walk from one over paths into one over nodes; fusing attention took 333k
    recursive calls from 1.4k roots before this. Identity keys are safe because
    every node visited stays reachable from `target` for the whole traversal.
    '''
    seen: set[int] = set()

    def walk(node: fd.GeneralTerm) -> Iterable[T]:
        if id(node) in seen:
            return
        seen.add(id(node))
        match node:
            case fd.Term() if predicate(node):
                yield node # type: ignore
                if not deep:
                    return
        match node:
            case fd.Term():
                segments = node.dict().values()
            case tuple():
                segments = node
            case _:
                return
        for segment in segments:
            yield from walk(segment)

    return util.unique_iterable(walk(target))

def search_repeat[T:fd.Term](
        target: fd.GeneralTerm, 
        predicate: Callable[[fd.Term], bool],
        deep: bool = True) -> Iterable[T]:
    match target:
        case fd.Term() if predicate(target):
            yield target # type: ignore
            if not deep:
                return
    match target:
        case fd.Term():
            segments = target.dict().values()
        case tuple():
            segments = target
        case _:
            return
    yield from (
        subterm
        for segment in segments
        for subterm in search_repeat(segment, predicate, deep)
    )