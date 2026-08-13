from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Sequence, Type, Iterable, Self

import data_structure.Term as fd # for 'foundations'
import data_structure.Numeric as nm
import utilities.utilities as util
import data_structure.Category as cat
import construction_helpers.product as chp

from functools import cache

@dataclass(frozen=True)
class ObjectNode(fd.UTerm):
    @classmethod
    def template(cls, target: Iterable):
        return tuple(cls() for _ in target)

@dataclass(frozen=True)
class HypergraphObject[L](fd.Term):
    obj: L
    node: ObjectNode = field(
        default_factory=lambda: ObjectNode())
    
    @classmethod
    def template(cls, target: Sequence[L], nodes: fd.Prod[ObjectNode] | None = None) -> fd.Prod[HypergraphObject[L]]:
        return tuple(
            HypergraphObject(obj=obj, node=node) 
            for obj, node in zip(target, nodes or ObjectNode.template(target))
        )
    
def is_rearrangement[L, M: cat.Morphism](graph: Hypergraph[L, M]) -> bool:
    match graph:
        case HypergraphRoot():
            return False
        case HypergraphBlock(body=body):
            return is_rearrangement(body)
        case AuxiliaryGraph():
            return all(is_rearrangement(subgraph) for subgraph, _ in graph._subgraphs)
        case _:
            raise ValueError(f"Unknown graph type: {type(graph)}")

@dataclass(frozen=True)
class Hypergraph[L, M:cat.Morphism](fd.UTerm):
    dom: fd.Prod[HypergraphObject[L]] = ()
    cod: fd.Prod[HypergraphObject[L]] = ()

    def subgraphs(self) -> fd.Prod[Hypergraph[L, M]]:
        return (self,)
    
    def udom(self) -> fd.Prod[ObjectNode]:
        return tuple(obj.node for obj in self.dom)
    
    def ucod(self) -> fd.Prod[ObjectNode]:
        return tuple(obj.node for obj in self.cod)

@dataclass(frozen=True)
class HypergraphRoot[L, M: cat.Morphism](Hypergraph[L, M]):
    wraps: M = None # type: ignore

    @classmethod
    def template(cls, 
                 target: M, 
                 udom: fd.Prod[ObjectNode] | None = None, 
                 ucod: fd.Prod[ObjectNode] | None = None
                 ):
        if isinstance(target, cat.Rearrangement):
            udom = udom or ()
            assert ucod is None or ucod == target.apply(udom)
            dom = HypergraphObject.template(target.dom(), udom)
            cod = target.apply(dom)
            return Multigraph(
                uid = fd.UID(Multigraph),
                dom = dom,
                cod = cod,
                _subgraphs = ()
            )
        return cls(
            uid=fd.UID(HypergraphRoot),
            dom=HypergraphObject.template(target.dom(), udom),
            cod=HypergraphObject.template(target.cod(), ucod),
            wraps=target
        )
    
    def subgraphs(self) -> fd.Prod[Hypergraph[L, M]]:
        return ()
    
@dataclass(frozen=True)
class HypergraphBlock[L, M: cat.Morphism](Hypergraph[L, M]):
    body: Hypergraph[L, M] = None # type: ignore
    block_tag: cat.BlockTag = None # type: ignore

    @classmethod
    def template(cls,
                 body: Hypergraph[L, M],
                 block_tag: cat.BlockTag,
                 reduce: bool = True
                 ) -> HypergraphBlock[L, M]:
        dom = body.dom if not reduce else util.unique_tuple(body.dom, unique_test=lambda obj: obj.node)
        return cls(
            uid=fd.UID(HypergraphBlock),
            dom=dom,
            cod=body.cod,
            body=body,
            block_tag=block_tag
        )
    
    def subgraphs(self) -> fd.Prod[Hypergraph[L, M]]:
        return (self.body,)

@dataclass(frozen=True)
class AuxiliaryGraph[L, M: cat.Morphism, A](Hypergraph[L, M]):
    _subgraphs: fd.Prod[tuple[Hypergraph[L, M], A]] = ()

    def subgraphs(self) -> fd.Prod[Hypergraph[L, M]]:
        return tuple(subgraph for subgraph, _ in self._subgraphs)
    
    @classmethod
    def construct(cls, 
                  _subgraphs: fd.Prod[tuple[Hypergraph[L, M], A]], 
                  dom: fd.Prod[HypergraphObject[L]], 
                  cod: fd.Prod[HypergraphObject[L]],
                  reduce_rearrangements: bool=False) -> Self:
        return cls(
            uid=fd.UID(cls),
            dom=dom,
            cod=cod,
            _subgraphs=(
                _subgraphs
                if not reduce_rearrangements else
                tuple((subgraph, aux) for subgraph, aux in _subgraphs
                if not is_rearrangement(subgraph))
            )
        )
    
    def replace(self, target: Hypergraph[L, M], replacement: Hypergraph[L, M]) -> Self:
        assert replacement.dom == target.dom and replacement.cod == target.cod
        new_subgraphs = tuple(
            (replacement if subgraph == target else subgraph, aux)
            for subgraph, aux in self._subgraphs
        )
        return self.construct(
            new_subgraphs,
            dom=self.dom,
            cod=self.cod
        )
    
    def replace_multiple(self, target_replacements: Iterable[tuple[Hypergraph[L, M], Hypergraph[L, M]]]) -> Self:
        target_replacements = tuple(target_replacements)
        for target, replacement in target_replacements:
            assert replacement.dom == target.dom and replacement.cod == target.cod
        target_replacements_dict = dict(target_replacements)
        new_subgraphs = tuple(
            (target_replacements_dict.get(subgraph, subgraph), aux)
            for subgraph, aux in self._subgraphs
        )
        return self.construct(
            new_subgraphs,
            dom=self.dom,
            cod=self.cod
        )
    
    def get_aux(self, target: Hypergraph[L, M]) -> A | None:
        for subgraph, aux in self._subgraphs:
            if subgraph == target:
                return aux
        return None
    
    def grab_single(self, target: Hypergraph[L, M] | int | fd.UID[Hypergraph]) -> Hypergraph[L, M]:
        match target:
            case int():
                return self.subgraphs()[target]
            case fd.UID():
                return next(subgraph for subgraph, _ in self._subgraphs if subgraph.uid == target)
            case Hypergraph():
                return self.grab_single(target.uid)
        raise ValueError(f"Target {target} not found in subgraphs.")

    
@dataclass(frozen=True)
class Multigraph[L, M: cat.Morphism](AuxiliaryGraph[L, M, None]):
    @classmethod
    def template(cls, 
                 dom: fd.Prod[HypergraphObject[L]], 
                 cod: fd.Prod[HypergraphObject[L]], 
                 subgraphs: Iterable[Hypergraph[L, M]] = ()) -> Multigraph[L, M]:
        return cls(
            uid=fd.UID(Multigraph),
            dom=dom,
            cod=cod,
            _subgraphs=tuple((subgraph, None) for subgraph in subgraphs)
        )
    
    @classmethod
    def construct(cls, # type: ignore
                  _subgraphs: fd.Prod[tuple[Hypergraph[L, M], None]], 
                  dom: fd.Prod[HypergraphObject[L]], 
                  cod: fd.Prod[HypergraphObject[L]]) -> Multigraph[L, M]:
        return cls(
            uid=fd.UID(Multigraph),
            dom=dom,
            cod=cod,
            _subgraphs=_subgraphs
        )

def flat_subgraphs[L, M: cat.Morphism](target: Hypergraph[L, M], remove_blocks: bool = False) -> fd.Prod[Hypergraph[L, M]]:
    match target:
        case HypergraphBlock(body=body):
            if remove_blocks and target.block_tag.repetition == nm.Integer(1):
                return flat_subgraphs(body, remove_blocks=True)
            return (target.reconstruct(body=flatten(body, remove_blocks)),)
        case AuxiliaryGraph(_subgraphs=subgraphs):
            return util.concat(flat_subgraphs(subgraph, remove_blocks) for subgraph, _ in subgraphs)
        case _:
            return (target,)
        

def flatten[L, M: cat.Morphism](target: Hypergraph[L, M], remove_blocks: bool = False) -> Hypergraph[L, M]:
    flattened_subgraphs = flat_subgraphs(target, remove_blocks=remove_blocks)
    if len(flattened_subgraphs) == 1:
        return flattened_subgraphs[0]
    return Multigraph.template(
        target.dom,
        target.cod,
        flattened_subgraphs
    )

def flatten_blocks[L, M: cat.Morphism](target: Hypergraph[L, M]) -> fd.Prod[Hypergraph[L, M]]:
    match target:
        case HypergraphBlock(body=body):
            return flatten_blocks(body)
        case AuxiliaryGraph(_subgraphs=subgraphs):
            return util.concat(flatten_blocks(subgraph) for subgraph, _ in subgraphs)
        case _:
            return (target,)


type LocationUnit = tuple[Type[cat.Composed] | Type[cat.ProductOfMorphisms], int] | cat.BlockTag
type Location = fd.Prod[LocationUnit]

@dataclass(frozen=True)
class StructuredHypergraph[L, M: cat.Morphism](AuxiliaryGraph[L, M, Location]):
    wraps: cat.ProdCategory[L, M] = None # type: ignore

    @classmethod
    def construct(cls, # type: ignore
                  _subgraphs: fd.Prod[tuple[Hypergraph[L, M], Location]], 
                  dom: fd.Prod[HypergraphObject[L]], 
                  cod: fd.Prod[HypergraphObject[L]],
                  ) -> Self: 
        return cls(
            uid=fd.UID(StructuredHypergraph),
            dom=dom,
            cod=cod,
            _subgraphs=_subgraphs,
            wraps=None, # type: ignore
        )


    @classmethod
    def from_morphism(
        cls,
        target: cat.ProdCategory[L, M],
        dom: fd.Prod[HypergraphObject[L]] | None = None,
        location: Location = (),
        reduce: bool = True,
    ) -> StructuredHypergraph[L, M]:
        dom = dom or HypergraphObject.template(target.dom())
        cod: fd.Prod[HypergraphObject[L]] = ()
        subgraphs = ()

        match target:
            case cat.Block():
                location = (*location, target.block_tag)
                subgraph = cls.from_morphism(
                    target.body, dom, location, reduce=reduce)
                subgraph = HypergraphBlock.template(
                    body=subgraph, block_tag=target.block_tag)
                subgraphs = ((subgraph, location),)
                cod = subgraph.cod
            case cat.ProductOfMorphisms(ms):
                for i, (m, _dom) in enumerate(target.partition(dom)):
                    subgraph = cls.from_morphism(
                        m, _dom, (*location, (cat.ProductOfMorphisms, i)), reduce=reduce)
                    cod = cod + subgraph.cod
                    subgraphs = (*subgraphs, *subgraph._subgraphs)
            case cat.Composed(ms):
                cod = dom
                for i, m in enumerate(ms):
                    subgraph = cls.from_morphism(
                        m, cod, (*location, (cat.Composed, i)), reduce=reduce)
                    cod = subgraph.cod
                    subgraphs = (*subgraphs, *subgraph._subgraphs)
            case cat.Rearrangement():
                subgraphs = ()
                cod = target.apply(dom)
            case _:
                subgraph = HypergraphRoot.template(
                    target, 
                    tuple(d.node for d in dom)
                )
                reduce = False
                subgraphs = ((subgraph, location),)
                cod = subgraph.cod
        dom = dom if not reduce else util.unique_tuple(dom)
        return cls(
            uid=fd.UID(StructuredHypergraph),
            dom=dom,
            cod=cod,
            wraps=target,
            _subgraphs=subgraphs
        )
        

        
############################
## HYPERGRAPH -> MORPHISM ##
############################
## This assumes that blocks are in their own Hypergraphs ##

def hypergraph2morphism[L, M: cat.Morphism](
    graph: Hypergraph[L, M]
) -> cat.ProdCategory[L, M]:
    match graph:
        case HypergraphRoot(wraps=wraps):# | StructuredHypergraph(wraps=cat.Morphism() as wraps):
            return wraps
        case HypergraphBlock(body=body, block_tag=block_tag):
            return cat.Block(
                body=hypergraph2morphism(body),
                block_tag=block_tag
            )
    sequence: fd.Prod[cat.ProdCategory[L, M]] = ()
    remaining = graph
    while remaining.subgraphs():
        sequence_front, remaining = cleave(remaining)
        sequence = (*sequence, *sequence_front)
        if not remaining.subgraphs() and remaining.dom == remaining.cod:
            break
    if len(sequence) == 1:
        return sequence[0]
    return cat.Composed(sequence)

def cleave[L, M: cat.Morphism](
    target: Hypergraph[L, M],
    verbose: bool = False
) -> tuple[fd.Prod[cat.ProdCategory[L, M]], Hypergraph[L, M]]:
    dom = list(target.udom())
    assert(len(dom) == len(set(dom)))
    subgraphs = target.subgraphs()
    cleaved_graphs, remaining_graphs = util.predicate_partition(
        subgraphs,
        lambda graph: set(graph.udom()) <= set(dom)
    )
    if verbose:
        print('CLEAVED: ', cleaved_graphs)
    future_dom = util.unique_iterable(
        target.cod + util.concat(graph.dom for graph in remaining_graphs),
        unique_test=lambda obj: obj.node
    )
    kept_obj = util.intersection(
        target.dom, 
        future_dom,
        unique_test=lambda obj: obj.node
    )
    column_dom = util.concat(c.dom for c in cleaved_graphs) + kept_obj
    column_cod = util.concat(c.cod for c in cleaved_graphs) + kept_obj

    mapping = [dom.index(obj.node) for obj in column_dom]
    if verbose:
        print(f'Mapping: {mapping}')
        print(f'Target udom: {target.udom()}')
        print(f'Column udom: {[obj.node for obj in column_dom]}')

    if mapping == list(range(len(dom))):
        rearrangement = None
    else:
        rearrangement = cat.Rearrangement(
            mapping = tuple(mapping),
            _dom = tuple(obj.obj for obj in target.dom)
        )
    
    column: cat.ProdCategory[L, M] = chp.morphism_product(
        (*(hypergraph2morphism(graph) for graph in cleaved_graphs),
         *(obj.obj for obj in kept_obj))
    ) # type: ignore

    multigraph = Multigraph.template(
        column_cod, target.cod, tuple(remaining_graphs)
    )
    return (
        (rearrangement, column) if rearrangement else (column,),
        multigraph
    )

