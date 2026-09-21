from __future__ import annotations
from typing import overload
from dataclasses import dataclass
from abc import ABC, abstractmethod

import data_structure.Category as cat
import data_structure.Term as fd
import graphs.data_structure.Hypergraph as hg
from construction_helpers import simple_helper as chsh

import utilities.utilities as util

def reduce_nodes[L, M: cat.Morphism](context: fd.Context, target: hg.Hypergraph[L, M]) -> hg.Hypergraph[L, M]:
    if isinstance(target, hg.HypergraphRoot):
        return context.apply(target)
    new_dom = util.unique_tuple(context.apply(d) for d in target.dom)
    new_cod = util.unique_tuple(context.apply(c) for c in target.cod)
    match target:
        case hg.HypergraphBlock(body=body):
            new_body = reduce_nodes(context, body)
            return target.reconstruct(body=new_body, dom=new_dom, cod=new_cod)
        case hg.Multigraph(_subgraphs=subgraphs):
            new_subgraphs = tuple(
                reduce_nodes(context, subgraph) for subgraph in subgraphs
            )
            return target.reconstruct(_subgraphs=new_subgraphs, dom=new_dom, cod=new_cod)
    raise NotImplementedError(f"Cannot reduce nodes for {type(target).__name__}.")
@dataclass
class Functor[L1, M1: cat.Morphism, L2, M2: cat.Morphism](ABC):

    # The essential operations
    @abstractmethod
    def apply_object(self, target: L1) -> L2: ... 

    @abstractmethod
    def apply_root(self, target: M1) -> cat.ProdCategory[L2, M2]: ...

    # Custom operations
    def apply_block_tag(self, target: cat.BlockTag) -> None | cat.BlockTag:
        return target

    def apply_prod_object(self, target: cat.ProdObject[L1]) -> cat.ProdObject[L2]:
        return cat.ProdObject.from_iter(self.apply_object(obj) for obj in target)

    def apply_block_morphism(self, target: cat.Block[L1, M1]) -> cat.ProdCategory[L2, M2]:
        new_body = self.apply_category(target.body)
        new_block_tag = self.apply_block_tag(target.block_tag)
        if new_block_tag is None:
            return new_body
        return cat.Block(body=new_body, block_tag=new_block_tag)

    def apply_block_graph(self, target: hg.HypergraphBlock[L1, M1]) -> tuple[hg.Hypergraph[L2, M2], fd.Context]:
        new_body, equality_classes = self.apply_hypergraph_with_eq(target.body)
        new_block_tag = self.apply_block_tag(target.block_tag)
        if new_block_tag is None:
            return new_body, equality_classes
        return hg.HypergraphBlock.template(
            new_body, target.block_tag
        ), equality_classes

    # Derived Operations
    def apply_category(self, target: cat.ProdCategory[L1, M1]) -> cat.ProdCategory[L2, M2]:
        match target:
            case cat.Composed(content=ms):
                new_content = tuple(self.apply_category(m) for m in ms)
                return chsh.make_composed(*new_content)
            case cat.ProductOfMorphisms(content=ms):
                new_content = tuple(self.apply_category(m) for m in ms)
                return chsh.make_product(*new_content)
            case cat.Rearrangement(mapping=mapping, _dom=dom):
                new_dom = tuple(self.apply_object(d) for d in dom)
                return cat.Rearrangement(mapping=mapping, _dom=new_dom)
            case cat.Block():
                return self.apply_block_morphism(target) #type: ignore
            case _:
                return self.apply_root(target)

    def apply_hypergraph_root(self, target: hg.HypergraphRoot[L1, M1]) -> tuple[hg.Hypergraph[L2, M2], fd.Context]:
        new_body = self.apply_root(target.wraps)
        as_graph = hg.Multigraph.from_morphism(
            new_body,
            hg.HypergraphObject.template(new_body.dom(), target.dom)
        )
        # An input may appear more than once here.
        equality_classes = fd.Context().append_buckets(
            fd.UIDRenaming.set_canonical(final, original)
            for original, final in zip(
                (*target.cod, *target.dom),
                (*as_graph.cod, *as_graph.dom))
        )
        return as_graph, equality_classes

    def apply_hypergraph_with_eq(self, target: hg.Hypergraph[L1, M1]) -> tuple[hg.Hypergraph[L2, M2], fd.Context]:
        match target:
            case hg.HypergraphRoot():
                return self.apply_hypergraph_root(target)
            case hg.Multigraph(_subgraphs=subgraphs):
                new_subgraphs_eq = tuple(
                    self.apply_hypergraph_with_eq(subgraph)
                    for subgraph in subgraphs
                )
                equality_classes = fd.Context().append_contexts(
                    eq_class for _, eq_class in new_subgraphs_eq)
                dom_obj = cat.ProdObject[L2].from_iter(self.apply_object(d.obj) for d in target.dom)
                cod_obj = cat.ProdObject[L2].from_iter(self.apply_object(c.obj) for c in target.cod)
                dom: fd.Prod[hg.HypergraphObject[L2]] = hg.HypergraphObject.template(
                    dom_obj,
                    target.dom
                )
                cod: fd.Prod[hg.HypergraphObject[L2]] = hg.HypergraphObject.template(
                    cod_obj,
                    target.cod
                )
                new_subgraphs = util.concat(
                    (subgraph,)
                    if not isinstance(subgraph, hg.Multigraph)
                    else subgraph._subgraphs
                    for subgraph, _ in new_subgraphs_eq
                    if not hg.is_rearrangement(subgraph)
                )
                return hg.Multigraph[L2, M2].template(
                    dom, cod, new_subgraphs
                ), equality_classes
            case hg.HypergraphBlock():
                return self.apply_block_graph(target)
            case _:
                raise NotImplementedError()

    def apply_hypergraph(self, target: hg.Hypergraph[L1, M1]) -> hg.Hypergraph[L2, M2]:
        new_graph, context = self.apply_hypergraph_with_eq(target)
        contextualized_graph = reduce_nodes(context, new_graph)
        return contextualized_graph

    @overload
    def __call__(self, target: hg.Hypergraph[L1, M1]) -> hg.Hypergraph[L2, M2]: ...

    @overload
    def __call__(self, target: cat.ProdCategory[L1, M1]) -> cat.ProdCategory[L2, M2]: ...

    @overload
    def __call__(self, target: cat.ProdObject[L1]) -> cat.ProdObject[L2]: ...

    def __call__(
        self,
        target: hg.Hypergraph[L1, M1] | cat.ProdCategory[L1, M1] | cat.ProdObject[L1],
    ) -> hg.Hypergraph[L2, M2] | cat.ProdCategory[L2, M2] | cat.ProdObject[L2]:
        match target:
            case hg.Hypergraph():
                return self.apply_hypergraph(target)
            case cat.Morphism():
                return self.apply_category(target) # type: ignore
            case cat.ProdObject():
                return self.apply_prod_object(target)
            case _:
                raise NotImplementedError(f"Cannot apply {type(self).__name__} to {type(target).__name__}.")
            
@dataclass
class Endofunctor[L1, M1: cat.Morphism](Functor[L1, M1, L1, M1]):
    # The essential operations
    def apply_object(self, target: L1) -> L1:
        return target

    def apply_root(self, target: M1) -> cat.ProdCategory[L1, M1]:
        return target
