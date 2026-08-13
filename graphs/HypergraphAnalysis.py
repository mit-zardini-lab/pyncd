from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Type, Iterable, Literal, Callable, NamedTuple

import data_structure.Term as fd # for 'foundations'
import data_structure.Numeric as nm
import utilities.utilities as util
import data_structure.Category as cat
import construction_helpers.product as chp

from functools import cached_property, cache
import graphs.Hypergraph as hg

import enum


# We have edges defined by;
# LeftEdges:  dict[HypergraphObject, (Hypergraph, int)]
# RightEdges: dict[HypergraphObject, (Hypergraph, int)]

# OUTSIDE = 'OUTSIDE' # type: Literal['OUTSIDE']
class HypergraphSpecialTag(enum.Enum):
    LEFT = 'LEFT'
    RIGHT = 'RIGHT'

type GraphTag[L, M: cat.Morphism] = hg.Hypergraph[L, M] | HypergraphSpecialTag

def link[L, M: cat.Morphism](g2n: Iterable[Graph2Node[L, M]], n2g: Iterable[Node2Graph[L, M]]) -> Iterable[Graph2Graph[L, M]]:
    return (
        Graph2Graph(left_graph, cod_segment, dom_segment, right_graph)
        for left_graph, cod_segment, left_node in g2n
        for right_node, dom_segment, right_graph in n2g
        if left_node == right_node
    )

class Graph2Node[L, M: cat.Morphism](NamedTuple):
    graph: hg.Hypergraph[L, M] | Literal[HypergraphSpecialTag.LEFT]
    segment: int
    node: hg.ObjectNode

class Node2Graph[L, M: cat.Morphism](NamedTuple):
    node: hg.ObjectNode
    segment: int
    graph: hg.Hypergraph[L, M] | Literal[HypergraphSpecialTag.RIGHT]

class Graph2Graph[L, M: cat.Morphism](NamedTuple):
    left_graph: hg.Hypergraph[L, M] | Literal[HypergraphSpecialTag.LEFT]
    cod_segment: int
    dom_segment: int
    right_graph: hg.Hypergraph[L, M] | Literal[HypergraphSpecialTag.RIGHT]

def cod_g2n[L, M: cat.Morphism](*target: hg.Hypergraph[L, M]) -> fd.Prod[Graph2Node[L, M]]:
    return tuple(
        Graph2Node(graph, i, cod.node)
        for graph in target
        for i, cod in enumerate(graph.cod)
    )

def dom_n2g[L, M: cat.Morphism](*target: hg.Hypergraph[L, M]) -> fd.Prod[Node2Graph[L, M]]:
    return tuple(
        Node2Graph(dom.node, i, graph)
        for graph in target
        for i, dom in enumerate(graph.dom)
    )
    
def graph_right_nodes(target: hg.Hypergraph) -> fd.Prod[Graph2Node]:
    return tuple(
        Graph2Node(graph=target, segment=i, node=cod.node)
        for i, cod in enumerate(target.cod)
    )

def graph_left_nodes(target: hg.Hypergraph) -> fd.Prod[Node2Graph]:
    return tuple(
        Node2Graph(node=dom.node, segment=i, graph=target)
        for i, dom in enumerate(target.dom)
    )

####
@dataclass
class HypergraphAnalysis[L, M: cat.Morphism, A = Any]:
    target: hg.Hypergraph[L, M]
    
    # includes codomain
    @cached_property
    def __node_right(self) -> util.Multidict[hg.ObjectNode, Node2Graph[L, M]]:
        _node_right = util.Multidict(
            (obj.node, Node2Graph(node=obj.node, segment=i, graph=subgraph))
            for subgraph in self.target.subgraphs()
            for i, obj in enumerate(subgraph.dom)
        ).update(
            (obj.node, Node2Graph(node=obj.node, segment=i, graph=HypergraphSpecialTag.RIGHT))
            for i, obj in enumerate(self.target.cod)
        )
        return _node_right
    
    @cached_property
    def __node_left(self) -> util.Multidict[hg.ObjectNode, Graph2Node[L, M]]:
        _node_left = util.Multidict(
            (obj.node, Graph2Node(graph=subgraph, segment=i, node=obj.node))
            for subgraph in self.target.subgraphs()
            for i, obj in enumerate(subgraph.cod)
        ).update(
            (obj.node, Graph2Node(graph=HypergraphSpecialTag.LEFT, segment=i, node=obj.node))
            for i, obj in enumerate(self.target.dom)
        )
        return _node_left
    
    ## Ensuring object order to avoid twisting extracted graphs.
    @cached_property
    def node_order(self) -> dict[hg.ObjectNode, int]:
        orders = {}
        objects = (
            obj for subgraph in self.subgraphs()
            for obj in subgraph.dom
        )
        counter = 0
        for obj in objects:
            if obj.node not in orders:
                orders[obj.node] = counter
                counter += 1
        return orders

    ## Discovering Objects
    @cached_property
    def __node_objects(self) -> util.Multidict[hg.ObjectNode, L]:
        return util.Multidict(
            (obj.node, obj.obj)
            for subgraph in (self.target, *self.subgraphs())
            for obj in subgraph.dom + subgraph.cod
        )
    
    def node_objects(self):
        return self.__node_objects
    
    def node_object(self, node: hg.ObjectNode) -> L:
        return util.iallequals(self.__node_objects[node])
    
    @cached_property
    def __node_as_cod_obj(self) -> dict[hg.ObjectNode, L]:
        return {
            **{
                obj.node: obj.obj
                for subgraph in self.target.subgraphs()
                for obj in subgraph.cod
            },
            **{
                obj.node: obj.obj
                for obj in self.target.dom
            }
        }
    
    @cached_property
    def __node_as_dom_obj(self) -> util.Multidict[hg.ObjectNode, L]:
        return util.Multidict(
            (obj.node, obj.obj)
            for subgraph in self.target.subgraphs()
            for obj in subgraph.dom
        )
    
    def node_as_cod_obj(self, node: hg.ObjectNode | hg.HypergraphObject) -> L:
        return self.__node_as_cod_obj[node.node if isinstance(node, hg.HypergraphObject) else node]
    def node_as_dom_obj(self, node: hg.ObjectNode | hg.HypergraphObject) -> L:
        candidates = tuple(
            self.__node_as_dom_obj[node.node if isinstance(node, hg.HypergraphObject) else node]
        )
        return util.iallequals(candidates)
    
    
    def nodes_right(self, *nodes: hg.ObjectNode | hg.HypergraphObject) -> fd.Prod[Node2Graph[L, M]]:
        return tuple(util.unique_concat(
            self.__node_right[
                node.node 
                if isinstance(node, hg.HypergraphObject) else node]
            for node in nodes
        ))
    
    def nodes_left(self, *nodes: hg.ObjectNode | hg.HypergraphObject) -> fd.Prod[Graph2Node[L, M]]:
        return tuple(util.unique_concat(
            self.__node_left[
                node.node 
                if isinstance(node, hg.HypergraphObject) else node]
            for node in nodes
        ))
    
    
    # def subgraph_right(self, target: hg.Hypergraph) -> fd.Prod[tuple[int, int, hg.Hypergraph[L, M] | Literal[HypergraphSpecialTag.RIGHT]]]:
    #     return tuple(
    #         (i, edge.segment, edge.graph)
    #         for i, cod in enumerate(target.cod)
    #         for edge in self.nodes_right(cod)
    #     )
    
    # def subgraph_left(self, target: hg.Hypergraph) -> fd.Prod[tuple[hg.Hypergraph[L, M] | Literal[HypergraphSpecialTag.LEFT], int, int]]:
    #     return tuple(
    #         (node.graph, j, i)
    #         for i, dom in enumerate(target.dom)
    #         for j, node in enumerate(self.nodes_left(dom))
    #     )
    
    # For finding things by treating outside as an origin / terminus
    def tag_udom(self, target: GraphTag[L, M]) -> fd.Prod[hg.ObjectNode]:
        return (
            target.udom() if isinstance(target, hg.Hypergraph)
            else self.target.ucod() if target == HypergraphSpecialTag.RIGHT
            else ()
        )
    def tag_ucod(self, target: GraphTag[L, M]) -> fd.Prod[hg.ObjectNode]:
        return (
            target.ucod() if isinstance(target, hg.Hypergraph)
            else self.target.udom() if target == HypergraphSpecialTag.LEFT
            else ()
        )

    # Finds subgraphs to the left and right of a target.
    def left_subgraphs(self, target: GraphTag[L, M]) -> fd.Prod[hg.Hypergraph[L, M] | Literal[HypergraphSpecialTag.LEFT]]:
        return tuple(util.unique_iterable(
            graph for graph, _, _ in self.nodes_left(*self.tag_udom(target))
        ))
    def right_subgraphs(self, target: GraphTag[L, M]) -> fd.Prod[hg.Hypergraph[L, M] | Literal[HypergraphSpecialTag.RIGHT]]:
        return tuple(util.unique_iterable(
            edge.graph for edge in self.nodes_right(*self.tag_ucod(target))
        ))
    
    def cod_nodes(self, target: hg.Hypergraph):
        return tuple(
            Graph2Node(graph=target, segment=i, node=cod.node)
            for i, cod in enumerate(target.cod)
        )
    
    def subgraphs(self) -> fd.Prod[hg.Hypergraph[L, M]]:
        return self.target.subgraphs()
    

    @cached_property
    def __hypergraphUID_to_aux(self) -> dict[fd.UID[hg.Hypergraph], A]:
        if isinstance(self.target, hg.AuxiliaryGraph):
            return {
                subgraph.uid: aux
                for subgraph, aux in self.target._subgraphs
            }
        return dict()
    def get_aux(self, target: hg.Hypergraph[L, M]) -> A | None:
        uid = target.uid
        return self.__hypergraphUID_to_aux.get(uid, None) # type: ignore
    
    # To extract exclusive

@dataclass
class AuxiliaryHypergraphAnalysis[L, M: cat.Morphism, A](HypergraphAnalysis[L, M]):
    target: hg.AuxiliaryGraph[L, M, A] # type: ignore

    def object_override(self, hg_obj: hg.HypergraphObject[L]) -> hg.HypergraphObject[L]:
        for dom_obj in self.target.dom + self.target.cod:
            if dom_obj.node == hg_obj.node:
                return dom_obj
        return hg_obj

    def extract_subgraph(self, *uid_targets: hg.Hypergraph[L, M] | fd.UID[hg.Hypergraph]) -> hg.AuxiliaryGraph[L, M, A]:
        uid_targets = tuple(
            t.uid if isinstance(t, hg.Hypergraph) else t
            for t in uid_targets
        )
        _subgraphs = tuple(
            subgraph_aux for subgraph_aux in self.target._subgraphs
            if subgraph_aux[0].uid in uid_targets
        )
        # NOTE: This causes twisting. The order of subgraph dom will not match the order of the objects before
        # it in the expression.
        subgraph_dom = tuple(util.unique_iterable(
            (self.object_override(dom)
            for subgraph, _ in _subgraphs
            for dom in subgraph.dom
            if any(
                subgraph_left == HypergraphSpecialTag.LEFT or
                subgraph_left.uid not in uid_targets
                for subgraph_left, _, _ in
                self.nodes_left(dom)
            )),
            unique_test=lambda obj: obj.node
        ))
        # The sorting fixes the twisting.
        subgraph_dom = tuple(
            sorted(subgraph_dom, key = lambda obj: self.node_order[obj.node], reverse=False)
        )
        subgraph_cod = util.unique_concat(
            ((cod
                for cod in self.target.cod
                if any(
                    graph.uid in uid_targets
                    for graph, _, _ in self.nodes_left(cod)
                    if isinstance(graph, hg.Hypergraph)
                )),
            (cod
                for subgraph, _ in _subgraphs
                for cod in subgraph.cod
                if any(
                    subgraph_right == HypergraphSpecialTag.RIGHT or
                    subgraph_right.uid not in uid_targets
                    for _, _, subgraph_right in
                    self.nodes_right(cod)
                    if isinstance(subgraph_right, hg.Hypergraph)
                )
            )),
            unique_test=lambda obj: obj.node
        )
        return type(self.target).construct(
            _subgraphs=_subgraphs,
            dom=tuple(subgraph_dom),
            cod=tuple(subgraph_cod)
        )

    def replace_multiple(self, 
        targets: Iterable[hg.Hypergraph[L, M] | fd.UID[hg.Hypergraph]],
        replacement: hg.Hypergraph[L, M],
        auxiliary_data: A
    ) -> hg.AuxiliaryGraph[L, M, A]:
        
        target_uids = tuple(
            t.uid if isinstance(t, hg.Hypergraph) else t
            for t in targets
        )

        # Need to check whether the object nodes match
        extracted_subgraph = self.extract_subgraph(*target_uids)
        if extracted_subgraph.udom() != replacement.udom() or extracted_subgraph.ucod() != replacement.ucod():
            raise ValueError("Replacement graph does not match the domain and codomain of the extracted subgraph.")
        
        new_subgraphs = tuple(
            (subgraph, aux) for subgraph, aux in
            self.target._subgraphs
            if subgraph.uid not in target_uids
        ) + ((replacement, auxiliary_data),)

        return type(self.target).construct(
            _subgraphs=new_subgraphs,
            dom=self.target.dom,
            cod=self.target.cod
        )

'''
For crawling we have the following approach:
- We have a dict[node, guide]
- We apply the operation to all subgraphs whose nodes fall within the guide
- We do this iteratively
- 
'''
@dataclass
class ReverseCrawler[L, M: cat.Morphism, G](ABC):

    @abstractmethod
    def seed_processor(self, target: M, guide: fd.Prod[G]) -> tuple[M, fd.Prod[G]]:
        raise NotImplementedError
    
    @abstractmethod
    def coalesce(self, guides: fd.Prod[G]) -> G:
        raise NotImplementedError
    
    @abstractmethod
    def generate_zero(self, source: fd.Prod[L]) -> G:
        raise NotImplementedError
    
    def update_aux[T](self, aux: T) -> T:
        return aux
    
    def __call__(self, target: hg.Hypergraph[L, M], guide: fd.Prod[G]) -> hg.Hypergraph[L, M]:
        graph, guide = self.crawl(target, guide)
        return graph

    def crawl(self, 
              target: hg.Hypergraph[L, M], 
              guide: fd.Prod[G]) -> tuple[hg.Hypergraph[L, M], fd.Prod[G]]:
        match target:
            case hg.HypergraphRoot(wraps=wraps):
                new_wraps, new_guide = self.seed_processor(wraps, guide)
                new_graph = hg.HypergraphRoot.template(
                    new_wraps, 
                    target.udom(),
                    target.ucod()
                )
                return new_graph, new_guide
            case hg.HypergraphBlock(body=body, block_tag=block_tag):
                new_body, new_guide = self.crawl(body, guide)
                new_graph = hg.HypergraphBlock.template(
                    body=new_body,
                    block_tag=block_tag
                )
                return new_graph, new_guide
            case hg.AuxiliaryGraph():
                return self.crawl_auxiliary(target, guide)
            case _:
                raise NotImplementedError(f"Unsupported graph type {type(target)} in crawl.")


    def object_generator(
        self,
        original_dom: fd.Prod[hg.HypergraphObject[L]],
        original_cod: fd.Prod[hg.HypergraphObject[L]],
        dom_guide: fd.Prod[G],
        cod_guide: fd.Prod[G],
    ) -> tuple[Iterable[L], Iterable[L]]:
        return (
            (obj_node.obj for obj_node in original_dom),
            (obj_node.obj for obj_node in original_cod)
        )

    def crawl_auxiliary[A](self, 
            target: hg.AuxiliaryGraph[L, M, A], 
            guide: fd.Prod[G]) -> tuple[hg.AuxiliaryGraph[L, M, A], fd.Prod[G]]:
        analyzer = HypergraphAnalysis(target)

        processed_node_guide = dict(zip(analyzer.target.ucod(), guide))
        unprocessed_node_guide: util.Multidict[hg.ObjectNode, G] = util.Multidict()

        unprocessed_subgraphs = set(target._subgraphs)
        processed_subgraphs: set[GraphTag[L, M]] = set((HypergraphSpecialTag.LEFT, HypergraphSpecialTag.RIGHT,))

        #new_graphs: list[tuple[hg.Hypergraph[L, M], A]] = []
        new_graph_key: dict[fd.UID[hg.Hypergraph], tuple[hg.Hypergraph[L, M], A]] = {}

        # Generate Zeros
        for node, objects in analyzer.node_objects().items():
            if not analyzer.nodes_right(node):
                processed_node_guide[node] = self.generate_zero(tuple(objects))

        while unprocessed_subgraphs:
            processable_subgraphs = tuple(
                (subgraph, aux) for subgraph, aux in set(unprocessed_subgraphs)
                if all(
                    node in processed_node_guide
                    for _, _, node in analyzer.cod_nodes(subgraph)
                )
            )
            if not processable_subgraphs:
                raise ValueError('No processable subgraphs found, but unprocessed subgraphs remain. This likely indicates a cycle in the graph.')
            for subgraph, aux in processable_subgraphs:
                unprocessed_subgraphs.remove((subgraph, aux))
                subgraph_guide = tuple(
                    processed_node_guide[node]
                    for node in subgraph.ucod()
                )
                new_subgraph, new_guide = self.crawl(subgraph, subgraph_guide)
                #new_graphs.append((new_subgraph, self.update_aux(aux)))
                new_graph_key[subgraph.uid] = (new_subgraph, self.update_aux(aux))

                unprocessed_node_guide.update(
                    (node, new_guide[i])
                    for i, node in enumerate(subgraph.udom())
                )
                processed_subgraphs.add(subgraph)

                for node in subgraph.udom():
                    if node not in processed_node_guide and \
                        all(edge.graph in processed_subgraphs for edge in analyzer.nodes_right(node)):
                        processed_node_guide[node] = self.coalesce(tuple(
                            unprocessed_node_guide[node]
                        ))

        cod_guides = tuple(
            processed_node_guide[cod]
            for cod in target.ucod()
        )
        dom_guides = tuple(
            processed_node_guide[dom]
            for dom in target.udom()
        )
        dom, cod = self.object_generator(
            target.dom,
            target.cod,
            dom_guides,
            cod_guides
        )
        graph_dom = hg.HypergraphObject.template(
            cat.ProdObject.from_iter(dom),
            tuple(obj.node for obj in target.dom)
        )
        graph_cod = hg.HypergraphObject.template(
            cat.ProdObject.from_iter(cod),
            tuple(obj.node for obj in target.cod)
        )
        return type(target).construct(
            _subgraphs=tuple(
                new_graph_key[subgraph.uid]
                for subgraph, _ in target._subgraphs
            ),
            dom=tuple(graph_dom),
            cod=tuple(graph_cod)
        ), tuple(
            processed_node_guide[dom] 
            for dom in target.udom()
        )