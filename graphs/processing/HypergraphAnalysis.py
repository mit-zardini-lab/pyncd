from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Iterable, Literal, NamedTuple

import data_structure.Term as fd # for 'foundations'
import data_structure.Numeric as nm
import utilities.utilities as util
import data_structure.Category as cat
import construction_helpers.product as chp

from functools import cached_property, cache
import graphs.data_structure.Hypergraph as hg

import enum


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
    node: hg.HypergraphObject[L]

class Node2Graph[L, M: cat.Morphism](NamedTuple):
    node: hg.HypergraphObject[L]
    segment: int
    graph: hg.Hypergraph[L, M] | Literal[HypergraphSpecialTag.RIGHT]

class Graph2Graph[L, M: cat.Morphism](NamedTuple):
    left_graph: hg.Hypergraph[L, M] | Literal[HypergraphSpecialTag.LEFT]
    cod_segment: int
    dom_segment: int
    right_graph: hg.Hypergraph[L, M] | Literal[HypergraphSpecialTag.RIGHT]

def cod_g2n[L, M: cat.Morphism](*target: hg.Hypergraph[L, M]) -> fd.Prod[Graph2Node[L, M]]:
    return tuple(
        Graph2Node(graph, i, cod)
        for graph in target
        for i, cod in enumerate(graph.cod)
    )

def dom_n2g[L, M: cat.Morphism](*target: hg.Hypergraph[L, M]) -> fd.Prod[Node2Graph[L, M]]:
    return tuple(
        Node2Graph(dom, i, graph)
        for graph in target
        for i, dom in enumerate(graph.dom)
    )

def graph_right_nodes(target: hg.Hypergraph) -> fd.Prod[Graph2Node]:
    return tuple(
        Graph2Node(graph=target, segment=i, node=cod)
        for i, cod in enumerate(target.cod)
    )

def graph_left_nodes(target: hg.Hypergraph) -> fd.Prod[Node2Graph]:
    return tuple(
        Node2Graph(node=dom, segment=i, graph=target)
        for i, dom in enumerate(target.dom)
    )

####
@dataclass
class HypergraphAnalysis[L, M: cat.Morphism]:
    target: hg.Hypergraph[L, M]

    # includes codomain
    @cached_property
    def __node_right(self) -> util.Multidict[hg.HypergraphObject[L], Node2Graph[L, M]]:
        _node_right = util.Multidict(
            (obj, Node2Graph(node=obj, segment=i, graph=subgraph))
            for subgraph in self.target.subgraphs()
            for i, obj in enumerate(subgraph.dom)
        ).update(
            (obj, Node2Graph(node=obj, segment=i, graph=HypergraphSpecialTag.RIGHT))
            for i, obj in enumerate(self.target.cod)
        )
        return _node_right

    @cached_property
    def __node_left(self) -> util.Multidict[hg.HypergraphObject[L], Graph2Node[L, M]]:
        _node_left = util.Multidict(
            (obj, Graph2Node(graph=subgraph, segment=i, node=obj))
            for subgraph in self.target.subgraphs()
            for i, obj in enumerate(subgraph.cod)
        ).update(
            (obj, Graph2Node(graph=HypergraphSpecialTag.LEFT, segment=i, node=obj))
            for i, obj in enumerate(self.target.dom)
        )
        return _node_left

    ## Ensuring object order to avoid twisting extracted graphs.
    @cached_property
    def node_order(self) -> dict[hg.HypergraphObject[L], int]:
        orders: dict[hg.HypergraphObject[L], int] = {}
        objects = (
            obj for subgraph in self.subgraphs()
            for obj in subgraph.dom
        )
        counter = 0
        for obj in objects:
            if obj not in orders:
                orders[obj] = counter
                counter += 1
        return orders

    ## Discovering Objects
    @cached_property
    def __node_objects(self) -> util.Multidict[hg.HypergraphObject[L], L]:
        return util.Multidict(
            (obj, obj.obj)
            for subgraph in (self.target, *self.subgraphs())
            for obj in subgraph.dom + subgraph.cod
        )

    def node_objects(self):
        return self.__node_objects

    def node_object(self, node: hg.HypergraphObject[L]) -> L:
        return util.iallequals(self.__node_objects[node])

    @cached_property
    def __node_as_cod_obj(self) -> dict[hg.HypergraphObject[L], L]:
        return {
            **{
                obj: obj.obj
                for subgraph in self.target.subgraphs()
                for obj in subgraph.cod
            },
            **{
                obj: obj.obj
                for obj in self.target.dom
            }
        }

    @cached_property
    def __node_as_dom_obj(self) -> util.Multidict[hg.HypergraphObject[L], L]:
        return util.Multidict(
            (obj, obj.obj)
            for subgraph in self.target.subgraphs()
            for obj in subgraph.dom
        )

    def node_as_cod_obj(self, node: hg.HypergraphObject[L]) -> L:
        return self.__node_as_cod_obj[node]
    def node_as_dom_obj(self, node: hg.HypergraphObject[L]) -> L:
        return util.iallequals(tuple(self.__node_as_dom_obj[node]))


    def nodes_right(self, *nodes: hg.HypergraphObject[L]) -> fd.Prod[Node2Graph[L, M]]:
        return tuple(util.unique_concat(
            self.__node_right[node] for node in nodes
        ))

    def nodes_left(self, *nodes: hg.HypergraphObject[L]) -> fd.Prod[Graph2Node[L, M]]:
        return tuple(util.unique_concat(
            self.__node_left[node] for node in nodes
        ))


    # For finding things by treating outside as an origin / terminus
    def tag_dom(self, target: GraphTag[L, M]) -> fd.Prod[hg.HypergraphObject[L]]:
        return (
            target.dom if isinstance(target, hg.Hypergraph)
            else self.target.cod if target == HypergraphSpecialTag.RIGHT
            else ()
        )
    def tag_cod(self, target: GraphTag[L, M]) -> fd.Prod[hg.HypergraphObject[L]]:
        return (
            target.cod if isinstance(target, hg.Hypergraph)
            else self.target.dom if target == HypergraphSpecialTag.LEFT
            else ()
        )

    # Finds subgraphs to the left and right of a target.
    def left_subgraphs(self, target: GraphTag[L, M]) -> fd.Prod[hg.Hypergraph[L, M] | Literal[HypergraphSpecialTag.LEFT]]:
        return tuple(util.unique_iterable(
            graph for graph, _, _ in self.nodes_left(*self.tag_dom(target))
        ))
    def right_subgraphs(self, target: GraphTag[L, M]) -> fd.Prod[hg.Hypergraph[L, M] | Literal[HypergraphSpecialTag.RIGHT]]:
        return tuple(util.unique_iterable(
            edge.graph for edge in self.nodes_right(*self.tag_cod(target))
        ))

    def cod_nodes(self, target: hg.Hypergraph):
        return tuple(
            Graph2Node(graph=target, segment=i, node=cod)
            for i, cod in enumerate(target.cod)
        )

    def subgraphs(self) -> fd.Prod[hg.Hypergraph[L, M]]:
        return self.target.subgraphs()

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

    def __call__(self, target: hg.Hypergraph[L, M], guide: fd.Prod[G]) -> hg.Hypergraph[L, M]:
        graph, guide = self.crawl(target, guide)
        return graph

    def crawl(self,
              target: hg.Hypergraph[L, M],
              guide: fd.Prod[G]) -> tuple[hg.Hypergraph[L, M], fd.Prod[G]]:
        match target:
            case hg.HypergraphRoot(wraps=wraps):
                new_wraps, new_guide = self.seed_processor(wraps, guide)
                # TODO: do we keep the UID?
                new_graph = hg.HypergraphRoot.template(
                    new_wraps,
                    target.dom,
                    target.cod
                )
                return new_graph, new_guide
            case hg.HypergraphBlock(body=body, block_tag=block_tag):
                new_body, new_guide = self.crawl(body, guide)
                new_graph = hg.HypergraphBlock.template(
                    body=new_body,
                    block_tag=block_tag
                )
                return new_graph, new_guide
            case hg.Multigraph():
                return self.crawl_multigraph(target, guide)
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

    def crawl_multigraph(self,
            target: hg.Multigraph[L, M],
            guide: fd.Prod[G]) -> tuple[hg.Multigraph[L, M], fd.Prod[G]]:
        analyzer = HypergraphAnalysis(target)

        processed_node_guide = dict(zip(target.cod, guide))
        unprocessed_node_guide: util.Multidict[hg.HypergraphObject[L], G] = util.Multidict()

        unprocessed_subgraphs = set(target._subgraphs)
        processed_subgraphs: set[GraphTag[L, M]] = set((HypergraphSpecialTag.LEFT, HypergraphSpecialTag.RIGHT,))

        new_graph_key: dict[fd.UID[hg.Hypergraph], hg.Hypergraph[L, M]] = {}

        # Generate Zeros
        for node, objects in analyzer.node_objects().items():
            if not analyzer.nodes_right(node):
                processed_node_guide[node] = self.generate_zero(tuple(objects))

        while unprocessed_subgraphs:
            processable_subgraphs = tuple(
                subgraph for subgraph in set(unprocessed_subgraphs)
                if all(
                    node in processed_node_guide
                    for _, _, node in analyzer.cod_nodes(subgraph)
                )
            )
            if not processable_subgraphs:
                raise ValueError('No processable subgraphs found, but unprocessed subgraphs remain. This likely indicates a cycle in the graph.')
            for subgraph in processable_subgraphs:
                unprocessed_subgraphs.remove(subgraph)
                subgraph_guide = tuple(
                    processed_node_guide[node]
                    for node in subgraph.cod
                )
                new_subgraph, new_guide = self.crawl(subgraph, subgraph_guide)
                new_graph_key[subgraph.uid] = new_subgraph

                unprocessed_node_guide.update(
                    (node, new_guide[i])
                    for i, node in enumerate(subgraph.dom)
                )
                processed_subgraphs.add(subgraph)

                for node in subgraph.dom:
                    if node not in processed_node_guide and \
                        all(edge.graph in processed_subgraphs for edge in analyzer.nodes_right(node)):
                        processed_node_guide[node] = self.coalesce(tuple(
                            unprocessed_node_guide[node]
                        ))

        cod_guides = tuple(
            processed_node_guide[cod]
            for cod in target.cod
        )
        dom_guides = tuple(
            processed_node_guide[dom]
            for dom in target.dom
        )
        dom, cod = self.object_generator(
            target.dom,
            target.cod,
            dom_guides,
            cod_guides
        )
        graph_dom = hg.HypergraphObject.template(
            cat.ProdObject.from_iter(dom),
            target.dom
        )
        graph_cod = hg.HypergraphObject.template(
            cat.ProdObject.from_iter(cod),
            target.cod
        )
        return hg.Multigraph.template(
            dom=tuple(graph_dom),
            cod=tuple(graph_cod),
            subgraphs=tuple(
                new_graph_key[subgraph.uid]
                for subgraph in target._subgraphs
            )
        ), tuple(
            processed_node_guide[dom]
            for dom in target.dom
        )
