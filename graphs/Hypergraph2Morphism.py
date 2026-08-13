from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Type, Iterable, Literal, Callable, NamedTuple, Self

import data_structure.Term as fd # for 'foundations'
import data_structure.Numeric as nm
import utilities.utilities as util
import data_structure.Category as cat
import construction_helpers.product as chp

from functools import cached_property, cache
import graphs.Hypergraph as hg
import graphs.HypergraphAnalysis as hga

from abc import ABC, abstractmethod

@dataclass
class Branch[L, M: cat.Morphism](ABC):
    @abstractmethod
    def morphism(self) -> cat.ProdCategory[L, M]: ...
    @abstractmethod
    def right_nodes(self) -> fd.Prod[hg.ObjectNode]: ...
    @abstractmethod
    def left_nodes(self) -> fd.Prod[hg.ObjectNode]: ...
    @abstractmethod
    def newly_processed(self) -> fd.Prod[hga.GraphTag[L, M]]: ...

@dataclass
class RootBranch[L, M: cat.Morphism](Branch[L, M]):
    _morphism: cat.ProdCategory[L, M]
    _right_nodes: fd.Prod[hg.ObjectNode]
    _left_nodes: fd.Prod[hg.ObjectNode]
    _newly_processed: fd.Prod[hga.GraphTag[L, M]]
    def morphism(self): return self._morphism
    def right_nodes(self): return self._right_nodes
    def left_nodes(self): return self._left_nodes
    def newly_processed(self): return self._newly_processed

@dataclass
class IdentityBranch[L, M: cat.Morphism](Branch[L, M]):
    _node: hg.ObjectNode
    _obj: L
    def morphism(self): return cat.ProdObject((self._obj,)).identity()
    def right_nodes(self): return (self._node,)
    def left_nodes(self): return (self._node,)
    def newly_processed(self): return ()

    @classmethod
    def from_node(cls, node: hg.ObjectNode, analysis: hga.HypergraphAnalysis):
        return cls(
            _node=node,
            _obj=analysis.node_as_cod_obj(node)
        )

@dataclass
class NestedBranch[L, M: cat.Morphism](Branch[L, M]):
    # A subgraph that is itself an AuxiliaryGraph is expanded by its own
    # analysis, so the branch it produces reports the graphs *inside* it. The
    # parent needs the subgraph itself, otherwise it never lands in `ignore`
    # and gets emitted again at the next node.
    _branch: Branch[L, M]
    _graph: hg.Hypergraph[L, M]
    def morphism(self): return self._branch.morphism()
    def right_nodes(self): return self._branch.right_nodes()
    def left_nodes(self): return self._branch.left_nodes()
    def newly_processed(self): return (self._graph,)

def exclusive(
        nodes: fd.Prod[hg.ObjectNode],
        local: fd.Prod[hga.GraphTag],
        ignore: fd.Prod[hga.GraphTag],
        analysis: hga.HypergraphAnalysis):
    graphs = util.unique_iterable(
        graph for graph, _, _ in 
        analysis.nodes_left(*nodes)
        if set(analysis.tag_ucod(graph)) <= set(nodes)
        and set(analysis.right_subgraphs(graph)) <= set(local)
        and graph not in ignore
    )
    return tuple(graphs)

def make_rearrangement(
    left_nodes: fd.Prod[hg.ObjectNode],
    right_nodes: fd.Prod[hg.ObjectNode],
    analysis: hga.HypergraphAnalysis
) -> tuple[()] | tuple[cat.Rearrangement]:
    if left_nodes == right_nodes:
        return ()
    dom = tuple(
        analysis.node_as_cod_obj(node) for node in left_nodes
    )
    list_left_nodes = list(left_nodes)

    mapping = tuple(list_left_nodes.index(node) for node in right_nodes)
    return (cat.Rearrangement(
        mapping=mapping,
        _dom=dom,
    ),)

def rolled_subbranches(
    nodes: fd.Prod[hg.ObjectNode],
    local: fd.Prod[hga.GraphTag],
    ignore: fd.Prod[hga.GraphTag],
    analysis: hga.HypergraphAnalysis
) -> fd.Prod[Branch]:
    branches = ()
    for n in range(len(nodes)):

        rolled = nodes[:n+1]
        current_node = nodes[n]

        exclusive_subgraphs = exclusive(
            rolled, 
            local, 
            ignore, 
            analysis)

        used_elsewhere = tuple(
            graph for graph, _, _
            in analysis.nodes_left(current_node)
            if graph == hga.HypergraphSpecialTag.LEFT
            or any(
                next_graph not in local
                for next_graph in analysis.right_subgraphs(graph)
            )
        )

        if (not exclusive_subgraphs) and used_elsewhere:
            branches += (IdentityBranch.from_node(current_node, analysis),)
            continue

        branches += tuple(
            make_subbranch(subgraph, analysis)
            for subgraph in exclusive_subgraphs
        )

        ignore += util.concat(
            branch.newly_processed() for branch in branches
        )

    return branches

@dataclass
class ProductBranch[L, M: cat.Morphism](Branch[L, M]):
    _branches: fd.Prod[Branch[L, M]]
    def morphism(self):
        return (
            self._branches[0].morphism()
            if len(self._branches) == 1
            else cat.ProductOfMorphisms.from_iter(
                branch.morphism() for branch in self._branches
            )
        )
    def left_nodes(self):
        return util.concat(
            branch.left_nodes() for branch in self._branches
        )
    def right_nodes(self):
        return util.concat(
            branch.right_nodes() for branch in self._branches
        )
    def newly_processed(self):
        return util.concat(
            branch.newly_processed() for branch in self._branches
        )
    @classmethod
    def template(cls, branches: fd.Prod[Branch[L, M]]) -> ProductBranch[L, M]:
        return cls(_branches=branches)


def make_stacks(
    right_nodes: fd.Prod[hg.ObjectNode],
    local: fd.Prod[hga.GraphTag],
    analysis: hga.HypergraphAnalysis
) -> fd.Prod[Branch]:
    
    right_branch = ProductBranch.template(
        rolled_subbranches(right_nodes, local, local, analysis)
    )
    newly_processed = right_branch.newly_processed()

    if not newly_processed:
        return ()
    
    unique_left_nodes = util.unique_tuple(right_branch.left_nodes())

    local = (*local, *newly_processed)

    return (*make_stacks(unique_left_nodes, local, analysis), right_branch)

def expand_composed[L, M: cat.Morphism](target: cat.ProdCategory[L, M]) -> fd.Prod[cat.ProdCategory[L, M]]:
    match target:
        case cat.ProductOfMorphisms(content=(single,)):
            return expand_composed(single)
        case cat.Composed(content=content):
            return util.concat(map(expand_composed, content))
        case _:
            return (target,)
        


@dataclass
class ComposedBranch[L, M: cat.Morphism](Branch[L, M]):
    _stacks: fd.Prod[Branch[L, M]]
    _left_nodes: fd.Prod[hg.ObjectNode] | None
    _analysis: hga.HypergraphAnalysis

    def left_nodes(self):
        return self._left_nodes or util.unique_tuple(self._stacks[0].left_nodes())
    def right_nodes(self):
        return self._stacks[-1].right_nodes()
    def newly_processed(self):
        return util.concat(
            branch.newly_processed() for branch in self._stacks
        )
    def morphism(self) -> cat.ProdCategory[L, M]:
        content = util.concat(
            (*make_rearrangement(
                util.unique_tuple(branch.left_nodes())
                if i != 0 else self.left_nodes(),
                branch.left_nodes(), self._analysis),
                branch.morphism())
            for i, branch in enumerate(self._stacks)
        )

        expanded = util.concat(map(expand_composed, content))
        match expanded:
            case ():
                return cat.ProdObject.from_iter(self._analysis.node_object(node) for node in self.left_nodes()).identity()
            case (single,):
                return single
            case content:
                return cat.Composed(content)

    @classmethod
    def template(cls, 
                 stacks: fd.Prod[Branch[L, M]], 
                 left_nodes: fd.Prod[hg.ObjectNode] | None,
                 analysis: hga.HypergraphAnalysis) -> ComposedBranch[L, M]:
        return ComposedBranch(
            stacks,
            left_nodes,
            analysis
        )

def make_subbranch(
    origin: hga.GraphTag,
    analysis: hga.HypergraphAnalysis
):
    nodes = analysis.tag_udom(origin)
    local = (origin, hga.HypergraphSpecialTag.LEFT)

    origin_branches = (
        () if isinstance(origin, hga.HypergraphSpecialTag)
        else (NestedBranch(graph_to_branch(origin), origin),)
    )

    stacks = (
        *make_stacks(nodes, local, analysis), 
        *origin_branches
    )
    left_nodes = (
        analysis.target.udom()
        if origin == hga.HypergraphSpecialTag.RIGHT
        else None
    )
    return ComposedBranch.template(
        stacks, left_nodes, analysis
    )

def graph_to_branch[L, M: cat.Morphism](
    target: hg.Hypergraph[L, M],
    analysis: hga.HypergraphAnalysis[L, M] | None = None
):
    match target:
        case hg.HypergraphRoot():
            return RootBranch(
                _morphism=target.wraps,
                _right_nodes=target.ucod(),
                _left_nodes=target.udom(),
                _newly_processed=(target,)
            )
        case hg.HypergraphBlock(body=body, block_tag=block_tag):
            return RootBranch(
                _morphism=cat.Block(
                    body=hypergraph_to_morphism(body),
                    block_tag=block_tag
                ),
                _right_nodes=target.ucod(),
                _left_nodes=target.udom(),
                _newly_processed=(target,) # type: ignore
            )
        case hg.AuxiliaryGraph():
            analysis = analysis or hga.HypergraphAnalysis(target)
            return make_subbranch(
                origin=hga.HypergraphSpecialTag.RIGHT,
                analysis=analysis
            )
    raise ValueError(f"Unknown hypergraph type: {target}")

def hypergraph_to_morphism[L, M: cat.Morphism](
    target: hg.Hypergraph[L, M],
):# -> Composed[Any, ProdCategory[Any, Any]] | Any | Rearrangement...:
    return graph_to_branch(target).morphism()

def recycle[L, M: cat.Morphism](
    target: cat.ProdCategory[L, M]
):
    graph = hg.StructuredHypergraph.from_morphism(target)
    return hypergraph_to_morphism(graph)