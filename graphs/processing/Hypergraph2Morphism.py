from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Type, Iterable, Literal, Callable, NamedTuple, Self

import data_structure.Term as fd # for 'foundations'
import data_structure.Numeric as nm
import utilities.utilities as util
import data_structure.Category as cat
import construction_helpers.product as chp
import term_utilities.term_utilities as tutil
from construction_helpers import simple_helper as chsh

from functools import cached_property, cache
import graphs.data_structure.Hypergraph as hg
import graphs.processing.HypergraphAnalysis as hga

from abc import ABC, abstractmethod

@dataclass
class Branch[L, M: cat.Morphism](ABC):
    @abstractmethod
    def morphism(self) -> cat.ProdCategory[L, M]: ...
    @abstractmethod
    def right_nodes(self) -> fd.Prod[hg.HypergraphObject]: ...
    @abstractmethod
    def left_nodes(self) -> fd.Prod[hg.HypergraphObject]: ...
    @abstractmethod
    def newly_processed(self) -> fd.Prod[hga.GraphTag[L, M]]: ...

@dataclass
class RootBranch[L, M: cat.Morphism](Branch[L, M]):
    _morphism: cat.ProdCategory[L, M]
    _right_nodes: fd.Prod[hg.HypergraphObject]
    _left_nodes: fd.Prod[hg.HypergraphObject]
    _newly_processed: fd.Prod[hga.GraphTag[L, M]]
    def morphism(self): return self._morphism
    def right_nodes(self): return self._right_nodes
    def left_nodes(self): return self._left_nodes
    def newly_processed(self): return self._newly_processed

@dataclass
class IdentityBranch[L, M: cat.Morphism](Branch[L, M]):
    _node: hg.HypergraphObject
    _obj: L
    def morphism(self): return cat.ProdObject((self._obj,)).identity()
    def right_nodes(self): return (self._node,)
    def left_nodes(self): return (self._node,)
    def newly_processed(self): return ()

    @classmethod
    def from_node(cls, node: hg.HypergraphObject, analysis: hga.HypergraphAnalysis):
        return cls(
            _node=node,
            _obj=analysis.node_as_cod_obj(node)
        )

@dataclass
class NestedBranch[L, M: cat.Morphism](Branch[L, M]):
    # A subgraph that is itself a Multigraph is expanded by its own
    # analysis, so the branch it produces reports the graphs *inside* it. The
    # parent needs the subgraph itself, otherwise it never lands in `ignore`
    # and gets emitted again at the next node.
    _branch: Branch[L, M]
    _graph: hg.Hypergraph[L, M]
    def morphism(self): return self._branch.morphism()
    def right_nodes(self): return self._branch.right_nodes()
    def left_nodes(self): return self._branch.left_nodes()
    def newly_processed(self): return (self._graph,)

def boundary_cod(
        graph: hga.GraphTag,
        analysis: hga.HypergraphAnalysis) -> fd.Prod[hg.HypergraphObject]:
    '''The codomain nodes that sequencing has to wait for.

    Two kinds are excluded. The first is a wire the graph passes through
    untouched. A block's passenger keeps its node, so the node appears in both
    the graph's domain and its codomain, the graph becomes its own neighbour,
    and a chain of such blocks forms a false cycle.

    The second is a dangling output, meaning a wire that is dropped and
    consumed by nothing else. Such a wire never enters a frontier, and
    `ComposedBranch._right_override` projects it away again at the end.
    '''
    dom = set(analysis.tag_dom(graph))
    return tuple(
        node for node in analysis.tag_cod(graph)
        if node not in dom
        and any(edge.graph != graph for edge in analysis.nodes_right(node))
    )

def gating_right(
        graph: hga.GraphTag,
        analysis: hga.HypergraphAnalysis) -> fd.Prod[hga.GraphTag]:
    '''The consumers that must already be placed for `graph` to be emitted:
    those of the outputs on its codomain alone.'''
    return tuple(util.unique_iterable(
        edge.graph
        for edge in analysis.nodes_right(*boundary_cod(graph, analysis))
    ))

def exclusive(
        nodes: fd.Prod[hg.HypergraphObject],
        local: fd.Prod[hga.GraphTag],
        ignore: fd.Prod[hga.GraphTag],
        analysis: hga.HypergraphAnalysis):
    graphs = util.unique_iterable(
        graph for graph, _, _ in
        analysis.nodes_left(*nodes)
        if set(boundary_cod(graph, analysis)) <= set(nodes)
        and set(gating_right(graph, analysis)) <= set(local)
        and graph not in ignore
    )
    return tuple(graphs)

def make_rearrangement(
    left_nodes: fd.Prod[hg.HypergraphObject],
    right_nodes: fd.Prod[hg.HypergraphObject],
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

@dataclass
class SinkPlacement:
    '''The sinks of one conversion, each waiting to be injected into a column.

    A sink is a subgraph with an empty codomain. `Para.Drop` is the case. It
    saves a value to a tape slot and returns no wire.

    The demand frontier starts at the graph's codomain and walks left, and a
    sink is never demanded, so `exclusive` never emits one. Each sink is
    instead built into a branch before the main walk starts, by
    `make_subbranch`, so the branch holds the sink and every producer that
    feeds the sink alone. The branch then waits in `waiting`, keyed by each
    node of its domain. When a column is built whose branches consume one of
    those nodes, `inject_waiting_sinks` places the branch in that column, next
    to the consumer. The walk builds columns from right to left, so the column
    that claims a sink is the rightmost one consuming a wire of its domain,
    which is as far right as the sink can go without holding the wire. The
    wire is delivered once by the rearrangement in front of the column, and
    the column copies it.

    `claimed` holds every sink already injected, so a sink waiting on two
    nodes is placed once.
    '''
    waiting: dict[hg.HypergraphObject, list[hg.Hypergraph]] = field(
        default_factory=dict)
    branches: dict[hg.Hypergraph, Branch] = field(default_factory=dict)
    claimed: set[hg.Hypergraph] = field(default_factory=set)

    def register(self, sink: hg.Hypergraph, branch: Branch) -> None:
        self.branches[sink] = branch
        for node in util.unique_tuple(branch.left_nodes()):
            self.waiting.setdefault(node, []).append(sink)

    def claim(self, node: hg.HypergraphObject) -> fd.Prod[Branch]:
        sinks = tuple(
            sink for sink in self.waiting.get(node, ())
            if sink not in self.claimed)
        self.claimed.update(sinks)
        return tuple(self.branches[sink] for sink in sinks)

    def unclaimed(self) -> fd.Prod[hg.Hypergraph]:
        return tuple(
            sink for sink in self.branches if sink not in self.claimed)


def inject_waiting_sinks(
    branches: fd.Prod[Branch],
    placement: SinkPlacement,
) -> fd.Prod[Branch]:
    '''`branches` with each waiting sink inserted after the first branch that
    consumes a node of its domain. An injected branch is scanned in turn, so a
    sink waiting on a wire that another sink's chain consumes follows it into
    the same column.'''
    result: list[Branch] = []
    queue = list(branches)
    while queue:
        branch = queue.pop(0)
        result.append(branch)
        injected = util.concat(
            placement.claim(node)
            for node in util.unique_tuple(branch.left_nodes()))
        queue = [*injected, *queue]
    return tuple(result)


def rolled_subbranches(
    nodes: fd.Prod[hg.HypergraphObject],
    local: fd.Prod[hga.GraphTag],
    ignore: fd.Prod[hga.GraphTag],
    analysis: hga.HypergraphAnalysis,
    placement: SinkPlacement,
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
                for next_graph in gating_right(graph, analysis)
            )
        )

        if (not exclusive_subgraphs) and used_elsewhere:
            # Only hold the wire if nothing already emitted carries it: a
            # branch that passes the node through delivers it itself, and a
            # redundant identity here becomes a copy that rides the whole
            # composition unused before being deleted at the end.
            if not any(current_node in branch.right_nodes()
                       for branch in branches):
                branches += (IdentityBranch.from_node(current_node, analysis),)
            continue

        branches += tuple(
            make_subbranch(subgraph, analysis, placement)
            for subgraph in exclusive_subgraphs
        )

        ignore += util.concat(
            branch.newly_processed() for branch in branches
        )

    return inject_waiting_sinks(branches, placement)

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
    right_nodes: fd.Prod[hg.HypergraphObject],
    local: fd.Prod[hga.GraphTag],
    analysis: hga.HypergraphAnalysis,
    placement: SinkPlacement,
) -> fd.Prod[Branch]:

    right_branch = ProductBranch.template(
        rolled_subbranches(right_nodes, local, local, analysis, placement)
    )
    newly_processed = right_branch.newly_processed()

    if not newly_processed:
        return ()

    unique_left_nodes = util.unique_tuple(right_branch.left_nodes())

    local = (*local, *newly_processed)

    return (*make_stacks(unique_left_nodes, local, analysis, placement),
            right_branch)

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
    _left_nodes: fd.Prod[hg.HypergraphObject] | None
    _analysis: hga.HypergraphAnalysis
    # The cod the caller requested. A dangling output, meaning a dropped wire,
    # rides the stacks, which `exclusive` tolerates, and is projected away
    # here.
    _right_override: fd.Prod[hg.HypergraphObject] | None = None

    def left_nodes(self):
        return self._left_nodes or util.unique_tuple(self._stacks[0].left_nodes())
    def right_nodes(self):
        return (self._right_override if self._right_override is not None
                else self._stacks[-1].right_nodes())
    def newly_processed(self):
        return util.concat(
            branch.newly_processed() for branch in self._stacks
        )
    def morphism(self) -> cat.ProdCategory[L, M]:
        # Each stack is aligned to what the previous stack actually emits,
        # rather than to unique_tuple of its own left nodes. The two agree
        # whenever rolled_subbranches delivers nodes at their requested
        # positions. A branch can deliver a node later than requested, which
        # happens for a subgraph with several outputs that becomes exclusive
        # only at its last output's roll position. The actual order then
        # differs from the assumed one, the assumed rearrangement misses the
        # permutation, and the wiring follows position rather than node. Sparse
        # expansion's TopK blocks, which have two outputs, surfaced it.
        content: fd.Prod[cat.ProdCategory[L, M]] = ()
        previous_right = self.left_nodes()
        for branch in self._stacks:
            content = (*content,
                       *make_rearrangement(
                           previous_right, branch.left_nodes(),
                           self._analysis),
                       branch.morphism())
            previous_right = branch.right_nodes()
        if self._right_override is not None and self._stacks:
            content = (*content, *make_rearrangement(
                self._stacks[-1].right_nodes(),
                self._right_override, self._analysis))

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
                 left_nodes: fd.Prod[hg.HypergraphObject] | None,
                 analysis: hga.HypergraphAnalysis,
                 right_override: fd.Prod[hg.HypergraphObject] | None = None,
                 ) -> ComposedBranch[L, M]:
        return ComposedBranch(
            stacks,
            left_nodes,
            analysis,
            right_override,
        )

def make_subbranch(
    origin: hga.GraphTag,
    analysis: hga.HypergraphAnalysis,
    placement: SinkPlacement,
) -> ComposedBranch:
    '''The branch of `origin`: the subgraph itself, after the chain of
    producers that feed it alone.

    The sinks waiting on a node of the origin's domain are injected beside the
    origin before its chain is built, so a producer gated on such a sink is
    exclusive inside this scope and lands one column to the left of both.

    `local` holds only this scope's origin and what the injection placed. A
    sink injected elsewhere is not local here, so a producer gated on it is
    held and emitted by the enclosing scope, where the sink is local, which
    keeps the producer to the left of the sink.
    '''
    if origin == hga.HypergraphSpecialTag.RIGHT:
        return make_root_branch(analysis, placement)
    origin_column = ProductBranch.template(inject_waiting_sinks(
        (NestedBranch(graph_to_branch(origin), origin),), placement))
    local = (origin, hga.HypergraphSpecialTag.LEFT,
             *origin_column.newly_processed())
    stacks = (
        *make_stacks(util.unique_tuple(origin_column.left_nodes()), local,
                     analysis, placement),
        origin_column,
    )
    return ComposedBranch.template(stacks, None, analysis)


def has_consumer_outside(
    node: hg.HypergraphObject,
    absorbed: set[hga.GraphTag],
    analysis: hga.HypergraphAnalysis,
) -> bool:
    return any(
        isinstance(edge.graph, hg.Hypergraph) and edge.graph not in absorbed
        for edge in analysis.nodes_right(node))


def edge_column(
    carried: fd.Prod[hg.HypergraphObject],
    sinks: fd.Prod[hg.Hypergraph],
    is_held: Callable[[hg.HypergraphObject], bool],
    analysis: hga.HypergraphAnalysis,
    placement: SinkPlacement,
) -> ProductBranch:
    '''Identities over `carried`, with each of `sinks` beside the identity of
    a wire it consumes, or after all of them when it consumes none. An
    identity is dropped when `is_held` is false for its node and a sink took
    it, because the sink is then the wire's last reader.'''
    identities = tuple(
        IdentityBranch.from_node(node, analysis) for node in carried)
    injected = inject_waiting_sinks(identities, placement)
    taken = {
        node for branch in injected
        if not isinstance(branch, IdentityBranch)
        for node in branch.left_nodes()}
    kept = tuple(
        branch for branch in injected
        if not (isinstance(branch, IdentityBranch)
                and branch._node in taken and not is_held(branch._node)))
    remaining = tuple(
        placement.branches[sink] for sink in sinks
        if sink not in placement.claimed)
    placement.claimed.update(sinks)
    return ProductBranch.template((*kept, *remaining))


def make_root_branch(
    analysis: hga.HypergraphAnalysis,
    placement: SinkPlacement,
) -> ComposedBranch:
    '''The branch of the whole graph.

    Every sink is built into its branch first and registered with
    `placement`. Building one sink's branch can inject a sink registered
    before it, when the earlier sink waits on a wire the later sink's chain
    consumes.

    A sink whose domain has no consumer outside the sink branches is never
    claimed by the main walk, because no column consumes a wire of it. Such a
    sink is placed at an edge. When every wire of its domain is an input of
    the graph, or its chain reads no wire at all, as a constant dropped onto a
    tape slot reads none, it is placed in a leading column beside the inputs.
    Otherwise it is placed in a final column beside the outputs, and the wires
    of its domain join the initial frontier so that the walk produces them.
    '''
    target = analysis.target
    sinks = tuple(
        subgraph for subgraph in target.subgraphs()
        if not analysis.tag_cod(subgraph))
    for sink in sinks:
        placement.register(sink, make_subbranch(sink, analysis, placement))

    absorbed: set[hga.GraphTag] = set(util.concat(
        placement.branches[sink].newly_processed()
        for sink in placement.unclaimed()))
    inputs = set(target.dom)
    outputs = set(target.cod)
    leading_sinks: tuple[hg.Hypergraph, ...] = ()
    final_sinks: tuple[hg.Hypergraph, ...] = ()
    for sink in placement.unclaimed():
        left = util.unique_tuple(placement.branches[sink].left_nodes())
        if any(has_consumer_outside(node, absorbed, analysis)
               for node in left):
            continue
        if all(node in inputs for node in left):
            leading_sinks += (sink,)
        else:
            final_sinks += (sink,)

    leading: tuple[ProductBranch, ...] = ()
    if leading_sinks:
        leading = (edge_column(
            util.unique_tuple(target.dom), leading_sinks,
            lambda node: node in outputs
            or has_consumer_outside(node, absorbed, analysis),
            analysis, placement),)
    final: tuple[ProductBranch, ...] = ()
    if final_sinks:
        final = (edge_column(
            util.unique_tuple(target.cod), final_sinks,
            lambda node: True, analysis, placement),)

    nodes = (
        util.unique_tuple(final[0].left_nodes()) if final
        else target.cod)
    local = (hga.HypergraphSpecialTag.RIGHT, hga.HypergraphSpecialTag.LEFT,
             *util.concat(column.newly_processed() for column in final))
    stacks = (
        *leading,
        *make_stacks(nodes, local, analysis, placement),
        *final,
    )
    unplaced = placement.unclaimed()
    if unplaced:
        raise ValueError(
            f'sinks whose domain no column consumes: {unplaced}')
    return ComposedBranch.template(
        stacks, target.dom, analysis, target.cod)


def graph_to_branch[L, M: cat.Morphism](
    target: hg.Hypergraph[L, M],
    analysis: hga.HypergraphAnalysis[L, M] | None = None
):
    match target:
        case hg.HypergraphRoot():
            return RootBranch(
                _morphism=target.wraps,
                _right_nodes=target.cod,
                _left_nodes=target.dom,
                _newly_processed=(target,)
            )
        case hg.HypergraphBlock(body=body, block_tag=block_tag):
            return RootBranch(
                _morphism=cat.Block(
                    body=hypergraph_to_morphism(body),
                    block_tag=block_tag
                ),
                _right_nodes=target.cod,
                _left_nodes=target.dom,
                _newly_processed=(target,) # type: ignore
            )
        case hg.Multigraph():
            analysis = analysis or hga.HypergraphAnalysis(target)
            # One conversion level, one `SinkPlacement`. A nested
            # `Multigraph` gets its own through its own `graph_to_branch`,
            # matching its own analysis.
            return make_root_branch(analysis, SinkPlacement())
    raise ValueError(f"Unknown hypergraph type: {target}")

##################################
## REARRANGEMENT NORMALIZATION  ##
##################################
# The branch construction makes each copy at the LAST possible column: a wire
# consumed here and also held for a later column is duplicated right where the
# hold begins. Semantically free, but drawn it puts a fan-out in the middle of
# the picture. The expanded MoE showed the router's input being copied at the
# router, where the model's own presentation fans once, at the top. Hoisting
# pulls every product child's LEADING rearrangement out of the product and
# fuses it into the rearrangement before it, so cascaded fans merge into one
# at the earliest point. Wiring is untouched: a fused rearrangement is the
# composite of the two it replaces.

def _fuse_rearrangements[L](
    first: cat.Rearrangement[L],
    second: cat.Rearrangement[L],
) -> cat.Rearrangement[L]:
    '''The composite `first ; second`, as one Rearrangement.'''
    return cat.Rearrangement(
        mapping=tuple(first.mapping[i] for i in second.mapping),
        _dom=first._dom)


def _leading_split[L, M: cat.Morphism](
    child: cat.ProdCategory[L, M],
) -> tuple[cat.Rearrangement[L] | None, cat.ProdCategory[L, M]]:
    '''A product child's leading rearrangement, split off; (None, child) if
    it has none worth taking.'''
    match child:
        case cat.Rearrangement() if not tutil.is_identity(child):
            return child, cat.ProdObject(child.apply(child._dom)).identity()
        case cat.Composed(content=(cat.Rearrangement() as lead, *rest)):
            return lead, chsh.make_composed(*rest)
        case _:
            return None, child


def _hoist_product[L, M: cat.Morphism](
    target: cat.ProductOfMorphisms[L, M],
) -> tuple[cat.Rearrangement[L] | None, cat.ProdCategory[L, M]]:
    '''Split a product into (combined leading rearrangement, remainder).'''
    splits = tuple(_leading_split(child) for child in target.content)
    if all(lead is None for lead, _ in splits):
        return None, target
    mapping: list[int] = []
    dom: list[L] = []
    for (lead, _), child in zip(splits, target.content):
        offset = len(dom)
        if lead is None:
            segment = tuple(child.dom())
            dom.extend(segment)
            mapping.extend(offset + i for i in range(len(segment)))
        else:
            dom.extend(lead._dom)
            mapping.extend(offset + i for i in lead.mapping)
    combined = cat.Rearrangement(tuple(mapping), tuple(dom))
    return combined, chsh.make_product(*(rest for _, rest in splits))


def _hoist_once[L, M: cat.Morphism](
    target: cat.ProdCategory[L, M],
) -> cat.ProdCategory[L, M]:
    match target:
        case cat.Block(body=body):
            new_body = _hoist_once(body)
            return target if new_body is body else target.reconstruct(body=new_body)
        case cat.ProductOfMorphisms(content=ms):
            new_content = tuple(_hoist_once(m) for m in ms)
            if all(new is old for new, old in zip(new_content, ms)):
                return target
            return chsh.make_product(*new_content)
        case cat.Composed(content=ms):
            new_content: fd.Prod[cat.ProdCategory[L, M]] = ()
            for element in ms:
                element = _hoist_once(element)
                if isinstance(element, cat.ProductOfMorphisms):
                    lead, element = _hoist_product(element)
                    if lead is not None:
                        if (new_content
                                and isinstance(new_content[-1], cat.Rearrangement)):
                            lead = _fuse_rearrangements(new_content[-1], lead)
                            new_content = new_content[:-1]
                        if not tutil.is_identity(lead):
                            new_content = (*new_content, lead)
                # Adjacent rearrangements fuse: two permute/copy layers in a
                # row are one statement, and hoisting routinely creates the
                # pair.
                if (isinstance(element, cat.Rearrangement) and new_content
                        and isinstance(new_content[-1], cat.Rearrangement)):
                    element = _fuse_rearrangements(new_content[-1], element)
                    new_content = new_content[:-1]
                    if tutil.is_identity(element):
                        continue
                new_content = (*new_content, element)
            if not new_content:
                return target.dom().identity()
            return chsh.make_composed(*new_content)
        case _:
            return target


def hoist_rearrangements[L, M: cat.Morphism](
    target: cat.ProdCategory[L, M],
) -> cat.ProdCategory[L, M]:
    '''Merge cascaded copy/permute rearrangements toward the front.'''
    for _ in range(8):
        hoisted = _hoist_once(target)
        if hoisted == target:
            return hoisted
        target = hoisted
    return target


def hypergraph_to_morphism[L, M: cat.Morphism](
    target: hg.Hypergraph[L, M],
):# -> Composed[Any, ProdCategory[Any, Any]] | Any | Rearrangement...:
    return hoist_rearrangements(graph_to_branch(target).morphism())

def recycle[L, M: cat.Morphism](
    target: cat.ProdCategory[L, M]
):
    graph = hg.Multigraph.from_morphism(target)
    return hypergraph_to_morphism(graph)