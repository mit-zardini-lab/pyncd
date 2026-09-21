'''Tying a tape's grabs and drops back into wires.

`para.algebra.para_sparse_expansion` leaves an expression whose pieces are
independent. A `Drop` saves a value, a `Grab` loads it, and nothing carries it
between them, so no block's domain or codomain mentions it and each seed
morphism was rewritten on its own.

This pass connects them. Every `Grab` and every `Drop` of one `TapeSlot` is
given the same `hg.HypergraphObject`, the tape operations are removed, and the
value travels on that wire.

    Drop<s>(%4)  ;  ...  ;  %7 = Grab<s>()     becomes   %4 read where %7 was

## What a scope does with a slot it only half holds

A slot whose grab and drop are both inside one scope is internal to it, and the
scope's signature does not change. A slot with only one of the two has to reach
the other across the boundary:

| the scope holds | the node goes |
|---|---|
| a grab and no drop | on the domain, ahead of the objects already there |
| a drop and no grab | on the codomain, after the objects already there |
| both | nowhere: the wire is internal |

A grab therefore reads a wire the enclosing expression supplies, and a drop
writes one the enclosing expression carries away. `deepseek.sparse_expansion`
puts an index wire on a boundary in the same order and for the same reason.

Membership is counted through nested scopes. The `Experts` block of a mixture of
experts holds three grabs and no drop, so the wire enters its domain. The layer around it holds that block and
the drop, so it holds both and its own signature is unchanged.

## Where the tape goes

Both operations are removed. A tie replaces the tape with the wire it stood
for, so an expression that has been tied has no slot left to read and the
untied form is what a later pass reads a slot from. Running this pass is
therefore a decision about which form the expression is in, and not a
normalisation.

`obsidian/07-para/Para Category.md` covers the tape, and
`obsidian/02-categories/Sparse Expansion.md` covers the wired form of a
selection, which is what this pass produces from the taped one.
'''
from __future__ import annotations
from dataclasses import dataclass, field

import data_structure.Term as fd
import data_structure.Category as cat
import graphs.data_structure.Hypergraph as hg
import graphs.processing.Hypergraph2Morphism as h2m
import graphs.processing.hypergraph_functor as functor
import para.data_structure.Para as Para
import term_utilities.term_utilities as tutil


class TapeSlotDisagrees(Exception):
    '''One slot grabbed and dropped at two different array shapes.'''


def tape_operation(root: hg.Hypergraph) -> Para.ParaMorphism | None:
    '''The `Grab` or `Drop` a root wraps, or None for anything else.'''
    if not isinstance(root, hg.HypergraphRoot):
        return None
    wraps = root.wraps
    return wraps if isinstance(wraps, (Para.Grab, Para.Drop)) else None


@dataclass
class TieTapes:
    '''Every slot in a graph, given a wire, with the tape operations removed.

    `wires` is minted in `survey` and read everywhere after, so one slot has one
    wire however many grabs and drops name it. `scopes` records what each
    subgraph holds, keyed by `id`, in the way
    `deepseek.sparse_expansion._Expansion` keys its own.
    '''
    wires: dict[fd.UID, hg.HypergraphObject] = field(default_factory=dict)
    scopes: dict[int, tuple[set[fd.UID], set[fd.UID]]] = field(
        default_factory=dict)
    splices: fd.Context = field(default_factory=fd.Context)

    # -- phase 0: one wire per slot ----------------------------------------
    def survey(self, graph: hg.Hypergraph) -> None:
        for operation in tutil.type_search(Para.ParaMorphism, graph):
            if isinstance(operation, (Para.StreamGrab, Para.StreamDrop)):
                raise ValueError(
                    f'{operation} is a loop variable, whose drop is read by the '
                    'next iteration, and a wire from the drop to the grab would '
                    'close a cycle')
            if isinstance(operation, (Para.LoopGrab, Para.LoopDrop)):
                raise ValueError(
                    f'{operation} names the member of its slot that one iteration '
                    'of a repeated block holds, and the block is one morphism, so '
                    'no wire inside it carries the member of another iteration')
            if isinstance(operation, (Para.ReductionGrab, Para.ReductionDrop)):
                raise ValueError(
                    f'{operation} is exchanged between processors, whose drop is '
                    'read by a partner in the next round, and no wire on one '
                    'processor carries it')
            slot = operation.tape
            if slot.uid not in self.wires:
                self.wires[slot.uid] = hg.HypergraphObject(obj=operation.size)
            elif self.wires[slot.uid].obj != operation.size:
                raise TapeSlotDisagrees(
                    f'slot {slot.uid.to_latex()} carries '
                    f'{self.wires[slot.uid].obj} at one operation and '
                    f'{operation.size} at another, so one wire cannot serve '
                    'both')

    # -- phase 1: which scope holds which half ----------------------------
    def analyze(self, graph: hg.Hypergraph) -> tuple[set[fd.UID], set[fd.UID]]:
        '''The slots grabbed and the slots dropped, this scope and below.'''
        grabbed: set[fd.UID] = set()
        dropped: set[fd.UID] = set()
        match tape_operation(graph), graph:
            case Para.Grab() as grab, _:
                grabbed.add(grab.tape.uid)
            case Para.Drop() as drop, _:
                dropped.add(drop.tape.uid)
            case _, hg.HypergraphBlock(body=body):
                grabbed, dropped = self.analyze(body)
            case _, hg.Multigraph():
                for subgraph in graph.subgraphs():
                    sub_grabbed, sub_dropped = self.analyze(subgraph)
                    grabbed |= sub_grabbed
                    dropped |= sub_dropped
        self.scopes[id(graph)] = (grabbed, dropped)
        return grabbed, dropped

    # -- phase 2: rebuild --------------------------------------------------
    def _sorted_wires(self, uids) -> tuple[hg.HypergraphObject, ...]:
        return tuple(self.wires[uid]
                     for uid in sorted(uids, key=lambda uid: uid._id))

    def _rebuild(self, graph: hg.Hypergraph) -> hg.Hypergraph | None:
        '''The graph with its tape operations spliced out. None removes it.'''
        operation = tape_operation(graph)
        if operation is not None:
            return self._splice_out(graph, operation)
        match graph:
            case hg.HypergraphRoot():
                return graph
            case hg.HypergraphBlock(body=body):
                return hg.HypergraphBlock.template(
                    self._rebuild(body), graph.block_tag)
            case hg.Multigraph():
                return self._rebuild_scope(graph)
        raise NotImplementedError(f'cannot tie {type(graph).__name__}')

    def _splice_out(
        self,
        root: hg.HypergraphRoot,
        operation: Para.ParaMorphism,
    ) -> None:
        '''The tape operation's own wire, made an alias of the slot's wire.'''
        wire = self.wires[operation.tape.uid]
        own = (root.cod if isinstance(operation, Para.Grab)
               else root.dom)
        self.splices.append_bucket(
            fd.UIDRenaming.set_canonical(wire, own[0]))
        return None

    def _rebuild_scope(self, graph: hg.Multigraph) -> hg.Hypergraph:
        grabbed, dropped = self.scopes[id(graph)]
        subgraphs = tuple(
            rebuilt for subgraph in graph._subgraphs
            if (rebuilt := self._rebuild(subgraph)) is not None)
        enters = self._sorted_wires(grabbed - dropped)
        exits = self._sorted_wires(dropped - grabbed)
        return hg.Multigraph.template(
            (*enters, *graph.dom),
            (*graph.cod, *exits),
            subgraphs)

    def run(self, graph: hg.Hypergraph) -> hg.Hypergraph:
        self.survey(graph)
        if not self.wires:
            return graph
        self.analyze(graph)
        tied = self._rebuild(graph)
        if self.splices.equality_classes:
            tied = functor.reduce_nodes(self.splices, tied)
        return tied


def tie_tapes[L, M: cat.Morphism](
    target: cat.ProdCategory[L, M] | hg.Hypergraph[L, M],
) -> cat.ProdCategory[L, M] | hg.Hypergraph[L, M]:
    '''Connect every grab to the drop of its slot, as a wire.

    Accepts a morphism or a hypergraph and returns the same kind. A target
    holding no tape operation comes back unchanged, as the same object.

    The domain and the codomain change wherever a slot is grabbed without being
    dropped in the same scope, or dropped without being grabbed, which is the
    table in the module docstring.
    '''
    if not any(True for _ in tutil.type_search(Para.ParaMorphism, target)):
        return target
    as_morphism = not isinstance(target, hg.Hypergraph)
    graph = (hg.Multigraph.from_morphism(target)
             if as_morphism else target)
    tied = TieTapes().run(graph)
    if tied is graph and as_morphism:
        return target
    return h2m.hypergraph_to_morphism(tied) if as_morphism else tied
