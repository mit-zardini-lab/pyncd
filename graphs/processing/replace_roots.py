'''Walking and rewriting the roots of a hypergraph through its blocks.

A `Multigraph` holds roots and blocks, and a block holds a body that is again
a `Multigraph`. A pass that keys on individual roots, such as one that removes
a `Drop` or points a `Grab` at another slot, has to reach roots at every depth,
and `Multigraph._subgraphs` lists the top level alone.

Neither function touches a wire. A root replaced here keeps its domain and its
codomain, and a root removed here was composition-neutral, so a block is
rebuilt with `reconstruct` and keeps the domain and codomain it had.
'''
from __future__ import annotations
from typing import Iterator, Mapping

import data_structure.Category as cat
import data_structure.Term as fd
import graphs.data_structure.Hypergraph as hg


def walk_roots[L, M: cat.Morphism](
    graph: hg.Hypergraph[L, M],
) -> Iterator[hg.HypergraphRoot[L, M]]:
    '''Every root of `graph`, at any depth, in order.'''
    match graph:
        case hg.HypergraphRoot():
            yield graph
        case hg.HypergraphBlock(body=body):
            yield from walk_roots(body)
        case hg.Multigraph():
            for subgraph in graph.subgraphs():
                yield from walk_roots(subgraph)


type Replacement[L, M: cat.Morphism] = (
    hg.Hypergraph[L, M] | fd.Prod[hg.Hypergraph[L, M]] | None)


def replace_roots[L, M: cat.Morphism](
    graph: hg.Hypergraph[L, M],
    replacement: Mapping[hg.HypergraphRoot[L, M], Replacement[L, M]],
) -> hg.Hypergraph[L, M] | None:
    '''`graph` with each root in `replacement` replaced, removed where its
    replacement is None, or replaced by several subgraphs spliced in its place
    where the replacement is a tuple. A graph holding none of them comes back
    as the same object. A root replaced by several subgraphs leaves the
    boundaries of the blocks around it stale, and
    `rewire_blocks.recompute_block_boundaries` rederives them.'''
    match graph:
        case hg.HypergraphRoot():
            replaced = replacement.get(graph, graph)
            if isinstance(replaced, tuple):
                return hg.Multigraph.template(graph.dom, graph.cod, replaced)
            return replaced
        case hg.HypergraphBlock(body=body):
            rebuilt_body = replace_roots(body, replacement)
            if rebuilt_body is body:
                return graph
            return graph.reconstruct(body=rebuilt_body)
        case hg.Multigraph():
            rebuilt: tuple[hg.Hypergraph[L, M], ...] = ()
            for subgraph in graph._subgraphs:
                replaced = replacement.get(subgraph, subgraph)
                if isinstance(subgraph, hg.HypergraphRoot):
                    if replaced is None:
                        continue
                    rebuilt += replaced if isinstance(replaced, tuple) else (replaced,)
                    continue
                kept = replace_roots(subgraph, replacement)
                if kept is not None:
                    rebuilt += (kept,)
            unchanged = (len(rebuilt) == len(graph._subgraphs) and all(
                kept is subgraph
                for kept, subgraph in zip(rebuilt, graph._subgraphs)))
            if unchanged:
                return graph
            return hg.Multigraph.template(graph.dom, graph.cod, rebuilt)
    raise NotImplementedError(f'cannot rewrite {type(graph).__name__}')
