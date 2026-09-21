'''Removing the zero cotangents a shift-invariant operator leaves in a backward pass.

The rule for `Maximum` in `para/registries/derivative.py` passes back the zero map
`0 * dm`, so the backward pass derived from a shifted softmax carries a chain that
computes the cotangent of the maximum, multiplies it by zero, and adds the result
to the cotangent of the scores. The chain is arithmetic that contributes nothing,
and the tape it reads, the maximum itself, is stored for nothing.

This pass removes it. Every pointwise map whose formula is the constant zero is
deleted, every addition it fed loses that operand, an addition left with one
operand of its own shape is replaced by a rename of that operand, and the roots
whose outputs nothing then reads are collected to a fixed point, which removes
the chain and the drop of any slot only the chain grabbed.

`pathway_collapse.collapse` runs it first. `obsidian/07-para/Backpropagation.md`
records the rule for the maximum it exists for.
'''
from __future__ import annotations

import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd
import graphs.data_structure.Hypergraph as hg
import graphs.processing.Hypergraph2Morphism as h2m
import graphs.processing.replace_roots as replace_roots
import graphs.processing.rewire_blocks as rewire_blocks
import para.data_structure.Para as para
import para.data_structure.contraction as pcon
import para.processing.backprop as backprop

PRUNE_LIMIT = 64


def is_zero_map(root: hg.Hypergraph) -> bool:
    '''Whether `root` computes the constant zero of its output shape.'''
    if not isinstance(root, hg.HypergraphRoot) or not isinstance(
            root.wraps, cat.Broadcasted):
        return False
    operator = root.wraps.operator
    if isinstance(operator, pcon.Zero):
        return True
    return isinstance(operator, ops.Arithmetic) and operator.formula == nm.Integer(0)


def zero_closure(
    graph: hg.Multigraph,
) -> tuple[set[hg.HypergraphRoot], set[hg.HypergraphObject]]:
    '''The zero maps, the nodes that only reindex a zero wire, and the wires
    they produce.'''
    zeros = {root for root in replace_roots.walk_roots(graph) if is_zero_map(root)}
    wires = {wire for root in zeros for wire in root.cod}
    changed = True
    while changed:
        changed = False
        for root in replace_roots.walk_roots(graph):
            if root in zeros or not isinstance(root.wraps, cat.Broadcasted):
                continue
            if (isinstance(root.wraps.operator, ops.View) and root.dom
                    and all(wire in wires for wire in root.dom)):
                zeros.add(root)
                wires.update(root.cod)
                changed = True
    return zeros, wires


def _shape_of(wire: hg.HypergraphObject) -> tuple[cat.Axis, ...]:
    return tuple(wire.obj.shape())


def _without_zero_operands(
    graph: hg.Multigraph, zero_wires: set[hg.HypergraphObject],
) -> tuple[hg.Multigraph, bool]:
    '''Every addition reading a zero wire rebuilt without it. An addition left
    with one operand of its own shape becomes a rename of that operand.'''
    edits: dict[hg.HypergraphRoot, replace_roots.Replacement] = {}
    renamings: list[fd.UIDRenaming] = []
    for root in replace_roots.walk_roots(graph):
        if not isinstance(root.wraps, cat.Broadcasted) or not isinstance(
                root.wraps.operator, ops.AdditionOp):
            continue
        kept = tuple(i for i, wire in enumerate(root.dom) if wire not in zero_wires)
        if len(kept) == len(root.dom):
            continue
        if len(kept) == 1 and _shape_of(root.dom[kept[0]]) == _shape_of(root.cod[0]):
            edits[root] = None
            renamings.append(fd.UIDRenaming.set_canonical(root.dom[kept[0]], root.cod[0]))
            continue
        morphism = root.wraps
        rebuilt = morphism.reconstruct(
            input_weaves=tuple(morphism.input_weaves[i] for i in kept),
            reindexings=tuple(morphism.reindexings[i] for i in kept))
        edits[root] = hg.HypergraphRoot.template(
            rebuilt, dom=tuple(root.dom[i] for i in kept), cod=root.cod)
    if not edits:
        return graph, False
    rebuilt = replace_roots.replace_roots(graph, edits)
    return fd.Context(renamings).apply(rebuilt), True


def _consumed(graph: hg.Multigraph) -> set[hg.HypergraphObject]:
    consumed = {wire for root in replace_roots.walk_roots(graph) for wire in root.dom}
    for block in _loop_blocks(graph):
        consumed.update(block.dom)
    return consumed | set(graph.cod)


def _loop_blocks(graph: hg.Hypergraph) -> list[hg.HypergraphBlock]:
    found: list[hg.HypergraphBlock] = []
    match graph:
        case hg.HypergraphBlock(body=body):
            if rewire_blocks.is_loop(graph):
                found.append(graph)
            else:
                found.extend(_loop_blocks(body))
        case hg.Multigraph():
            for subgraph in graph.subgraphs():
                found.extend(_loop_blocks(subgraph))
    return found


def collect_dead_roots(graph: hg.Multigraph) -> hg.Multigraph:
    '''Roots whose every output is unread, removed to a fixed point.'''
    for _ in range(PRUNE_LIMIT):
        consumed = _consumed(graph)
        dead = {root for root in replace_roots.walk_roots(graph)
                if root.cod and not any(wire in consumed for wire in root.cod)}
        if not dead:
            return rewire_blocks.recompute_block_boundaries(graph)
        graph = replace_roots.replace_roots(graph, {root: None for root in dead})
    raise RuntimeError(f'{PRUNE_LIMIT} sweeps without a fixed point')


def prune_zero_cotangents[L, M: cat.Morphism](
    taped: backprop.Taped[L, M],
) -> backprop.Taped[L, M]:
    '''`taped` with every zero cotangent, the chain that computed it, and the
    slots only that chain read removed from both passes.'''
    backward_graph = hg.Multigraph.from_morphism(taped.backward)
    zeros, zero_wires = zero_closure(backward_graph)
    if not zeros:
        return taped
    backward_graph, _ = _without_zero_operands(backward_graph, zero_wires)
    backward_graph = collect_dead_roots(replace_roots.replace_roots(
        backward_graph, {root: None for root in zeros}))
    grabbed = {root.wraps.tape for root in replace_roots.walk_roots(backward_graph)
               if isinstance(root.wraps, para.Grab)}
    forward_graph = hg.Multigraph.from_morphism(taped.forward)
    unread = {root for root in replace_roots.walk_roots(forward_graph)
              if isinstance(root.wraps, para.Drop) and root.wraps.tape not in grabbed}
    forward_graph = collect_dead_roots(replace_roots.replace_roots(
        forward_graph, {root: None for root in unread}))
    return backprop.Taped.from_passes(
        h2m.hypergraph_to_morphism(forward_graph),
        h2m.hypergraph_to_morphism(backward_graph))
