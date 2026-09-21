'''Merging the roots of a hypergraph that compute one value twice.

Two roots are duplicates when they wrap equal morphisms and read the same
wires. The first in walk order is kept, in the innermost scope enclosing every
duplicate, and the wires the others produced are renamed onto its wires. The
domain and codomain of every block on the way are then recomputed from the
roots inside it, so a wire the kept root produces enters each block that reads
it.

A loop block is a boundary. A root inside one computes once per iteration, so
it is never merged with a root outside it, and two roots inside the same loop
merge within its body. `obsidian/03-hypergraphs/Functors.md` states the rule.
'''
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator

import data_structure.Category as cat
import data_structure.Term as fd
import graphs.data_structure.Hypergraph as hg
import graphs.processing.replace_roots as replace_roots
import graphs.processing.rewire_blocks as rewire_blocks
import utilities.utilities as util

type BlockChain = rewire_blocks.BlockChain
type Wire = hg.HypergraphObject


@dataclass(frozen=True)
class _PlacedRoot[L, M: cat.Morphism]:
    root: hg.HypergraphRoot[L, M]
    enclosing_blocks: fd.Prod[hg.HypergraphBlock[L, M]]

    def chain(self) -> BlockChain:
        return rewire_blocks.chain_of(self.enclosing_blocks)

    def loop_chain(self) -> BlockChain:
        return rewire_blocks.chain_of(tuple(
            block for block in self.enclosing_blocks if rewire_blocks.is_loop(block)))


@dataclass(frozen=True)
class _Merge[L, M: cat.Morphism]:
    kept: hg.HypergraphRoot[L, M]
    removed: fd.Prod[hg.HypergraphRoot[L, M]]
    kept_chain: BlockChain
    target_chain: BlockChain

    def is_hoisted(self) -> bool:
        return self.kept_chain != self.target_chain

    def renamings(self) -> Iterator[fd.UIDRenaming[Wire]]:
        for root in self.removed:
            for kept_wire, removed_wire in zip(self.kept.cod, root.cod):
                yield fd.UIDRenaming.set_canonical(kept_wire, removed_wire)


def _merge_of[L, M: cat.Morphism](
    group: fd.Prod[_PlacedRoot[L, M]], target_chain: BlockChain,
) -> _Merge[L, M]:
    kept, *removed = group
    return _Merge(
        kept=kept.root,
        removed=tuple(placed.root for placed in removed),
        kept_chain=kept.chain(),
        target_chain=target_chain)


def _merges_of[L, M: cat.Morphism](
    group: fd.Prod[_PlacedRoot[L, M]],
    produced_at: dict[Wire, BlockChain],
) -> Iterator[_Merge[L, M]]:
    '''One merge into the innermost scope enclosing the group, or, when a
    wire the group reads is produced inside a block below that scope, one
    merge per scope the group occupies.

    A root placed in the enclosing scope that read a wire from inside a child
    block and fed a reader in that same block would make the block and the
    root read each other, which no order of the scope can carry.
    '''
    target = rewire_blocks.common_prefix(placed.chain() for placed in group)
    produced_below = any(
        produced_at.get(wire, ())[:len(target)] == target
        and len(produced_at.get(wire, ())) > len(target)
        for wire in group[0].root.dom)
    if not produced_below:
        yield _merge_of(group, target)
        return
    by_chain: dict[BlockChain, list[_PlacedRoot[L, M]]] = {}
    for placed in group:
        by_chain.setdefault(placed.chain(), []).append(placed)
    for chain, members in by_chain.items():
        if len(members) > 1:
            yield _merge_of(tuple(members), chain)


def find_duplicate_roots[L, M: cat.Morphism](
    graph: hg.Hypergraph[L, M],
) -> fd.Prod[_Merge[L, M]]:
    '''Every group of roots wrapping equal morphisms on the same wires, under the
    same loop blocks, in walk order.'''
    groups: dict[tuple, list[_PlacedRoot[L, M]]] = {}
    produced_at: dict[Wire, BlockChain] = {}
    for root, blocks in rewire_blocks.walk_roots_with_blocks(graph):
        placed = _PlacedRoot(root, blocks)
        for wire in root.cod:
            produced_at.setdefault(wire, placed.chain())
        key = (root.wraps, root.dom, placed.loop_chain())
        groups.setdefault(key, []).append(placed)
    return tuple(util.concat(
        tuple(_merges_of(tuple(group), produced_at))
        for group in groups.values() if len(group) > 1))


def merge_duplicate_roots[L, M: cat.Morphism](
    graph: hg.Multigraph[L, M],
) -> hg.Multigraph[L, M]:
    '''`graph` with each set of duplicate roots merged into one, or the same
    object when it has none.

    Merging two roots makes their readers read one wire, so two readers that
    wrap equal morphisms become duplicates in turn. The merge is repeated until
    a round finds none, and every round removes at least one root.
    '''
    merged = _merge_duplicate_roots_once(graph)
    while merged is not graph:
        graph, merged = merged, _merge_duplicate_roots_once(merged)
    return merged


def _merge_duplicate_roots_once[L, M: cat.Morphism](
    graph: hg.Multigraph[L, M],
) -> hg.Multigraph[L, M]:
    merges = find_duplicate_roots(graph)
    if not merges:
        return graph
    edits: dict[hg.HypergraphRoot[L, M], replace_roots.Replacement[L, M]] = {
        root: None for merge in merges for root in merge.removed}
    hoisted: dict[BlockChain, tuple[hg.HypergraphRoot[L, M], ...]] = {}
    for merge in merges:
        if merge.is_hoisted():
            edits[merge.kept] = None
            hoisted[merge.target_chain] = (
                *hoisted.get(merge.target_chain, ()), merge.kept)
    moved = rewire_blocks.insert_roots(
        replace_roots.replace_roots(graph, edits), hoisted)
    renamed = fd.Context(list(util.concat(
        tuple(merge.renamings()) for merge in merges))).apply(moved)
    return rewire_blocks.recompute_block_boundaries(renamed)
