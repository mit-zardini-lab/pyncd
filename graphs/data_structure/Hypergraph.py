'''The hypergraph form that an expression is rewritten in.

A morphism is the form an expression is read in. Inserting an operation into a morphism
means restructuring the `Composed` and `ProductOfMorphisms` nesting around it. Inserting
one into a hypergraph means adding a subgraph and renaming a wire, so a rewrite is
written against the hypergraph form.

A wire is a `HypergraphObject`, whose uid is its identity and whose `.obj` is the
object carried on it. Equality and hashing read the uid alone, so two
`HypergraphObject`s with the same uid are the same wire even when their `.obj`
differs. The two differ constantly, because a functor pass rewrites `.obj` while the
wiring stays fixed. A splice therefore renames wire identity, with
`fd.UIDRenaming.set_canonical(new, old)` inside an `fd.Context`, which keeps each
occurrence's `.obj` and replaces its uid.

There are three kinds of graph.

  HypergraphRoot   One morphism, in `.wraps`. A leaf.
  HypergraphBlock  A body and a `cat.BlockTag`. The tag carries `repetition`,
                   which denotes a loop when it is not 1, and `aesthetics`,
                   which carries the title and the description a figure draws.
  Multigraph       A set of subgraphs with a domain and a codomain.

`Multigraph.from_morphism` converts a morphism into a graph.
`graphs.processing.Hypergraph2Morphism.hypergraph_to_morphism` converts a graph back
into a morphism. The conversion back reads the wiring alone, so a `Multigraph` records
nothing about where its subgraphs sat in the morphism it was read from.

The theory is in `obsidian/03-hypergraphs/Hypergraphs.md`.
'''
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence, Iterable

import data_structure.Term as fd # for 'foundations'
import data_structure.Numeric as nm
import utilities.utilities as util
import data_structure.Category as cat

@dataclass(frozen=True, eq=False)
class HypergraphObject[L](fd.UTerm):
    '''A wire of a hypergraph, carrying the object that travels on it.

    The uid is the wire, so equality and hashing read the uid alone. The `.obj`
    fields of two objects on one wire may differ, and a rewrite that changes
    which wire the neighbours name is a `fd.UIDRenaming` rather than an
    `fd.EqualityClass`, so that each occurrence keeps its own `.obj`.
    '''
    _memoize_hash = False
    obj: L = None # type: ignore

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HypergraphObject) and self.uid == other.uid

    def __hash__(self) -> int:
        return hash(self.uid)

    def __repr__(self) -> str:
        return (f'HypergraphObject({self.uid._id:X}, '
                f'{type(self.obj).__name__})')

    @classmethod
    def template(
        cls,
        target: Sequence[L],
        identities: Sequence[HypergraphObject[Any]] | None = None
    ) -> fd.Prod[HypergraphObject[L]]:
        '''One object per entry of `target`, each on the wire the matching entry
        of `identities` names, or on a fresh wire when none are given.'''
        if not identities:
            return tuple(cls(obj=obj) for obj in target)
        return tuple(
            cls(uid=identity.uid, obj=obj)
            for obj, identity in zip(target, identities)
        )

def is_rearrangement[L, M: cat.Morphism](graph: Hypergraph[L, M]) -> bool:
    '''Whether `graph` holds no `HypergraphRoot`, and so only rewires.

    `HypergraphRoot.template` turns a `cat.Rearrangement` into a `Multigraph` with no
    subgraphs, and a graph built from one is empty in exactly that sense. A
    `Multigraph` with no subgraphs passes the `all` here for the same reason.
    '''
    match graph:
        case HypergraphRoot():
            return False
        case HypergraphBlock(body=body):
            return is_rearrangement(body)
        case Multigraph(_subgraphs=subgraphs):
            return all(is_rearrangement(subgraph) for subgraph in subgraphs)
        case _:
            raise ValueError(f"Unknown graph type: {type(graph)}")

@dataclass(frozen=True)
class Hypergraph[L, M:cat.Morphism](fd.UTerm):
    dom: fd.Prod[HypergraphObject[L]] = ()
    cod: fd.Prod[HypergraphObject[L]] = ()

    def subgraphs(self) -> fd.Prod[Hypergraph[L, M]]:
        return (self,)

@dataclass(frozen=True)
class HypergraphRoot[L, M: cat.Morphism](Hypergraph[L, M]):
    wraps: M = None # type: ignore

    @classmethod
    def template(cls,
                 target: M,
                 dom: Sequence[HypergraphObject[Any]] | None = None,
                 cod: Sequence[HypergraphObject[Any]] | None = None
                 ) -> HypergraphRoot[L, M] | Multigraph[L, M]:
        '''A leaf wrapping `target`, or a `Multigraph` when `target` rewires.

        `dom` and `cod` name the wires the leaf sits on, and a wire missing from
        them is fresh. A `cat.Rearrangement` permutes, copies and deletes wires and
        computes nothing, so it becomes an empty `Multigraph` whose codomain names
        the wires the rearrangement sends its domain wires to. The wiring is then
        carried by wire identity alone.
        '''
        if isinstance(target, cat.Rearrangement):
            dom_objects = HypergraphObject.template(target.dom(), dom)
            cod_objects = target.apply(dom_objects)
            assert cod is None or tuple(cod) == cod_objects
            return Multigraph.template(dom=dom_objects, cod=cod_objects)
        return cls(
            uid=fd.UID(HypergraphRoot),
            dom=HypergraphObject.template(target.dom(), dom),
            cod=HypergraphObject.template(target.cod(), cod),
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
        '''A block over `body`. Under `reduce`, its domain lists each wire once.

        The body's own domain is left as it is, so a wire entering the block twice is
        listed once on the block and twice on the body. A guide crossing between the
        two is realigned by wire in `hypergraph_crawler.realign_guide`.
        '''
        dom = (
            body.dom if not reduce
            else util.unique_tuple(body.dom)
        )
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
class Multigraph[L, M: cat.Morphism](Hypergraph[L, M]):
    _subgraphs: fd.Prod[Hypergraph[L, M]] = ()

    def subgraphs(self) -> fd.Prod[Hypergraph[L, M]]:
        return self._subgraphs

    @classmethod
    def template(cls,
                 dom: fd.Prod[HypergraphObject[L]],
                 cod: fd.Prod[HypergraphObject[L]],
                 subgraphs: Iterable[Hypergraph[L, M]] = ()) -> Multigraph[L, M]:
        return cls(
            uid=fd.UID(Multigraph),
            dom=dom,
            cod=cod,
            _subgraphs=tuple(subgraphs)
        )

    @classmethod
    def from_morphism(
        cls,
        target: cat.ProdCategory[L, M],
        dom: fd.Prod[HypergraphObject[L]] | None = None,
        reduce: bool = True,
    ) -> Multigraph[L, M]:
        dom = dom or HypergraphObject.template(target.dom())
        cod: fd.Prod[HypergraphObject[L]] = ()
        subgraphs: fd.Prod[Hypergraph[L, M]] = ()

        match target:
            case cat.Block():
                block = HypergraphBlock.template(
                    body=cls.from_morphism(target.body, dom, reduce=reduce),
                    block_tag=target.block_tag)
                subgraphs = (block,)
                cod = block.cod
            case cat.ProductOfMorphisms():
                for m, _dom in target.partition(dom):
                    subgraph = cls.from_morphism(m, _dom, reduce=reduce)
                    cod = cod + subgraph.cod
                    subgraphs = (*subgraphs, *subgraph._subgraphs)
            case cat.Composed(content=ms):
                cod = dom
                for m in ms:
                    subgraph = cls.from_morphism(m, cod, reduce=reduce)
                    cod = subgraph.cod
                    subgraphs = (*subgraphs, *subgraph._subgraphs)
            case cat.Rearrangement():
                subgraphs = ()
                cod = target.apply(dom)
            case _:
                root = HypergraphRoot.template(target, dom)
                reduce = False
                subgraphs = (root,)
                cod = root.cod
        dom = dom if not reduce else util.unique_tuple(dom)
        return cls(
            uid=fd.UID(Multigraph),
            dom=dom,
            cod=cod,
            _subgraphs=subgraphs
        )

def flat_subgraphs[L, M: cat.Morphism](target: Hypergraph[L, M], remove_blocks: bool = False) -> fd.Prod[Hypergraph[L, M]]:
    '''The subgraphs of `target`, with the nesting of `Multigraph`s removed.

    `remove_blocks` does not remove every block. A block is descended through only when
    its `repetition` is 1, so a loop block is returned whole, as a `HypergraphBlock`
    with its body flattened. A walk that expects a `HypergraphRoot` and reads `.wraps`
    raises on one. Recurse into `.body` instead. `leaf_splicing.all_leaves` is the
    version that does.
    '''
    match target:
        case HypergraphBlock(body=body):
            if remove_blocks and target.block_tag.repetition == nm.Integer(1):
                return flat_subgraphs(body, remove_blocks=True)
            return (target.reconstruct(body=flatten(body, remove_blocks)),)
        case Multigraph(_subgraphs=subgraphs):
            return util.concat(flat_subgraphs(subgraph, remove_blocks) for subgraph in subgraphs)
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
    '''The leaves of `target`, descending through every block whatever its repetition.

    `flat_subgraphs` stops at a loop block. `flatten_blocks` does not, so the loop is
    lost and only the body's leaves are returned.
    '''
    match target:
        case HypergraphBlock(body=body):
            return flatten_blocks(body)
        case Multigraph(_subgraphs=subgraphs):
            return util.concat(flatten_blocks(subgraph) for subgraph in subgraphs)
        case _:
            return (target,)
