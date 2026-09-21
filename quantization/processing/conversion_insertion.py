'''Reading a hypergraph wire by wire, rewriting the datatype of every wire, and
inserting a `TypeConvert` where a consumer requires a datatype the wire does not carry.

Written by Claude Fable 5.1, effort 80, and moved here at effort 25 on 2026-09-19, so
that the quantization of a model and any other pass that assigns a datatype to every
wire share these steps. A pass assigns a datatype to every wire by dataflow, finds the
first wire a consumer requires another datatype of, and inserts a conversion there.

A conversion stands either beside the producer of the wire it reads or in front of the
operations that read it, and `insert_conversions_beside_the_producer` and
`insert_conversions_beside_the_consumer` are the two placements. When nothing is
required any more, `rewrite_datatypes` writes the assignment onto every wire.
'''
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import data_structure.Category as cat
import data_structure.Term as fd
import graphs.data_structure.Hypergraph as hg
import graphs.processing.leaf_splicing as leaf_splicing
import para.data_structure.Para as Para
import quantization.data_structure.Quantization as Quantization


class ImpositionError(ValueError):
    '''A pass could not write the datatype it assigned onto a wire.'''


def leaves_in_dataflow_order(graph: hg.Hypergraph) -> tuple[hg.HypergraphRoot, ...]:
    '''Every operation, each after the operations producing what it reads.'''
    leaves = list(leaf_splicing.all_leaves(graph))
    produced = {obj for leaf in leaves for obj in leaf.cod}
    available: set[hg.HypergraphObject] = set(graph.dom)
    ordered: list[hg.HypergraphRoot] = []
    remaining = list(leaves)
    while remaining:
        ready = [leaf for leaf in remaining
                 if all(obj in available or obj not in produced for obj in leaf.dom)]
        if not ready:
            raise ImpositionError(
                f'{len(remaining)} operations wait on each other through their wires')
        chosen = ready[0]
        remaining.remove(chosen)
        ordered.append(chosen)
        available.update(chosen.cod)
    return tuple(ordered)


def producers(graph: hg.Hypergraph) -> dict[hg.HypergraphObject, hg.HypergraphRoot]:
    return {obj: leaf for leaf in leaf_splicing.all_leaves(graph) for obj in leaf.cod}


def consumers(graph: hg.Hypergraph
              ) -> dict[hg.HypergraphObject, list[tuple[hg.HypergraphRoot, int]]]:
    found: dict[hg.HypergraphObject, list[tuple[hg.HypergraphRoot, int]]] = {}
    for leaf in leaf_splicing.all_leaves(graph):
        for index, obj in enumerate(leaf.dom):
            found.setdefault(obj, []).append((leaf, index))
    return found


def wire_array(graph: hg.Hypergraph, wire: hg.HypergraphObject) -> cat.Array:
    '''The array a wire carries, as its producer writes it.'''
    for leaf in leaf_splicing.all_leaves(graph):
        for obj in leaf.cod:
            if obj == wire:
                return obj.obj
    for obj in graph.dom:
        if obj == wire:
            return obj.obj
    raise ImpositionError(f'{wire} is produced by nothing in the graph')


def all_wires(graph: hg.Hypergraph) -> tuple[hg.HypergraphObject, ...]:
    '''Every wire once, the graph's inputs first and then each operation's
    outputs in dataflow order, each carrying the array its producer writes.'''
    seen: dict[hg.HypergraphObject, hg.HypergraphObject] = {obj: obj for obj in graph.dom}
    for leaf in leaves_in_dataflow_order(graph):
        for obj in leaf.cod:
            seen.setdefault(obj, obj)
    return tuple(seen.values())


def is_conversion(leaf: hg.HypergraphRoot) -> bool:
    return (isinstance(leaf.wraps, cat.Broadcasted)
            and isinstance(leaf.wraps.operator, Quantization.TypeConvert))


def with_datatypes(morphism: cat.Morphism,
                   dom_datatypes: tuple[cat.Datatype, ...],
                   cod_datatypes: tuple[cat.Datatype, ...]) -> cat.Morphism:
    '''One operation with the datatypes of its operands and its results replaced.'''
    match morphism:
        case cat.Broadcasted():
            operator = morphism.operator
            if isinstance(operator, Quantization.TypeConvert):
                operator = operator.reconstruct(
                    source=dom_datatypes[0], target=cod_datatypes[0])
            return morphism.reconstruct(
                operator=operator,
                input_weaves=tuple(
                    weave.reconstruct(datatype=datatype)
                    for weave, datatype in zip(morphism.input_weaves, dom_datatypes)),
                output_weaves=tuple(
                    weave.reconstruct(datatype=datatype)
                    for weave, datatype in zip(morphism.output_weaves, cod_datatypes)))
        case Para.Grab():
            return morphism.reconstruct(
                size=morphism.size.reconstruct(datatype=cod_datatypes[0]))
        case Para.Drop():
            return morphism.reconstruct(
                size=morphism.size.reconstruct(datatype=dom_datatypes[0]))
    raise ImpositionError(
        f'{type(morphism).__qualname__} carries arrays this pass cannot rewrite')


type Rebuild = Callable[
    [cat.Morphism, tuple[cat.Datatype, ...], tuple[cat.Datatype, ...]], cat.Morphism]


def rewrite_datatypes(
    graph: hg.Hypergraph,
    datatype_of: Callable[[hg.HypergraphObject], cat.Datatype],
    rebuild: Rebuild = with_datatypes,
) -> hg.Hypergraph:
    '''Every wire's array rebuilt with the datatype `datatype_of` gives it, on
    the operation writing it and on every operation reading it.

    `rebuild` writes the datatypes onto one operation and is `with_datatypes`
    unless a pass has more to write. `quantization.processing.quantise_model`
    passes a rebuild that also quantises the body of a box, so that the box and
    its body carry one set of quantisations.
    '''
    replacements: dict[fd.UID, fd.Prod[hg.Hypergraph]] = {}
    for leaf in leaf_splicing.all_leaves(graph):
        rebuilt = rebuild(
            leaf.wraps,
            tuple(datatype_of(obj) for obj in leaf.dom),
            tuple(datatype_of(obj) for obj in leaf.cod))
        replacements[leaf.uid] = (
            hg.HypergraphRoot.template(rebuilt, dom=leaf.dom, cod=leaf.cod),)
    inputs = tuple(
        hg.HypergraphObject(uid=obj.uid, obj=obj.obj.reconstruct(datatype=datatype_of(obj)))
        for obj in graph.dom)
    if not isinstance(graph, hg.Multigraph):
        raise ImpositionError(
            f'{type(graph).__qualname__} is not a Multigraph whose inputs can be retyped')
    retyped = hg.Multigraph.template(
        dom=inputs, cod=graph.cod, subgraphs=graph.subgraphs())
    return leaf_splicing.splice(retyped, replacements)


@dataclass(frozen=True)
class Conversion:
    '''A `TypeConvert` to insert: the wire it reads, the datatype it writes,
    and the consumers redirected onto its result. Where the convert stands is
    the choice of the function inserting it.'''
    wire: hg.HypergraphObject
    target: cat.Datatype
    name: str
    redirected: frozenset[fd.UID]


def containers(graph: hg.Hypergraph) -> dict[fd.UID, fd.UID]:
    '''The scope each operation sits in, keyed by the operation's uid and named by the
    uid of the block or graph that holds it. Two operations in one scope may read one
    wire and be spliced beside each other, and two in different scopes may not.'''
    found: dict[fd.UID, fd.UID] = {}

    def walk(current: hg.Hypergraph, holder: fd.UID) -> None:
        match current:
            case hg.HypergraphRoot():
                found[current.uid] = holder
            case hg.HypergraphBlock(body=body):
                walk(body, current.uid)
            case hg.Multigraph():
                for subgraph in current.subgraphs():
                    walk(subgraph, current.uid)

    walk(graph, graph.uid)
    return found


def _conversion_root(graph: hg.Hypergraph,
                     conversion: Conversion) -> hg.HypergraphRoot:
    array = wire_array(graph, conversion.wire)
    convert = Quantization.TypeConvert.over_shape(
        array.datatype, conversion.target, tuple(array.shape()), conversion.name)
    root = hg.HypergraphRoot.template(
        convert, dom=(conversion.wire,), cod=(hg.HypergraphObject(),))
    if not isinstance(root, hg.HypergraphRoot):
        raise ImpositionError(f'a conversion of {conversion.wire} is not one operation')
    return root


def _spliced_with_conversions(
    graph: hg.Hypergraph,
    following: Mapping[fd.UID, list[hg.HypergraphRoot]],
    preceding: Mapping[fd.UID, list[hg.HypergraphRoot]],
    redirections: Mapping[fd.UID, dict[int, hg.HypergraphObject]],
) -> hg.Hypergraph:
    replacements: dict[fd.UID, fd.Prod[hg.Hypergraph]] = {}
    for leaf in leaf_splicing.all_leaves(graph):
        touched = (leaf.uid in following or leaf.uid in preceding
                   or leaf.uid in redirections)
        if not touched:
            continue
        rebuilt = leaf
        if leaf.uid in redirections:
            new_dom = tuple(
                redirections[leaf.uid].get(index, obj)
                for index, obj in enumerate(leaf.dom))
            rebuilt = hg.HypergraphRoot.template(leaf.wraps, dom=new_dom, cod=leaf.cod)
            if not isinstance(rebuilt, hg.HypergraphRoot):
                raise ImpositionError(
                    f'{leaf} rebuilt with redirected operands is not one operation')
        replacements[leaf.uid] = (
            *preceding.get(leaf.uid, ()), rebuilt, *following.get(leaf.uid, ()))
    return leaf_splicing.splice(graph, replacements)


def insert_conversions_beside_the_producer(
    graph: hg.Hypergraph, conversions: tuple[Conversion, ...]) -> hg.Hypergraph:
    '''Each conversion standing beside the producer of the wire it reads, so a value
    converted once is converted once however many loops its readers sit inside.

    The conversion and the operations redirected onto its result then sit in different
    scopes wherever the producer does, and the readers of a value produced in one scope
    and read in another must therefore be the whole of what reads it. A pass converting
    one operand at a time meets that condition, and
    `insert_conversions_beside_the_consumer` is the placement for a pass that does not.
    '''
    if not conversions:
        return graph
    made_by = producers(graph)
    read_by = consumers(graph)
    following: dict[fd.UID, list[hg.HypergraphRoot]] = {}
    preceding: dict[fd.UID, list[hg.HypergraphRoot]] = {}
    redirections: dict[fd.UID, dict[int, hg.HypergraphObject]] = {}
    for conversion in conversions:
        root = _conversion_root(graph, conversion)
        producer = made_by.get(conversion.wire)
        redirected = [
            (leaf, index) for leaf, index in read_by.get(conversion.wire, ())
            if leaf.uid in conversion.redirected]
        if producer is not None:
            following.setdefault(producer.uid, []).append(root)
        elif redirected:
            preceding.setdefault(redirected[0][0].uid, []).append(root)
        else:
            raise ImpositionError(
                f'a conversion of {conversion.wire} has neither a producer to '
                'follow nor a consumer to precede')
        for leaf, index in redirected:
            redirections.setdefault(leaf.uid, {})[index] = root.cod[0]
    return _spliced_with_conversions(graph, following, preceding, redirections)


def insert_conversions_beside_the_consumer(
    graph: hg.Hypergraph, conversions: tuple[Conversion, ...]) -> hg.Hypergraph:
    '''Each conversion standing in front of the operations redirected onto its result,
    inside the scope those operations sit in.

    A value produced in one scope and read at another datatype in another is converted
    where it is read, which is the placement a pass converting a whole graph needs. A
    conversion whose redirected consumers sit in two scopes raises, because the one
    wire it writes would leave the scope it was written in.
    '''
    if not conversions:
        return graph
    read_by = consumers(graph)
    scope_of = containers(graph)
    preceding: dict[fd.UID, list[hg.HypergraphRoot]] = {}
    redirections: dict[fd.UID, dict[int, hg.HypergraphObject]] = {}
    for conversion in conversions:
        redirected = [
            (leaf, index) for leaf, index in read_by.get(conversion.wire, ())
            if leaf.uid in conversion.redirected]
        if not redirected:
            raise ImpositionError(
                f'a conversion of {conversion.wire} has no consumer to precede')
        scopes = {scope_of[leaf.uid] for leaf, _ in redirected}
        if len(scopes) != 1:
            raise ImpositionError(
                f'a conversion of {conversion.wire} is read in {len(scopes)} scopes')
        root = _conversion_root(graph, conversion)
        preceding.setdefault(redirected[0][0].uid, []).append(root)
        for leaf, index in redirected:
            redirections.setdefault(leaf.uid, {})[index] = root.cod[0]
    return _spliced_with_conversions(graph, {}, preceding, redirections)
