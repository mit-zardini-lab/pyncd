'''Leaving the quantisation on the wires where it changes.

Written by Claude Opus 5 (1M context), effort high.

`quantization/processing/quantise_model.py` writes a quantisation onto every wire,
and tsncd labels every array carrying one above the names of its axes. Most
operations hand the quantisation of their inputs on, so most labels repeat the label
before them. `strip_unchanged_quantisations` keeps the quantisation on a wire where
it changes: a wire entering the figure, a wire written by a `TypeConvert`, a wire
written by an operation with no input, and a wire written by an operation at a
quantisation that differs from the quantisation of one of its inputs. Every other
wire loses its quantisation, on the wire and in the weave or tape operation at each
of its ends, so both ends of the wire draw alike and tsncd needs no knowledge of the
choice. A `TypeConvert` keeps the full quantisations in its `source` and `target`.
The choice belongs to the display, and `DiagramSettings.clean_quantisation_labels`
makes it.
'''
from __future__ import annotations

import data_structure.Category as cat
import data_structure.Term as fd
import graphs.data_structure.Hypergraph as hg
import para.data_structure.MultiCategory as multi_category
import para.data_structure.Para as Para
import quantization.data_structure.Quantization as Quantization
import term_utilities.term_utilities as tutil


def without_quantisation(datatype: cat.Datatype) -> cat.Datatype:
    '''`datatype` with its quantisation removed and every wrapper standing outside
    the quantisation kept, so a complex number quantised in FP32 is returned as a
    complex number of the reals.'''
    if isinstance(datatype, Quantization.Quantified):
        return without_quantisation(datatype.wraps)
    for field_name in Quantization.WRAPPING_FIELDS:
        inner = getattr(datatype, field_name, None)
        if isinstance(inner, cat.Datatype):
            return datatype.reconstruct(
                **{field_name: without_quantisation(inner)})
    return datatype


def array_without_quantisation[B: cat.Datatype, A: cat.Axis](
    array: cat.Array[B, A],
) -> cat.Array[cat.Datatype, A]:
    return array.reconstruct(datatype=without_quantisation(array.datatype))


def outputs_with_a_new_quantisation(broadcasted: cat.Broadcasted) -> tuple[bool, ...]:
    '''For each output of `broadcasted`, whether its quantisation is new: every
    output of a `TypeConvert` or of an operation with no input, and otherwise an
    output whose quantisation differs from the quantisation of one of the inputs.'''
    if isinstance(broadcasted.operator, Quantization.TypeConvert):
        return tuple(True for _ in broadcasted.output_weaves)
    read = [Quantization.quantisation_of(weave.datatype)
            for weave in broadcasted.input_weaves]
    return tuple(
        not read or any(
            quantisation != Quantization.quantisation_of(weave.datatype)
            for quantisation in read)
        for weave in broadcasted.output_weaves)


def wires_with_a_new_quantisation(graph: hg.Hypergraph) -> frozenset[fd.UID]:
    '''The uid of every wire of `graph` whose quantisation is new where it is
    written, with the wires entering `graph` among them.'''
    new = {node.uid for node in graph.dom}
    for root in tutil.type_search(hg.HypergraphRoot, graph):
        if isinstance(root.wraps, cat.Broadcasted):
            new.update(
                node.uid for node, is_new
                in zip(root.cod, outputs_with_a_new_quantisation(root.wraps)) if is_new)
    return frozenset(new)


def weave_as_its_wire[B: cat.Datatype, A: cat.Axis](
    weave: cat.Weave[B, A], wire: hg.HypergraphObject, new: frozenset[fd.UID],
) -> cat.Weave[cat.Datatype, A]:
    if wire.uid in new:
        return weave
    return weave.reconstruct(datatype=without_quantisation(weave.datatype))


def operation_as_its_wires[M: cat.Morphism](
    root: hg.HypergraphRoot, new: frozenset[fd.UID],
) -> M:
    '''The morphism `root` wraps, with the quantisation left on each operand and
    result exactly where it is left on its wire.'''
    match root.wraps:
        case cat.Broadcasted(input_weaves=inputs, output_weaves=outputs):
            return root.wraps.reconstruct(
                input_weaves=tuple(weave_as_its_wire(weave, wire, new)
                                   for weave, wire in zip(inputs, root.dom)),
                output_weaves=tuple(weave_as_its_wire(weave, wire, new)
                                    for weave, wire in zip(outputs, root.cod)))
        case Para.ParaMorphism(size=cat.Array() as size):
            (wire,) = (*root.dom, *root.cod)
            if wire.uid in new:
                return root.wraps
            return root.wraps.reconstruct(size=array_without_quantisation(size))
    return root.wraps


def strip_quantisations_where_unchanged[G: hg.Hypergraph](graph: G) -> G:
    new = wires_with_a_new_quantisation(graph)
    rewritten: dict[int, object] = {}

    def rewrite(target: object) -> object:
        if id(target) in rewritten:
            return rewritten[id(target)]
        match target:
            case hg.HypergraphObject(obj=cat.Array() as array) if target.uid not in new:
                rebuilt = target.reconstruct(obj=array_without_quantisation(array))
            case hg.HypergraphObject():
                rebuilt = target
            case hg.HypergraphRoot():
                rebuilt = target.reconstruct(
                    dom=tuple(rewrite(wire) for wire in target.dom),
                    cod=tuple(rewrite(wire) for wire in target.cod),
                    wraps=operation_as_its_wires(target, new))
            case _:
                rebuilt = fd.deep_reconstruct(target, rewrite)
        rewritten[id(target)] = rebuilt
        return rebuilt

    return rewrite(graph)  # type: ignore[return-value]


def strip_unchanged_quantisations[L, M: cat.Morphism](
    term: cat.ProdCategory[L, M] | hg.Hypergraph[L, M]
    | multi_category.MultiCategory[L, M],
) -> cat.ProdCategory[L, M] | hg.Hypergraph[L, M] | multi_category.MultiCategory[L, M]:
    '''`term` with the quantisation of its datatypes left only on the wires where it
    changes. A term holding no quantisation, and a row of morphisms, is returned as
    it stands, and a morphism holding quantisations is returned as the hypergraph it
    converts to.'''
    if not any(True for _ in tutil.type_search(Quantization.Quantified, term)):
        return term
    match term:
        case multi_category.MultiCategory():
            return term
        case hg.Hypergraph():
            return strip_quantisations_where_unchanged(term)
        case _:
            return strip_quantisations_where_unchanged(hg.Multigraph.from_morphism(term))
