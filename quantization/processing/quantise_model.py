# Claude Opus 5 (1M context), effort high. Extended by Claude Fable 5.1, effort 80, on
# 2026-09-20, with the policy of a box, the quantisation returned by a titled block,
# and the count of casts as written operations.
'''A quantisation on every wire of a model, and a cast wherever an operation requires
another.

A model is written in the reals, and a released model runs in a handful of
quantisations. A quantisation is the number format a value is held in, together with
the size of that format in bits. The pass here writes the quantisations onto the
expression itself, before any machine is chosen, so that a figure of the model shows
the quantisation of each value and the place of each rounding. `QuantizationPolicy`
states the quantisations, `quantization.registries.operator_quantisations` says which
quantisation each operation uses, and `quantise_model` applies the two.

The pass works on the hypergraph, because the question answered by it is about wires.
It assigns a quantisation to every wire by dataflow, finds each operand required by a
rule at another quantisation, inserts a `TypeConvert` named `cast` on that wire, and
repeats until nothing is required. It then writes the assignment onto every wire. The
two steps are `quantization.processing.conversion_insertion`.

A box is a morphism of its own. The pass descends into the body of every
`ops.BlockOperator` and of every `para.data_structure.ParaBlockOperator
.ParaBlockOperator` inside a `ParaWrap`, quantising the body with the quantisations
carried by the operands of the box, under the policy selected by the name of the box,
and requiring the quantisations of its results, which puts a cast at the end of a body
computing something else. A repeated block and a loop are descended into by
`graphs.processing.leaf_splicing.all_leaves`, which reads them as part of the
graph. A block whose title is named by the policy returns each result named there at
the quantisation named for it, with the cast standing beside the operation computing
the result, because a released module ending in `.to(dtype)` need not be a box of the
expression.

A slot of the tape carries the quantisation dropped onto it. A slot dropped inside a
box and grabbed outside it is not visible to the pass, because a box states which slot
is carried by each of its ports and says nothing about the quantisation behind the
port, so such a slot takes the activation quantisation.

`obsidian/04-quantization/Quantization.md` states the package, the policy and the rules.
'''
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import data_structure.Category as cat
import data_structure.Operators as ops
import data_structure.Term as fd
import graphs.data_structure.Hypergraph as hg
import graphs.processing.Hypergraph2Morphism as h2m
import graphs.processing.leaf_splicing as leaf_splicing
import para.data_structure.Para as Para
import para.data_structure.ParaBlockOperator as ParaBlockOperator
import para.data_structure.ParaWrap as para_wrap
import quantization.data_structure.Quantization as Quantization
import quantization.processing.conversion_insertion as conversion_insertion
import quantization.registries.operator_quantisations as operator_quantisations
import term_utilities.term_utilities as tutil

CAST_NAME = 'cast'
CONVERSION_LIMIT = 256

type Quantisation = Quantization.Quantified[Any]
type SlotQuantisations = Mapping[fd.UID, Quantisation | None]

NO_SLOTS: SlotQuantisations = MappingProxyType({})

ArithmeticQuantisation = operator_quantisations.ArithmeticQuantisation
ContractionQuantisation = operator_quantisations.ContractionQuantisation
BoxPolicy = operator_quantisations.BoxPolicy
DEFAULT_BOX_POLICY = operator_quantisations.DEFAULT_BOX_POLICY


class QuantizationError(ValueError):
    '''A model given no quantisation on some wire by the pass.'''


@dataclass(frozen=True)
class QuantizationPolicy:
    '''The quantisations given to a model.

    `inputs` is the quantisation of the model's own operands on arrival.
    `activations` is the quantisation of a tensor held between two modules or
    kernels, which is the quantisation returned by a box and held by a slot of the
    tape. `scalars` is the quantisation of arithmetic, inside a module upcasting on
    entry and inside every kernel. `rounded_operands` is the quantisation written by
    the rounding kernel in front of a projection, when the weight of the projection
    has fewer bits than an activation. `results` is the quantisation required of the
    model's own results, and `None` requires nothing of them.

    `weights` gives the read quantisation of a weight, by the text of the name carried
    by the operator holding it, as `ops.Linear` and `ops.Embedding` hold theirs, and
    `weights_by_default` is the quantisation of a weight not named in the mapping. A
    weight has no wire, so the pass records its quantisation rather than writing it
    onto the expression.

    `boxes` gives the `BoxPolicy` of a box by its name, for the few modules handing
    FP32 on or never upcasting, and for a fused kernel. `block_results` gives the
    quantisation returned by a block at each position of its codomain, by its title,
    with `None` at a position returned as computed, for a released module ending in
    `.to(dtype)` on some of its returns and standing as a block of the expression
    rather than a box.

    `integers` is the quantisation of an index: the model's own token identifiers on
    arrival, every index computed by an operation, a slot of the tape holding one,
    and a table of integers not named in `weights`. A box returns or requires another
    where its `BoxPolicy` names one.

    `slots` gives the quantisation held by a slot of the tape, by the text of the
    slot's name, for a slot dropped inside one box and grabbed inside another, which
    the pass cannot follow through the ports of the boxes. A value dropped onto a
    named slot is cast to the named quantisation in front of the drop, and a grab of
    the slot returns it.
    '''
    inputs: Quantisation = Quantization.BF16
    activations: Quantisation = Quantization.BF16
    scalars: Quantisation = Quantization.FP32
    rounded_operands: Quantisation = Quantization.E4M3
    results: Quantisation | None = Quantization.BF16
    weights: Mapping[str, Quantisation] = field(default_factory=dict)
    weights_by_default: Quantisation = Quantization.BF16
    boxes: Mapping[str, BoxPolicy] = field(default_factory=dict)
    block_results: Mapping[str, fd.Prod[Quantisation | None]] = field(
        default_factory=dict)
    integers: Quantisation = Quantization.INT64
    slots: Mapping[str, Quantisation] = field(default_factory=dict)

    def weight_quantisation(self, name: fd.DynamicName | None) -> Quantisation:
        '''The read quantisation of the weight held by an operator named `name`.'''
        bodies = None if name is None else name.to_bodies()
        return self.weights.get(bodies or '', self.weights_by_default)

    def integer_weight_quantisation(self, name: fd.DynamicName | None) -> Quantisation:
        '''The quantisation of a table of integers held by an operator named `name`,
        which is the integer quantisation unless `weights` names the table.'''
        bodies = None if name is None else name.to_bodies()
        return self.weights.get(bodies or '', self.integers)

    def named_slot_quantisation(self, slot: Para.TapeSlot) -> Quantisation | None:
        '''The quantisation `slots` names for `slot`, and `None` where it names
        none.'''
        name = getattr(slot.uid, '_name', None)
        return None if name is None else self.slots.get(name.to_bodies())

    def box_policy(self, name: str | None) -> BoxPolicy:
        return DEFAULT_BOX_POLICY if name is None else self.boxes.get(
            name, DEFAULT_BOX_POLICY)

    def box_results(self, name: str | None) -> Quantisation:
        '''The quantisation returned by a box named `name`.'''
        return self.box_policy(name).results or self.activations


@dataclass(frozen=True)
class QuantisedModel:
    '''A model with a quantisation on every wire, beside the quantisation of every
    weight held by it.

    `weight_quantisations` is keyed by the text of the weight's name, which is the key
    of `websocket_transfer.auxiliary_information.OperatorRole`, so a notebook can print
    the table and write each quantisation into the role shown by an inspection box.
    '''
    morphism: cat.Morphism
    weight_quantisations: Mapping[str, Quantisation]


@dataclass(frozen=True)
class Scope:
    '''What one body is quantised under: the policy, the quantisation already carried
    by each slot of the tape, and the name of the box owning the body, which is `None`
    for the model itself.'''
    policy: QuantizationPolicy
    slots: SlotQuantisations = NO_SLOTS
    enclosing_box: str | None = None


def quantise_model(morphism: cat.Morphism,
                   policy: QuantizationPolicy) -> QuantisedModel:
    '''`morphism` with a quantisation on every wire holding a number, a cast
    wherever an operation requires another quantisation, and the quantisation of
    every weight held by it.'''
    return QuantisedModel(
        morphism=quantise_morphism(morphism, policy),
        weight_quantisations=weight_quantisations(morphism, policy))


def quantise_morphism(
    morphism: cat.Morphism,
    policy: QuantizationPolicy,
    input_quantisations: fd.Prod[Quantisation | None] | None = None,
    required_results: fd.Prod[Quantisation | None] | None = None,
    slots: SlotQuantisations = NO_SLOTS,
    enclosing_box: str | None = None,
) -> cat.Morphism:
    '''`morphism` with a quantisation on every wire holding a number.

    `input_quantisations` is the quantisation of each operand on arrival and defaults
    to the input quantisation of the policy at every operand holding a real number.
    `required_results` is the quantisation required of each result and defaults to
    the result quantisation of the policy at every result holding one, or to no
    requirement where the policy names none. `slots` gives the quantisation already
    carried by a slot of the tape, which is how a box is told the quantisations behind
    its ports, and `enclosing_box` names the box owning `morphism` as its body.
    '''
    scope = Scope(policy=policy, slots=slots, enclosing_box=enclosing_box)
    graph = hg.Multigraph.from_morphism(morphism)
    given = (input_quantisations if input_quantisations is not None
             else _default_quantisations(graph.dom, policy.inputs, policy.integers))
    required = (required_results if required_results is not None
                else _default_quantisations(graph.cod, policy.results, None))
    with_block_results = _cast_block_results(graph, scope, given)
    converted = _convert_until_settled(with_block_results, scope, given)
    with_results = _cast_results(converted, scope, given, required)
    return h2m.hypergraph_to_morphism(
        _written_onto_every_wire(with_results, scope, given))


def weight_quantisations(morphism: cat.Morphism, policy: QuantizationPolicy
                         ) -> Mapping[str, Quantisation]:
    '''The quantisation of every weight held by `morphism`, keyed by the text of its
    name. A weight is the array contracted against or read by an `ops.Linear` or an
    `ops.Embedding`, held by the operator rather than read on a wire. A table whose
    entries are integers takes the integer quantisation of the policy unless the
    policy names the table.'''
    return {
        name: (policy.weight_quantisation(node.operator.name)
               if any(Quantization.holds_real_numbers(weave.datatype)
                      for weave in node.output_weaves)
               else policy.integer_weight_quantisation(node.operator.name))
        for node in tutil.type_search(cat.Broadcasted, morphism)
        if isinstance(node.operator, (ops.Linear, ops.Embedding))
        and node.operator.name is not None
        and any(Quantization.holds_a_number(weave.datatype)
                for weave in node.output_weaves)
        for name in (node.operator.name.to_bodies(),)}


# ==========================================================================
# The quantisation of every wire.
# ==========================================================================
def _default_quantisations(
    wires: fd.Prod[hg.HypergraphObject], real: Quantisation | None,
    natural: Quantisation | None,
) -> fd.Prod[Quantisation | None]:
    '''`real` at every wire holding a real number and `natural` at every wire
    holding an index.'''
    return tuple(_by_kind(wire.obj.datatype, real, natural) for wire in wires)


def _by_kind(datatype: cat.Datatype, real: Quantisation | None,
             natural: Quantisation | None) -> Quantisation | None:
    if Quantization.holds_real_numbers(datatype):
        return real
    if Quantization.holds_natural_numbers(datatype):
        return natural
    return None


def _slot_uid(morphism: Para.ParaMorphism) -> fd.UID:
    return morphism.tape.uid


def _quantisation_of(morphism: cat.Broadcasted, scope: Scope,
                     operands: fd.Prod[Quantisation | None],
                     ) -> operator_quantisations.OperatorQuantisation:
    rule = operator_quantisations.rule_for(morphism.operator)
    return rule(operator_quantisations.QuantisationQuestion(
        morphism=morphism, policy=scope.policy, operand_quantisations=operands,
        enclosing_box=scope.enclosing_box))


def _wrapped_operands(
    wrap: para_wrap.ParaWrap, kept: fd.Prod[Quantisation | None],
    slots: SlotQuantisations, policy: QuantizationPolicy,
) -> fd.Prod[Quantisation | None]:
    '''The quantisation of every operand of the body of `wrap`, taking a taped operand
    from its slot and a wired one from `kept`, in the order of the wrap's own
    domain.'''
    wired = iter(kept)
    return tuple(
        next(wired) if entry is None
        else _slot_quantisation(entry, array, slots, policy)
        for entry, array in zip(wrap.grabs, wrap.body.dom()))


def _slot_quantisation(entry: Para.SlotEntry, array: cat.Array,
                       slots: SlotQuantisations, policy: QuantizationPolicy,
                       ) -> Quantisation | None:
    '''The quantisation held by the slot of a tape operation: the one named for the
    slot by the policy, otherwise the one already written to the slot, and otherwise
    the activation quantisation for a real number and the integer quantisation for
    an index.'''
    held = _by_kind(array.datatype, policy.activations, policy.integers)
    if held is None or entry is None:
        return held
    slot = Para.slot_of(entry)
    return policy.named_slot_quantisation(slot) or slots.get(slot.uid, held)


@dataclass(frozen=True)
class WireQuantisations:
    '''The quantisation assigned to every wire of a graph, and to every slot written
    by a drop in it.'''
    wires: Mapping[hg.HypergraphObject, Quantisation | None]
    slots: Mapping[fd.UID, Quantisation | None]


def _assign_quantisations(
    graph: hg.Hypergraph, scope: Scope, given: fd.Prod[Quantisation | None],
) -> WireQuantisations:
    '''The quantisation of every wire by dataflow: the domain as given, a tape
    operation passing the quantisation of its slot, and every operation the
    quantisation declared by its rule.'''
    policy = scope.policy
    assigned: dict[hg.HypergraphObject, Quantisation | None] = dict(
        zip(graph.dom, given))
    written: dict[fd.UID, Quantisation | None] = dict(scope.slots)
    for leaf in conversion_insertion.leaves_in_dataflow_order(graph):
        morphism = leaf.wraps
        match morphism:
            case Para.Grab():
                assigned[leaf.cod[0]] = _slot_quantisation(
                    Para.entry_of(morphism), morphism.size, written, policy)
            case Para.Drop():
                written[_slot_uid(morphism)] = assigned.get(leaf.dom[0])
            case para_wrap.ParaWrap(body=cat.Broadcasted() as body):
                operands = _wrapped_operands(
                    morphism, tuple(assigned.get(obj) for obj in leaf.dom),
                    written, policy)
                results = _quantisation_of(body, scope, operands).results
                kept = iter(leaf.cod)
                for entry, result in zip(morphism.drops, results):
                    if entry is None:
                        assigned[next(kept)] = result
                    else:
                        written[Para.slot_of(entry).uid] = result
            case cat.Broadcasted():
                results = _quantisation_of(
                    morphism, scope,
                    tuple(assigned.get(obj) for obj in leaf.dom)).results
                for obj, result in zip(leaf.cod, results):
                    assigned[obj] = result
            case _:
                raise QuantizationError(
                    f'{type(morphism).__qualname__} has no rule for the '
                    'quantisation of its results')
    return WireQuantisations(wires=assigned, slots=written)


# ==========================================================================
# The casts required by an operation.
# ==========================================================================
def _required_operands(
    leaf: hg.HypergraphRoot, scope: Scope, assigned: WireQuantisations,
) -> fd.Prod[tuple[hg.HypergraphObject, Quantisation]]:
    '''Each wire read by `leaf` and required by its rule at another quantisation,
    beside the quantisation required. A taped operand is left out, because a value
    off the tape arrives at the quantisation held by the slot and no wire carries
    it. A drop onto a slot named by the policy requires the value dropped at the
    quantisation named for the slot.'''
    morphism = leaf.wraps
    match morphism:
        case para_wrap.ParaWrap(body=cat.Broadcasted() as body):
            operands = _wrapped_operands(
                morphism, tuple(assigned.wires.get(obj) for obj in leaf.dom),
                assigned.slots, scope.policy)
            required = _quantisation_of(body, scope, operands).operands
            wired = tuple(required[position] for position in morphism.kept_inputs())
        case cat.Broadcasted():
            wired = _quantisation_of(
                morphism, scope,
                tuple(assigned.wires.get(obj) for obj in leaf.dom)).operands
        case Para.Drop():
            wired = (scope.policy.named_slot_quantisation(
                Para.slot_of(Para.entry_of(morphism))),)
        case _:
            return ()
    return tuple(
        (obj, quantisation)
        for obj, quantisation in zip(leaf.dom, wired)
        if quantisation is not None
        and assigned.wires.get(obj) != quantisation)


def _conversions_required(
    graph: hg.Hypergraph, scope: Scope, assigned: WireQuantisations,
) -> fd.Prod[conversion_insertion.Conversion]:
    '''One conversion per wire, required quantisation and scope, reading the
    operations of that scope requiring that quantisation off the wire.

    The operations of one scope share a cast, so a value read four times inside one
    block at one quantisation is cast once there. The operations of another scope
    take a cast of their own, because the wire written by a cast is read in the scope
    of the cast.
    '''
    arrays = {wire: wire.obj for wire in conversion_insertion.all_wires(graph)}
    scope_of = conversion_insertion.containers(graph)
    readers: dict[
        tuple[hg.HypergraphObject, Quantisation, fd.UID], set[fd.UID]] = {}
    for leaf in conversion_insertion.leaves_in_dataflow_order(graph):
        for wire, quantisation in _required_operands(leaf, scope, assigned):
            readers.setdefault(
                (wire, quantisation, scope_of[leaf.uid]), set()).add(leaf.uid)
    return tuple(
        conversion_insertion.Conversion(
            wire=wire,
            target=Quantization.with_quantisation(
                arrays[wire].datatype, quantisation),
            name=CAST_NAME,
            redirected=frozenset(uids))
        for (wire, quantisation, _), uids in readers.items())


def _convert_until_settled(
    graph: hg.Hypergraph, scope: Scope, given: fd.Prod[Quantisation | None],
) -> hg.Hypergraph:
    current = graph
    for _ in range(CONVERSION_LIMIT):
        assigned = _assign_quantisations(current, scope, given)
        conversions = _conversions_required(current, scope, assigned)
        if not conversions:
            return current
        current = conversion_insertion.insert_conversions_beside_the_consumer(
            current, conversions)
    raise QuantizationError(
        f'the quantisations did not settle in {CONVERSION_LIMIT} rounds of casts')


def _cast_results(
    graph: hg.Hypergraph, scope: Scope,
    given: fd.Prod[Quantisation | None],
    required: fd.Prod[Quantisation | None],
) -> hg.Hypergraph:
    '''`graph` with a `TypeConvert` on every result computed at a quantisation other
    than the required one, standing at the end of the graph, where the result is
    available whichever block produced it.'''
    if not isinstance(graph, hg.Multigraph):
        raise QuantizationError(
            f'{type(graph).__qualname__} is not a Multigraph whose results can be cast')
    assigned = _assign_quantisations(graph, scope, given)
    casts: list[hg.Hypergraph] = []
    results: list[hg.HypergraphObject] = []
    for wire, quantisation in zip(graph.cod, required):
        carried = assigned.wires.get(wire)
        if quantisation is None or carried == quantisation:
            results.append(wire)
            continue
        array = wire.obj
        source = Quantization.with_quantisation(array.datatype, carried) \
            if carried is not None else array.datatype
        convert = Quantization.TypeConvert.over_shape(
            source, Quantization.with_quantisation(array.datatype, quantisation),
            tuple(array.shape()), CAST_NAME)
        cast_result = hg.HypergraphObject(obj=cat.Array(
            Quantization.with_quantisation(array.datatype, quantisation),
            array._shape))
        casts.append(hg.HypergraphRoot.template(
            convert, dom=(wire,), cod=(cast_result,)))
        results.append(cast_result)
    if not casts:
        return graph
    return hg.Multigraph.template(
        dom=graph.dom, cod=tuple(results),
        subgraphs=(*graph.subgraphs(), *casts))


# ==========================================================================
# The quantisation returned by a titled block.
# ==========================================================================
def _blocks(graph: hg.Hypergraph) -> Iterator[hg.HypergraphBlock]:
    match graph:
        case hg.HypergraphBlock(body=body):
            yield graph
            yield from _blocks(body)
        case hg.Multigraph():
            for subgraph in graph.subgraphs():
                yield from _blocks(subgraph)


def block_title(block: hg.HypergraphBlock) -> str | None:
    aesthetics = block.block_tag.aesthetics
    return None if aesthetics is None else aesthetics.title


def _cast_block_results(
    graph: hg.Hypergraph, scope: Scope, given: fd.Prod[Quantisation | None],
) -> hg.Hypergraph:
    '''`graph` with a `TypeConvert` on every result of a block whose title is named by
    the policy, at each position given a quantisation there, where the result is
    computed at another quantisation.

    The cast stands beside the operation computing the result, inside the block, and
    writes the wire returned by the block, so every reader of that wire reads the
    converted value. The operation writes a fresh wire read by the cast alone.
    '''
    titled = scope.policy.block_results
    if not titled:
        return graph
    assigned = _assign_quantisations(graph, scope, given)
    made_by = conversion_insertion.producers(graph)
    replacements: dict[fd.UID, fd.Prod[hg.Hypergraph]] = {}
    renamed: dict[fd.UID, dict[hg.HypergraphObject, hg.HypergraphObject]] = {}
    for block in _blocks(graph):
        title = block_title(block)
        if title not in titled:
            continue
        quantisations = titled[title]
        if len(quantisations) != len(block.cod):
            raise QuantizationError(
                f'{len(quantisations)} result quantisations for the block titled '
                f'{title}, which returns {len(block.cod)} values')
        for wire, required in zip(block.cod, quantisations):
            carried = assigned.wires.get(wire)
            if (required is None
                    or not Quantization.holds_a_number(wire.obj.datatype)
                    or carried == required):
                continue
            producer = made_by.get(wire)
            if producer is None:
                raise QuantizationError(
                    f'the block titled {title} returns {wire}, which no operation of '
                    'the graph writes')
            fresh = hg.HypergraphObject(obj=wire.obj)
            renamed.setdefault(producer.uid, {})[wire] = fresh
            array = wire.obj
            source = (array.datatype if carried is None
                      else Quantization.with_quantisation(array.datatype, carried))
            convert = Quantization.TypeConvert.over_shape(
                source, Quantization.with_quantisation(array.datatype, required),
                tuple(array.shape()), CAST_NAME)
            cast = hg.HypergraphRoot.template(convert, dom=(fresh,), cod=(wire,))
            replacements.setdefault(producer.uid, ())
            replacements[producer.uid] = (*replacements[producer.uid], cast)
    if not replacements:
        return graph
    for leaf in leaf_splicing.all_leaves(graph):
        if leaf.uid not in replacements:
            continue
        rebuilt = hg.HypergraphRoot.template(
            leaf.wraps, dom=leaf.dom,
            cod=tuple(renamed[leaf.uid].get(obj, obj) for obj in leaf.cod))
        if not isinstance(rebuilt, hg.HypergraphRoot):
            raise QuantizationError(
                f'{leaf} rebuilt with a renamed result is not one operation')
        replacements[leaf.uid] = (rebuilt, *replacements[leaf.uid])
    return leaf_splicing.splice(graph, replacements)


# ==========================================================================
# Writing the quantisations onto the expression.
# ==========================================================================
def _written_onto_every_wire(
    graph: hg.Hypergraph, scope: Scope, given: fd.Prod[Quantisation | None],
) -> hg.Hypergraph:
    assigned = _assign_quantisations(graph, scope, given)
    arrays = {wire: wire.obj for wire in conversion_insertion.all_wires(graph)}

    def datatype_of(wire: hg.HypergraphObject) -> cat.Datatype:
        datatype = arrays[wire].datatype
        quantisation = assigned.wires.get(wire)
        return (datatype if quantisation is None
                else Quantization.with_quantisation(datatype, quantisation))

    def rebuild(morphism: cat.Morphism,
                dom_datatypes: tuple[cat.Datatype, ...],
                cod_datatypes: tuple[cat.Datatype, ...]) -> cat.Morphism:
        return _rebuilt_operation(
            morphism, dom_datatypes, cod_datatypes, scope.policy, assigned.slots)

    return conversion_insertion.rewrite_datatypes(graph, datatype_of, rebuild)


def _box_name(operator: ops.BlockOperator) -> str | None:
    return None if operator.name is None else operator.name.to_bodies()


def _rebuilt_operation(
    morphism: cat.Morphism,
    dom_datatypes: tuple[cat.Datatype, ...],
    cod_datatypes: tuple[cat.Datatype, ...],
    policy: QuantizationPolicy,
    slots: SlotQuantisations,
) -> cat.Morphism:
    '''One operation with the quantisations of its operands and results written onto
    it, and with the body of a box quantised at those quantisations.'''
    match morphism:
        case para_wrap.ParaWrap(body=cat.Broadcasted() as box):
            return morphism.reconstruct(body=_quantised_wrapped_box(
                morphism, box, dom_datatypes, cod_datatypes, policy, slots))
        case cat.Broadcasted(operator=ops.BlockOperator() as operator):
            return conversion_insertion.with_datatypes(
                morphism.reconstruct(operator=operator.reconstruct(
                    block=_quantised_block(
                        operator.block, dom_datatypes, cod_datatypes,
                        Scope(policy=policy, slots=slots,
                              enclosing_box=_box_name(operator))))),
                dom_datatypes, cod_datatypes)
    return conversion_insertion.with_datatypes(
        morphism, dom_datatypes, cod_datatypes)


def _quantised_block(
    block: cat.Block, dom_datatypes: tuple[cat.Datatype, ...],
    cod_datatypes: tuple[cat.Datatype, ...], scope: Scope,
) -> cat.Block:
    '''`block` with its body quantised, reading its operands at `dom_datatypes` and
    required to return its results at `cod_datatypes`.

    The tag is derived from the tag of the block and the body now held by it. One
    block written once and composed at two quantisations has two quantised bodies,
    and a figure recycling blocks draws one body per tag, so the two need two tags.
    Two sites quantising to the same body keep one tag between them and are drawn
    once.
    '''
    quantised = quantise_morphism(
        block.body, scope.policy,
        input_quantisations=tuple(map(Quantization.quantisation_of, dom_datatypes)),
        required_results=tuple(map(Quantization.quantisation_of, cod_datatypes)),
        slots=scope.slots, enclosing_box=scope.enclosing_box)
    tag = block.block_tag
    return block.reconstruct(
        body=quantised,
        block_tag=tag.reconstruct(uid=tag.uid.reconstruct(
            _id=fd.hash_id((tag.uid._id, hash(quantised))))))


def _quantised_wrapped_box(
    wrap: para_wrap.ParaWrap, box: cat.Broadcasted,
    dom_datatypes: tuple[cat.Datatype, ...],
    cod_datatypes: tuple[cat.Datatype, ...],
    policy: QuantizationPolicy, slots: SlotQuantisations,
) -> cat.Broadcasted:
    '''The box of `wrap` with its block quantised and its taped ports written at the
    quantisations of their slots.

    The seeds recorded by the operator are re-derived from the quantised block, so
    the box and the block state one morphism, which is what
    `ParaBlockOperator.bare_box` establishes when the box is first built.
    '''
    operator = box.operator
    port_datatypes = _ported_datatypes(wrap, box, dom_datatypes, cod_datatypes,
                                       policy, slots)
    if not isinstance(operator, ParaBlockOperator.ParaBlockOperator):
        return conversion_insertion.with_datatypes(
            box, port_datatypes.operands, port_datatypes.results)
    quantised = _quantised_block(
        operator.block, tuple(dom_datatypes), tuple(cod_datatypes),
        Scope(policy=policy, slots=slots, enclosing_box=_box_name(operator)))
    exposed = ParaBlockOperator.expose_tape_as_ports(quantised)
    _check_the_ports_still_stand(operator, exposed)
    return conversion_insertion.with_datatypes(
        box.reconstruct(operator=operator.reconstruct(
            block=quantised, grabs=exposed.grabs, drops=exposed.drops)),
        port_datatypes.operands, port_datatypes.results)


@dataclass(frozen=True)
class PortDatatypes:
    '''The datatype of every operand and every result of a box, the taped ports
    included.'''
    operands: tuple[cat.Datatype, ...]
    results: tuple[cat.Datatype, ...]


def _ported_datatypes(
    wrap: para_wrap.ParaWrap, box: cat.Broadcasted,
    dom_datatypes: tuple[cat.Datatype, ...],
    cod_datatypes: tuple[cat.Datatype, ...],
    policy: QuantizationPolicy, slots: SlotQuantisations,
) -> PortDatatypes:
    wired_operands, wired_results = iter(dom_datatypes), iter(cod_datatypes)
    return PortDatatypes(
        operands=tuple(
            next(wired_operands) if entry is None
            else _taped_datatype(entry, array, slots, policy)
            for entry, array in zip(wrap.grabs, box.dom())),
        results=tuple(
            next(wired_results) if entry is None
            else _taped_datatype(entry, array, slots, policy)
            for entry, array in zip(wrap.drops, box.cod())))


def _taped_datatype(entry: Para.SlotEntry, array: cat.Array,
                    slots: SlotQuantisations, policy: QuantizationPolicy,
                    ) -> cat.Datatype:
    quantisation = _slot_quantisation(entry, array, slots, policy)
    return (array.datatype if quantisation is None
            else Quantization.with_quantisation(array.datatype, quantisation))


class PortsMoved(QuantizationError):
    '''A quantised block whose tape seeds no longer stand at the ports recorded for
    them by the box, so the box and the block would state two different morphisms.'''


def _check_the_ports_still_stand(
    operator: ParaBlockOperator.ParaBlockOperator,
    exposed: ParaBlockOperator.TapeAsPorts,
) -> None:
    before = tuple(Para.entry_of(seed)
                   for seed in (*operator.grabs, *operator.drops))
    after = tuple(Para.entry_of(seed)
                  for seed in (*exposed.grabs, *exposed.drops))
    if before != after:
        raise PortsMoved(
            f'{len(before)} taped ports before quantising and {len(after)} after, '
            f'on the slots {[Para.slot_of(entry).uid.to_latex() for entry in after]}')


# ==========================================================================
# Reading a quantised model.
# ==========================================================================
def operations_of(morphism: cat.Morphism) -> fd.Prod[cat.Broadcasted]:
    '''Every operation written in `morphism`, in graph order, entering the body of
    each box once per block tag. A repeated block counts its operations once, because
    the block is written once, and one sublayer written in twenty layers counts its
    operations twenty times, because each layer writes them.

    `term_utilities.type_search` visits each distinct term once instead, and two
    operations written over the same axes with the same operator are one term, so a
    count through it collapses the twenty sublayers into one.
    '''
    found: list[cat.Broadcasted] = []
    entered: set[fd.UID] = set()

    def walk(current: cat.Morphism) -> None:
        for leaf in leaf_splicing.all_leaves(hg.Multigraph.from_morphism(current)):
            operation = leaf.wraps
            if isinstance(operation, para_wrap.ParaWrap):
                operation = operation.body
            if not isinstance(operation, cat.Broadcasted):
                continue
            found.append(operation)
            operator = operation.operator
            if isinstance(operator, ops.BlockOperator):
                tag = operator.block.block_tag.uid
                if tag not in entered:
                    entered.add(tag)
                    walk(operator.block.body)

    walk(morphism)
    return tuple(found)


def conversions_of(morphism: cat.Morphism) -> fd.Prod[cat.Broadcasted]:
    '''Every operation of `morphism` whose operator is a `TypeConvert`, counted as
    `operations_of` counts.'''
    return tuple(
        node for node in operations_of(morphism)
        if isinstance(node.operator, Quantization.TypeConvert))


def casts_of(morphism: cat.Morphism) -> fd.Prod[cat.Broadcasted]:
    '''Every conversion of `morphism` changing the quantisation of its operand, which
    is every one rounding a value or reading it into a format with more bits.'''
    return tuple(node for node in conversions_of(morphism)
                 if Quantization.is_cast(node.operator))


def cast_counts(morphism: cat.Morphism) -> Mapping[tuple[str, str], int]:
    '''How many casts of `morphism` read each pair of formats, keyed by the name of
    the format read and the name of the format written. A datatype holding no format
    is counted as `unquantised`.'''
    counts: dict[tuple[str, str], int] = {}
    for node in casts_of(morphism):
        operator = node.operator
        pair = (_format_name(operator.source), _format_name(operator.target))
        counts[pair] = counts.get(pair, 0) + 1
    return dict(sorted(counts.items()))


def conversions_between_two_quantisations(morphism: cat.Morphism) -> bool:
    '''Whether every conversion of `morphism` reads one quantisation into another,
    which is what a conversion of a quantised model is. A conversion into or out of a
    datatype carrying no quantisation is a conversion left unwritten by the pass.'''
    return all(
        Quantization.quantisation_of(node.operator.source) is not None
        and Quantization.quantisation_of(node.operator.target) is not None
        for node in conversions_of(morphism))


def _format_name(datatype: cat.Datatype) -> str:
    quantisation = Quantization.quantisation_of(datatype)
    if quantisation is None:
        return 'unquantised'
    return Quantization.format_name(quantisation)


def unquantised_arrays(morphism: cat.Morphism) -> fd.Prod[cat.Array]:
    '''Every array of `morphism` holding a real number and carrying no quantisation,
    which is empty for a model given a quantisation on every wire by the pass.'''
    return tuple(
        array for array in tutil.type_search(cat.Array, morphism)
        if Quantization.holds_a_number(array.datatype)
        and Quantization.quantisation_of(array.datatype) is None)


def unquantised_weaves(morphism: cat.Morphism) -> fd.Prod[cat.Weave]:
    '''Every weave of `morphism` whose datatype holds a number and carries no
    quantisation. A weave is the source of the label on a wire of a figure.'''
    return tuple(
        weave for weave in tutil.type_search(cat.Weave, morphism)
        if Quantization.holds_a_number(weave.datatype)
        and Quantization.quantisation_of(weave.datatype) is None)


def leaves_without_a_rule(morphism: cat.Morphism) -> fd.Prod[type[cat.Operator]]:
    '''Every operator class of `morphism` given no rule by the registry.'''
    found: dict[type[cat.Operator], None] = {}
    for node in tutil.type_search(cat.Broadcasted, morphism):
        if not operator_quantisations.has_a_rule(node.operator):
            found[type(node.operator)] = None
    return tuple(found)


def descends_into(morphism: cat.Morphism) -> fd.Prod[cat.Morphism]:
    '''Every box of `morphism` whose body is quantised by the pass, which is every
    `ops.BlockOperator` held by it.'''
    return tuple(
        node for node in tutil.type_search(cat.Broadcasted, morphism)
        if isinstance(node.operator, ops.BlockOperator))


def graph_leaves(morphism: cat.Morphism) -> fd.Prod[hg.HypergraphRoot]:
    '''Every operation of the graph of `morphism`, for a check reading the model wire
    by wire.'''
    return tuple(leaf_splicing.all_leaves(hg.Multigraph.from_morphism(morphism)))
