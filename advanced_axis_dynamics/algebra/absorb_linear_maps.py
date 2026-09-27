# Claude Opus 5.5 (1M context), effort 40.
'''Contracting a linear map against the other operand of the contraction that reads its
result, where that order costs fewer operations.

A contraction whose result is read once, by a second contraction, directly or through a
view, is a chain of two contractions. The operands of the chain are the operands of the
first contraction, which is called the producer, followed by the other operands of the
second, which is called the consumer. Multi-head latent attention holds two such
chains. The keys `K[x, h, d_n] = sum_l W^{UK}[l, h, d_n] lat[x, l]` are read through the
causal mask by the scores, and the values `V[x, h, d_v] = sum_l W^{UV}[l, h, d_v]
lat[x, l]` are read through the mask by the sum over the slots.

A chain can be evaluated in more than one order. The order it is written in computes
the producer first. Every other order is reached in three steps. The view is carried
back onto the operands of the producer by `move_reads_backwards`, which is legal
because the producer is broadcast over the axes the view reads. The producer is then
merged into the consumer as one contraction by `einops_rearrange.merge_einops`. That
contraction is finally split by `contract_pair_first`, so that two of its operands, at
least one of them an operand of the consumer, are contracted before the rest. In
multi-head latent attention the split contracts the queries against `W^{UK}` before the
latent is read, and the softmax weights against the latent before `W^{UV}` is applied.
DeepSeek-V3 calls that order `absorb`.

The operation count of an order is a sum of products of axis sizes, read from each of
its contractions by `morphism_work.read_symbolic_work`. No order is cheaper at every
size. The order as written expands a key and a value for every position once, and the
absorbed order computes a score over the latent width for every query and every slot.
`absorb_linear_maps` binds the sizes it is given, and rewrites a chain only where
another order costs fewer operations than the order it is written in.

The search runs inside each scope of the hypergraph, so a producer and a consumer in
different blocks are not paired. `discovering_broadcasts.remove_grouping_blocks` writes
the blocks of a model out first. A `Linear` whose weight is inside the operator is not
a contraction, and `linear_expansion.expand_linear_root` writes it as one first.

`merge_einops` is described in `obsidian/02-categories/Einops Rearrangement.md`.
'''
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping
import itertools

import data_structure.Term as fd
import data_structure.Numeric as nm
import data_structure.Category as cat
import data_structure.Operators as ops
import graphs.data_structure.Hypergraph as hg
import graphs.processing.Hypergraph2Morphism as h2m
import graphs.processing.hypergraph_functor as hypergraph_functor
import term_utilities.term_utilities as tutil
import construction_helpers.simple_helper as chsh
import algebra.einops_rearrange as einops_rearrange
import algebra.einops_simplification as einops_simplification
import algebra.merge_into_consumer as merge_into_consumer
import algebra.reindexing_absorption as reindexing_absorption
import performance_modeling.morphism_work as morphism_work
import advanced_axis_dynamics.algebra.move_reads_backwards as move_reads_backwards


@dataclass(frozen=True)
class ContractionChain:
    '''A contraction `producer` whose result `consumer` reads at `port`, directly or
    through `read`, a view whose result nothing else reads.'''
    producer: hg.HypergraphRoot
    read: hg.HypergraphRoot | None
    consumer: hg.HypergraphRoot
    port: int

    def other_consumer_positions(self) -> fd.Prod[int]:
        return tuple(k for k in range(len(self.consumer.dom)) if k != self.port)

    def operand_wires(self) -> fd.Prod[hg.HypergraphObject]:
        '''The wires of the producer's operands, then of the consumer's other
        operands.'''
        return (*self.producer.dom,
                *(self.consumer.dom[k] for k in self.other_consumer_positions()))

    def other_consumer_operands(self) -> fd.Prod[cat.Array]:
        arrays = tuple(self.consumer.wraps.dom())
        return tuple(arrays[k] for k in self.other_consumer_positions())


@dataclass(frozen=True)
class Association:
    '''One order of evaluating a chain, as a morphism from the operands of the chain to
    the consumer's result, and its operation count with every axis size a symbol.

    `contracted_first` gives the positions, among the operands of the chain, of the
    operands contracted first. The order the chain is written in contracts the
    producer's operands first.'''
    contracted_first: fd.Prod[int]
    morphism: cat.BroadcastedCategory
    operations: nm.Numeric


@dataclass(frozen=True)
class UnifiedChain:
    '''A chain as one contraction over its operands, each operand read first through
    the morphism at its position in `operand_reads`, which is a view where the read of
    the chain was carried onto that operand and the identity elsewhere.'''
    operand_reads: fd.Prod[cat.BroadcastedCategory]
    contraction: cat.Broadcasted


def is_contraction_root(graph: hg.Hypergraph) -> bool:
    return (isinstance(graph, hg.HypergraphRoot)
            and einops_simplification.is_einsum(graph.wraps))


def is_view_root(graph: hg.Hypergraph) -> bool:
    return (isinstance(graph, hg.HypergraphRoot)
            and reindexing_absorption.is_node(graph.wraps))


def identity_on(arrays: fd.Prod[cat.Array]) -> cat.Rearrangement:
    return cat.ProdObject(tuple(arrays)).identity()


def roots_of(morphism: cat.Morphism) -> fd.Prod[cat.Broadcasted]:
    '''Every `Broadcasted` of `morphism`, once per occurrence.'''
    match morphism:
        case cat.Broadcasted():
            return (morphism,)
        case cat.Composed(content=parts) | cat.ProductOfMorphisms(content=parts):
            return tuple(root for part in parts for root in roots_of(part))
        case cat.Block(body=body):
            return roots_of(body)
    return ()


def operation_count(morphism: cat.BroadcastedCategory) -> nm.Numeric:
    '''The operations of every `Broadcasted` of `morphism`, symbolic in the axis sizes.

    A view performs no arithmetic, which is the count `operation_work` registers for
    one, and its reindexing may be strided, which `read_symbolic_work` cannot read, so
    a view is left out of the sum.'''
    counts = tuple(morphism_work.read_symbolic_work(root).operations
                   for root in roots_of(morphism)
                   if not isinstance(root.operator, ops.View))
    return nm.Addition.template(nm.Integer(0), *counts)


def operations_at(association: Association, sizes: Mapping[nm.Numeric, int]) -> float:
    return morphism_work.numeric_value(association.operations, sizes)


def written_order(chain: ContractionChain) -> cat.BroadcastedCategory:
    '''The chain as it is written: the producer, then the view where there is one,
    then the consumer, over the operands of the chain.'''
    result = (chain.producer.wraps if chain.read is None
              else chsh.make_composed(chain.producer.wraps, chain.read.wraps))
    others = chain.other_consumer_operands()
    if not others:
        return chsh.make_composed(result, chain.consumer.wraps)
    placed = tuple(0 if k == chain.port else 1 + (k if k < chain.port else k - 1)
                   for k in range(len(chain.consumer.dom)))
    return chsh.make_composed(
        chsh.make_product(result, identity_on(others)),
        cat.Rearrangement(placed, (*result.cod(), *others)),
        chain.consumer.wraps)


def merges_as_one_contraction(
    producer: cat.Broadcasted, consumer: cat.Broadcasted, port: int,
) -> bool:
    '''Whether `merge_einops` accepts the pair, which are the refusals of
    `einops_rearrange.merge_rule`.'''
    return (einops_simplification.is_einsum(producer)
            and einops_simplification.is_einsum(consumer)
            and all(entry is cat.WeaveMode.TILED
                    for entry in producer.output_weaves[0]._shape)
            and (len(producer.output_weaves[0]._shape)
                 == len(consumer.input_weaves[port]._shape))
            and tutil.is_mappable_broadcast(producer)
            and tutil.is_mappable_broadcast(consumer))


def unified_chain(chain: ContractionChain) -> UnifiedChain | None:
    '''The chain as one contraction, with its read carried onto the producer's
    operands, or `None` where the read cannot be carried or the two contractions do
    not merge.'''
    producer = chain.producer.wraps
    if chain.read is None:
        carried_producer: cat.BroadcastedCategory = producer
        operand_reads: fd.Prod[move_reads_backwards.Read] = (
            (None,) * len(producer.dom()))
    else:
        moved = move_reads_backwards.move_reads_backwards(
            producer,
            (move_reads_backwards.as_stride_morphism(chain.read.wraps.reindexings[0]),))
        if moved.stops:
            return None
        carried_producer, operand_reads = moved.expression, moved.domain_reads
    if not (isinstance(carried_producer, cat.Broadcasted) and merges_as_one_contraction(
            carried_producer, chain.consumer.wraps, chain.port)):
        return None
    views = tuple(
        identity_on((array,)) if read is None
        else move_reads_backwards.view_of(array, read)
        for array, read in zip(producer.dom(), operand_reads))
    return UnifiedChain(
        operand_reads=(*views, *(identity_on((array,))
                                 for array in chain.other_consumer_operands())),
        contraction=einops_rearrange.merge_einops(
            carried_producer, chain.consumer.wraps, chain.port))


def contract_pair_first(
    contraction: cat.Broadcasted, first: tuple[int, int],
) -> cat.BroadcastedCategory:
    '''`contraction` evaluated as two contractions. The operands at the positions
    `first` are contracted first, keeping every index the other operands or the result
    still read, and that intermediate is contracted against the other operands.'''
    inputs, output, _ = einops_simplification.index_shapes(contraction)
    rest = tuple(k for k in range(len(inputs)) if k not in first)
    read_later = (*(inputs[k] for k in rest), output)
    kept = einops_simplification.unique(
        variable for k in first for variable in inputs[k]
        if any(variable in shape for shape in read_later))
    datatype = contraction.output_weaves[0].datatype
    pair = einops_simplification.einsum(
        tuple(inputs[k] for k in first), kept, datatype)
    remainder = einops_simplification.einsum(
        (kept, *(inputs[k] for k in rest)), output, datatype)
    operands = tuple(contraction.dom())
    return chsh.make_composed(
        cat.Rearrangement((*first, *rest), operands),
        chsh.make_product(pair, *((identity_on(tuple(operands[k] for k in rest)),)
                                  if rest else ())),
        remainder)


def associations(chain: ContractionChain) -> fd.Prod[Association]:
    '''The order the chain is written in, followed by every order that contracts a
    pair of the unified contraction first, one of the pair an operand of the consumer.

    The pair of the producer's own operands, contracted after the read was carried onto
    them, is left out. It computes the producer once for every position the read
    returns, and the written order computes it once for every position of its own
    degree.'''
    written = written_order(chain)
    written_association = Association(
        contracted_first=tuple(range(len(chain.producer.dom))),
        morphism=written, operations=operation_count(written))
    unified = unified_chain(chain)
    if unified is None:
        return (written_association,)
    producer_width = len(chain.producer.dom)
    reordered = []
    for pair in itertools.combinations(range(len(unified.operand_reads)), 2):
        if pair[1] < producer_width:
            continue
        morphism = chsh.make_composed(
            chsh.make_product(*unified.operand_reads),
            contract_pair_first(unified.contraction, pair))
        reordered.append(Association(
            contracted_first=pair, morphism=morphism,
            operations=operation_count(morphism)))
    return (written_association, *reordered)


def cheaper_association(
    chain: ContractionChain, sizes: Mapping[nm.Numeric, int],
) -> Association | None:
    '''The order of the chain with the fewest operations at `sizes`, where it costs
    fewer than the order the chain is written in, and `None` otherwise.'''
    written, *others = associations(chain)
    if not others:
        return None
    cheapest = min(others, key=lambda association: operations_at(association, sizes))
    if operations_at(cheapest, sizes) < operations_at(written, sizes):
        return cheapest
    return None


def chains_in_scope(
    subgraphs: fd.Prod[hg.Hypergraph], handed_on: fd.Prod[hg.HypergraphObject],
) -> Iterable[ContractionChain]:
    '''Every chain whose three roots are siblings in one scope, in the order the
    consumers stand in `subgraphs`. `handed_on` are the wires the scope passes out of
    itself, which count as read.'''
    reads: Counter = Counter(wire for subgraph in subgraphs for wire in subgraph.dom)
    reads.update(handed_on)
    producer_of = {wire: subgraph for subgraph in subgraphs
                   if isinstance(subgraph, hg.HypergraphRoot) for wire in subgraph.cod}
    for consumer in subgraphs:
        if not is_contraction_root(consumer):
            continue
        for port, wire in enumerate(consumer.dom):
            if consumer.dom.count(wire) != 1 or reads[wire] != 1:
                continue
            source = producer_of.get(wire)
            read = None
            if source is not None and is_view_root(source):
                read, wire = source, source.dom[0]
                source = producer_of.get(wire) if reads[wire] == 1 else None
            if (source is not None and is_contraction_root(source)
                    and len(source.cod) == 1):
                yield ContractionChain(source, read, consumer, port)


def spliced(scope: hg.Multigraph, chain: ContractionChain,
            association: Association) -> hg.Multigraph:
    '''`scope` with the roots of `chain` replaced by the roots of `association`, on the
    operand wires of the chain and the result wire of its consumer.'''
    replacement = hg.Multigraph.from_morphism(
        association.morphism,
        dom=hg.HypergraphObject.template(
            tuple(association.morphism.dom()), chain.operand_wires()))
    removed = {chain.producer.uid, chain.consumer.uid}
    if chain.read is not None:
        removed.add(chain.read.uid)
    kept = tuple(subgraph for subgraph in scope._subgraphs
                 if subgraph.uid not in removed)
    renaming = fd.Context([fd.UIDRenaming.set_canonical(
        chain.consumer.cod[0], replacement.cod[0])])
    return renaming.apply(scope.reconstruct(_subgraphs=(
        *kept, *hg.flat_subgraphs(replacement, remove_blocks=True))))


def rewritten_scope(scope: hg.Multigraph,
                    sizes: Mapping[nm.Numeric, int]) -> hg.Multigraph:
    '''`scope` with one chain at a time rewritten into its cheapest order, until no
    chain has an order cheaper than the one it is written in. Each rewrite lowers the
    operation count of the scope at `sizes`, so the search ends.'''
    for _ in range(merge_into_consumer.MERGE_LIMIT):
        rewrite = next(
            ((chain, association)
             for chain in chains_in_scope(scope._subgraphs, scope.cod)
             if (association := cheaper_association(chain, sizes)) is not None),
            None)
        if rewrite is None:
            return scope
        scope = spliced(scope, *rewrite)
    raise RuntimeError(
        f'{merge_into_consumer.MERGE_LIMIT} chains rewritten without reaching a fixed '
        'point')


def rewritten_graph[L, M: cat.Morphism](
    graph: hg.Hypergraph[L, M], sizes: Mapping[nm.Numeric, int],
) -> hg.Hypergraph[L, M]:
    '''`graph` with every scope rewritten, the innermost first.'''
    match graph:
        case hg.HypergraphBlock(body=body):
            rewritten = rewritten_graph(body, sizes)
            return graph if rewritten is body else graph.reconstruct(body=rewritten)
        case hg.Multigraph(_subgraphs=subgraphs):
            descended = tuple(rewritten_graph(subgraph, sizes)
                              for subgraph in subgraphs)
            scope = (graph if all(before is after
                                  for before, after in zip(subgraphs, descended))
                     else graph.reconstruct(_subgraphs=descended))
            return rewritten_scope(scope, sizes)
    return graph


def absorb_linear_maps[L, M: cat.Morphism](
    target: cat.ProdCategory[L, M] | hg.Hypergraph[L, M],
    sizes: Mapping[nm.Numeric, int],
) -> cat.ProdCategory[L, M] | hg.Hypergraph[L, M]:
    '''Every chain of `target` rewritten into the order that costs the fewest
    operations at `sizes`, which binds the size of every axis the chains carry.

    Accepts a morphism or a hypergraph and returns the same kind, and returns `target`
    itself where no chain has a cheaper order.'''
    as_morphism = not isinstance(target, hg.Hypergraph)
    graph = hg.Multigraph.from_morphism(target) if as_morphism else target
    rewritten = rewritten_graph(graph, sizes)
    if rewritten is graph:
        return target
    return h2m.hypergraph_to_morphism(rewritten) if as_morphism else rewritten


@dataclass
class ReadThroughOneStrideMorphism[B: cat.Datatype, A: cat.Axis](
        hypergraph_functor.Endofunctor[cat.Array[B, A], cat.Broadcasted[B, A]]):
    '''Every view whose reindexing is strided, rewritten to read through one
    `sc.StrideMorphism`.

    `mark_sparse_domains.guarded_view` writes the causal mask as the product of a
    stride morphism over the positions and the identity on the other axes.
    `concatenation_expansion.reads_the_position_alone` reads a strided reindexing only
    when it is one `sc.StrideMorphism`, so it refuses to carry a concatenation through
    the mask in that form.'''
    def apply_root(self, target: cat.Broadcasted[B, A]) -> cat.Broadcasted[B, A]:
        if not (isinstance(target, cat.Broadcasted)
                and isinstance(target.operator, ops.View)
                and len(target.reindexings) == 1
                and not tutil.is_mappable(target.reindexings[0])):
            return target
        return target.reconstruct(reindexings=(
            move_reads_backwards.as_stride_morphism(target.reindexings[0]),))


def read_through_one_stride_morphism[L, M: cat.Morphism](
    target: cat.ProdCategory[L, M],
) -> cat.ProdCategory[L, M]:
    return ReadThroughOneStrideMorphism()(target)
