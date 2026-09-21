'''Rewriting a consumer of a concatenated axis into the consumers of the parts.

Written by Claude Opus 5 (1M context), reasoning effort medium.

`F(Concat(x, y)) = B_F(F(x), F(y))` holds whenever `F` treats the positions of the
concatenated axis one at a time. `F` then either computes at each position on its own,
in which case its results over the parts concatenate again, or folds the axis away, in
which case `part_combination.combination_for` gives the operator that folds the two
partial results into one. `ConcatenatedAxisRole` names the two cases, and an operator
that reads two positions of the axis at once, as a `ops.SoftMax` does, belongs to
neither and raises.

Every consumer of the concatenated wire changes, so the rewrite works on the
hypergraph, as `deepseek.sparse_expansion` does. One pass replaces each consumer by one
root per part and one root for the combination, on the consumer's own wires, and leaves
out every concatenation nothing reads any more. A pass that makes a new concatenation,
which is what the streamed case does, is followed by another, so a chain of streamed
operations between the concatenation and the fold that consumes the axis unwinds one
operation at a time.

A consumer standing inside the body of a block is reached by `expand_concatenations`
only when the concatenation stands in the same body. A concatenation outside a block
whose consumer is inside it is reached by `expand_concatenations_through_blocks`, which
writes the block out first, so that the body and the concatenation stand in one scope.
The rule it applies is `x ⋮ y >> F = (x >> F * y >> F) ; concat`, read from the
concatenation towards the consumers: a block computed once per index of a degree reads
the concatenated array at every index, and writing the block out is what puts one copy
of the concatenation in front of the consumers of each part.

The attention core of `notebooks/sota/DeepSeekV41Flash/attention_core.py` is the
expression the rewrite was written for. Written with the window latents and the
selected-entry latents concatenated along the slot axis, it holds one contraction of
the queries against the
concatenated latents, one exponential, one sum and one contraction of the weights
against the same latents, and the rewrite turns it into the two-branch core
`scaled_attention_core.py` writes by hand.

`obsidian/02-categories/Advanced Axis Dynamics.md` states the rewrite.
'''
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

import data_structure.Term as fd # for 'foundations'
import data_structure.Numeric as nm
import data_structure.Category as cat
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import graphs.data_structure.Hypergraph as hg
import graphs.processing.Hypergraph2Morphism as h2m
import term_utilities.term_utilities as tutil
import algebra.discovering_broadcasts as discovering_broadcasts
import advanced_axis_dynamics.data_structure.Operators as aops
import advanced_axis_dynamics.registries.part_combination as part_combination


class ConcatenatedAxisIsNotStreamed(Exception):
    '''A consumer of a concatenated axis that neither computes at each position of it
    on its own nor folds it away, so its result over a part is no part of its result
    over the whole.'''


class OperandsConcatenateDifferently(Exception):
    '''Two operands of one consumer concatenated along axes that are not the same, or
    from parts that are not the same, so no part of one matches a part of the other.'''


class ConcatenationHasSeveralOutputs(Exception):
    '''A `ConcatenateAxes` or one of its consumers with more than one output, which the
    rewrite does not wire.'''


class ConcatenatedAxisRole(Enum):
    '''What a consumer does with the concatenated axis.'''
    STREAMED = 'streamed'
    FOLDED = 'folded'


@dataclass(frozen=True)
class Concatenation:
    '''One `aops.ConcatenateAxes` root, read off the weaves it carries.

    `position` is where the concatenated axis stands in the arrays, `axis` is that
    axis and `parts` the axis each input holds there. They are read from the weaves
    rather than from the operator's own reindexings, because composition aligns the
    axes of the expression by uid, so the axis in a weave and the axis in a reindexing
    carry one uid and need not be one object. Every comparison below is by uid for
    the same reason.
    '''
    root: hg.HypergraphRoot
    position: int
    axis: cat.Axis
    parts: fd.Prod[cat.Axis]

    @classmethod
    def of(cls, root: hg.HypergraphRoot) -> Concatenation:
        weave = root.wraps.output_weaves[0]
        position = next(index for index, entry in enumerate(weave._shape)
                        if not isinstance(entry, cat.WeaveMode))
        return cls(root=root, position=position, axis=weave._shape[position],
                   parts=tuple(operand._shape[position]
                               for operand in root.wraps.input_weaves))


def expand_concatenations[L, M: cat.Morphism](
    target: cat.ProdCategory[L, M] | hg.Hypergraph[L, M],
) -> cat.ProdCategory[L, M] | hg.Hypergraph[L, M]:
    '''Every consumer of a concatenated axis in `target` rewritten into the consumers
    of the parts.

    Accepts a morphism or a hypergraph and returns the same kind. A target holding no
    concatenation, or one whose consumers the rewrite cannot reach, comes back as the
    same object.
    '''
    if not any(True for _ in tutil.type_search(aops.ConcatenateAxes, target)):
        return target
    as_morphism = not isinstance(target, hg.Hypergraph)
    graph = hg.Multigraph.from_morphism(target) if as_morphism else target
    rewritten = expanded_graph(graph)
    if rewritten is graph:
        return target
    return h2m.hypergraph_to_morphism(rewritten) if as_morphism else rewritten


def expand_concatenations_through_blocks[B: cat.Datatype, A: cat.Axis](
    target: cat.BroadcastedCategory[B, A],
) -> cat.BroadcastedCategory[B, A]:
    '''Every consumer of a concatenated axis in `target` rewritten into the consumers
    of the parts, a consumer standing inside the body of a block included.

    A block whose degree is not empty computes its body once per index of that degree,
    and a concatenated wire on its domain is the same array at every index, so the
    parts have to reach the body without the concatenation being performed once per
    index. `expand_blocks_reading_concatenations` supplies them by writing the block
    out: the body is lifted over the degree, each operand is read through the `View`
    of its own reindexing, and the block that held the body is dropped. The
    concatenation and its consumers then stand in one scope and
    `expand_concatenations` rewrites them.

    The alternative, composing the concatenation onto the head of the body, is what
    the degree rules out. The attention core of DeepSeek-V4.1-Flash is computed once
    per head and reads latents that every head shares, so a body holding the
    concatenation would concatenate the latents once per head, and
    `discovering_broadcasts.confirm_broadcast_expansion` refuses such a body against
    the core written out over every head.

    The result is the flat expression, because every block the concatenation reaches
    is written out. A model composes the expanded form and states the broadcast again
    over it, which is what `notebooks/sota/DeepSeekV41Flash/attention_core.py` does.
    '''
    return expand_concatenations(expand_blocks_reading_concatenations(target))


def expand_blocks_reading_concatenations[B: cat.Datatype, A: cat.Axis](
    target: cat.BroadcastedCategory[B, A],
) -> cat.BroadcastedCategory[B, A]:
    '''`target` with every broadcast of a `ops.BlockOperator` that reads a
    concatenated axis replaced by its body lifted over the degree.

    The axes are the ones a `aops.ConcatenateAxes` of `target` writes its parts onto,
    compared by uid because composition aligns the axes of the expression and leaves
    the operator's own reindexings as they were built.
    '''
    return expand_blocks_reading_the_axes(target, {
        concatenation.concatenated_axis().uid
        for concatenation in tutil.type_search(aops.ConcatenateAxes, target)})


def expand_blocks_reading_the_axes[B: cat.Datatype, A: cat.Axis](
    target: cat.BroadcastedCategory[B, A],
    axes: set[fd.UID],
) -> cat.BroadcastedCategory[B, A]:
    '''`target` with every broadcast of a `ops.BlockOperator` one of whose operands
    carries an axis of `axes` written out, and the block that held its body dropped.

    A body that itself holds such a broadcast is written out too, because the
    expansion is descended into.
    '''
    match target:
        case cat.Broadcasted(operator=ops.BlockOperator()) if any(
                entry.uid in axes
                for array in target.dom() for entry in array.shape()):
            return expand_blocks_reading_the_axes(
                discovering_broadcasts.remove_grouping_blocks(
                    discovering_broadcasts.expand_broadcast_of_block(target)),
                axes)
        case cat.Block(body=body):
            written_out = expand_blocks_reading_the_axes(body, axes)
            return (target if written_out is body
                    else target.reconstruct(body=written_out))
        case cat.Composed(content=segments) | cat.ProductOfMorphisms(content=segments):
            return type(target).from_iter(
                expand_blocks_reading_the_axes(segment, axes)
                for segment in segments)
    return target


def expanded_graph[L, M: cat.Morphism](
    graph: hg.Hypergraph[L, M]) -> hg.Hypergraph[L, M]:
    '''`graph` with the rewrite applied inside every scope, a scope being a
    `hg.Multigraph` and the body of every block below it.'''
    match graph:
        case hg.HypergraphBlock(body=body):
            rewritten = expanded_graph(body)
            return graph if rewritten is body else graph.reconstruct(body=rewritten)
        case hg.Multigraph(_subgraphs=subgraphs):
            descended = tuple(expanded_graph(subgraph) for subgraph in subgraphs)
            rewritten = rewritten_scope(descended, graph.cod)
            if all(before is after for before, after in zip(subgraphs, rewritten)) and (
                    len(subgraphs) == len(rewritten)):
                return graph
            return graph.reconstruct(_subgraphs=rewritten)
    return graph


def rewritten_scope[L, M: cat.Morphism](
    subgraphs: fd.Prod[hg.Hypergraph[L, M]],
    handed_on: fd.Prod[hg.HypergraphObject[L]],
) -> fd.Prod[hg.Hypergraph[L, M]]:
    '''`subgraphs` with every consumer of a concatenated wire replaced, run until no
    consumer is left that the rewrite reaches.

    `handed_on` are the wires the scope passes out of itself, so a concatenation
    producing one of them stays however few consumers it has here.
    '''
    rewriting = True
    while rewriting:
        rewriting = False
        concatenations = concatenation_roots(subgraphs)
        if not concatenations:
            break
        rebuilt: list[hg.Hypergraph[L, M]] = []
        for subgraph in subgraphs:
            replacement = replaced_consumer(subgraph, concatenations)
            if replacement is None:
                rebuilt.append(subgraph)
                continue
            rebuilt.extend(replacement)
            rewriting = True
        subgraphs = unread_concatenations_dropped(tuple(rebuilt), handed_on)
    return tuple(subgraphs)


def concatenation_roots[L, M: cat.Morphism](
    subgraphs: fd.Prod[hg.Hypergraph[L, M]],
) -> dict[fd.UID, Concatenation]:
    '''The roots of `subgraphs` holding a `ConcatenateAxes`, keyed by the wire each
    one produces.'''
    roots = {}
    for subgraph in subgraphs:
        if not isinstance(subgraph, hg.HypergraphRoot):
            continue
        wraps = subgraph.wraps
        if not isinstance(wraps, cat.Broadcasted):
            continue
        if not isinstance(wraps.operator, aops.ConcatenateAxes):
            continue
        if len(subgraph.cod) != 1:
            raise ConcatenationHasSeveralOutputs(
                f'a concatenation with {len(subgraph.cod)} outputs')
        roots[subgraph.cod[0].uid] = Concatenation.of(subgraph)
    return roots


def replaced_consumer[L, M: cat.Morphism](
    subgraph: hg.Hypergraph[L, M],
    concatenations: dict[fd.UID, Concatenation],
) -> fd.Prod[hg.Hypergraph[L, M]] | None:
    '''The roots replacing `subgraph`, one per part and one for the combination, or
    `None` where `subgraph` is not a consumer the rewrite reaches.

    A block, a rearrangement and a concatenation of concatenations are all left alone:
    a block would need the part wires carried across it, and a concatenation reads its
    parts in its target rather than streaming over them. So is a consumer one of whose
    operands carries the concatenated axis and does not arrive from a concatenation,
    which is the operand a streamed consumer of the same axis is about to produce, and
    a later pass reaches it once that concatenation exists.
    '''
    if not isinstance(subgraph, hg.HypergraphRoot):
        return None
    consumer = subgraph.wraps
    if not isinstance(consumer, cat.Broadcasted):
        return None
    if isinstance(consumer.operator, aops.ConcatenateAxes):
        return None
    operands = {position: concatenations[wire.uid]
                for position, wire in enumerate(subgraph.dom)
                if wire.uid in concatenations}
    if not operands:
        return None
    if len(subgraph.cod) != 1:
        raise ConcatenationHasSeveralOutputs(
            f'{consumer.operator} consumes a concatenated axis and has '
            f'{len(subgraph.cod)} outputs')
    axis, parts = the_shared_concatenation(tuple(operands.values()))
    if any(position not in operands
           for position, array in enumerate(consumer.dom())
           if carries_the_axis(array, axis)):
        return None
    role = role_of(consumer, axis)
    part_roots = tuple(
        hg.HypergraphRoot.template(
            consumer_of_the_part(consumer, axis, part),
            dom=tuple(operands[position].root.dom[index]
                      if position in operands else wire
                      for position, wire in enumerate(subgraph.dom)))
        for index, part in enumerate(parts))
    return (*part_roots, hg.HypergraphRoot.template(
        combination(role, consumer, part_roots, axis, operands),
        dom=tuple(root.cod[0] for root in part_roots),
        cod=subgraph.cod))


def the_shared_concatenation(
    concatenations: fd.Prod[Concatenation],
) -> tuple[cat.Axis, fd.Prod[cat.Axis]]:
    '''The axis every operand of one consumer is concatenated along, and its parts.

    The concatenated axis need not stand at one position in every operand: the scores
    of an attention core carry it last and the values carry it in the middle. The uids
    are what have to agree.
    '''
    axes = {concatenation.axis.uid for concatenation in concatenations}
    parts = {tuple(part.uid for part in concatenation.parts)
             for concatenation in concatenations}
    if len(axes) != 1 or len(parts) != 1:
        raise OperandsConcatenateDifferently(
            f'{len(axes)} concatenated axes and {len(parts)} sets of parts among the '
            'operands of one consumer')
    return concatenations[0].axis, concatenations[0].parts


def role_of(consumer: cat.Broadcasted, axis: cat.Axis) -> ConcatenatedAxisRole:
    '''What `consumer` does with `axis`, raising where it does neither.

    The axis stands either in the degree, so that the operator is broadcast over it and
    computes at each position on its own, or in the target of an operand and nowhere in
    the output, so that the operator folds it away. An operator holding it in the
    target on both sides reads every position to write every position, and an operand
    holding it twice reads a diagonal of it, and neither has a result over a part.
    '''
    degree = tuple(consumer.degree())
    in_degree = [position for position, entry in enumerate(degree)
                 if entry.uid == axis.uid]
    into = [target_count(weave, axis) for weave in consumer.input_weaves]
    out_of = [target_count(weave, axis) for weave in consumer.output_weaves]
    if in_degree and not any(into) and not any(out_of):
        if len(in_degree) == 1 and streams_over(consumer, in_degree[0], axis):
            return ConcatenatedAxisRole.STREAMED
        raise ConcatenatedAxisIsNotStreamed(
            f'{consumer.operator} is broadcast over {len(in_degree)} positions of a '
            'concatenated axis, or reads one position of it at another')
    if any(into) and not any(out_of) and not in_degree:
        if all(count <= 1 for count in into):
            return ConcatenatedAxisRole.FOLDED
        raise ConcatenatedAxisIsNotStreamed(
            f'{consumer.operator} holds a concatenated axis at {max(into)} positions '
            'of one operand, which is a diagonal of it')
    raise ConcatenatedAxisIsNotStreamed(
        f'{consumer.operator} holds a concatenated axis in the target of '
        f'{sum(1 for count in into if count)} operands and '
        f'{sum(1 for count in out_of if count)} outputs, and neither streams over it '
        'nor folds it away')


def carries_the_axis(array: cat.Array, axis: cat.Axis) -> bool:
    '''Whether `array` holds `axis` at any position, target or broadcast.'''
    return any(entry.uid == axis.uid for entry in array.shape())


def target_count(weave: cat.Weave, axis: cat.Axis) -> int:
    '''How many positions of the weave's target carry `axis`.'''
    return sum(1 for entry in weave.target().shape() if entry.uid == axis.uid)


def streams_over(consumer: cat.Broadcasted, position: int, axis: cat.Axis) -> bool:
    '''Whether every operand of `consumer` reads the degree position `position` at its
    own index, so that each position of `axis` is computed from that position alone.'''
    return all(reads_the_position_alone(reindexing, position, axis)
               for reindexing in consumer.reindexings)


def reads_the_position_alone(reindexing: cat.StrideCategory, position: int,
                             axis: cat.Axis) -> bool:
    '''Whether `reindexing` carries the degree position `position` to at most one
    position of its operand, at the same index.

    A reindexing that does not read the position at all broadcasts the operand over the
    axis, which reads one position of it as readily. A row that reads the position at a
    stride, at a shift, or beside another axis mixes the positions of the axis, and a
    row onto a different axis reads the position of one axis as the position of
    another.

    A reindexing that permutes, copies and deletes is read through
    `term_utilities.get_mapping` rather than matched, because the expansion of a block
    hands each operand the composition of its own reindexing with the rearrangement
    separating the weave's degree from its target, and a composition of rearrangements
    is neither a `cat.Rearrangement` nor a `sc.StrideMorphism`.
    '''
    try:
        mapping = tuple(tutil.get_mapping(reindexing))
    except (ValueError, KeyError):
        mapping = None
    if mapping is not None:
        return sum(1 for entry in mapping if entry == position) <= 1
    match reindexing:
        case sc.StrideMorphism(_cod_stride_shift=rows):
            reading = [row for row in rows if not nm.is_zero(row[1][position])]
            if not reading:
                return True
            if len(reading) != 1:
                return False
            row_axis, strides, shift = reading[0]
            return (row_axis.uid == axis.uid
                    and strides[position] == nm.Integer(1)
                    and nm.is_zero(shift)
                    and all(nm.is_zero(stride)
                            for index, stride in enumerate(strides)
                            if index != position))
    return False


def consumer_of_the_part(consumer: cat.Broadcasted, axis: cat.Axis,
                         part: cat.Axis) -> cat.Broadcasted:
    '''`consumer` with the concatenated axis replaced by `part` in every weave and
    every reindexing, which is the consumer of that part alone.'''
    substitution = fd.Context([fd.EqualityClass(
        _type=type(axis), bucket={axis.uid}, canonical=part)])
    return substitution.apply(consumer)


def combination[L, M: cat.Morphism](
    role: ConcatenatedAxisRole,
    consumer: cat.Broadcasted,
    part_roots: fd.Prod[hg.HypergraphRoot[L, M]],
    axis: cat.Axis,
    operands: dict[int, Concatenation],
) -> cat.Broadcasted:
    '''`B_F`: the morphism taking the result of the consumer on each part to the result
    on the whole axis.

    A streamed consumer hands out an array carrying the part, so the results
    concatenate again, along the same axis and under the name the concatenation of the
    operands carried. A folded consumer hands out an array without the axis, and the
    results fold pointwise by the accumulator of that fold, which
    `part_combination.combination_for` reads from `algebra.registries.accumulator`.
    '''
    results = tuple(root.wraps.cod()[0] for root in part_roots)
    datatype = consumer.output_weaves[0].datatype
    if role is ConcatenatedAxisRole.STREAMED:
        named = next(iter(operands.values())).root.wraps.operator.name
        return aops.ConcatenateAxes.template(
            tuple(tuple(result.shape()) for result in results),
            base=datatype, concatenated=axis, name=named)
    return pointwise_fold(
        part_combination.combination_for(consumer.operator),
        results[0], len(results))


def pointwise_fold(operator: cat.Operator, result: cat.Array,
                   partials: int) -> cat.Broadcasted:
    '''`operator` applied over every position of `result`, with `partials` operands of
    that shape and one output, which is how two partial results of a fold are
    combined into one.'''
    weave = cat.Weave(result.datatype,
                      (cat.WeaveMode.TILED,) * len(result.shape()))
    identity = cat.ProdObject(tuple(result.shape())).identity()
    return cat.Broadcasted(
        operator=operator, input_weaves=(weave,) * partials,
        output_weaves=(weave,), reindexings=(identity,) * partials)


def unread_concatenations_dropped[L, M: cat.Morphism](
    subgraphs: fd.Prod[hg.Hypergraph[L, M]],
    handed_on: fd.Prod[hg.HypergraphObject[L]],
) -> fd.Prod[hg.Hypergraph[L, M]]:
    '''`subgraphs` without the concatenations whose wire nothing in the scope reads and
    the scope does not hand on.'''
    concatenations = concatenation_roots(subgraphs)
    read = {wire.uid for subgraph in subgraphs for wire in subgraph.dom}
    read |= {wire.uid for wire in handed_on}
    return tuple(subgraph for subgraph in subgraphs
                 if not isinstance(subgraph, hg.HypergraphRoot)
                 or subgraph.cod[0].uid not in concatenations
                 or subgraph.cod[0].uid in read)
