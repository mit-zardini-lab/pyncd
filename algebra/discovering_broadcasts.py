# Claude Opus 5 (1M context), high effort.
'''Writing a repeated operation as one block broadcast over a degree.

A `Broadcasted` over an `ops.BlockOperator` states that one body is computed
once per index of a degree. `discover_broadcast_over_axes` builds that statement
from a block whose arrays carry the degree axes at their head, and
`confirm_broadcast_expansion` proves it by expanding the statement back out and
comparing the expansion against the expression it came from.

The mathematics, and what the comparison establishes, are in
`obsidian/02-categories/Discovering Broadcasts.md`.
'''
from __future__ import annotations
from dataclasses import dataclass

import construction_helpers.lift as lift
import construction_helpers.simple_helper as chsh
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd
import graphs.processing.Hypergraph2Morphism as h2m
import term_utilities.term_utilities as tutil

import algebra.reindexing_absorption as reindexing_absorption


class DegreeAxesNotRemovable(ValueError):
    '''A block does not read the declared degree axes as one broadcast, so no
    body with those axes deleted states the same expression.'''


@dataclass(frozen=True)
class BroadcastConfirmation:
    '''What the comparison of a candidate against its original found.

    `normalised_original` and `normalised_expansion` are the two expressions the
    comparison ran on, so a caller that wants to read them prints them through
    `agent_display.listing`.
    '''
    candidate: cat.Broadcasted
    normalised_original: cat.BroadcastedCategory
    normalised_expansion: cat.BroadcastedCategory
    named_difference: str | None

    def is_confirmed(self) -> bool:
        return self.named_difference is None


@dataclass(frozen=True)
class DiscoveredBroadcast:
    '''A candidate built from a block and the axes declared to be its degree,
    with the confirmation that the candidate expands back to the block.'''
    candidate: cat.Broadcasted
    body: ops.BBlock
    confirmation: BroadcastConfirmation

    def is_confirmed(self) -> bool:
        return self.confirmation.is_confirmed()


def discover_broadcast_over_axes[B: cat.Datatype, A: cat.Axis](
    block: ops.BBlock[B, A],
    degree_axes: fd.Prod[A],
    name: str | None | fd.DynamicName = None,
) -> DiscoveredBroadcast:
    '''`block` written as its body without `degree_axes`, broadcast over them.

    Every array of the block's codomain carries `degree_axes` at its head, and
    each array of its domain either carries them at its head or carries none of
    them. An operand that carries them is read through the identity on the
    degree and an operand that does not is read through the rearrangement that
    deletes it, which is the reindexing tuple `((0,), (0,), (), ())` of an
    attention core whose queries and sink carry the head axis and whose latents
    do not.
    '''
    body = remove_leading_degree_axes(block, degree_axes)
    whole_degree = tuple(range(len(degree_axes)))
    candidate = broadcast_block_over_axes(
        body,
        degree_axes,
        tuple(whole_degree if reduced != carried else ()
              for carried, reduced in zip(block.dom(), body.dom())),
        name)
    return DiscoveredBroadcast(
        candidate=candidate,
        body=body,
        confirmation=confirm_broadcast_expansion(block, candidate))


def broadcast_block_over_axes[B: cat.Datatype, A: cat.Axis](
    body: ops.BBlock[B, A],
    degree_axes: fd.Prod[A],
    degree_readings: fd.Prod[fd.Prod[int]],
    name: str | None | fd.DynamicName = None,
) -> cat.Broadcasted[B, A, ops.BlockOperator[B, A]]:
    '''`body`, which carries none of `degree_axes`, as one operator computed
    once per index of them.

    `degree_readings` holds one tuple per operand of `body`, giving the degree
    positions that operand is read at, so the attention core whose queries and
    sink carry the head axis and whose two latents do not is
    `((0,), (0,), (), ())`. Each operand's weave tiles as many positions as its
    reading names, in front of the target the body receives, and every result
    carries the whole degree because an operation is computed once per index of
    it.
    '''
    degree = tuple(degree_axes)
    if len(degree_readings) != len(body.dom()):
        raise ValueError(
            f'{len(degree_readings)} degree readings for '
            f'{len(body.dom())} operands of the body')
    operator_name = fd.DynamicName.from_str(name) if name is not None else None
    if operator_name is not None:
        operator_name = operator_name.reconstruct(
            settings=fd.DynamicNameSettings(bold=True))
    return cat.Broadcasted(
        operator=ops.BlockOperator(name=operator_name, block=body),
        input_weaves=tuple(
            cat.Weave(array.datatype,
                      (cat.WeaveMode.TILED,) * len(reading)
                      + tuple(array.shape()))
            for array, reading in zip(body.dom(), degree_readings)),
        output_weaves=tuple(
            cat.Weave(array.datatype,
                      (cat.WeaveMode.TILED,) * len(degree)
                      + tuple(array.shape()))
            for array in body.cod()),
        reindexings=tuple(
            cat.Rearrangement(tuple(reading), degree)
            for reading in degree_readings))


def confirm_broadcast_expansion[B: cat.Datatype, A: cat.Axis](
    original: cat.BroadcastedCategory[B, A],
    candidate: cat.Broadcasted[B, A, ops.BlockOperator[B, A]],
) -> BroadcastConfirmation:
    '''Whether expanding `candidate` gives back `original`.

    A candidate whose degree is empty states no broadcast, so it is refused
    before anything is expanded.
    '''
    if len(candidate.degree()) == 0:
        return BroadcastConfirmation(
            candidate=candidate,
            normalised_original=original,
            normalised_expansion=original,
            named_difference='the candidate has an empty degree, so it states '
                             'no broadcast')
    expansion = expand_broadcast_of_block(candidate)
    normalised_original = normalise_for_comparison(original)
    normalised_expansion = normalise_for_comparison(expansion)
    return BroadcastConfirmation(
        candidate=candidate,
        normalised_original=normalised_original,
        normalised_expansion=normalised_expansion,
        named_difference=name_difference(
            normalised_original, normalised_expansion))


def expand_broadcast_of_block[B: cat.Datatype, A: cat.Axis](
    candidate: cat.Broadcasted[B, A, ops.BlockOperator[B, A]],
) -> cat.BroadcastedCategory[B, A]:
    '''The body lifted over the degree, with a `View` per operand carrying that
    operand's reindexing and a `View` per result putting the degree back where
    the output weave holds it.

    An operand whose reindexing deletes the degree is read through a repeat, so
    the lifted body reads every operand at the degree, which is the one shape
    `lift.morphism_object_lift` gives it.

    `ops.BlockOperator.expand` states the same expansion and builds it wrongly:
    it hands `ops.View.template` both the composed reindexing and the weave's
    target as the base, and `View.template` products the base's shape onto the
    reindexing a second time, so the operand arrives carrying its target axes
    twice over. It also closes with `weave.inverse_rearrangement(degree)`, whose
    domain is the weave's own shape where the body hands out the degree followed
    by the target.
    '''
    degree = candidate.degree()
    operand_views = chsh.make_product(*(
        ops.View.template(
            base=weave.datatype,
            reindexing=chsh.make_composed(
                chsh.make_product(
                    reindexing, weave.target().shape().identity()),
                weave.rearrangement(reindexing.cod())))
        for reindexing, weave
        in zip(candidate.reindexings, candidate.input_weaves)))
    result_views = chsh.make_product(*(
        ops.View.template(
            base=weave.datatype,
            reindexing=weave.inverse_rearrangement(degree))
        for weave in candidate.output_weaves))
    return chsh.make_composed(
        operand_views,
        lift.morphism_object_lift(candidate.operator.block, degree),
        result_views)


def normalise_for_comparison[B: cat.Datatype, A: cat.Axis](
    target: cat.BroadcastedCategory[B, A],
) -> cat.BroadcastedCategory[B, A]:
    '''`target` with its wiring recycled, its blocks removed, its repeats
    absorbed into the operations that read them, and every reindexing that is a
    rearrangement rewritten as one `cat.Rearrangement`.

    The blocks go because an expansion prepends its repeats outside the block it
    lifts, and `reindexing_absorption.absorb_nodes` merges siblings alone, so a
    repeat and the operation inside the block that reads it never meet while the
    block stands.
    '''
    return canonicalise_rearranging_reindexings(
        reindexing_absorption.absorb_nodes(
            remove_grouping_blocks(h2m.recycle(target))))


def remove_leading_degree_axes[B: cat.Datatype, A: cat.Axis](
    target: cat.BroadcastedCategory[B, A],
    degree_axes: fd.Prod[A],
) -> cat.BroadcastedCategory[B, A]:
    '''`target` with `degree_axes` deleted from the head of the degree of every
    operation and from the head of every array that begins with them.

    It is the inverse of `lift.morphism_object_lift` over those axes. A degree
    axis read twice by one operand is a diagonal rather than a broadcast, and a
    reindexing that computes rather than rearranges holds the axes in a row
    that deleting a position would change, so both are refused.
    '''
    match target:
        case cat.Block():
            return target.reconstruct(
                body=remove_leading_degree_axes(target.body, degree_axes))
        case cat.Rearrangement(mapping=mapping, _dom=arrays):
            return cat.Rearrangement(
                mapping=mapping,
                _dom=tuple(_array_without_leading_axes(array, degree_axes)
                           for array in arrays))
        case cat.Composed(content=segments) | cat.ProductOfMorphisms(content=segments):
            return type(target).from_iter(
                remove_leading_degree_axes(segment, degree_axes)
                for segment in segments)
        case cat.Broadcasted():
            return _broadcasted_without_leading_degree(target, degree_axes)
    raise DegreeAxesNotRemovable(
        f'a {type(target).__name__} has no rule for deleting a leading degree')


def remove_grouping_blocks[B: cat.Datatype, A: cat.Axis](
    target: cat.BroadcastedCategory[B, A],
) -> cat.BroadcastedCategory[B, A]:
    '''`target` with every block whose repetition is one replaced by its body.

    A block whose repetition is not one denotes a loop over the body, so it
    states something the body alone does not and it stays.
    '''
    match target:
        case cat.Block(body=body, block_tag=tag):
            inner = remove_grouping_blocks(body)
            if tag.repetition == nm.Integer(1):
                return inner
            return target.reconstruct(body=inner)
        case cat.Composed(content=segments) | cat.ProductOfMorphisms(content=segments):
            return type(target).from_iter(
                remove_grouping_blocks(segment) for segment in segments)
    return target


def canonicalise_rearranging_reindexings[B: cat.Datatype, A: cat.Axis](
    target: cat.BroadcastedCategory[B, A],
) -> cat.BroadcastedCategory[B, A]:
    '''`target` with every reindexing that permutes, copies and deletes written
    as one `cat.Rearrangement` over the degree.

    Two morphisms that read the same positions may hold the reading as a
    product, as a composition or as one rearrangement, so an equality test on
    the terms answers no on a pair that states one map. A reindexing that
    computes is left where it is, because there is no mapping to rewrite it as.
    '''
    match target:
        case cat.Block():
            return target.reconstruct(
                body=canonicalise_rearranging_reindexings(target.body))
        case cat.Composed(content=segments) | cat.ProductOfMorphisms(content=segments):
            return type(target).from_iter(
                canonicalise_rearranging_reindexings(segment)
                for segment in segments)
        case cat.Broadcasted(reindexings=reindexings):
            degree = tuple(target.degree())
            return target.reconstruct(reindexings=tuple(
                _canonical_reindexing(reindexing, degree)
                for reindexing in reindexings))
    return target


def name_difference[B: cat.Datatype, A: cat.Axis](
    original: cat.BroadcastedCategory[B, A],
    expansion: cat.BroadcastedCategory[B, A],
) -> str | None:
    '''The first way the two expressions disagree, or `None` when they agree.'''
    for side, left, right in (
            ('domain', original.dom(), expansion.dom()),
            ('codomain', original.cod(), expansion.cod())):
        difference = _name_object_difference(side, tuple(left), tuple(right))
        if difference is not None:
            return difference
    if original == expansion:
        return None
    return _name_operation_difference(original, expansion)


def _canonical_reindexing[A: cat.Axis](
    reindexing: cat.StrideCategory[A],
    degree: fd.Prod[A],
) -> cat.StrideCategory[A]:
    try:
        return cat.Rearrangement(tuple(tutil.get_mapping(reindexing)), degree)
    except (ValueError, KeyError):
        return reindexing


def _array_without_leading_axes[B: cat.Datatype, A: cat.Axis](
    array: cat.Array[B, A],
    degree_axes: fd.Prod[A],
) -> cat.Array[B, A]:
    shape = tuple(array.shape())
    if shape[:len(degree_axes)] != tuple(degree_axes):
        return array
    return cat.Array(array.datatype, shape[len(degree_axes):])


def _broadcasted_without_leading_degree[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A],
    degree_axes: fd.Prod[A],
) -> cat.Broadcasted[B, A]:
    count = len(degree_axes)
    degree = tuple(target.degree())
    if degree[:count] != tuple(degree_axes):
        misplaced = tuple(axis for axis in degree_axes if axis in degree)
        if misplaced:
            raise DegreeAxesNotRemovable(
                f'{_operation_text(target)} carries {_axis_names(misplaced)} at '
                'a later degree position, and a deletion takes a leading degree '
                'alone')
        return target
    remaining = cat.ProdObject(degree[count:])
    reindexings = []
    input_weaves = []
    for reindexing, weave in zip(target.reindexings, target.input_weaves):
        mapping = _mapping_of_reindexing(reindexing, target)
        read_twice = [position for position in range(count)
                      if sum(1 for read in mapping if read == position) > 1]
        if read_twice:
            raise DegreeAxesNotRemovable(
                f'{_operation_text(target)} reads degree position '
                f'{read_twice[0]} twice in one operand, which is a diagonal '
                'rather than a broadcast')
        reindexings.append(cat.Rearrangement(
            tuple(read - count for read in mapping if read >= count),
            degree[count:]))
        input_weaves.append(_weave_without_tiled_slots(
            weave, tuple(slot for slot, read in enumerate(mapping)
                         if read < count)))
    return target.reconstruct(
        input_weaves=tuple(input_weaves),
        output_weaves=tuple(
            _weave_without_tiled_slots(weave, tuple(range(count)))
            for weave in target.output_weaves),
        reindexings=tuple(reindexings),
        backup_degree=remaining if not reindexings else None)


def _mapping_of_reindexing[B: cat.Datatype, A: cat.Axis](
    reindexing: cat.StrideCategory[A],
    target: cat.Broadcasted[B, A],
) -> fd.Prod[int]:
    try:
        return tuple(tutil.get_mapping(reindexing))
    except (ValueError, KeyError) as reason:
        raise DegreeAxesNotRemovable(
            f'{_operation_text(target)} reads an operand through a reindexing '
            'that computes rather than rearranges, so no degree position can be '
            'deleted from it') from reason


def _weave_without_tiled_slots[B: cat.Datatype, A: cat.Axis](
    weave: cat.Weave[B, A],
    dropped_slots: fd.Prod[int],
) -> cat.Weave[B, A]:
    '''`weave` without the tiled slots numbered in `dropped_slots`, the
    numbering running over the tiled slots alone.'''
    kept = []
    slot = 0
    for entry in weave._shape:
        if isinstance(entry, cat.WeaveMode):
            if slot in dropped_slots:
                slot += 1
                continue
            slot += 1
        kept.append(entry)
    return cat.Weave(weave.datatype, tuple(kept))


def _name_object_difference[B: cat.Datatype, A: cat.Axis](
    side: str,
    original: fd.Prod[cat.Array[B, A]],
    expansion: fd.Prod[cat.Array[B, A]],
) -> str | None:
    if len(original) != len(expansion):
        return (f'the original has {len(original)} arrays in its {side} and the '
                f'expansion has {len(expansion)}')
    for position, (left, right) in enumerate(zip(original, expansion)):
        if left != right:
            return (f'{side} position {position} is {_array_text(left)} in the '
                    f'original and {_array_text(right)} in the expansion')
    return None


def _name_operation_difference[B: cat.Datatype, A: cat.Axis](
    original: cat.BroadcastedCategory[B, A],
    expansion: cat.BroadcastedCategory[B, A],
) -> str:
    left = operations_of(original)
    right = operations_of(expansion)
    if len(left) != len(right):
        return (f'the original has {len(left)} operations, '
                f'{_operation_tally(left)}, and the expansion has {len(right)}, '
                f'{_operation_tally(right)}')
    for position, (one, other) in enumerate(zip(left, right)):
        if one != other:
            return (f'operation {position} is {_operation_text(one)} in the '
                    f'original and {_operation_text(other)} in the expansion')
    return ('the operations agree and the expressions differ, so the wiring or '
            'the blocks differ')


def operations_of[B: cat.Datatype, A: cat.Axis](
    target: cat.BroadcastedCategory[B, A],
) -> fd.Prod[cat.Broadcasted[B, A]]:
    match target:
        case cat.Broadcasted():
            return (target,)
        case cat.Composed(content=segments) | cat.ProductOfMorphisms(content=segments):
            return tuple(operation for segment in segments
                         for operation in operations_of(segment))
        case cat.Block(body=body):
            return operations_of(body)
    return ()


def _operation_tally[B: cat.Datatype, A: cat.Axis](
    operations: fd.Prod[cat.Broadcasted[B, A]],
) -> str:
    counts: dict[str, int] = {}
    for operation in operations:
        kind = type(operation.operator).__name__
        counts[kind] = counts.get(kind, 0) + 1
    return ', '.join(f'{count}x {kind}' for kind, count in sorted(counts.items()))


def _operation_text[B: cat.Datatype, A: cat.Axis](
    operation: cat.Broadcasted[B, A],
) -> str:
    operands = ' '.join(_array_text(array) for array in operation.dom())
    results = ' '.join(_array_text(array) for array in operation.cod())
    return (f'{type(operation.operator).__name__}({operands}) -> {results} at '
            f'degree [{_axis_names(tuple(operation.degree()))}]')


def _array_text[B: cat.Datatype, A: cat.Axis](array: cat.Array[B, A]) -> str:
    return f'[{_axis_names(tuple(array.shape()))}]'


def _axis_names[A: cat.Axis](axes: fd.Prod[A]) -> str:
    return ', '.join(
        axis.uid._name.to_bodies() if axis.uid._name is not None else '?'
        for axis in axes)
