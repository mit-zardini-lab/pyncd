'''The operators that read a reindexing covariantly.

Written by Claude Opus 5 (1M context), reasoning effort medium.

A `ops.View` reads a reindexing contravariantly: its reindexing sends each output
position to the input position it reads, which is the direction every `reindexings`
entry of a `cat.Broadcasted` points. The two operators here read a reindexing the other
way, sending each input position to the output position it writes.

`CovariantView` is one reindexing read that way, which is the merge of a split, and
`mark_sparse_codomains.mark_sparse_codomain` marks the codomain axes its image does not
fill. `ConcatenateAxes` is one reindexing per input read that way, their images disjoint
and together filling one codomain axis, which is a concatenation.
`concatenation_expansion.expand_concatenations` rewrites a consumer that streams over
the concatenated axis into the consumers of the parts. The axis a concatenation
produces is an `AxisConcatenation.ConcatenatedAxis`, which references its parts.

`DeconcatenateAxes` cuts one axis into the parts that fill it, with the same reindexings
read contravariantly, one per output. It is the reverse derivative of a concatenation,
a concatenation is its reverse derivative, and
`advanced_axis_dynamics/registries/standard_expansions.py` writes it out as a copy
followed by one `ops.View` per part.

This module holds the operators advanced axis dynamics introduces, so a reader who
meets `aops.` knows which feature declared the operator.

`obsidian/02-categories/Advanced Axis Dynamics.md` states the feature, and
`obsidian/02-categories/Operators.md` lists every operator in the package.
'''
from __future__ import annotations
from dataclasses import dataclass

import data_structure.Term as fd # for 'foundations'
import data_structure.Numeric as nm
import data_structure.Category as cat
import advanced_axis_dynamics.data_structure.AffineGuards as AffineGuards
import advanced_axis_dynamics.data_structure.AxisConcatenation as AxisConcatenation
import advanced_axis_dynamics.algebra.disentangle_reindexings as disentangle_reindexings
import advanced_axis_dynamics.algebra.mark_sparse_codomains as mark_sparse_codomains


class InputAxesDoNotMatchTheDomain(Exception):
    '''Input axes handed to a merge that are neither one per domain axis of the
    reindexing, which `CovariantView.template` merges, nor the leading domain axes,
    which `CovariantView.broadcast_over_absent_axes_and_merge` merges while
    broadcasting the input over the rest.'''


@dataclass(frozen=True)
class CovariantView(cat.Operator):
    '''A reindexing read covariantly: the output at `reindexing(i)` is the input at `i`.

    A `View` reads its reindexing contravariantly. Its reindexing maps each output
    position to the input position it reads, which is the direction every
    `reindexings` entry of a `Broadcasted` points. Read the other way, a reindexing
    sends each input position to the output position it writes, and that reading is
    a function only when the reindexing is a bijection between its two index boxes.
    `mark_sparse_codomains.merge_groups` tests it: in some order of the rows, each row
    writes its own domain axes as a signed mixed-radix number, with a shift, beside the
    axes the rows before it determine. The split of an axis into blocks and offsets,
    `i_B = |u| i_P + i_u`, is the case with one row, and this operator is its merge,
    which no `View` states because the inverse of the split rounds down and is not
    affine. A codomain position no live domain position writes holds the universal
    unit, and `mark_sparse_codomains.mark_sparse_codomain` derives the
    `AffineGuards.AffineSparseAxis` each such codomain axis becomes.

    The input need not carry every axis the reindexing merges.
    `broadcast_over_absent_axes_and_merge` takes the axes the input does carry, which
    stand at the head of the domain, and broadcasts the input over the rest, so one
    input position is written to one output position per index of the absent axes. The
    merge `(b, a) -> x` of the lightning indexer, at
    `i_x = |a| i_b + i_a + (|a| - 1)`, is that case. Its input carries the compressed
    entries `b` and not the offsets `a`, group `b` holds the `|a|` queries whose newest
    reachable entry is `b`, and each entry's slots are written to every query of its
    group, so the merge produces `[x, r|x]` from `[b, r|b]`. The image of the whole
    domain box is what the marking reads, so the absent axes enter the mixed-radix
    test beside the axes the input carries.

    The reindexing is a field of the operator rather than a `reindexings` entry, for
    the reason `para.data_structure.transpose.ReindexTranspose` gives: an entry points
    from the output back to the input, and this one runs the other way. The axes the
    input carries are the target of the input weave and the reindexing's codomain axis
    is the target of the output weave, so the operator consumes the axes it merges and
    produces the merged one, and the degree carries the rest. Its reverse derivative
    is the `View` of the same reindexing, per
    `advanced_axis_dynamics/registries/derivative.py`, and a merge of a sparse axis
    expands per `obsidian/02-categories/Sparse Expansion.md`.
    '''
    reindexing: cat.StrideMorphism

    @classmethod
    def template[B: cat.Datatype, A: cat.Axis](
        cls,
        reindexing: cat.StrideMorphism[A],
        base: B = cat.Reals(),
        name: str | fd.DynamicName | None = None,
        input_axes: fd.Prod[A] | None = None,
        output_axes: fd.Prod[A] | None = None,
        degree: fd.Prod[A] = (),
    ) -> cat.Broadcasted[B, A]:
        '''The merge along `reindexing` of an input carrying every axis it merges,
        broadcast over `degree`, which follows the merged axes on both sides.

        `input_axes` and `output_axes` default to the reindexing's own domain and to
        its codomain with every partly written axis marked by
        `mark_sparse_codomains.mark_sparse_codomain`.
        A caller merging an axis that carries a selection passes the arriving axes in
        their place, as `deepseek.merge_selected_axis` does, because a
        `deepseek.SparseAxis` and its parent are two axes of one size. An input
        carrying fewer axes than the reindexing merges raises
        `InputAxesDoNotMatchTheDomain`, and
        `broadcast_over_absent_axes_and_merge` merges it.
        '''
        if input_axes is not None and len(input_axes) != len(reindexing._dom):
            raise InputAxesDoNotMatchTheDomain(
                f'{len(input_axes)} input axes for the {len(reindexing._dom)} domain '
                f'axes of {reindexing}, and an input carrying fewer of them is '
                'merged by broadcast_over_absent_axes_and_merge')
        return cls.merge_of_input_target(
            reindexing,
            tuple(reindexing.dom()) if input_axes is None else tuple(input_axes),
            base, name, output_axes, degree)

    @classmethod
    def broadcast_over_absent_axes_and_merge[B: cat.Datatype, A: cat.Axis](
        cls,
        reindexing: cat.StrideMorphism[A],
        input_axes: fd.Prod[A],
        base: B = cat.Reals(),
        name: str | fd.DynamicName | None = None,
        output_axes: fd.Prod[A] | None = None,
        degree: fd.Prod[A] = (),
    ) -> cat.Broadcasted[B, A]:
        '''The merge along `reindexing` of an input carrying `input_axes`, the leading
        axes of its domain, broadcast over the domain axes the input does not carry and
        over `degree`.

        The input is written to one output position per index of the absent axes, which
        is how each compressed entry's slots reach every query of its group in the
        lightning indexer. The absent axes enter
        `mark_sparse_codomains.merge_groups` and `mark_sparse_codomain` with the axes
        the input carries, because what the marking reads is the image of the whole
        domain box.
        '''
        check_input_axes_lead_the_domain(reindexing, input_axes)
        return cls.merge_of_input_target(
            reindexing, tuple(input_axes), base, name, output_axes, degree)

    @classmethod
    def merge_of_input_target[B: cat.Datatype, A: cat.Axis](
        cls,
        reindexing: cat.StrideMorphism[A],
        input_target: fd.Prod[A],
        base: B,
        name: str | fd.DynamicName | None,
        output_axes: fd.Prod[A] | None,
        degree: fd.Prod[A],
    ) -> cat.Broadcasted[B, A]:
        '''The merge along `reindexing` of an input whose weave target is
        `input_target`, broadcast over `degree`.

        A degree axis that is an `AffineGuards.AffineSparseAxis` guided by an axis the
        merge consumes leaves guided by the codomain, through a unit-stride reindexing
        pairing the output's axis with the input's, so the slots `r|b` of the lightning
        indexer leave the merge of `(b, a)` into `x` as `r|x`.
        '''
        carrying = mark_sparse_codomains.merge_carrying(reindexing, degree)
        marked_rows = mark_sparse_codomains.mark_sparse_codomain(
            carrying)._cod_stride_shift
        merged_width = len(reindexing._cod_stride_shift)
        marked = reindexing.reconstruct(_cod_stride_shift=tuple(
            (axis, strides[:len(reindexing._dom)], shift)
            for axis, strides, shift in marked_rows[:merged_width]))
        output_degree = tuple(axis for axis, _, _ in marked_rows[merged_width:])
        cod = tuple(marked.cod()) if output_axes is None else tuple(output_axes)
        if name is None:
            name = reindexing.name
        tiled = (cat.WeaveMode.TILED,) * len(degree)
        return cat.Broadcasted(
            operator=cls(reindexing=marked,
                         name=(fd.DynamicName.from_str(name)
                               if name is not None else None)),
            input_weaves=(cat.Weave(base, (*input_target, *tiled)),),
            output_weaves=(cat.Weave(base, (*cod, *tiled)),),
            reindexings=(degree_reindexing(output_degree, degree),),
        )


def check_input_axes_lead_the_domain[A: cat.Axis](
    reindexing: cat.StrideMorphism[A], input_axes: fd.Prod[A]) -> None:
    '''Raise unless `input_axes` are the first axes of the reindexing's domain, in
    its order, so that the axes the input does not carry are the last of them.'''
    dom = tuple(reindexing.dom())
    if len(input_axes) > len(dom):
        raise InputAxesDoNotMatchTheDomain(
            f'{len(input_axes)} input axes for the {len(dom)} domain axes of '
            f'{reindexing}')
    disagreeing = [position for position, axis in enumerate(input_axes)
                   if axis != dom[position]]
    if disagreeing:
        raise InputAxesDoNotMatchTheDomain(
            f'the input axes disagree with the domain of {reindexing} at the '
            f'positions {disagreeing}, and a merge broadcasts over the last domain '
            'axes alone')


def degree_reindexing[A: cat.Axis](output_degree: fd.Prod[A],
                                   input_degree: fd.Prod[A]) -> cat.StrideCategory[A]:
    '''The identity on `input_degree` where the two agree, and otherwise the product of
    a unit-stride map per position whose axis changed beside an identity on every other
    position, which is how an axis whose guide a merge consumes changes its guide.

    Each position is a map of its own, so
    `disentangle_reindexings.disentangle_reindexing` splits the rows apart and the
    re-guiding is drawn on the wire it re-guides, with a straight wire through every
    axis the merge leaves alone.
    '''
    if all(out is inp for out, inp in zip(output_degree, input_degree)):
        return cat.ProdObject(input_degree).identity()
    return disentangle_reindexings.disentangle_reindexing(cat.StrideMorphism(
        _dom=output_degree,
        _cod_stride_shift=tuple(
            (inp, tuple(nm.Integer(1 if j == i else 0)
                        for j in range(len(input_degree))),
             nm.Integer(0))
            for i, inp in enumerate(input_degree))))


class PartsDisagreeOutsideTheConcatenatedAxis(Exception):
    '''Two shapes handed to `ConcatenateAxes.template` that differ at more than one
    position, so no single axis is the one being concatenated.'''


class PartsDoNotFillTheAxis(Exception):
    '''Part reindexings whose images are not disjoint runs that together fill the
    concatenated axis.'''


class ConcatenatedAxisHasOtherParts(Exception):
    '''A `ConcatenatedAxis` handed to `ConcatenateAxes.template` whose parts are not
    the axes the shapes hold at the concatenated position.'''


@dataclass(frozen=True)
class ConcatenateAxes(cat.Operator):
    '''One reindexing per input, read covariantly, their images filling one axis.

    Part `k` holds one axis and one row, which writes position `j` of that axis to
    position `j + offset_k` of the concatenated axis, where `offset_k` is the sum of
    the sizes of the parts before it. The images are therefore disjoint runs in order,
    and they fill the concatenated axis because its size is the sum of theirs. The
    window scores `[h, x, w]` and the selected-entry scores `[h, x, s]` of
    DeepSeek-V4.1-Flash concatenate into `[h, x, t]` with `|t| = |w| + |s|`.

    The operator is the multi-input `CovariantView`, and the reindexings are fields of
    the operator for the same reason: a `reindexings` entry of a `cat.Broadcasted`
    points from the output back to the input, and these run the other way.

    The concatenated axis is an `AxisConcatenation.ConcatenatedAxis` by default. It
    references its parts, its size is the sum of theirs, and it is labelled by their
    labels, so the window slots `w|x` beside the selected slots `s|x` concatenate into
    `w|x + s|x`. It stays dense, whatever the parts carry. Each part may be an
    `AffineGuards.AffineSparseAxis` with a form of its own, as those two slots both
    are, and the live positions of the concatenation are the union of two runs under
    two forms, which no single affine form states.
    `concatenation_expansion.expand_concatenations` restores each part and its own
    form by rewriting every consumer that streams over the concatenated axis, so nothing
    downstream needs the union.

    The concatenated axis may instead be an axis the model declared, whose size is the
    sum of the sizes of the parts. A `DeconcatenateAxes` cuts a declared axis `c` into
    parts, and the concatenation of those parts onto `c` returns the array to the axis
    every other wire of the model carries. `concatenation_expansion` reads the parts
    off the weaves of the operator, so it treats both kinds of axis alike.

    Its reverse derivative is the product of the `ops.View`s of the part reindexings,
    per `advanced_axis_dynamics/registries/derivative.py`, which is the cotangent on
    the concatenated axis cut back into the parts.
    '''
    part_reindexings: fd.Prod[cat.StrideMorphism]

    @classmethod
    def template[B: cat.Datatype, A: cat.Axis](
        cls,
        shapes: fd.Prod[fd.Prod[A]],
        base: B = cat.Reals(),
        name: str | fd.DynamicName | None = '\\Vert',
        concatenated: A | None = None,
    ) -> cat.Broadcasted[B, A]:
        '''The concatenation of arrays of `shapes`, which agree at every position but
        one and hold the parts at that one.

        The shapes are the arrays the caller has in hand, so the concatenated axis
        stands where the parts stood and every other axis is the degree.
        `concatenated` defaults to a fresh `AxisConcatenation.ConcatenatedAxis` over the
        parts, and a caller that holds the axis already, as the second concatenation of
        a rewrite does, passes it. A `ConcatenatedAxis` passed over other parts raises
        `ConcatenatedAxisHasOtherParts`. Any other axis passed for it is a declared
        axis the parts are laid end to end onto, as the rotary embedding of
        DeepSeek-V4.1-Flash lays the channels it left alone and the channels it
        rotated onto the latent axis `c` they were cut from, and
        `check_parts_fill_the_axis` raises unless its size is the sum of theirs.
        '''
        position = differing_position(shapes)
        parts = tuple(shape[position] for shape in shapes)
        degree = (*shapes[0][:position], *shapes[0][position + 1:])
        axis = whole_axis_of(parts, concatenated)
        part_reindexings = part_reindexings_onto(axis, parts)
        check_parts_fill_the_axis(part_reindexings)
        leading = (cat.WeaveMode.TILED,) * position
        trailing = (cat.WeaveMode.TILED,) * (len(degree) - position)
        return cat.Broadcasted(
            operator=cls(part_reindexings=part_reindexings,
                         name=(fd.DynamicName.from_str(name)
                               if name is not None else None)),
            input_weaves=tuple(
                cat.Weave(base, (*leading, part, *trailing)) for part in parts),
            output_weaves=(cat.Weave(base, (*leading, axis, *trailing)),),
            reindexings=(cat.ProdObject(degree).identity(),) * len(parts))

    def parts(self) -> fd.Prod[cat.Axis]:
        '''The axis each part reindexing writes into the concatenated axis.'''
        return tuple(reindexing._dom[0] for reindexing in self.part_reindexings)

    def concatenated_axis(self) -> cat.Axis:
        '''The axis the parts fill.'''
        axis, _, _ = self.part_reindexings[0]._cod_stride_shift[0]
        return axis


@dataclass(frozen=True)
class DeconcatenateAxes(cat.Operator):
    '''One axis cut into the parts that fill it, one reindexing per output.

    Part `k` holds one axis and one row, and the row reads position `j` of that part
    from position `j + offset_k` of the axis being cut, where `offset_k` is the sum of
    the sizes of the parts before it. The reindexings are the ones a `ConcatenateAxes`
    of the same parts holds. A concatenation reads them covariantly, from each part
    onto the whole, and a deconcatenation reads them contravariantly, which is the
    direction an `ops.View` reads. The 512 channels of a DeepSeek-V4.1-Flash latent
    deconcatenate into the 448 channels the rotary embedding leaves alone and the 64 it
    rotates.

    The axis being cut need not be a `ConcatenatedAxis`. A model holds a latent on a
    declared axis `c` and cuts it into parts whose sizes sum to `|c|`, and
    `check_parts_fill_the_axis` raises where they do not.

    Its reverse derivative is the `ConcatenateAxes` of the same reindexings, and the
    reverse derivative of a `ConcatenateAxes` is this operator, per
    `advanced_axis_dynamics/registries/derivative.py`. Its standard expansion is a copy
    of the input followed by one `ops.View` per part, per
    `advanced_axis_dynamics/registries/standard_expansions.py`.
    '''
    part_reindexings: fd.Prod[cat.StrideMorphism]

    @classmethod
    def template[B: cat.Datatype, A: cat.Axis](
        cls,
        shapes: fd.Prod[fd.Prod[A]],
        base: B = cat.Reals(),
        name: str | fd.DynamicName | None = '\\Vert^{-1}',
        concatenated: A | None = None,
    ) -> cat.Broadcasted[B, A]:
        '''The deconcatenation of one array into arrays of `shapes`, which agree at
        every position but one and hold the parts at that one.

        `concatenated` is the axis being cut. It defaults to a fresh
        `AxisConcatenation.ConcatenatedAxis` over the parts, a `ConcatenatedAxis`
        passed for it has to be over the same parts, and any other axis passed for it
        has to have the size the parts sum to.
        '''
        position = differing_position(shapes)
        parts = tuple(shape[position] for shape in shapes)
        degree = (*shapes[0][:position], *shapes[0][position + 1:])
        axis = whole_axis_of(parts, concatenated)
        part_reindexings = part_reindexings_onto(axis, parts)
        check_parts_fill_the_axis(part_reindexings)
        leading = (cat.WeaveMode.TILED,) * position
        trailing = (cat.WeaveMode.TILED,) * (len(degree) - position)
        return cat.Broadcasted(
            operator=cls(part_reindexings=part_reindexings,
                         name=(fd.DynamicName.from_str(name)
                               if name is not None else None)),
            input_weaves=(cat.Weave(base, (*leading, axis, *trailing)),),
            output_weaves=tuple(
                cat.Weave(base, (*leading, part, *trailing)) for part in parts),
            reindexings=(cat.ProdObject(degree).identity(),))

    def parts(self) -> fd.Prod[cat.Axis]:
        '''The axis each part reindexing reads out of the axis being cut.'''
        return tuple(reindexing._dom[0] for reindexing in self.part_reindexings)

    def concatenated_axis(self) -> cat.Axis:
        '''The axis the parts are cut from.'''
        axis, _, _ = self.part_reindexings[0]._cod_stride_shift[0]
        return axis


def part_reindexings_onto[A: cat.Axis](
    axis: A, parts: fd.Prod[A]) -> fd.Prod[cat.StrideMorphism[A]]:
    '''One unit-stride row per part, from the part onto `axis`, shifted by the sum of
    the sizes of the parts before it.'''
    return tuple(
        cat.StrideMorphism(
            _dom=(part,),
            _cod_stride_shift=((axis, (nm.Integer(1),), offset),))
        for part, offset in zip(parts, running_offsets(parts)))


def differing_position[A: cat.Axis](shapes: fd.Prod[fd.Prod[A]]) -> int:
    '''The one position at which `shapes` do not all carry the same axis.'''
    widths = {len(shape) for shape in shapes}
    if len(widths) != 1:
        raise PartsDisagreeOutsideTheConcatenatedAxis(
            f'the shapes of a concatenation have the widths {sorted(widths)}')
    differing = [position for position in range(widths.pop())
                 if len({shape[position] for shape in shapes}) != 1]
    if len(differing) != 1:
        raise PartsDisagreeOutsideTheConcatenatedAxis(
            f'{len(differing)} positions of {shapes} differ, and a concatenation '
            'joins the parts at one')
    return differing[0]


def running_offsets[A: cat.Axis](parts: fd.Prod[A]) -> fd.Prod[nm.Numeric]:
    '''The position of the concatenated axis at which each part starts, which is the
    sum of the sizes of the parts before it.'''
    offsets = [nm.Integer(0)]
    for part in parts[:-1]:
        offsets.append(nm.Addition.template(offsets[-1], part.local_size()))
    return tuple(offsets)


def whole_axis_of[A: cat.Axis](
    parts: fd.Prod[A],
    concatenated: A | None,
) -> A | AxisConcatenation.ConcatenatedAxis:
    '''The axis `parts` fill. A declared axis passed as `concatenated` is that axis
    as it stands, and `check_parts_fill_the_axis` is what compares its size with the
    sum of the parts. A `ConcatenatedAxis`, or no axis, goes through
    `concatenated_axis_over`.'''
    if concatenated is not None and not isinstance(
            concatenated, AxisConcatenation.ConcatenatedAxis):
        return concatenated
    return concatenated_axis_over(parts, concatenated)


def concatenated_axis_over[A: cat.Axis](
    parts: fd.Prod[A],
    concatenated: A | None,
) -> AxisConcatenation.ConcatenatedAxis:
    '''`concatenated` where it is given and is a `ConcatenatedAxis` over `parts`, and a
    fresh `ConcatenatedAxis` over `parts` where it is not given.

    The parts are compared by uid, because composition rewrites the axis and its parts
    together and a caller may hold the axis from before the rewrite.
    '''
    if concatenated is None:
        return AxisConcatenation.ConcatenatedAxis(parts=tuple(parts))
    if not isinstance(concatenated, AxisConcatenation.ConcatenatedAxis):
        raise ConcatenatedAxisHasOtherParts(
            f'{type(concatenated).__name__} handed over where a ConcatenatedAxis over '
            f'the {len(parts)} parts is expected, and `whole_axis_of` is the function '
            'that accepts a declared axis')
    if [part.uid for part in concatenated.parts] != [part.uid for part in parts]:
        raise ConcatenatedAxisHasOtherParts(
            f'a concatenated axis over {len(concatenated.parts)} parts handed to a'
            f' concatenation of {len(parts)} parts, and the parts do not agree by uid')
    return concatenated


def check_parts_fill_the_axis(
    part_reindexings: fd.Prod[cat.StrideMorphism]) -> None:
    '''Raise unless every part writes one axis at unit stride onto one shared axis,
    starting where the part before it ended, and the last of them ends at the size of
    the shared axis.'''
    if not part_reindexings:
        raise PartsDoNotFillTheAxis('a concatenation of no parts')
    axes = set()
    reached: nm.Numeric = nm.Integer(0)
    for reindexing in part_reindexings:
        if len(reindexing._dom) != 1 or len(reindexing._cod_stride_shift) != 1:
            raise PartsDoNotFillTheAxis(
                f'a part of a concatenation reads {len(reindexing._dom)} axes onto '
                f'{len(reindexing._cod_stride_shift)} rows, and one part is one axis '
                'on one row')
        axis, strides, shift = reindexing._cod_stride_shift[0]
        axes.add(axis)
        if strides[0] != nm.Integer(1):
            raise PartsDoNotFillTheAxis(
                f'a part of a concatenation is written at the stride '
                f'{strides[0].to_latex()}, and a part is laid out in order')
        gap = nm.collect_like_terms(nm.Addition.template(
            shift, nm.Multiplication.template(nm.Integer(-1), reached)))
        if not nm.is_zero(gap):
            raise PartsDoNotFillTheAxis(
                f'a part of a concatenation starts {gap.to_latex()} past where the '
                'part before it ended')
        reached = nm.Addition.template(reached, reindexing._dom[0].local_size())
    if len(axes) != 1:
        raise PartsDoNotFillTheAxis(
            f'the parts of a concatenation are written onto {len(axes)} axes')
    axis = axes.pop()
    unfilled = nm.collect_like_terms(nm.Addition.template(
        axis.local_size(), nm.Multiplication.template(nm.Integer(-1), reached)))
    if not nm.is_zero(unfilled):
        raise PartsDoNotFillTheAxis(
            f'the parts of a concatenation fill {reached.to_latex()} of the '
            f'{axis.local_size().to_latex()} positions of the axis they are written '
            'onto')
