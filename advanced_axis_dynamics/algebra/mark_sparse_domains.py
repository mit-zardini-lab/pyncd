'''Marking the domain of a reindexing from the rows that read outside their axis.

Written by Claude Opus 5 (1M context), reasoning effort medium.

`rows_reading_outside` finds the rows of a stride morphism whose form leaves
`[0, size)` of its codomain axis somewhere in the domain box. Each such row marks the
last domain axis it reads as an `AffineGuards.AffineSparseAxis` stating its form, and
`mark_sparse_domain` performs the replacement over a composite. `guarded_view` is the
`ops.View` of a reindexing with its domain marked, which is how a model writes a window,
a pad or a relative read.

Two forms travel further. `pulled_back_sparse_axis` substitutes a row into the form of a
codomain axis that already carries one, so a read of a guarded axis guards its own
domain. `sparse_axis_after_fold` substitutes the folded axis's most favourable position,
so a fold over a guarded axis leaves the form on its last guide. `live_positions`
evaluates a form at concrete positions and sizes.

`obsidian/02-categories/Advanced Axis Dynamics.md` states the feature, and
`obsidian/02-categories/Padding and Masks as Sparse Axes.md` the rules the unit's laws
give.
'''
from __future__ import annotations
from typing import Mapping

import data_structure.Term as fd # for 'foundations'
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Category as cat
import data_structure.ProductCategory as pc
import data_structure.StrideCategory as sc
import construction_helpers.product as chp
import advanced_axis_dynamics.data_structure.AffineGuards as AffineGuards


class AxisMarkedTwice(Exception):
    '''Two rows of one stride morphism that would each make the same domain axis
    sparse.'''


def rows_reading_outside[A: sc.Axis](
    morphism: sc.StrideMorphism[A],
) -> fd.Prod[tuple[fd.Prod[nm.Numeric], nm.Numeric, nm.Numeric | None]]:
    '''The strides, the shift and the extent of every codomain row of `morphism` whose
    form leaves `[0, size)` of its axis at some position of the domain box.

    A row whose codomain axis also stands in the domain, which is the statement that
    the map preserves the axis's length, is examined for reading past the end of that
    axis, and the extent returned is the axis's size. A row onto a codomain axis whose
    size is a symbol nothing in the row relates to, per
    `AffineGuards.extent_is_independent`, is taken to size that axis, as a group view
    or a block split sizes its codomain, so it never reads past the end and the extent
    returned is `None`. A row onto a codomain axis
    whose size is written in the row's own symbols, as the query axis sized
    `|a| |b|` is for the view reading queries by group and offset, is examined for
    reading past the end, and the extent returned is the axis's size where it does.
    '''
    dom = tuple(morphism._dom)
    outside = []
    for axis, strides, shift in morphism._cod_stride_shift:
        size = axis.local_size()
        if any(axis == domain_axis for domain_axis in dom):
            past_end = AffineGuards.reads_past_end(strides, shift, dom, size)
            extent = size
        elif AffineGuards.extent_is_independent(size, strides, shift, dom):
            past_end = False
            extent = None
        else:
            past_end = AffineGuards.reads_past_end(strides, shift, dom, size)
            extent = size if past_end else None
        if AffineGuards.reads_before_start(strides, shift, dom) or past_end:
            outside.append((strides, shift, extent))
    return tuple(outside)


def mark_sparse_domain[A: sc.Axis](
    morphism: sc.StrideCategory[A]) -> sc.StrideCategory[A]:
    '''`morphism` with every domain axis that a row reading outside its axis marks
    replaced by the `AffineGuards.AffineSparseAxis` that row states. The domain axes
    are replaced and the codomain rows are kept, so a map naming one axis on both sides
    reads the dense axis and produces the sparse one. A morphism no row of which reads
    outside is returned as the same object.

    A composite is marked from its last factor back to its first. Each factor's marks
    become the sparse codomain axes of the factor before it, which a `sc.StrideMorphism`
    pulls back through its rows and a `Rearrangement` carries to the domain position
    it copies from, so a reindexing written as a copy, a product with the identity and
    a deletion marks the composite's domain as one stride morphism would.
    '''
    marked, _ = mark_sparse_domain_under(morphism, {})
    return marked


def mark_sparse_domain_under[A: sc.Axis](
    morphism: sc.StrideCategory[A],
    codomain_marks: Mapping[int, AffineGuards.AffineSparseAxis],
) -> tuple[sc.StrideCategory[A], dict[int, AffineGuards.AffineSparseAxis]]:
    '''`morphism` with the codomain positions in `codomain_marks` read as the sparse
    axes given and its own domain marked in turn, beside the marks on its domain
    positions.'''
    match morphism:
        case sc.StrideMorphism():
            return mark_stride_morphism_under(morphism, codomain_marks)
        case pc.Rearrangement(mapping=mapping, _dom=dom):
            domain_marks: dict[int, AffineGuards.AffineSparseAxis] = {}
            for codomain_position, domain_position in enumerate(mapping):
                if codomain_position not in codomain_marks:
                    continue
                mark = codomain_marks[codomain_position]
                if domain_marks.get(domain_position, mark) != mark:
                    raise AxisMarkedTwice(
                        f'{dom[domain_position]} is copied to two positions marked '
                        f'{domain_marks[domain_position]} and {mark}')
                domain_marks[domain_position] = mark
            if not domain_marks:
                return morphism, {}
            return morphism.reconstruct(
                _dom=tuple(domain_marks.get(i, axis) for i, axis in enumerate(dom))
            ), domain_marks
        case pc.ProductOfMorphisms(content=content):
            marked_content = []
            domain_marks = {}
            codomain_offset = 0
            domain_offset = 0
            for factor in content:
                codomain_width = len(factor.cod())
                factor_marks = {
                    position - codomain_offset: mark
                    for position, mark in codomain_marks.items()
                    if codomain_offset <= position < codomain_offset + codomain_width}
                marked_factor, factor_domain_marks = mark_sparse_domain_under(
                    factor, factor_marks)
                marked_content.append(marked_factor)
                domain_marks.update({domain_offset + position: mark
                                     for position, mark in factor_domain_marks.items()})
                codomain_offset += codomain_width
                domain_offset += len(factor.dom())
            if all(before is after for before, after in zip(content, marked_content)):
                return morphism, domain_marks
            return pc.ProductOfMorphisms.from_iter(marked_content), domain_marks
        case pc.Composed(content=content):
            marks: dict[int, AffineGuards.AffineSparseAxis] = dict(codomain_marks)
            marked_content = []
            for factor in reversed(content):
                marked_factor, marks = mark_sparse_domain_under(factor, marks)
                marked_content.append(marked_factor)
            marked_content.reverse()
            if all(before is after for before, after in zip(content, marked_content)):
                return morphism, marks
            return pc.Composed.from_iter(marked_content), marks
    return morphism, {}


def mark_stride_morphism_under[A: sc.Axis](
    morphism: sc.StrideMorphism[A],
    codomain_marks: Mapping[int, AffineGuards.AffineSparseAxis],
) -> tuple[sc.StrideCategory[A], dict[int, AffineGuards.AffineSparseAxis]]:
    rows = tuple(
        (codomain_marks.get(position, axis), strides, shift)
        for position, (axis, strides, shift) in enumerate(morphism._cod_stride_shift))
    reading = (morphism.reconstruct(_cod_stride_shift=rows)
               if codomain_marks else morphism)
    dom = tuple(reading._dom)
    marks = [AffineGuards.sparse_axis_for_row(strides, shift, dom, extent)
             for strides, shift, extent in rows_reading_outside(reading)]
    marks.extend(
        pulled_back_sparse_axis(axis, reading)
        for axis, _, _ in reading._cod_stride_shift
        if isinstance(axis, AffineGuards.AffineSparseAxis))
    marks = [(position, sparse) for position, sparse in marks
             if sparse.crosses_start() or sparse.crosses_end()]
    if not marks:
        return reading, {}
    marked: dict[int, AffineGuards.AffineSparseAxis] = {}
    for position, sparse in marks:
        if position in marked:
            raise AxisMarkedTwice(
                f'{dom[position]} is marked by two rows of {morphism}, and one axis '
                'carries one affine form')
        marked[position] = sparse
    return reading.reconstruct(
        _dom=tuple(marked.get(i, axis) for i, axis in enumerate(dom))), marked


def pull_back_sparse_axis[A: sc.Axis](
    axis: AffineGuards.AffineSparseAxis, split: sc.StrideMorphism[A],
) -> fd.Prod[sc.Axis]:
    '''The domain of `split` with the form of `axis` pulled back through it.

    `split` has one codomain row, which reads `axis` at `sum(s * i) + t` for its
    domain positions `i`. Substituting that position into the form of `axis` gives a
    form over the guides and the domain of `split`, and the last domain axis of
    `split` the row reads becomes the `AffineGuards.AffineSparseAxis` stating it,
    guided by the guides of `axis` and by the other domain axes. A block split
    `i_B = |u| i_P + i_u` of entries live where `i_x - i_B >= 0` gives offsets live
    where `i_x - |u| i_P - i_u >= 0`.
    '''
    if len(split._cod_stride_shift) != 1:
        raise AffineGuards.GuardReadsNoAxis(
            f'{len(split._cod_stride_shift)} codomain rows in {split}, where one reads '
            f'{axis}')
    _, split_strides, split_shift = split._cod_stride_shift[0]
    position, sparse = pulled_back_sparse_axis(
        axis,
        split.reconstruct(_cod_stride_shift=((axis, split_strides, split_shift),)))
    return tuple(sparse if i == position else domain_axis
                 for i, domain_axis in enumerate(split._dom))


def codomain_row[A: sc.Axis](morphism: sc.StrideMorphism[A],
                             axis: sc.Axis) -> int | None:
    '''The index of the codomain row of `morphism` that produces `axis`, and `None`
    where no row does.'''
    for index, (produced, _, _) in enumerate(morphism._cod_stride_shift):
        if produced == axis:
            return index
    return None


def pulled_back_sparse_axis[A: sc.Axis](
    axis: AffineGuards.AffineSparseAxis, morphism: sc.StrideMorphism[A]
) -> tuple[int, AffineGuards.AffineSparseAxis]:
    '''The domain position of `morphism` that reading `axis` through it marks, and
    the `AffineGuards.AffineSparseAxis` marking it.

    `axis` is a codomain axis of `morphism`, read at the affine form its row states.
    Substituting that form for the position of `axis`, and the row of every guide
    that `morphism` also produces for that guide's position, writes the form of
    `axis` over the domain of `morphism` and the guides it does not produce. A guide
    that `morphism` does not produce but carries in its domain is the identity on
    that position, which is how a copy and an identity factor pass it round the
    morphism. The last domain axis with a stride in the result is marked, guided by
    the rest. A result
    whose form stays in range at every position, as a relative read undone by the
    read that made it gives, is dropped by `mark_stride_morphism_under`.
    '''
    dom = tuple(morphism._dom)
    own = codomain_row(morphism, axis)
    if own is None:
        raise AffineGuards.GuardReadsNoAxis(f'{morphism} produces no {axis}')
    strides: list[nm.Numeric] = [nm.Integer(0)] * len(dom)
    external_guides: list[sc.Axis] = []
    external_strides: list[nm.Numeric] = []

    def substitute(coefficient: nm.Numeric, row: int) -> nm.Numeric:
        _, row_strides, row_shift = morphism._cod_stride_shift[row]
        for i, stride in enumerate(row_strides):
            strides[i] = nm.Addition.template(
                strides[i], nm.Multiplication.template(coefficient, stride))
        return nm.Multiplication.template(coefficient, row_shift)

    shift = nm.Addition.template(axis.shift, substitute(axis.stride, own))
    for guide, guide_stride in zip(axis.guides, axis.guide_strides):
        row = codomain_row(morphism, guide)
        in_domain = [i for i, domain_axis in enumerate(dom) if domain_axis == guide]
        if row is not None:
            shift = nm.Addition.template(shift, substitute(guide_stride, row))
        elif in_domain:
            strides[in_domain[0]] = nm.Addition.template(
                strides[in_domain[0]], guide_stride)
        else:
            external_guides.append(guide)
            external_strides.append(guide_stride)
    combined_dom = (*external_guides, *dom)
    combined_strides = (*external_strides,
                        *(nm.collect_like_terms(stride) for stride in strides))
    position, sparse = AffineGuards.sparse_axis_for_row(
        combined_strides, nm.collect_like_terms(shift), combined_dom, axis.extent)
    if position < len(external_guides):
        raise AffineGuards.GuardReadsNoAxis(
            f'{morphism} reads {axis} along no axis of its domain')
    return position - len(external_guides), sparse


def sparse_axis_after_fold(
    axis: AffineGuards.AffineSparseAxis) -> AffineGuards.AffineSparseAxis:
    '''The sparse axis the last guide of `axis` becomes once `axis` is folded.

    A fold ignores the unit, so its result at the guides' positions holds a value
    where some position of `axis` does, which is where the form holds at the position
    most favourable to it: the first position where `stride` is negative and the last
    where it is positive. Substituting that position removes `axis` from the form and
    leaves a form over the guides, whose last member the result states. Folding the
    offsets `i_u - |u| i_P + i_x >= 0` of a block split by a maximum gives blocks live
    where `i_x - |u| i_P >= 0`.
    '''
    if not axis.guides:
        raise AffineGuards.GuardReadsNoAxis(
            f'{axis} has no guide to carry its form after a fold')
    favourable = (nm.Integer(0)
                  if nm.is_negative_for_positive_symbols(axis.stride)
                  else AffineGuards.last_position(axis))
    shift = nm.collect_like_terms(nm.Addition.template(
        axis.shift, nm.Multiplication.template(axis.stride, favourable)))
    _, sparse = AffineGuards.sparse_axis_for_row(
        axis.guide_strides, shift, axis.guides, axis.extent)
    return sparse


def live_positions(axis: AffineGuards.AffineSparseAxis, guide_positions: fd.Prod[int],
                   sizes: Mapping[nm.Numeric, int]) -> list[int]:
    '''The positions of `axis` holding a value when its guides stand at
    `guide_positions`, with every size symbol bound by `sizes`.'''
    guides = tuple(nm.FreeNumeric() for _ in axis.guides)
    position = nm.FreeNumeric()
    form = axis.guard_form(guides, position)
    extent = (None if axis.extent is None
              else nm.evaluate_integer(axis.extent, sizes))
    bound = {**sizes, **dict(zip(guides, guide_positions))}
    live = []
    for candidate in range(nm.evaluate_integer(axis.local_size(), sizes)):
        value = nm.evaluate_integer(form, {**bound, position: candidate})
        if value >= 0 and (extent is None or value < extent):
            live.append(candidate)
    return live


def guarded_view[B: cat.Datatype = cat.Reals, A: cat.Axis = cat.RawAxis](
    reindexing: chp.ProductMorphismTarget[A, sc.StrideCategory[A]],
    base: chp.ProductObjectTarget[cat.Array[B, A], B] = cat.Reals(),
    name: str | None | fd.DynamicName = None,
) -> cat.BroadcastedCategory[B, A]:
    '''The `ops.View` of `reindexing`, with every domain axis a row reading outside its
    axis marks replaced by the `AffineGuards.AffineSparseAxis` that row states.

    `ops.Elementwise.template` composed its reindexing with the shape of `base` and
    marked it until 2026-09-15, so every view acquired a guard whether the model wanted
    one or not. It now marks nothing, and a model that reads a window, a pad or a
    relative read calls this instead. The sliding window
    `i_x + j_w + 1 - |w| -> i_x` built here produces `[x, w|x, c]`, and composition
    carries `w|x` onto every array that aligns with it.
    '''
    return ops.View.template(
        base=base, name=name,
        reindexing=mark_sparse_domain(chp.morphism_product(reindexing)))
