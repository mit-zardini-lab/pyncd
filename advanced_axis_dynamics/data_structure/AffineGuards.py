'''An axis whose live positions are stated by an affine form of its position.

Written by Claude Opus 5 (1M context), reasoning effort medium.

A codomain index of a `sc.StrideMorphism` outside `[0, size)` of its axis names no
position, and a read there yields the universal unit. `AffineSparseAxis` records the
consequence on the array read: position `j` of the axis, at positions `i` of its
`guides`, holds a value where `0 <= sum(guide_strides * i) + stride * j + shift <
extent`, and the unit elsewhere. The form is the row of the stride morphism whose read
produced the axis, and `sparse_axis_for_row` builds the axis from that row.

The rest of the module is the reach of an affine row over a domain box.
`reads_before_start` and `reads_past_end` evaluate the row at the corner most
favourable to leaving the axis, treating every size symbol as a positive integer, and
`marked_position` names the domain axis the row's form is carried on. The axis's own
`crosses_start`, `crosses_end` and `empty_end` are written in those tests, which is why
they sit beside it rather than in `algebra/`.

`obsidian/02-categories/Advanced Axis Dynamics.md` states the feature, and
`obsidian/02-categories/Padding and Masks as Sparse Axes.md` the three reads that
produce such an axis.
'''
from __future__ import annotations
from dataclasses import dataclass
from typing import Self
from enum import Enum

import data_structure.Term as fd # for 'foundations'
import data_structure.Numeric as nm
import data_structure.StrideCategory as sc


class EmptyEnd(Enum):
    '''Which end of an `AffineSparseAxis` holds the unit at the guide positions
    where some of its positions do.'''
    FIRST = 'first'
    LAST = 'last'
    BOTH = 'both'


class GuardReadsNoAxis(Exception):
    '''A row of a stride morphism every one of whose strides is zero.'''


class NotAPrefix(Exception):
    '''An `AffineSparseAxis` whose live positions are not a run from its first
    position, asked for the axis a selection over it fills.'''


@dataclass(frozen=True)
class AffineSparseAxis(sc.Axis):
    '''An axis whose positions hold a value where an affine form of the position lies
    in a range, and the universal unit elsewhere.

    Position `j` of the axis, taken at positions `i` of the `guides`, holds a value
    where `0 <= sum(guide_strides * i) + stride * j + shift < extent`. The form is the
    row of the stride morphism whose read produced the axis, with `extent` the size of
    the axis that row indexed. The sliding window of DeepSeek-V4.1-Flash reads token
    `i_x + j_w + 1 - |w|`, so its window axis is live where
    `i_x + j_w + 1 - |w| >= 0`, with the query axis as its one guide, and the first
    slots of an early query hold the unit. A query reaches a compressed entry `i_b` at
    ratio `|a|` where `i_x - |a| i_b - (|a| - 1) >= 0`, so the entry axis is live on a
    run from its first position and the last entries hold the unit.

    Which end holds the unit follows from the sign of `stride` and from which bound of
    the range the form can cross, per `empty_end`. The number of live positions is a
    floor of the form and is not affine, so the axis carries the form and derives the
    count. A `SparseAxis` in `deepseek/data_structure.py` states an activity decided
    by data. This axis states an active set decided by position alone.

    `extent` is `None` where the row that produced the axis indexed a codomain axis
    that stands nowhere in its domain. Such a row sizes that axis, as a group view or a
    block split sizes its codomain, so the form never reaches the end and only its
    fall below zero empties positions.
    '''
    guides: fd.Prod[sc.Axis] = ()
    guide_strides: fd.Prod[nm.Numeric] = ()
    stride: nm.Numeric = nm.Integer(1)
    shift: nm.Numeric = nm.Integer(0)
    extent: nm.Numeric | None = None

    def guard_form(self, guide_positions: fd.Prod[nm.Numeric],
                   position: nm.Numeric) -> nm.Numeric:
        '''The affine form at `position` of this axis and at `guide_positions` of the
        guides, in the order of `guides`.'''
        if len(guide_positions) != len(self.guides):
            raise GuardReadsNoAxis(
                f'{len(guide_positions)} guide positions for {len(self.guides)} guides')
        return nm.collect_like_terms(nm.Addition.template(
            self.shift,
            nm.Multiplication.template(self.stride, position),
            *(nm.Multiplication.template(guide_stride, guide_position)
              for guide_stride, guide_position
              in zip(self.guide_strides, guide_positions))))

    def row(self) -> tuple[fd.Prod[sc.Axis], fd.Prod[nm.Numeric], nm.Numeric]:
        '''The domain, the strides and the shift of the form, with this axis last.'''
        return ((*self.guides, self), (*self.guide_strides, self.stride), self.shift)

    def crosses_start(self) -> bool:
        dom, strides, shift = self.row()
        return reads_before_start(strides, shift, dom)

    def crosses_end(self) -> bool:
        '''Whether the form reaches `extent` at some position. Where `extent` is a size
        symbol the form's own symbols do not relate to, reaching the end turns on a
        relation the sizes cannot settle, so only a provably positive overshoot
        crosses. Where `extent` is written in the form's own symbols, as `|a| |b|` is
        for a query axis sized by the entries and the ratio, an overshoot not provably
        at most zero crosses, as a form not provably at least zero crosses the start.'''
        if self.extent is None:
            return False
        dom, strides, shift = self.row()
        overshoot = nm.Addition.template(
            highest_value(strides, shift, dom),
            nm.Multiplication.template(nm.Integer(-1), self.extent), nm.Integer(1))
        if extent_is_independent(self.extent, strides, shift, dom):
            return nm.is_positive_for_positive_symbols(overshoot)
        return not nm.is_nonpositive_for_positive_symbols(overshoot)

    def empty_end(self) -> EmptyEnd:
        '''The end of the axis holding the unit. A form below its range with a
        positive stride empties the first positions, and with a negative stride the
        last. A form past its range does the opposite.'''
        positive = nm.is_positive_for_positive_symbols(self.stride)
        low_end = EmptyEnd.FIRST if positive else EmptyEnd.LAST
        high_end = EmptyEnd.LAST if positive else EmptyEnd.FIRST
        ends = ({low_end} if self.crosses_start() else set()) | (
            {high_end} if self.crosses_end() else set())
        if len(ends) == 2:
            return EmptyEnd.BOTH
        if not ends:
            raise GuardReadsNoAxis(f'{self} is live at every position')
        return ends.pop()

    def selected_slots(self, size: nm.Numeric,
                       label: str | fd.DynamicName | None = None) -> AffineSparseAxis:
        '''An axis of `size` slots filled from this axis's live positions in order, so
        that the first as many slots as there are live positions hold a value.

        Where this axis is live on a run from its first position, slot `j` is live
        where position `j` is, and the form carries over. Where it is live on a run to
        its last position, slot `j` is live where position `|axis| - 1 - j` is, which
        substitutes that position into the form. A selection over this axis hands its
        slots out on the result, as `deepseek.TopK.template` does.
        '''
        match self.empty_end():
            case EmptyEnd.LAST:
                stride, shift = self.stride, self.shift
            case EmptyEnd.FIRST:
                stride = nm.Multiplication.template(nm.Integer(-1), self.stride)
                shift = nm.collect_like_terms(nm.Addition.template(
                    self.shift,
                    nm.Multiplication.template(self.stride, last_position(self))))
            case EmptyEnd.BOTH:
                raise NotAPrefix(f'{self} holds the unit at both ends')
        return sparse_axis_named(
            label if label is not None else fd.DynamicName('k'),
            self.guides,
            AffineSparseAxis(_size=size, guides=self.guides,
                             guide_strides=self.guide_strides, stride=stride,
                             shift=shift, extent=self.extent))


def axis_body(axis: sc.Axis) -> str:
    name = getattr(axis.uid, '_name', None)
    bodies = name.to_bodies() if name is not None else None
    return bodies if bodies else '?'


def sparse_axis_named[S: AffineSparseAxis](
    label: str | fd.DynamicName, guides: fd.Prod[sc.Axis], axis: S) -> S:
    '''`axis` named after `label` and its guides as `label|guide,guide`.'''
    body = fd.DynamicName.from_str(label).to_bodies()
    guide_bodies = ','.join(axis_body(guide) for guide in guides)
    return fd.DynamicName.from_str(f'{body}|{guide_bodies}').capture(axis)


def last_position(axis: sc.Axis) -> nm.Numeric:
    return nm.Addition.template(axis.local_size(), nm.Integer(-1))


def lowest_value(strides: fd.Prod[nm.Numeric], shift: nm.Numeric,
                 dom: fd.Prod[sc.Axis]) -> nm.Numeric:
    '''The form at the corner of the domain box where every axis with a negative
    stride stands at its last position and every other axis at its first.'''
    return nm.Addition.template(shift, *(
        nm.Multiplication.template(stride, last_position(axis))
        for stride, axis in zip(strides, dom)
        if nm.is_negative_for_positive_symbols(stride)))


def highest_value(strides: fd.Prod[nm.Numeric], shift: nm.Numeric,
                  dom: fd.Prod[sc.Axis]) -> nm.Numeric:
    '''The form at the corner of the domain box where every axis with a positive
    stride stands at its last position and every other axis at its first.'''
    return nm.Addition.template(shift, *(
        nm.Multiplication.template(stride, last_position(axis))
        for stride, axis in zip(strides, dom)
        if nm.is_positive_for_positive_symbols(stride)))


def has_settled_sign(stride: nm.Numeric) -> bool:
    return (nm.is_zero(stride)
            or nm.is_positive_for_positive_symbols(stride)
            or nm.is_negative_for_positive_symbols(stride))


def reads_before_start(strides: fd.Prod[nm.Numeric], shift: nm.Numeric,
                       dom: fd.Prod[sc.Axis]) -> bool:
    '''Whether the form falls below zero at some position of the domain box, for
    some assignment of positive sizes. A stride whose sign the sizes do not settle
    is reported as reading before the start.'''
    if not all(has_settled_sign(stride) for stride in strides):
        return True
    return not nm.is_nonnegative_for_positive_symbols(lowest_value(strides, shift, dom))


def reads_past_end(strides: fd.Prod[nm.Numeric], shift: nm.Numeric,
                   dom: fd.Prod[sc.Axis], extent: nm.Numeric) -> bool:
    '''Whether the form reaches `extent` at some position of the domain box, for
    some assignment of positive sizes.'''
    if not all(has_settled_sign(stride) for stride in strides):
        return True
    overshoot = nm.Addition.template(
        highest_value(strides, shift, dom),
        nm.Multiplication.template(nm.Integer(-1), extent),
        nm.Integer(1))
    return not nm.is_nonpositive_for_positive_symbols(overshoot)


def extent_is_independent(extent: nm.Numeric, strides: fd.Prod[nm.Numeric],
                          shift: nm.Numeric, dom: fd.Prod[sc.Axis]) -> bool:
    '''Whether `extent` is written in no symbol the row or the sizes of its domain axes
    mention, so that no relation between the row's reach and the extent is stated
    anywhere. The block split `(P, u) -> r|x` with `|r| = |a| |b|` has an independent
    extent, and the query view `(b_0, a) -> x` with `|x| = |a| |b|` does not.'''
    mentioned = set().union(
        *(nm.free_symbols(stride) for stride in strides),
        nm.free_symbols(shift),
        *(nm.free_symbols(axis.local_size()) for axis in dom))
    return nm.free_symbols(extent).isdisjoint(mentioned)


def marked_position(strides: fd.Prod[nm.Numeric]) -> int:
    '''The index of the last domain axis a row reads, meaning the last with a stride
    that is not zero.'''
    for position in reversed(range(len(strides))):
        if not nm.is_zero(strides[position]):
            return position
    raise GuardReadsNoAxis(f'every stride of {strides} is zero')


def sparse_axis_for_row(strides: fd.Prod[nm.Numeric], shift: nm.Numeric,
                        dom: fd.Prod[sc.Axis], extent: nm.Numeric | None
                        ) -> tuple[int, AffineSparseAxis]:
    '''The position in `dom` of the last axis the row reads, and the
    `AffineSparseAxis` that replaces it, guided by the other axes the row reads.
    `extent` is `None` where the row sizes its codomain axis.'''
    position = marked_position(strides)
    guided = tuple(i for i, stride in enumerate(strides)
                   if i != position and not nm.is_zero(stride))
    marked = dom[position]
    axis = AffineSparseAxis(
        _size=marked.local_size(),
        guides=tuple(dom[i] for i in guided),
        guide_strides=tuple(strides[i] for i in guided),
        stride=strides[position],
        shift=shift,
        extent=extent)
    return position, sparse_axis_named(axis_body(marked), axis.guides, axis)
