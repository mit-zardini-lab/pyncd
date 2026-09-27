'''Dragging an index backwards through an expression.

Written by Claude Fable 5.1, reasoning effort 80.

An expression lifted over an axis computes the same function at every index of that
axis, so reading its result at one index is the same as reading its input at that
index and computing once: `[F; x](input)[t_x] = F(input[t_x])`, with `[F; x]` the
morphism `F` broadcast over `x` and `t_x` an index of `x`. The rule holds because a
`Broadcasted` reads each input at the degree position its reindexing maps the output
position to, so a fixed output index fixes the input index the row computes from it.
The index therefore travels from the codomain of an expression back towards its
domain, and the crawl here carries it.

An axis of an array is *pinned* at an index when the result being computed reads that
axis at that one index. The pins of a wire are one entry per axis of its array: the
index the axis is pinned at, or `None` where every index is read. `IndexPinCrawler` is
a `ReverseCrawler` over the broadcasted category whose guide is the pins of each wire.
At a `Broadcasted` it drops the pin of every target position, because the operator
reads the whole target, and carries the pins of the degree through each reindexing
with `ReindexingPinCrawler`, a `ForwardCrawler` over the stride category, because a
reindexing maps the degree of the output to the degree of the input. A row of a stride
morphism pins its codomain axis where every domain axis it reads is pinned, at the
value the row takes there, and leaves it free otherwise. A `ops.BlockOperator`
computes its body on the targets of its operands, so the pins of its target positions
are carried through the body and out to the operands, and only a pin on a target the
body itself consumes is dropped. A wire read by several consumers carries the pins
they agree on and is free wherever they differ, per `agreed_pins`, which is the rule
the requester stated on 2026-09-25: an index that meets an expression at several
points continues only where every one of them holds the same value.

The crawl rebuilds the expression. Every pinned position carries the axis
`AffineGuards.axis_pinned_at` builds, live at that index alone, so the result draws
with `x[t_x]` on every wire the index reached. Wherever a consumer asks for a pin the
wire before it does not carry, which happens after a copy whose branches disagree and
after an operator whose target the index does not cross, a `ops.View` reading the
pinned axis from the dense one is written between the two, so the rebuilt expression
composes and states where the index stopped. `DraggedIndex.stops` lists those views in
the order the crawl wrote them.

`pin_guards` runs after the crawl. A guarded axis such as `w|x` names the axes its form
reads, and where every array carrying it also carries the pinned form of one of those
guides, the index passed through that guide, so the guard is pinned: the index is
substituted into the form, the guide is dropped from it, and the axis is named
`w|x[t_x]`. The requester asked for the substitution as a separate step on 2026-09-25,
because a covariant view relates its axes by a map the pin cannot state, and a step
that reads the rebuilt expression can be told what it needs.

The crawl runs on the morphism form. A hypergraph merges the readers of a wire by
wire identity and the view a stop needs would have to be spliced in, which is not
written.

`obsidian/02-categories/Advanced Axis Dynamics.md` states the feature.
'''
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import data_structure.Term as fd  # for 'foundations'
import data_structure.Numeric as nm
import data_structure.Category as cat
import data_structure.Operators as ops
import data_structure.StrideCategory as sc
import utilities.utilities as util
import construction_helpers.simple_helper as chsh
import graphs.processing.hypergraph_crawler as hypergraph_crawler
import term_utilities.term_utilities as tutil
import advanced_axis_dynamics.data_structure.AffineGuards as AffineGuards

type Pins = fd.Prod[nm.Numeric | None]
'''One entry per axis of an array: the index the axis is pinned at, or `None` where
every index is read.'''


class PinNotCarried(Exception):
    '''A wire carrying a pin at a position where its reader asks for a different one,
    which the crawl never produces, because a wire carries the pins its readers agree
    on.'''


def free_pins(array: cat.Array) -> Pins:
    return (None,) * len(array.shape())


def agreed_pins(pins: Iterable[Pins]) -> Pins:
    '''The pins several readers of one array agree on, position by position, with a
    position free wherever two of them differ.'''
    return tuple(util.iallequals(column, fallback=None)
                 for column in zip(*tuple(pins)))


def pins_carried(array: cat.Array) -> Pins:
    '''The pins an array of a rebuilt expression carries, read off its pinned axes.'''
    return tuple(
        AffineGuards.pinned_index(axis) if AffineGuards.is_pinned_axis(axis) else None
        for axis in array.shape())


def row_pin(strides: fd.Prod[nm.Numeric], shift: nm.Numeric,
            pins: Pins) -> nm.Numeric | None:
    '''The index the row `sum(strides * i) + shift` computes when every domain axis it
    reads is pinned, and `None` where one of them is free. A row reading no axis pins
    its codomain axis at its shift.'''
    read = tuple((stride, pin) for stride, pin in zip(strides, pins)
                 if not nm.is_zero(stride))
    if any(pin is None for _, pin in read):
        return None
    return nm.collect_like_terms(nm.Addition.template(
        shift, *(nm.Multiplication.template(stride, pin) for stride, pin in read)))


def target_count(weave: cat.Weave) -> int:
    return len(tuple(weave.select_target(weave._shape)))


def pins_over_weave(weave: cat.Weave, degree_pins: Pins, target_pins: Pins) -> Pins:
    '''The pins of the array `weave` stands for: the degree pins at its tiled positions
    and the target pins at its target positions.'''
    degree = iter(degree_pins)
    target = iter(target_pins)
    return tuple(next(degree) if isinstance(mode, cat.WeaveMode) else next(target)
                 for mode in weave._shape)


def pins_over_degree(weave: cat.Weave, degree_pins: Pins) -> Pins:
    '''The pins of the array `weave` stands for: the degree pins at its tiled
    positions, and its target positions free.'''
    return pins_over_weave(weave, degree_pins, (None,) * target_count(weave))


@dataclass
class PinnedAxes:
    '''One pinned axis per axis and index, so that every wire pinned at one index of
    one axis carries one term and the rebuilt expression composes. The memo is keyed
    by the identity of the axis because it makes one replacement one term. Which
    positions are replaced is decided by position, in the crawl.'''
    _by_axis_and_index: dict[tuple[sc.Axis, nm.Numeric], AffineGuards.AffineSparseAxis] = (
        field(default_factory=dict))

    def pinned(self, axis: sc.Axis, index: nm.Numeric | None) -> sc.Axis:
        if index is None:
            return axis
        key = (axis, index)
        if key not in self._by_axis_and_index:
            self._by_axis_and_index[key] = AffineGuards.axis_pinned_at(axis, index)
        return self._by_axis_and_index[key]

    def pinned_array[B: cat.Datatype, A: cat.Axis](
            self, array: cat.Array[B, A], pins: Pins) -> cat.Array[B, A]:
        return array.reconstruct(_shape=tuple(
            self.pinned(axis, pin) for axis, pin in zip(array.shape(), pins)))

    def items(self) -> fd.Prod[tuple[sc.Axis, nm.Numeric, AffineGuards.AffineSparseAxis]]:
        '''Every axis the crawl pinned, with the index and the pinned axis.'''
        return tuple((axis, index, pinned)
                     for (axis, index), pinned in self._by_axis_and_index.items())


@dataclass
class ReindexingPinCrawler(
        hypergraph_crawler.ForwardCrawler[sc.Axis, sc.StrideMorphism, nm.Numeric | None]):
    '''Carries the pins of a degree through a reindexing to the degree of the input
    the reindexing reads, and rebuilds the reindexing with the pinned axes on both
    sides. The guide is one pin per axis.'''
    pinned_axes: PinnedAxes

    def object_processor(self, target: sc.Axis, guide: nm.Numeric | None
                         ) -> tuple[sc.Axis, nm.Numeric | None]:
        return self.pinned_axes.pinned(target, guide), guide

    def generate_guide_for_zeros(self, target: sc.Axis) -> None:
        return None

    def root_processor(self, target: sc.StrideMorphism, guide: Sequence[nm.Numeric | None]
                       ) -> tuple[sc.StrideMorphism, Sequence[nm.Numeric | None]]:
        domain_pins = tuple(guide)
        new_dom = tuple(self.pinned_axes.pinned(axis, pin)
                        for axis, pin in zip(target._dom, domain_pins))
        codomain_pins = tuple(row_pin(strides, shift, domain_pins)
                              for _, strides, shift in target._cod_stride_shift)
        rows = tuple(
            (self.pinned_axes.pinned(axis, pin), strides, shift)
            for (axis, strides, shift), pin in zip(target._cod_stride_shift, codomain_pins))
        return target.reconstruct(_dom=new_dom, _cod_stride_shift=rows), codomain_pins


@dataclass(frozen=True)
class TargetPins:
    '''What an operator does with the pins asked of its targets: the operator with
    the pins carried through its body where it has one, the pins each result's target
    carries, and the pins each operand's target is asked for.'''
    operator: cat.Operator
    output_pins: fd.Prod[Pins]
    input_pins: fd.Prod[Pins]


@dataclass
class IndexPinCrawler[B: cat.Datatype, A: cat.Axis](
        hypergraph_crawler.ReverseCrawler[cat.Array[B, A], cat.Broadcasted[B, A], Pins]):
    '''Carries the pins of every wire from the codomain of an expression to its
    domain, and rebuilds the expression with a pinned axis at every pinned position
    and a view where the index stops.'''
    pinned_axes: PinnedAxes = field(default_factory=PinnedAxes)
    stops: list[cat.Broadcasted[B, A]] = field(default_factory=list)

    def merge_guides(self, guides: Iterable[Pins]) -> Pins:
        return agreed_pins(guides)

    def generate_guide_for_zeros(self, target: cat.Array[B, A]) -> Pins:
        return free_pins(target)

    def object_processor(self, target: cat.Array[B, A], guide: Pins
                         ) -> tuple[cat.Array[B, A], Pins]:
        return self.pinned_axes.pinned_array(target, guide), guide

    def read_pinned_positions(self, array: cat.Array[B, A], carried: Pins,
                              asked: Pins) -> cat.Broadcasted[B, A]:
        '''The view reading `array` at the pinned index of every axis `asked` pins and
        `carried` does not: the pinned axis on its domain, the dense axis on its
        codomain and the identity row between them, beside the identity on every other
        axis. The view and each row are named by the index they read, and the view
        is recorded as a stop.'''
        pieces: list[sc.StrideCategory[A]] = []
        indices: list[str] = []
        for axis, have, want in zip(array.shape(), carried, asked):
            if have == want:
                pieces.append(cat.ProdObject((axis,)).identity())
            elif have is None:
                pieces.append(sc.StrideMorphism(
                    _dom=(self.pinned_axes.pinned(axis, want),),
                    _cod_stride_shift=((axis, (nm.Integer(1),), nm.Integer(0)),),
                    name=fd.DynamicName(f'[{want.to_latex()}]')))
                indices.append(want.to_latex())
            else:
                raise PinNotCarried(
                    f'{axis} carries the pin {have.to_latex()} and is asked for '
                    f'{want.to_latex() if want is not None else "every index"}')
        stop = ops.View.template(
            base=array.datatype, reindexing=tuple(pieces),
            name=fd.DynamicName('[' + ', '.join(indices) + ']'))
        self.stops.append(stop)
        return stop

    def read_asked_pins(self, morphism: cat.BroadcastedCategory[B, A],
                        carried: Sequence[Pins], asked: Sequence[Pins]
                        ) -> cat.BroadcastedCategory[B, A]:
        '''`morphism` followed by a view on every codomain array whose reader asks
        for a pin the array does not carry, and `morphism` itself where the pins
        agree.'''
        readers = tuple(
            cat.ProdObject((array,)).identity() if have == want
            else self.read_pinned_positions(array, have, want)
            for array, have, want in zip(morphism.cod(), carried, asked))
        if all(tutil.is_identity(reader) for reader in readers):
            return morphism
        return chsh.make_composed(morphism, chsh.make_product(*readers))

    def degree_pins(self, target: cat.Broadcasted[B, A], guide: Sequence[Pins]) -> Pins:
        '''The pins of the degree, which every output carries at its tiled positions
        and which the outputs must agree on.'''
        if not target.output_weaves:
            return (None,) * len(target.degree())
        return agreed_pins(
            tuple(weave.select_degree(pins))
            for weave, pins in zip(target.output_weaves, guide))

    def target_pins(self, target: cat.Broadcasted[B, A], guide: Sequence[Pins]
                    ) -> TargetPins:
        '''A block operator carries the pins of its targets through its body, and
        every other operator reads its whole target and drops them.'''
        if isinstance(target.operator, ops.BlockOperator):
            asked = tuple(tuple(weave.select_target(pins))
                          for weave, pins in zip(target.output_weaves, guide))
            block, operand_pins = self.propagate_category(target.operator.block, asked)
            return TargetPins(target.operator.reconstruct(block=block), asked,
                              tuple(operand_pins))
        return TargetPins(
            target.operator,
            tuple((None,) * target_count(weave) for weave in target.output_weaves),
            tuple((None,) * target_count(weave) for weave in target.input_weaves))

    def root_processor(self, target: cat.Broadcasted[B, A], guide: Sequence[Pins]
                       ) -> tuple[cat.BroadcastedCategory[B, A], Sequence[Pins]]:
        degree_pins = self.degree_pins(target, guide)
        targets = self.target_pins(target, guide)
        carried = tuple(
            ReindexingPinCrawler(self.pinned_axes).propagate_category(
                reindexing, degree_pins)
            for reindexing in target.reindexings)
        backup_degree = (
            cat.ProdObject.from_iter(
                self.pinned_axes.pinned(axis, pin)
                for axis, pin in zip(target.backup_degree, degree_pins))
            if target.has_empty_domain() else None)
        rebuilt = target.reconstruct(
            operator=targets.operator,
            input_weaves=tuple(
                weave.imprint_target(self.pinned_axes.pinned_array(weave.target(), pins))
                for weave, pins in zip(target.input_weaves, targets.input_pins)),
            output_weaves=tuple(
                weave.imprint_target(self.pinned_axes.pinned_array(weave.target(), pins))
                for weave, pins in zip(target.output_weaves, targets.output_pins)),
            reindexings=tuple(reindexing for reindexing, _ in carried),
            backup_degree=backup_degree)
        provided = tuple(
            pins_over_weave(weave, degree_pins, pins)
            for weave, pins in zip(target.output_weaves, targets.output_pins))
        input_pins = tuple(
            pins_over_weave(weave, codomain_pins, pins)
            for weave, (_, codomain_pins), pins
            in zip(target.input_weaves, carried, targets.input_pins))
        return self.read_asked_pins(rebuilt, provided, guide), input_pins

    def propagate_category(self, target: cat.BroadcastedCategory[B, A],
                           guide: Sequence[Pins]
                           ) -> tuple[cat.BroadcastedCategory[B, A], Sequence[Pins]]:
        if isinstance(target, cat.Rearrangement):
            return self.propagate_rearrangement(target, guide)
        return super().propagate_category(target, guide)

    def propagate_rearrangement(self, target: cat.Rearrangement[cat.Array[B, A]],
                                guide: Sequence[Pins]
                                ) -> tuple[cat.BroadcastedCategory[B, A], Sequence[Pins]]:
        '''A copied wire carries the pins its readers agree on, and a reader asking
        for more reads its pinned positions through a view after the copy. A deleted
        wire is free.'''
        dom_to_cod = util.Multidict(target.pairwise())
        agreed = tuple(
            self.merge_guides(guide[j] for j in dom_to_cod[i])
            if dom_to_cod[i] else free_pins(array)
            for i, array in enumerate(target._dom))
        rebuilt = cat.Rearrangement(
            mapping=target.mapping,
            _dom=tuple(self.pinned_axes.pinned_array(array, pins)
                       for array, pins in zip(target._dom, agreed)))
        return self.read_asked_pins(rebuilt, rebuilt.apply(agreed), guide), agreed


@dataclass(frozen=True)
class DraggedIndex[B: cat.Datatype, A: cat.Axis]:
    '''The expression rebuilt with every position the index reached pinned, the pins
    of its domain, the views written where the index stopped, in the order the
    crawl wrote them, and the axes the crawl pinned.'''
    expression: cat.BroadcastedCategory[B, A]
    domain_pins: fd.Prod[Pins]
    stops: fd.Prod[cat.Broadcasted[B, A]]
    pinned_axes: PinnedAxes


def drag_index_backwards[B: cat.Datatype, A: cat.Axis](
        morphism: cat.BroadcastedCategory[B, A], codomain_pins: Sequence[Pins]
        ) -> DraggedIndex[B, A]:
    '''`morphism` with the pins of its codomain carried back to its domain, one pins
    entry per codomain array.'''
    crawl = IndexPinCrawler[B, A]()
    expression, domain_pins = crawl.propagate_category(morphism, tuple(codomain_pins))
    return DraggedIndex(expression, tuple(domain_pins), tuple(crawl.stops),
                        crawl.pinned_axes)


def codomain_pinned_at[B: cat.Datatype, A: cat.Axis](
        morphism: cat.BroadcastedCategory[B, A], output: int, axis_position: int,
        index: nm.Numeric) -> fd.Prod[Pins]:
    '''The pins of the codomain of `morphism` with the axis at `axis_position` of the
    array at `output` pinned at `index`, and every other axis free.'''
    return tuple(
        tuple(index if k == output and p == axis_position else None
              for p in range(len(array.shape())))
        for k, array in enumerate(morphism.cod()))


def guard_pinned_at(guarded: AffineGuards.AffineSparseAxis, guide: sc.Axis,
                    index: nm.Numeric,
                    pinned: AffineGuards.AffineSparseAxis) -> AffineGuards.AffineSparseAxis:
    '''`guarded` with `index` substituted for its guide `guide` in its form, so the
    form no longer reads that guide, named after its own letter and its guides with
    `pinned` in the place of `guide`, as one body, so that the underscore of the index
    is not read as a lineage.'''
    kept = tuple(position for position, other in enumerate(guarded.guides)
                 if other != guide)
    dropped = tuple(position for position, other in enumerate(guarded.guides)
                    if other == guide)
    shift = nm.collect_like_terms(nm.Addition.template(
        guarded.shift,
        *(nm.Multiplication.template(guarded.guide_strides[position], index)
          for position in dropped)))
    letter = AffineGuards.axis_body(guarded).split('|')[0]
    guide_bodies = ','.join(
        AffineGuards.axis_body(pinned if position in dropped else guarded.guides[position])
        for position in range(len(guarded.guides)))
    return fd.DynamicName.from_str(f'{letter}|{guide_bodies}', lineage=False).capture(
        AffineGuards.AffineSparseAxis(
            _size=guarded.local_size(),
            guides=tuple(guarded.guides[position] for position in kept),
            guide_strides=tuple(guarded.guide_strides[position] for position in kept),
            stride=guarded.stride, shift=shift, extent=guarded.extent))


def pin_guards[B: cat.Datatype, A: cat.Axis](
        dragged: DraggedIndex[B, A]) -> cat.BroadcastedCategory[B, A]:
    '''The expression of `dragged` with every guard pinned whose guide the index
    passed through: a guarded axis that stands on no array without the pinned form of
    one of its guides beside it is replaced everywhere by `guard_pinned_at` of that
    guide.

    The arrays are the domain and the codomain of every operation, because an
    operator stores weaves and reindexings and computes its arrays. The test is by
    identity, because it asks whether the pinned form of the guide stands on the same
    wires as the guard, and the replacement runs through an `fd.Context`, so the
    reindexings and the weaves naming the axis change with the arrays.'''
    arrays = tuple(
        array
        for operation in tutil.type_search(cat.Morphism, dragged.expression)
        if isinstance(operation, cat.Broadcasted)
        or (isinstance(operation, cat.Rearrangement)
            and all(isinstance(wire, cat.Array) for wire in operation._dom))
        for array in (*operation.dom(), *operation.cod()))
    context = fd.Context()
    replaced = 0
    for guide, index, pinned in dragged.pinned_axes.items():
        guarded_axes = util.unique_tuple(
            axis for array in arrays for axis in array.shape()
            if isinstance(axis, AffineGuards.AffineSparseAxis)
            and any(other == guide for other in axis.guides))
        for guarded in guarded_axes:
            carrying = tuple(array for array in arrays
                             if any(axis == guarded for axis in array.shape()))
            if not all(any(axis == pinned for axis in array.shape())
                       for array in carrying):
                continue
            context.append_bucket(
                fd.EqualityClass.template(guarded, priority=0).merge(
                    fd.EqualityClass.template(
                        guard_pinned_at(guarded, guide, index, pinned), priority=1)))
            replaced += 1
    if replaced == 0:
        return dragged.expression
    return context.apply(dragged.expression)
