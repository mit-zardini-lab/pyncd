'''Moving a read backwards through an expression.

Written by Claude Fable 5.1, reasoning effort 80.

A read is an `ops.View`, and its reindexing maps the axes of the array it returns to
the axes of the array it reads. An operator broadcast over its degree computes the same
function at every index of the degree, so reading its result through a reindexing is
the same as reading each operand through that reindexing composed with the operand's
own, and computing once over the read's domain. That is the Yoneda trick of
`obsidian/06-practice/Yoneda and Cartesian Tricks.md`, which the requester calls
Yoneda sliding, and this module runs it mechanically, from the result of an expression
towards its inputs. A read that meets a view composes into it, and the view is dropped
where the composite reads the operand at the identity. The requester asked for the
process on 2026-09-25 as the second way of dragging an index backwards, beside
`drag_index_backwards`. The index morphism, a stride morphism with no domain axis whose
one row is the shift `t_x`, slides from the result of an attention towards its inputs,
the causal mask composes into it as the read `t_x - j_w` of one axis by one axis, the
composite moves in front of the key and value projections, and at the copy the reads
the branches agree on become one view whose result is copied to them.

`ReadCrawler` is a `ReverseCrawler` over the broadcasted category whose guide is the
pending read of each wire: the reindexing of the view that would stand on the wire, an
`sc.StrideMorphism` whose codomain is the wire's axes in order, or `None` where nothing
is read. Its domain is the array the wire carries once the read is written, ordered by
the first row that reads each axis, so two branches that read one wire the same way
carry equal reads. At a `Broadcasted` the read splits by the output weave. A row onto a
target position must be the identity on that axis, because the operator reads the whole
target, and the rows onto the degree positions form the read of the degree, which
composes with each operand's reindexing. A composite whose every row selects one degree
axis at unit stride, and reads that axis as itself, stays in the operator as its
reindexing. A row reading one axis as another is a renaming, and it moves onto the
operand as any other read does, so that the operand's wire carries the axis the operator
now reads. Where any row does more, the composite moves onto the operand's wire as its
pending read, and the operator keeps the projection onto the axes the read needs. A read the operator cannot pass, because a
target row is not the identity, is written after the operator as a view, and the
operands are read as they were.

Two operators pass more. A `ops.BlockOperator` computes its body on the targets of its
operands, so the rows onto its target positions are a read of the body's result, and
the crawl carries that read through the body and out to the operands, while the rows
onto its degree positions slide as they do for any operator. An `ops.Arrange` writes the
index of every position of its axis, so its result read through a row is the value of
that row: a row with no domain axis is its shift, written by an `ops.ConstantOp`, and a
row over one axis is the arrangement of that axis followed by the affine map of the
row, written by an `ops.Arithmetic`. The requester stated the rule on 2026-09-25 as an
index that goes through an arrangement becoming the covariant view of the index, and
noted that an arrangement is over one axis, so a row over several axes is not written
and stops the read.

`written_out` writes the operators of given classes out by their standard expansions,
inside every box, so that the rotary table of a model is met as the arrangement of its
positions and the read continues through it.

At a `Rearrangement` the reads asked of the wires copied from one wire are grouped by
equality. A wire whose readers all ask for one read carries it further. A wire whose
readers disagree stops the reads, and each distinct read is written once after the
copy, as a view whose result is copied to the branches that asked for it.

The crawl runs on the morphism form, as `drag_index_backwards` does.
`obsidian/02-categories/Advanced Axis Dynamics.md` states the feature, and
`slide_causal_reads_backwards` runs the crawl from every causal read of an
expression.
'''
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import data_structure.Term as fd  # for 'foundations'
import data_structure.Numeric as nm
import data_structure.Category as cat
import data_structure.Operators as ops
import data_structure.ProductCategory as pc
import data_structure.StrideCategory as sc
import utilities.utilities as util
import construction_helpers.simple_helper as chsh
import graphs.processing.hypergraph_crawler as hypergraph_crawler
import graphs.processing.hypergraph_functor as hypergraph_functor
import algebra.registries.standard_expansions as standard_expansions
import advanced_axis_dynamics.algebra.disentangle_reindexings as disentangle_reindexings

type Read = sc.StrideMorphism | None
'''The pending read of a wire: the reindexing of the view that would stand on it, whose
codomain is the wire's axes in order, or `None` where nothing is read.'''

type Row = tuple[sc.Axis, fd.Prod[nm.Numeric], nm.Numeric]


class ReadDoesNotFitTheWire(Exception):
    '''A read with a codomain row for every axis of one array, applied to an array or a
    morphism with another number of axes.'''


class NotAStrideCategoryMorphism(Exception):
    '''A term met where a morphism of the stride category was expected.'''


def unit_row(axis: sc.Axis, width: int, position: int) -> Row:
    '''The row reading `axis` from domain position `position` of `width` at unit
    stride and no shift.'''
    return (axis, tuple(nm.Integer(1) if i == position else nm.Integer(0)
                        for i in range(width)), nm.Integer(0))


def selects_one_axis(strides: fd.Prod[nm.Numeric], shift: nm.Numeric) -> int | None:
    '''The one domain position a row reads at unit stride and no shift, and `None` for
    a row that reads nothing, several axes, a stride other than one, or a shift.'''
    read = tuple((position, stride) for position, stride in enumerate(strides)
                 if not nm.is_zero(stride))
    if len(read) == 1 and read[0][1] == nm.Integer(1) and nm.is_zero(shift):
        return read[0][0]
    return None


def positions_read(strides: fd.Prod[nm.Numeric]) -> fd.Prod[int]:
    return tuple(position for position, stride in enumerate(strides)
                 if not nm.is_zero(stride))


def one_name(morphisms: Iterable[sc.StrideMorphism]) -> fd.DynamicName | None:
    '''The name the named morphisms among `morphisms` share, and `None` where none is
    named or two names differ.'''
    names = util.unique_tuple(morphism.name for morphism in morphisms
                              if morphism.name is not None)
    return names[0] if len(names) == 1 else None


def compose_stride_morphisms(first: sc.StrideMorphism,
                             second: sc.StrideMorphism) -> sc.StrideMorphism:
    '''`first` followed by `second`: the map whose rows are the rows of `second` with
    the rows of `first` substituted for the positions they read. It is named after
    `second`, and after `first` where `second` carries no name.'''
    if len(first._cod_stride_shift) != len(second._dom):
        raise ReadDoesNotFitTheWire(
            f'{len(first._cod_stride_shift)} rows composed onto {len(second._dom)} axes')
    width = len(first._dom)
    rows: list[Row] = []
    for axis, strides, shift in second._cod_stride_shift:
        summed: list[list[nm.Numeric]] = [[] for _ in range(width)]
        shifts = [shift]
        for stride, (_, first_strides, first_shift) in zip(strides, first._cod_stride_shift):
            if nm.is_zero(stride):
                continue
            for position, first_stride in enumerate(first_strides):
                if not nm.is_zero(first_stride):
                    summed[position].append(nm.Multiplication.template(stride, first_stride))
            shifts.append(nm.Multiplication.template(stride, first_shift))
        rows.append((
            axis,
            tuple(nm.collect_like_terms(nm.Addition.template(nm.Integer(0), *terms))
                  for terms in summed),
            nm.collect_like_terms(nm.Addition.template(*shifts))))
    return sc.StrideMorphism(
        _dom=first._dom, _cod_stride_shift=tuple(rows),
        name=second.name if second.name is not None else first.name)


def as_stride_morphism(morphism: sc.StrideCategory) -> sc.StrideMorphism:
    '''`morphism` as one `sc.StrideMorphism`: a rearrangement as its unit rows, a
    product as the rows of its factors side by side, and a composite composed.'''
    match morphism:
        case sc.StrideMorphism():
            return morphism
        case pc.Rearrangement(mapping=mapping, _dom=dom):
            return sc.StrideMorphism(
                _dom=tuple(dom),
                _cod_stride_shift=tuple(unit_row(dom[i], len(dom), i) for i in mapping))
        case pc.ProductOfMorphisms(content=content):
            factors = tuple(as_stride_morphism(factor) for factor in content)
            dom = util.concat(factor._dom for factor in factors)
            rows: list[Row] = []
            offset = 0
            for factor in factors:
                width = len(factor._dom)
                for axis, strides, shift in factor._cod_stride_shift:
                    rows.append((axis, (*(nm.Integer(0),) * offset, *strides,
                                        *(nm.Integer(0),) * (len(dom) - offset - width)),
                                 shift))
                offset += width
            return sc.StrideMorphism(_dom=dom, _cod_stride_shift=tuple(rows),
                                     name=one_name(factors))
        case pc.Composed(content=content):
            composed = as_stride_morphism(content[0])
            for factor in content[1:]:
                composed = compose_stride_morphisms(composed, as_stride_morphism(factor))
            return composed
        case pc.Block(body=body):
            return as_stride_morphism(body)
    raise NotAStrideCategoryMorphism(f'{type(morphism).__name__} is not a reindexing')


def in_reading_order(read: sc.StrideMorphism) -> tuple[sc.StrideMorphism, fd.Prod[int]]:
    '''`read` with its domain axes in the order its rows first read them, a domain
    axis no row reads last, beside the old position of each new domain position.'''
    order: list[int] = []
    for _, strides, _ in read._cod_stride_shift:
        for position in positions_read(strides):
            if position not in order:
                order.append(position)
    order.extend(position for position in range(len(read._dom)) if position not in order)
    if order == list(range(len(read._dom))):
        return read, tuple(order)
    return read.reconstruct(
        _dom=tuple(read._dom[position] for position in order),
        _cod_stride_shift=tuple(
            (axis, tuple(strides[position] for position in order), shift)
            for axis, strides, shift in read._cod_stride_shift)), tuple(order)


def same_read(one: Read, other: Read) -> bool:
    '''Whether two pending reads state one map, whatever their names.'''
    if one is None or other is None:
        return one is other
    return one._dom == other._dom and one._cod_stride_shift == other._cod_stride_shift


def read_domain_array[B: cat.Datatype, A: cat.Axis](
        array: cat.Array[B, A], read: sc.StrideMorphism[A]) -> cat.Array[B, A]:
    '''The array `read` returns when it reads `array`.'''
    if len(read._cod_stride_shift) != len(array.shape()):
        raise ReadDoesNotFitTheWire(
            f'{len(read._cod_stride_shift)} rows for the {len(array.shape())} axes of '
            f'{array}')
    return cat.Array(array.datatype, tuple(read._dom))


def view_of[B: cat.Datatype, A: cat.Axis](
        array: cat.Array[B, A], read: sc.StrideMorphism[A]) -> cat.Broadcasted[B, A]:
    '''The view reading `array` through `read`, with the read split into its
    independent factors so that the axes it leaves alone draw as straight wires. A read
    that only copies, permutes or deletes axes is written as rearrangements, and its
    name names the view alone, so a named diagonal draws as a dot on its wire.'''
    named_rearrangement = (read.name is not None
                           and disentangle_reindexings.states_a_rearrangement(
                               tuple(read._dom), read._cod_stride_shift))
    return ops.View.template(
        base=array.datatype,
        reindexing=disentangle_reindexings.disentangle_reindexing(
            read.reconstruct(name=None) if named_rearrangement else read),
        name=read.name)


@dataclass(frozen=True)
class SplitRead:
    '''A read split at an output weave: the read of the degree, from the axes the read
    returns at its tiled positions to the degree, the read of the target, from the
    axes the read returns at its target positions to the target, which is `None` where
    the target is read at the identity, and the weave of the array the read returns.'''
    degree_read: sc.StrideMorphism
    target_read: Read
    weave: cat.Weave


def split_at_weave(read: sc.StrideMorphism, weave: cat.Weave,
                   into_the_target: bool = False) -> SplitRead | None:
    '''`read` split at `weave`, or `None` where the operator cannot pass it. A row
    onto a target position must be the identity on that axis, unless
    `into_the_target` allows the target to be read, and a domain axis read by a target
    row and by a degree row cannot be split between the two.'''
    if len(read._cod_stride_shift) != len(weave._shape):
        raise ReadDoesNotFitTheWire(
            f'{len(read._cod_stride_shift)} rows for a weave of {len(weave._shape)}')
    target_positions = tuple(position for position, mode in enumerate(weave._shape)
                             if not isinstance(mode, cat.WeaveMode))
    degree_positions = tuple(position for position, mode in enumerate(weave._shape)
                             if isinstance(mode, cat.WeaveMode))
    target_sources: set[int] = set()
    for position in target_positions:
        target_sources.update(positions_read(read._cod_stride_shift[position][1]))
    for position in degree_positions:
        if target_sources & set(positions_read(read._cod_stride_shift[position][1])):
            return None
    sources = [selects_one_axis(strides, shift)
               for _, strides, shift in (read._cod_stride_shift[p] for p in target_positions)]
    identity_on_targets = (
        all(source is not None for source in sources)
        and len(set(sources)) == len(sources)
        and all(read._dom[source] == read._cod_stride_shift[position][0]
                for source, position in zip(sources, target_positions)))
    if not identity_on_targets and not into_the_target:
        return None
    target_domain = tuple(i for i in range(len(read._dom)) if i in target_sources)
    degree_domain = tuple(i for i in range(len(read._dom)) if i not in target_sources)
    degree_read = sc.StrideMorphism(
        _dom=tuple(read._dom[i] for i in degree_domain),
        _cod_stride_shift=tuple(
            (axis, tuple(strides[i] for i in degree_domain), shift)
            for axis, strides, shift in (read._cod_stride_shift[p] for p in degree_positions)),
        name=read.name)
    target_read = None if identity_on_targets else sc.StrideMorphism(
        _dom=tuple(read._dom[i] for i in target_domain),
        _cod_stride_shift=tuple(
            (axis, tuple(strides[i] for i in target_domain), shift)
            for axis, strides, shift in (read._cod_stride_shift[p] for p in target_positions)),
        name=read.name)
    return SplitRead(
        degree_read=degree_read,
        target_read=target_read,
        weave=cat.Weave(weave.datatype, tuple(
            cat.WeaveMode.TILED if i in degree_domain else read._dom[i]
            for i in range(len(read._dom)))))


@dataclass(frozen=True)
class CarriedRead:
    '''What one operand keeps of a read carried through its operator: the reindexing
    the operator now reads the operand through, the operand's weave, and the pending
    read on the operand's wire.'''
    reindexing: sc.StrideCategory
    weave: cat.Weave
    read: Read


VIEW_NAME_GIVEN_BY_DEFAULT = fd.DynamicName('\\sigma')
'''The name `ops.View.template` gives a view its caller named nothing.'''


def reindexing_named_after_its_view(
        target: cat.Broadcasted, reindexing: sc.StrideCategory) -> sc.StrideCategory:
    '''The reindexing of `target` as one stride morphism named after the view, where
    `target` is a view its caller named whose reindexing carries no name, and
    `reindexing` itself otherwise.

    A view written with a `cat.Rearrangement`, such as the Diagonal of a mixture of
    experts, holds its name on the `ops.View` alone. A read composed into it is named
    after its reindexing, so without the name the view written where the read stops
    would be drawn unnamed. Added by Claude Opus 5.5 (1M context), effort 40, on
    2026-09-27, for the cached pass of Mixtral-8x7B.'''
    if (not isinstance(target.operator, ops.View)
            or target.operator.name in (None, VIEW_NAME_GIVEN_BY_DEFAULT)):
        return reindexing
    as_one = as_stride_morphism(reindexing)
    if as_one.name is not None:
        return reindexing
    return as_one.reconstruct(name=target.operator.name)


def carry_through_operand(degree_read: sc.StrideMorphism, reindexing: sc.StrideCategory,
                          weave: cat.Weave, target_read: Read = None) -> CarriedRead:
    '''The read of the degree composed with the operand's reindexing, kept in the
    operator where every row selects one axis and reads it as itself, and moved onto
    the operand's wire otherwise, with the operator keeping the projection onto the axes the read needs,
    beside `target_read`, the read of the operand's target that came out of a box, or
    the identity on the target where there is none.'''
    composite = compose_stride_morphisms(degree_read, as_stride_morphism(reindexing))
    sources = tuple(selects_one_axis(strides, shift)
                    for _, strides, shift in composite._cod_stride_shift)
    plain = all(source is not None and composite._dom[source] == axis
                for (axis, _, _), source in zip(composite._cod_stride_shift, sources))
    if plain and target_read is None:
        return CarriedRead(
            pc.Rearrangement(mapping=tuple(sources), _dom=tuple(degree_read._dom)),
            weave, None)
    if plain:
        degree_domain = tuple(axis for axis, _, _ in composite._cod_stride_shift)
        degree_rows: fd.Prod[Row] = tuple(
            unit_row(axis, len(degree_domain), position)
            for position, axis in enumerate(degree_domain))
        needed: fd.Prod[int] = ()
    else:
        needed = tuple(
            position for position in range(len(degree_read._dom))
            if any(not nm.is_zero(strides[position])
                   for _, strides, _ in composite._cod_stride_shift))
        degree_domain = tuple(degree_read._dom[position] for position in needed)
        degree_rows = tuple(
            (axis, tuple(strides[position] for position in needed), shift)
            for axis, strides, shift in composite._cod_stride_shift)
    targets = tuple(weave.select_target(weave._shape))
    target_domain = targets if target_read is None else tuple(target_read._dom)
    target_rows: fd.Prod[Row] = (
        tuple(unit_row(axis, len(targets), position)
              for position, axis in enumerate(targets))
        if target_read is None else target_read._cod_stride_shift)
    dom = (*degree_domain, *target_domain)
    degree_row = iter(degree_rows)
    target_row = iter(target_rows)
    rows: list[Row] = []
    for mode in weave._shape:
        if isinstance(mode, cat.WeaveMode):
            axis, strides, shift = next(degree_row)
            rows.append((axis, (*strides, *(nm.Integer(0),) * len(target_domain)), shift))
        else:
            axis, strides, shift = next(target_row)
            rows.append((axis, (*(nm.Integer(0),) * len(degree_domain), *strides), shift))
    read, order = in_reading_order(sc.StrideMorphism(
        _dom=dom, _cod_stride_shift=tuple(rows),
        name=target_read.name if target_read is not None and target_read.name is not None
        else composite.name))
    degree_part = tuple(position for position, old in enumerate(order)
                        if old < len(degree_domain))
    new_weave = cat.Weave(weave.datatype, tuple(
        cat.WeaveMode.TILED if position in degree_part else read._dom[position]
        for position in range(len(read._dom))))
    if plain:
        projection: sc.StrideCategory = pc.Rearrangement(
            mapping=tuple(sources), _dom=tuple(degree_read._dom))
    else:
        projection = pc.Rearrangement(
            mapping=tuple(needed[order[position]] for position in degree_part),
            _dom=tuple(degree_read._dom))
    return CarriedRead(projection, new_weave, read)


def index_values[B: cat.Datatype, A: cat.Axis](
        arranged: cat.Broadcasted[B, A], read: sc.StrideMorphism[A]
        ) -> cat.BroadcastedCategory[B, A] | None:
    '''The values of `read`'s one row over its domain, which is what an `ops.Arrange`
    returns when its result is read through `read`: the shift alone, from an
    `ops.ConstantOp`, where the row reads no axis, and the affine map of one axis's
    positions, from an `ops.Arrange` of that axis and an `ops.Arithmetic`, where it
    reads one. `None` where it reads several.'''
    datatype = arranged.output_weaves[0].datatype
    (_, strides, shift), = read._cod_stride_shift
    if len(read._dom) == 0:
        return cat.Broadcasted(
            operator=ops.ConstantOp(name=fd.DynamicName(shift.to_latex()), value=shift),
            input_weaves=(), output_weaves=(cat.Weave(datatype, ()),),
            reindexings=(), backup_degree=cat.ProdObject())
    if len(read._dom) != 1:
        return None
    positions = ops.Arrange.template(read._dom[0])
    formula = nm.collect_like_terms(nm.Addition.template(
        shift, nm.Multiplication.template(strides[0], nm.x)))
    return chsh.make_composed(
        positions,
        ops.Arithmetic.template(
            formula, base=positions.cod()[0], output_datatype=datatype,
            name=fd.DynamicName(formula.to_latex())))


@dataclass(frozen=True)
class ReadGroup:
    '''The branches of a copy that ask one wire for one read.'''
    wire: int
    read: Read
    branches: fd.Prod[int]


def read_groups(target: cat.Rearrangement, reads: Sequence[Read]) -> fd.Prod[ReadGroup]:
    '''The reads asked of each domain wire of `target`, grouped by equality, in the
    order the first branch asking for each appears in the codomain.'''
    groups: list[ReadGroup] = []
    for branch, wire in enumerate(target.mapping):
        for index, group in enumerate(groups):
            if group.wire == wire and same_read(group.read, reads[branch]):
                groups[index] = ReadGroup(wire, group.read, (*group.branches, branch))
                break
        else:
            groups.append(ReadGroup(wire, reads[branch], (branch,)))
    return tuple(groups)


@dataclass
class ReadCrawler[B: cat.Datatype, A: cat.Axis](
        hypergraph_crawler.ReverseCrawler[cat.Array[B, A], cat.Broadcasted[B, A], Read]):
    '''Carries the pending read of every wire from the codomain of an expression to its
    domain, and rebuilds the expression over the arrays the reads return, with a view
    written wherever a read stops.'''
    stops: list[cat.Broadcasted[B, A]] = field(default_factory=list)

    def merge_guides(self, guides: Iterable[Read]) -> Read:
        reads = tuple(guides)
        return reads[0] if all(same_read(reads[0], read) for read in reads) else None

    def read_yields_its_name(self, read: sc.StrideMorphism) -> bool:
        '''Whether a view its caller named gives its name to `read` composed into it,
        which holds where `read` carries no name. A crawl naming the reads it carries
        for itself lets those names yield as well.'''
        return read.name is None

    def generate_guide_for_zeros(self, target: cat.Array[B, A]) -> Read:
        return None

    def object_processor(self, target: cat.Array[B, A], guide: Read
                         ) -> tuple[cat.Array[B, A], Read]:
        return (target if guide is None else read_domain_array(target, guide)), guide

    def stopped(self, target: cat.Broadcasted[B, A], reads: Sequence[Read]
                ) -> tuple[cat.BroadcastedCategory[B, A], Sequence[Read]]:
        '''`target` followed by the view of every read it cannot pass, and its operands
        read as they were.'''
        views = tuple(
            cat.ProdObject((array,)).identity() if read is None else view_of(array, read)
            for array, read in zip(target.cod(), reads))
        for view in views:
            if not isinstance(view, cat.Rearrangement):
                self.stops.append(view)
        return (chsh.make_composed(target, chsh.make_product(*views)),
                (None,) * len(tuple(target.dom())))

    def through_tape_seed(self, seed: cat.Morphism[cat.Array[B, A]],
                          reads: Sequence[Read]
                          ) -> tuple[cat.BroadcastedCategory[B, A], Sequence[Read]]:
        '''A grab or a drop of a tape slot, which holds its array whole. A drop reads
        its operand whole, and a grab read through a view is followed by the view.'''
        if all(read is None for read in reads):
            return seed, (None,) * len(tuple(seed.dom()))
        return self.stopped(seed, reads)

    def rebuilt(self, target: cat.Broadcasted[B, A], splits: Sequence[SplitRead],
                carried: Sequence[CarriedRead], operator: cat.Operator
                ) -> cat.BroadcastedCategory[B, A]:
        rebuilt = target.reconstruct(
            operator=operator,
            input_weaves=tuple(carry.weave for carry in carried),
            output_weaves=tuple(split.weave for split in splits),
            reindexings=tuple(carry.reindexing for carry in carried),
            backup_degree=(splits[0].degree_read.dom()
                           if target.has_empty_domain() else None))
        if ops.is_identity(rebuilt):
            return cat.ProdObject(tuple(rebuilt.dom())).identity()
        return rebuilt

    def splits_agreeing_on_the_degree(
            self, target: cat.Broadcasted[B, A], reads: Sequence[Read],
            into_the_target: bool) -> fd.Prod[SplitRead] | None:
        '''The reads of every result split at its weave, or `None` where a result is
        not read, a split fails, or two results read the degree differently.'''
        if any(read is None for read in reads):
            return None
        splits = tuple(split_at_weave(read, weave, into_the_target)
                       for read, weave in zip(reads, target.output_weaves))
        if any(split is None for split in splits):
            return None
        if not all(same_read(splits[0].degree_read, split.degree_read)
                   for split in splits):
            return None
        return splits

    def through_box(self, target: cat.Broadcasted[B, A], reads: Sequence[Read]
                    ) -> tuple[cat.BroadcastedCategory[B, A], Sequence[Read]]:
        '''The reads of the box's targets carried through its body and out to the
        operands, beside the read of its degree carried as for any operator.'''
        splits = self.splits_agreeing_on_the_degree(target, reads, True)
        if splits is None:
            return self.stopped(target, reads)
        block, body_reads = self.propagate_category(
            target.operator.block, tuple(split.target_read for split in splits))
        if block != target.operator.block:
            block = with_the_tag_of_its_body(block)
        carried = tuple(
            carry_through_operand(splits[0].degree_read, reindexing, weave, body_read)
            for reindexing, weave, body_read
            in zip(target.reindexings, target.input_weaves, body_reads))
        return (self.rebuilt(target, splits, carried,
                             target.operator.reconstruct(block=block)),
                tuple(carry.read for carry in carried))

    def root_processor(self, target: cat.Broadcasted[B, A], guide: Sequence[Read]
                       ) -> tuple[cat.BroadcastedCategory[B, A], Sequence[Read]]:
        reads = tuple(guide)
        if not isinstance(target, cat.Broadcasted):
            return self.through_tape_seed(target, reads)
        if all(read is None for read in reads):
            return target, (None,) * len(target.input_weaves)
        if isinstance(target.operator, ops.Arrange):
            values = index_values(target, reads[0])
            return (values, ()) if values is not None else self.stopped(target, reads)
        if isinstance(target.operator, ops.BlockOperator):
            return self.through_box(target, reads)
        splits = self.splits_agreeing_on_the_degree(target, reads, False)
        if splits is None:
            return self.stopped(target, reads)
        yields = self.read_yields_its_name(splits[0].degree_read)
        carried = tuple(
            carry_through_operand(
                splits[0].degree_read,
                reindexing_named_after_its_view(target, reindexing) if yields
                else reindexing,
                weave)
            for reindexing, weave in zip(target.reindexings, target.input_weaves))
        return (self.rebuilt(target, splits, carried, target.operator),
                tuple(carry.read for carry in carried))

    def propagate_category(self, target: cat.BroadcastedCategory[B, A],
                           guide: Sequence[Read]
                           ) -> tuple[cat.BroadcastedCategory[B, A], Sequence[Read]]:
        if isinstance(target, cat.Rearrangement):
            return self.propagate_rearrangement(target, guide)
        return super().propagate_category(target, guide)

    def propagate_rearrangement(self, target: cat.Rearrangement[cat.Array[B, A]],
                                guide: Sequence[Read]
                                ) -> tuple[cat.BroadcastedCategory[B, A], Sequence[Read]]:
        '''A wire whose branches all ask for one read carries it further. A wire whose
        branches disagree is read once per distinct read after the copy, and each
        read's result is copied to the branches that asked for it.'''
        groups = read_groups(target, tuple(guide))
        groups_of_wire = {wire: tuple(group for group in groups if group.wire == wire)
                          for wire in range(len(target._dom))}
        carried = {wire: own[0].read for wire, own in groups_of_wire.items()
                   if len(own) == 1 and own[0].read is not None}
        new_dom = tuple(
            read_domain_array(array, carried[wire]) if wire in carried else array
            for wire, array in enumerate(target._dom))
        domain_reads = tuple(carried.get(wire) for wire in range(len(target._dom)))
        if all(group.wire in carried or group.read is None for group in groups):
            return cat.Rearrangement(mapping=target.mapping, _dom=new_dom), domain_reads
        views = tuple(
            cat.ProdObject((new_dom[group.wire],)).identity()
            if group.wire in carried or group.read is None
            else view_of(new_dom[group.wire], group.read)
            for group in groups)
        for view in views:
            if not isinstance(view, cat.Rearrangement):
                self.stops.append(view)
        group_of_branch = {branch: index for index, group in enumerate(groups)
                           for branch in group.branches}
        spread = cat.Rearrangement(mapping=tuple(group.wire for group in groups),
                                   _dom=new_dom)
        copied = cat.Rearrangement(
            mapping=tuple(group_of_branch[branch] for branch in range(len(target.mapping))),
            _dom=tuple(view.cod()[0] for view in views))
        return (chsh.make_composed(spread, chsh.make_product(*views), copied),
                domain_reads)


def with_the_tag_of_its_body[B: cat.Datatype, A: cat.Axis](
        block: cat.Block[B, A]) -> cat.Block[B, A]:
    '''`block` under a tag derived from its tag and its body. A body the crawl rebuilt
    at one site of a box is then drawn, and opened on a page, apart from the body the
    box keeps at another site, since a figure and a page draw one body per tag. Two
    sites rebuilt alike keep one tag. `quantise_model` derives the tag of a quantised
    block the same way.'''
    tag = block.block_tag
    return block.reconstruct(block_tag=tag.reconstruct(uid=tag.uid.reconstruct(
        _id=fd.hash_id((tag.uid._id, hash(block.body))))))


@dataclass
class WriteOutOperators(hypergraph_functor.Endofunctor[cat.Array, cat.Broadcasted]):
    '''Every operator of the classes in `kinds` written out by its standard expansion,
    inside the body of every block operator as well, so that a table an arrangement
    stands behind is met as that arrangement.'''
    kinds: tuple[type[cat.Operator], ...] = ()

    def apply_root(self, target: cat.Broadcasted) -> cat.BroadcastedCategory:
        if not isinstance(target, cat.Broadcasted):
            return target
        if isinstance(target.operator, ops.BlockOperator):
            return target.reconstruct(operator=target.operator.reconstruct(
                block=self.apply_category(target.operator.block)))
        if isinstance(target.operator, self.kinds):
            expanded = standard_expansions.expand_standard(target)
            return target if expanded is None else expanded
        return target


def written_out[B: cat.Datatype, A: cat.Axis](
        morphism: cat.BroadcastedCategory[B, A], *kinds: type[cat.Operator]
        ) -> cat.BroadcastedCategory[B, A]:
    '''`morphism` with every operator of the classes `kinds` written out by its
    standard expansion, inside every box as well.'''
    return WriteOutOperators(kinds=kinds)(morphism)


@dataclass(frozen=True)
class MovedReads[B: cat.Datatype, A: cat.Axis]:
    '''The expression rebuilt over the arrays the reads return, the reads that reached
    its domain, one per domain array, and the views written where a read stopped, in
    the order the crawl wrote them.'''
    expression: cat.BroadcastedCategory[B, A]
    domain_reads: fd.Prod[Read]
    stops: fd.Prod[cat.Broadcasted[B, A]]


def move_reads_backwards[B: cat.Datatype, A: cat.Axis](
        morphism: cat.BroadcastedCategory[B, A], codomain_reads: Sequence[Read]
        ) -> MovedReads[B, A]:
    '''`morphism` with the reads of its codomain arrays moved back towards its domain,
    one read per codomain array.'''
    crawl = ReadCrawler[B, A]()
    expression, domain_reads = crawl.propagate_category(morphism, tuple(codomain_reads))
    return MovedReads(expression, tuple(domain_reads), tuple(crawl.stops))


def index_read[B: cat.Datatype, A: cat.Axis](
        morphism: cat.BroadcastedCategory[B, A], output: int, axis_position: int,
        index: nm.Numeric) -> fd.Prod[Read]:
    '''The reads of the codomain of `morphism` that read the axis at `axis_position` of
    the array at `output` at `index`, through the row with no domain axis and the shift
    `index`, beside the identity on every other axis, and read no other array.'''
    reads: list[Read] = []
    for k, array in enumerate(morphism.cod()):
        if k != output:
            reads.append(None)
            continue
        shape = tuple(array.shape())
        kept = tuple(position for position in range(len(shape)) if position != axis_position)
        rows = tuple(
            (axis, (nm.Integer(0),) * len(kept), index) if position == axis_position
            else unit_row(axis, len(kept), kept.index(position))
            for position, axis in enumerate(shape))
        reads.append(sc.StrideMorphism(
            _dom=tuple(shape[position] for position in kept),
            _cod_stride_shift=rows, name=fd.DynamicName(index.to_latex())))
    return tuple(reads)
