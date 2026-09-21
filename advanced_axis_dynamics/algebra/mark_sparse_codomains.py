'''Marking the codomain of a stride morphism read covariantly.

Written by Claude Opus 5 (1M context), reasoning effort medium.

Read covariantly, as `aops.CovariantView` reads it, a stride morphism writes the input
at each live domain position to the codomain position its rows compute, and a codomain
position no live domain position writes holds the universal unit. `merge_groups` tests
that the read is a function on positions, by finding an order of the rows in which each
row writes its own domain axes as a signed mixed-radix number beside the axes the rows
before it determine. `mark_sparse_codomain` then writes every constraint on the domain,
the range of each group and the form of each domain guard, over the codomain, and marks
the codomain axes the image does not fill. `merge_carrying` lets the axes a merge is
broadcast over enter that marking, so a slot axis guided by an axis the merge consumes
leaves guided by the axis the merge produces.

`obsidian/02-categories/Advanced Axis Dynamics.md` states the feature, and
`obsidian/02-categories/Padding and Masks as Sparse Axes.md` the reachability the merge
of the relative indexer derives.
'''
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import data_structure.Term as fd # for 'foundations'
import data_structure.Numeric as nm
import data_structure.StrideCategory as sc
import advanced_axis_dynamics.data_structure.AffineGuards as AffineGuards
import advanced_axis_dynamics.algebra.mark_sparse_domains as mark_sparse_domains


class GuardNotAffineAfterMerge(Exception):
    '''An `AffineGuards.AffineSparseAxis` whose guides a mixed-radix merge consumes,
    where no affine form over the merged axis has the same live positions.'''


class NotAMixedRadixSplit(Exception):
    '''A stride morphism that is not an injection from its domain box into its codomain
    box when read covariantly: no order of its rows writes each row's own domain axes
    as a signed mixed-radix number beside axes the rows before it already determine.'''


@dataclass(frozen=True)
class MergeGroup:
    '''The domain axes one codomain row of a merge writes as a signed mixed-radix
    number, from the finest digit to the coarsest.

    `row` indexes the codomain row. `digits` are the domain positions of the axes the
    row writes, ordered so that the radix of each is the product of the sizes of the
    digits before it, `radices` are those products and `signs` the sign of each stride,
    so the row reads digit `d` at `signs[d] * radices[d]`. The row may read the digits
    of earlier groups at any stride. The row `i_B = |u| i_P + i_u` has the digits
    `(u, P)` with the radices `(1, |u|)`, and `i_b = i_{b_0} - i_r` beside a row that
    writes `b_0` has the one digit `r` with the sign `-1`.
    '''
    row: int
    digits: fd.Prod[int]
    radices: fd.Prod[nm.Numeric]
    signs: fd.Prod[int]

    def box(self, dom: fd.Prod[sc.Axis]) -> tuple[nm.Numeric, nm.Numeric]:
        '''The lowest and the highest value the digits' part of the row takes over
        the domain box.'''
        strides = tuple(signed_radix(sign, radix)
                        for sign, radix in zip(self.signs, self.radices))
        axes = tuple(dom[d] for d in self.digits)
        return (AffineGuards.lowest_value(strides, nm.Integer(0), axes),
                AffineGuards.highest_value(strides, nm.Integer(0), axes))

    def size(self, dom: fd.Prod[sc.Axis]) -> nm.Numeric:
        return nm.Multiplication.template(*(dom[d].local_size() for d in self.digits))


def signed_radix(sign: int, radix: nm.Numeric) -> nm.Numeric:
    return nm.Multiplication.template(nm.Integer(sign), radix)


def signed_mixed_radix(
    positions: Sequence[int], strides: fd.Prod[nm.Numeric], dom: fd.Prod[sc.Axis]
) -> tuple[fd.Prod[int], fd.Prod[nm.Numeric], fd.Prod[int]] | None:
    '''`positions` ordered from the finest digit to the coarsest, with the radices
    and the signs, where the strides at `positions` are plus or minus a mixed radix
    over the axes there, and `None` where they are not.'''
    remaining = list(positions)
    ordered: list[int] = []
    radices: list[nm.Numeric] = []
    signs: list[int] = []
    radix: nm.Numeric = nm.Integer(1)
    while remaining:
        matches = [(position, sign) for position in remaining for sign in (1, -1)
                   if strides[position] == signed_radix(sign, radix)]
        if len(matches) != 1:
            return None
        position, sign = matches[0]
        remaining.remove(position)
        ordered.append(position)
        radices.append(radix)
        signs.append(sign)
        radix = nm.Multiplication.template(radix, dom[position].local_size())
    return tuple(ordered), tuple(radices), tuple(signs)


def merge_groups[A: sc.Axis](morphism: sc.StrideMorphism[A]) -> fd.Prod[MergeGroup]:
    '''The groups of `morphism` read covariantly, in an order where every row reads,
    beside its own digits, only digits of the groups before it.

    Given the codomain position, the first row's digits are the signed mixed-radix
    digits of its value less its shift, the second row's digits follow once the first
    group's are subtracted, and so on, so a morphism with such an order is an injection
    from its domain box into its codomain box, and the covariant read of it is a
    function on positions. A mixed-radix split with one row, `i_B = |u| i_P + i_u`, is
    the case with one group. Raises `NotAMixedRadixSplit` where no order exists.
    '''
    dom = tuple(morphism._dom)
    rows = morphism._cod_stride_shift
    unassigned = set(range(len(dom)))
    remaining_rows = list(range(len(rows)))
    groups: list[MergeGroup] = []
    while remaining_rows:
        for row in remaining_rows:
            _, strides, _ = rows[row]
            own = [position for position in sorted(unassigned)
                   if not nm.is_zero(strides[position])]
            if not own:
                continue
            ordered = signed_mixed_radix(own, strides, dom)
            if ordered is not None:
                digits, radices, signs = ordered
                groups.append(MergeGroup(row=row, digits=digits, radices=radices,
                                         signs=signs))
                remaining_rows.remove(row)
                unassigned -= set(digits)
                break
        else:
            raise NotAMixedRadixSplit(
                f'the rows {[rows[row][0] for row in remaining_rows]} of {morphism} '
                f'write no signed mixed-radix number over the axes '
                f'{[dom[position] for position in sorted(unassigned)]}')
    if unassigned:
        raise NotAMixedRadixSplit(
            f'{[dom[position] for position in sorted(unassigned)]} are read by no row '
            f'of {morphism}')
    return tuple(groups)


@dataclass(frozen=True)
class AffineConstraint:
    '''An affine form over a fixed tuple of axes that is at least zero at every live
    position, as `strides` over the axes in order and a `shift`.'''
    strides: fd.Prod[nm.Numeric]
    shift: nm.Numeric

    def collected(self) -> AffineConstraint:
        return AffineConstraint(
            tuple(nm.collect_like_terms(stride) for stride in self.strides),
            nm.collect_like_terms(self.shift))

    def holds_everywhere(self, axes: fd.Prod[sc.Axis]) -> bool:
        return not AffineGuards.reads_before_start(self.strides, self.shift, axes)

    def scaled(self, factor: nm.Numeric) -> AffineConstraint:
        return AffineConstraint(
            tuple(nm.Multiplication.template(factor, stride)
                  for stride in self.strides),
            nm.Multiplication.template(factor, self.shift))

    def plus(self, other: AffineConstraint) -> AffineConstraint:
        return AffineConstraint(
            tuple(nm.Addition.template(a, b)
                  for a, b in zip(self.strides, other.strides)),
            nm.Addition.template(self.shift, other.shift))

    def implies(self, other: AffineConstraint, axes: fd.Prod[sc.Axis]) -> bool:
        '''Whether `other` is at least zero wherever this constraint is, because
        their difference is at least zero over the whole box.'''
        difference = other.plus(self.scaled(nm.Integer(-1))).collected()
        return difference.holds_everywhere(axes)


def constant_constraint(axes_count: int, shift: nm.Numeric) -> AffineConstraint:
    return AffineConstraint((nm.Integer(0),) * axes_count, shift)


type MergeRows = fd.Prod[tuple[sc.Axis, fd.Prod[nm.Numeric], nm.Numeric]]


def mark_sparse_codomain[A: sc.Axis](
    morphism: sc.StrideMorphism[A]) -> sc.StrideMorphism[A]:
    '''`morphism`, read covariantly, with every codomain axis some of whose positions
    receive no value replaced by the `AffineGuards.AffineSparseAxis` stating which
    positions do.

    The covariant read writes the input at each live domain position to the codomain
    position the rows compute, and a codomain position no live domain position writes
    holds the unit. The live codomain positions are the image of the domain box, less
    the positions the domain's own `AffineGuards.AffineSparseAxis`es empty. Each group
    of `merge_groups` writes its digits' part of its row onto an interval, so the row's
    value less its shift and less the digits of earlier groups lies in that interval,
    and each domain sparse axis holds its form in its range. Every one of those
    constraints is written over the codomain by eliminating the domain axes it reads. A
    constraint reading a group's digits in proportion to their signed radices reads a
    multiple of the group's part of its row, which is exact. A constraint reading only
    the coarsest digit, at plus or minus one, reads a floor of that part, and
    `floor(y) >= m` is `y >= m` for an integer `m`, which is how a group's number
    reaches the query axis it is merged into. A range constraint of any other shape
    raises `GuardNotAffineAfterMerge`, because the image of the box is then no affine
    sparse axis. A domain axis's own form of any other shape is dropped and the
    codomain axis it would have marked stays dense, which is the candidate axis `C`
    of the V4.1 notebook: the kept blocks `p|x` are live where `i_x - |u| i_p >= 0`,
    the layout `i_C = |u| i_p + i_u` reads the block at the stride `|u|`, and the empty
    candidates of a query are no affine form of a candidate's position. A codomain axis
    that already states the form derived for it is kept, so marking is idempotent, and
    the axes a merge is broadcast over enter through `merge_carrying`.

    A constraint that holds over the whole box is dropped, before and after the
    elimination, and a constraint another one implies is dropped. The box gives each
    codomain axis the size of the group that writes it, as the split `View` sizes its
    codomain, so the axis `x` merged from `(b_0, a)` is tested at the size `|a| |b_0|`.
    What remains marks the last codomain axis it reads, guided by the others. Two
    constraints on one axis whose sum is a constant are one axis with an extent. The
    merge `(b_0, a, r|b_0) -> (x, b)` with `i_x = |a| i_{b_0} + i_a + |a| - 1` and
    `i_b = i_{b_0} - i_r` marks `b|x`, live where `i_x - |a| i_b - (|a| - 1) >= 0`,
    which is the reachability of entry `b` from query `x` at compression ratio `|a|`,
    and leaves `x` dense because the form on `b` implies the one on `x`. A morphism
    that marks nothing is returned as the same object.
    '''
    dom = tuple(morphism._dom)
    rows = morphism._cod_stride_shift
    cod = tuple(axis for axis, _, _ in rows)
    groups = merge_groups(morphism)
    guides = external_guides(dom, cod)
    offset = len(guides)
    cod_offset = offset + len(dom)
    axes_count = cod_offset + len(cod)
    boxed = (*guides, *dom, *codomain_sized_by_groups(cod, groups, dom))

    constraints = [
        *((constraint, True) for constraint
          in range_constraints(rows, groups, dom, axes_count, offset, cod_offset)),
        *((constraint, False) for constraint
          in domain_constraints(dom, (*guides, *dom, *cod), axes_count, offset))]
    eliminated: list[AffineConstraint] = []
    for constraint, states_the_image in constraints:
        constraint = constraint.collected()
        if constraint.holds_everywhere(boxed):
            continue
        try:
            constraint = eliminate_domain_axes(
                constraint, rows, groups, dom, offset, cod_offset).collected()
        except GuardNotAffineAfterMerge:
            if states_the_image:
                raise
            continue
        if constraint.holds_everywhere(boxed):
            continue
        if constraint not in eliminated:
            eliminated.append(constraint)
    kept = [constraint for constraint in eliminated
            if not any(other is not constraint and other.implies(constraint, boxed)
                       for other in eliminated)]

    marks: dict[int, list[AffineConstraint]] = {}
    for constraint in kept:
        position = AffineGuards.marked_position(constraint.strides)
        if position < cod_offset:
            raise AffineGuards.GuardReadsNoAxis(
                f'{morphism} leaves the constraint {constraint} on no codomain axis')
        marks.setdefault(position - cod_offset, []).append(constraint)
    replacements: dict[int, AffineGuards.AffineSparseAxis] = {}
    for position, forms in marks.items():
        sparse = sparse_axis_for_constraints(
            [without_domain_axes(form, offset, cod_offset) for form in forms],
            (*guides, *cod), offset, cod[position])
        if states_same_form(cod[position], sparse):
            continue
        if sparse.crosses_start() or sparse.crosses_end():
            replacements[position] = sparse
    if not replacements:
        return morphism
    return morphism.reconstruct(_cod_stride_shift=tuple(
        (replacements.get(position, axis), strides, shift)
        for position, (axis, strides, shift) in enumerate(rows)))


def without_domain_axes(constraint: AffineConstraint, offset: int,
                        cod_offset: int) -> AffineConstraint:
    '''`constraint` over the guides and the codomain alone, once the domain axes have
    been eliminated and their strides are zero.'''
    return AffineConstraint(
        (*constraint.strides[:offset], *constraint.strides[cod_offset:]),
        constraint.shift)


def external_guides(dom: fd.Prod[sc.Axis], cod: fd.Prod[sc.Axis]) -> fd.Prod[sc.Axis]:
    '''The guides of the sparse axes in `dom` that stand neither in `dom` nor in
    `cod`.'''
    guides: list[sc.Axis] = []
    for axis in dom:
        if isinstance(axis, AffineGuards.AffineSparseAxis):
            for guide in axis.guides:
                if not any(guide == member for member in (*dom, *cod, *guides)):
                    guides.append(guide)
    return tuple(guides)


def states_same_form(axis: sc.Axis, sparse: AffineGuards.AffineSparseAxis) -> bool:
    '''Whether `axis` is already the sparse axis `sparse` states, so that marking it
    again would replace it by an equal axis under a new identity.'''
    if not isinstance(axis, AffineGuards.AffineSparseAxis):
        return False
    guides = tuple(nm.FreeNumeric() for _ in sparse.guides)
    position = nm.FreeNumeric()
    return (axis.guides == sparse.guides
            and axis.extent == sparse.extent
            and axis.guard_form(guides, position)
            == sparse.guard_form(guides, position))


def merge_carrying[A: sc.Axis](morphism: sc.StrideMorphism[A],
                               carried: fd.Prod[sc.Axis]) -> sc.StrideMorphism[A]:
    '''`morphism` with a unit-stride row for every axis of `carried`, so that the axes
    a merge is broadcast over enter its marking.

    A carried `AffineGuards.AffineSparseAxis` guided by an axis the merge consumes is
    written onto a fresh axis of its own body and size, which the marking then guides
    by the codomain: the slots `r|b_0` carried past the merge of `(b_0, a)` into `x`
    come out as `r|x`. Every other carried axis is written onto itself.
    '''
    dom = tuple(morphism._dom)
    width = len(dom) + len(carried)
    rows = [(axis, (*strides, *(nm.Integer(0),) * len(carried)), shift)
            for axis, strides, shift in morphism._cod_stride_shift]
    for position, axis in enumerate(carried):
        strides = [nm.Integer(0)] * width
        strides[len(dom) + position] = nm.Integer(1)
        reguided = isinstance(axis, AffineGuards.AffineSparseAxis) and any(
            guide == domain_axis for guide in axis.guides for domain_axis in dom)
        body = fd.DynamicName.from_str(AffineGuards.axis_body(axis).split('|')[0])
        target = (body.capture(sc.RawAxis(_size=axis.local_size()))
                  if reguided else axis)
        rows.append((target, tuple(strides), nm.Integer(0)))
    return morphism.reconstruct(_dom=(*dom, *carried), _cod_stride_shift=tuple(rows))


def codomain_sized_by_groups(cod: fd.Prod[sc.Axis], groups: fd.Prod[MergeGroup],
                             dom: fd.Prod[sc.Axis]) -> fd.Prod[sc.Axis]:
    '''`cod` with every axis a group writes replaced by an axis of that group's size,
    for testing a constraint over the box a merge writes.'''
    sizes = {group.row: group.size(dom) for group in groups}
    return tuple(sc.RawAxis(_size=sizes[row]) if row in sizes else axis
                 for row, axis in enumerate(cod))


def range_constraints(rows: MergeRows, groups: fd.Prod[MergeGroup],
                      dom: fd.Prod[sc.Axis], axes_count: int, offset: int,
                      cod_offset: int) -> list[AffineConstraint]:
    '''For every group, the two constraints that its row's value, less its shift and
    less what the row reads of other groups, lies between the lowest and the highest
    value its own digits take.'''
    constraints = []
    for group in groups:
        part = group_part_of_row(rows, group, axes_count, offset, cod_offset)
        lowest, highest = group.box(dom)
        constraints.append(part.plus(constant_constraint(
            axes_count, nm.Multiplication.template(nm.Integer(-1), lowest))))
        constraints.append(part.scaled(nm.Integer(-1)).plus(
            constant_constraint(axes_count, highest)))
    return constraints


def group_part_of_row(rows: MergeRows, group: MergeGroup, axes_count: int,
                      offset: int, cod_offset: int) -> AffineConstraint:
    '''The row's codomain axis less the row's shift and less the domain axes the row
    reads outside `group`, which equals the signed mixed-radix number the group's
    digits form.'''
    _, strides, shift = rows[group.row]
    form = [nm.Integer(0)] * axes_count
    form[cod_offset + group.row] = nm.Integer(1)
    for position, stride in enumerate(strides):
        if position not in group.digits and not nm.is_zero(stride):
            form[offset + position] = nm.Multiplication.template(nm.Integer(-1), stride)
    return AffineConstraint(
        tuple(form), nm.Multiplication.template(nm.Integer(-1), shift))


def domain_constraints(dom: fd.Prod[sc.Axis], members: fd.Prod[sc.Axis],
                       axes_count: int, offset: int) -> list[AffineConstraint]:
    '''For every sparse axis of `dom`, its form at least zero, and below its extent
    where it has one, over `members`, the guides, the domain and the codomain in
    the order the constraints index them.'''
    constraints = []
    for position, axis in enumerate(dom):
        if not isinstance(axis, AffineGuards.AffineSparseAxis):
            continue
        strides = [nm.Integer(0)] * axes_count
        strides[offset + position] = axis.stride
        for guide, stride in zip(axis.guides, axis.guide_strides):
            where = next(i for i, member in enumerate(members) if member == guide)
            strides[where] = nm.Addition.template(strides[where], stride)
        form = AffineConstraint(tuple(strides), axis.shift)
        constraints.append(form)
        if axis.extent is not None:
            constraints.append(form.scaled(nm.Integer(-1)).plus(constant_constraint(
                axes_count, nm.Addition.template(axis.extent, nm.Integer(-1)))))
    return constraints


def eliminate_domain_axes(constraint: AffineConstraint, rows: MergeRows,
                          groups: fd.Prod[MergeGroup], dom: fd.Prod[sc.Axis],
                          offset: int, cod_offset: int) -> AffineConstraint:
    '''`constraint` written over the guides and the codomain alone, eliminating the
    domain axes it reads from the latest group backwards, since eliminating a group's
    digits introduces the digits of earlier groups only.'''
    axes_count = len(constraint.strides)
    while True:
        read = {position for position in range(len(dom))
                if not nm.is_zero(constraint.strides[offset + position])}
        if not read:
            return constraint
        latest = max(index for index, group in enumerate(groups)
                     if set(group.digits) & read)
        group = groups[latest]
        part = group_part_of_row(rows, group, axes_count, offset, cod_offset)
        constraint = eliminate_group(constraint, group, part, dom, offset)


def eliminate_group(constraint: AffineConstraint, group: MergeGroup,
                    part: AffineConstraint, dom: fd.Prod[sc.Axis], offset: int
                    ) -> AffineConstraint:
    '''`constraint` with the digits of `group` replaced by `part`, the group's number
    written over the codomain and the earlier groups.'''
    coefficients = tuple(nm.collect_like_terms(constraint.strides[offset + digit])
                         for digit in group.digits)
    rest = AffineConstraint(
        tuple(nm.Integer(0) if position - offset in group.digits else stride
              for position, stride in enumerate(constraint.strides)),
        constraint.shift)
    factor = signed_radix(group.signs[0], coefficients[0])
    proportional = all(
        nm.is_zero(nm.collect_like_terms(nm.Addition.template(
            coefficient,
            nm.Multiplication.template(
                nm.Integer(-1), factor, signed_radix(sign, radix)))))
        for coefficient, sign, radix in zip(coefficients, group.signs, group.radices))
    if proportional:
        return rest.plus(part.scaled(factor))
    coarsest = coefficients[-1]
    reads_coarsest_alone = (
        all(nm.is_zero(coefficient) for coefficient in coefficients[:-1])
        and coarsest in (nm.Integer(1), nm.Integer(-1)))
    if not reads_coarsest_alone:
        raise GuardNotAffineAfterMerge(
            f'a constraint reads the digits {[dom[d] for d in group.digits]} at '
            f'{[coefficient.to_latex() for coefficient in coefficients]}, which is '
            f'neither a multiple of their radices nor the coarsest digit alone')
    radix = group.radices[-1]
    lower_strides = tuple(signed_radix(sign, lower_radix) for sign, lower_radix
                          in zip(group.signs[:-1], group.radices[:-1]))
    low = AffineGuards.lowest_value(lower_strides, nm.Integer(0),
                       tuple(dom[d] for d in group.digits[:-1]))
    direction = group.signs[-1] * (1 if coarsest == nm.Integer(1) else -1)
    scaled_rest = rest.scaled(radix)
    axes_count = len(constraint.strides)
    if direction == 1:
        return scaled_rest.plus(part).plus(constant_constraint(
            axes_count, nm.Multiplication.template(nm.Integer(-1), low)))
    return scaled_rest.plus(part.scaled(nm.Integer(-1))).plus(constant_constraint(
        axes_count, nm.Addition.template(radix, nm.Integer(-1), low)))


def sparse_axis_for_constraints(
    forms: Sequence[AffineConstraint], axes: fd.Prod[sc.Axis], guides_count: int,
    marked: sc.Axis) -> AffineGuards.AffineSparseAxis:
    '''The sparse axis that `forms`, each at least zero on the live positions, state
    on `marked`: one form gives the axis with no extent, and two whose sum is a
    constant give the axis live between the one with the positive stride and that
    constant.'''
    if len(forms) == 1:
        form, = forms
        extent = None
    elif len(forms) == 2:
        total = forms[0].plus(forms[1]).collected()
        if not all(nm.is_zero(stride) for stride in total.strides):
            raise mark_sparse_domains.AxisMarkedTwice(
                f'{marked} is marked by two constraints of a merge, {forms[0]} and '
                f'{forms[1]}, and one axis carries one affine form')
        position = AffineGuards.marked_position(forms[0].strides)
        leading = forms[0].strides[position]
        form = (forms[0] if nm.is_positive_for_positive_symbols(leading)
                else forms[1])
        extent = nm.Addition.template(total.shift, nm.Integer(1))
    else:
        raise mark_sparse_domains.AxisMarkedTwice(
            f'{marked} is marked by {len(forms)} constraints of a merge')
    position, sparse = AffineGuards.sparse_axis_for_row(
        form.strides, form.shift, axes, extent)
    if position < guides_count:
        raise AffineGuards.GuardReadsNoAxis(
            f'a merge leaves {form} on the guide {axes[position]}')
    return sparse
