# Claude Fable 5.1, effort 80.
'''Wrapping a named reindexing in a block that explains it and is drawn as the
reindexing alone.

`notebooks/display/explain_operators.py` explains an operator through a block drawn in
place. The reviewer asked on 2026-09-17 for the same for a reindexing: "We should have
a Block for reindexings, which can also carry hover / explanation information." A
`cat.Block` is generic over the category of its body, so a block whose body is a
`cat.StrideMorphism` is already a morphism of the stride category, and no new term is
needed. `present` replaces each named `cat.StrideMorphism` standing in the
`reindexings` of a `cat.Broadcasted` by such a block, whose aesthetics say
`cat.BlockDrawing.BODY_IN_PLACE`, and tsncd draws the reindexing as it did and opens an
inspection box over it.

The formula of the box is written here from the rows of the reindexing, so a table
gives the sentences alone. A reindexing of a `cat.Broadcasted` maps each position of
the result to the position of the operand it reads, so the split of the distances into
blocks of `|u|` reads

    y[i_{P}, i_{u}] = x[|u| i_{P} + i_{u}]

in the index notation of `algebra.write_index_notation`. Only the axes the reindexing
names appear, and every other axis of the view is carried unchanged.

The wrapping is a display pass, applied to the term that is sent, under
`AdvancedDisplay.INTERACTIVE` alone and after the auxiliary information of the
operators has been assembled. Every pass of the algebra reads a reindexing through its
`mapping` or its rows, which a block does not have, so the model is never edited. A
reindexing held in a field of an operator, as an `aops.CovariantView` holds one, is
left as it stands, because the operator's own box draws it and
`explain_operators` explains the operator.

The table belongs to the model that is drawn and is keyed by the text of the
reindexing's name. `notebooks/sota/DeepSeekV41Flash/operator_explanations.py` holds the
table of DeepSeek-V4.1-Flash.

A view that copies, deletes or permutes axes is written with a `cat.Rearrangement`,
which has no name field, so the name stands on the `ops.View` alone. tsncd draws a
rearrangement as wires with no label, and before 2026-09-26 no box opened over the
repeat of DeepSeek-V4.1-Flash, the repeat of GLM-5.3 or the diagonal of either model,
although the table of GLM-5.3 held a row for its repeat and its diagonal. `present`
therefore writes the rearrangement of a view whose name the table explains as the
stride morphism of the same map, which carries the view's name. The stride morphism
covers the positions the rearrangement moves, copies or deletes, and the trailing
positions it carries unchanged stand beside it as an identity, as the carried axes of a
causal mask do. Where the stride morphism has one row, as a repeat has, tsncd draws it
as a pentagon holding the name, and `present` wraps it as it wraps any named reindexing.
tsncd draws any other stride morphism as a red hexagon whose width does not
fit the name, so a diagonal or a transpose keeps its wires, and the block wraps the
rearrangement itself under the view's name and the formula of the stride morphism.
The model is not edited, because every pass of the algebra reads a rearrangement as
the Cartesian structure of **St**.
'''
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import algebra.write_index_notation as write_index_notation
import construction_helpers.product as chp
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd
import term_utilities.term_utilities as tutil


@dataclass(frozen=True)
class ReindexingExplanation:
    '''What the inspection box over a reindexing says under its formula.'''
    description: str
    references: fd.Prod[cat.CodeReference] = ()


type ReindexingExplanations = Mapping[str, ReindexingExplanation]


def scaled_index(stride: nm.Numeric, index: str) -> str | None:
    '''`index` multiplied by `stride` in LaTeX, and `None` where the stride is zero.'''
    if stride == nm.Integer(0):
        return None
    if stride == nm.Integer(1):
        return index
    if stride == nm.Integer(-1):
        return f'-{index}'
    return rf'{stride.to_latex()}\, {index}'


def position_read(
    strides: fd.Prod[nm.Numeric], shift: nm.Numeric, indices: fd.Prod[str],
) -> str:
    '''The position one row of a reindexing reads, as the sum of its scaled indices
    and its shift.'''
    terms = [term for term in map(scaled_index, strides, indices) if term is not None]
    if shift != nm.Integer(0) or not terms:
        terms.append(shift.to_latex())
    return ' + '.join(terms).replace('+ -', '- ')


def reindexing_formula[A: cat.Axis](reindexing: cat.StrideMorphism[A]) -> str:
    '''The result at each position of the domain of `reindexing` as the operand read
    at the position each row names.'''
    letters = write_index_notation.axis_letters(tuple(reindexing._dom))
    indices = tuple(write_index_notation.index_of(letter) for letter in letters)
    read = ', '.join(position_read(strides, shift, indices)
                     for _, strides, shift in reindexing._cod_stride_shift)
    return f'{write_index_notation.read_at("y", letters)} = x[{read}]'


def block_drawn_in_place[A: cat.Axis, M: cat.Morphism](
    body: M, title: str, formula: str, explanation: ReindexingExplanation,
) -> cat.Block[A, M]:
    return cat.Block.template(
        body,
        title=title,
        description=explanation.description,
        fill_color=None,
        references=explanation.references or None,
        formula=formula,
        drawing=cat.BlockDrawing.BODY_IN_PLACE)


def drawn_in_place[A: cat.Axis](
    reindexing: cat.StrideMorphism[A], explanation: ReindexingExplanation,
) -> cat.Block[A, cat.StrideMorphism[A]]:
    return block_drawn_in_place(
        reindexing, reindexing.name.to_latex(), reindexing_formula(reindexing),
        explanation)


def count_carried_trailing_positions[A: cat.Axis](
    rearrangement: cat.Rearrangement[A],
) -> int:
    '''How many trailing positions of the domain of `rearrangement` the trailing
    positions of its codomain read in order, where no other position of the codomain
    reads one of them.'''
    domain_length = len(rearrangement._dom)
    mapping = rearrangement.mapping
    for carried in range(min(domain_length, len(mapping)), 0, -1):
        leading, trailing = mapping[:-carried], mapping[-carried:]
        if (trailing == tuple(range(domain_length - carried, domain_length))
                and all(read < domain_length - carried for read in leading)):
            return carried
    return 0


def moved_positions_as_stride_morphism[A: cat.Axis](
    rearrangement: cat.Rearrangement[A], name: fd.DynamicName,
) -> cat.StrideMorphism[A] | None:
    '''The stride morphism `name` of the positions `rearrangement` moves, copies or
    deletes, whose rows each read one of those positions at stride one. The trailing
    positions it carries unchanged are left out. `None` for an identity, which moves
    nothing.'''
    carried = count_carried_trailing_positions(rearrangement)
    moved = rearrangement._dom[:len(rearrangement._dom) - carried]
    if not moved:
        return None
    rows = tuple(
        (moved[read],
         tuple(nm.Integer(1) if position == read else nm.Integer(0)
               for position in range(len(moved))),
         nm.Integer(0))
        for read in rearrangement.mapping[:len(rearrangement.mapping) - carried])
    return cat.StrideMorphism(_dom=moved, _cod_stride_shift=rows, name=name)


def presented_rearrangement[A: cat.Axis](
    rearrangement: cat.Rearrangement[A], name: fd.DynamicName,
    explanation: ReindexingExplanation,
) -> cat.StrideCategory[A]:
    '''`rearrangement` as the view named `name` presents it. A stride morphism of one
    row stands beside the identity on the positions carried unchanged, for `present`
    to wrap by its name, and a rearrangement whose stride morphism has no row or
    several rows is itself wrapped under `name`, because tsncd draws a pentagon only for
    a single row.'''
    moved = moved_positions_as_stride_morphism(rearrangement, name)
    if moved is None:
        return rearrangement
    if len(moved._cod_stride_shift) != 1:
        return block_drawn_in_place(
            rearrangement, name.to_latex(), reindexing_formula(moved), explanation)
    carried = rearrangement._dom[len(moved._dom):]
    if not carried:
        return moved
    return chp.morphism_product((moved, cat.ProdObject(carried).identity()))


def is_named_by_its_view_alone(reindexing: object) -> bool:
    '''Whether `reindexing` moves, copies or deletes a position through a
    `cat.Rearrangement` and holds no named stride morphism, so that the name of its
    view is the one name `present` can draw it under. A reindexing that holds a named
    stride morphism opens a box through that name.'''
    return (any(count_carried_trailing_positions(rearrangement) < len(rearrangement._dom)
                for rearrangement in tutil.type_search(cat.Rearrangement, reindexing))
            and not any(stride.name is not None
                        for stride in tutil.type_search(cat.StrideMorphism, reindexing)))


def names_of_unexplained_views(presented: fd.GeneralTerm) -> tuple[str, ...]:
    '''The names of the named views of `presented`, a term `present` has written, whose
    reindexings hold no block drawn in place, so that no box opens over them.'''
    return tuple(sorted({
        node.operator.name.to_bodies()
        for node in tutil.type_search(cat.Broadcasted, presented)
        if isinstance(node.operator, ops.View) and node.operator.name is not None
        and not any(block.aesthetics is not None
                    and block.aesthetics.drawing is cat.BlockDrawing.BODY_IN_PLACE
                    for reindexing in node.reindexings
                    for block in tutil.type_search(cat.Block, reindexing))}))


def explained_view_name(
    node: cat.Broadcasted, explanations: ReindexingExplanations,
) -> fd.DynamicName | None:
    '''The name of `node` where it is a view whose name `explanations` explains.'''
    name = node.operator.name if isinstance(node.operator, ops.View) else None
    return name if name is not None and name.to_bodies() in explanations else None


def with_rearrangements_presented[T: fd.GeneralTerm](
    reindexing: T, name: fd.DynamicName, explanation: ReindexingExplanation,
) -> T:
    '''`reindexing` with every `cat.Rearrangement` in it replaced by
    `presented_rearrangement`.'''
    if isinstance(reindexing, cat.Rearrangement):
        return presented_rearrangement(  # type: ignore[return-value]
            reindexing, name, explanation)
    return fd.deep_reconstruct(
        reindexing,
        lambda node: with_rearrangements_presented(node, name, explanation))


def present[T: fd.GeneralTerm](
    term: T, explanations: ReindexingExplanations | None,
) -> T:
    '''`term` with every named `cat.StrideMorphism` that `explanations` explains, and
    that stands in the `reindexings` of a `cat.Broadcasted`, wrapped by
    `drawn_in_place`, inside the body of every box as well. A rearrangement in the
    reindexings of a view whose name `explanations` explains is first written as the
    stride morphism of that name, or wrapped under it, by
    `with_rearrangements_presented`.

    Each node is rewritten once, by identity, so the sharing of the term is kept.
    '''
    if not explanations:
        return term
    rewritten: dict[int, object] = {}

    def explain(node: object) -> object:
        rebuilt = fd.deep_reconstruct(node, explain)
        if isinstance(rebuilt, cat.StrideMorphism) and rebuilt.name is not None:
            explanation = explanations.get(rebuilt.name.to_bodies())
            if explanation is not None:
                return drawn_in_place(rebuilt, explanation)
        return rebuilt

    def presented_and_explained(
        reindexing: cat.StrideCategory, view_name: fd.DynamicName | None,
    ) -> object:
        if view_name is None or not is_named_by_its_view_alone(reindexing):
            return explain(reindexing)
        return explain(with_rearrangements_presented(
            reindexing, view_name, explanations[view_name.to_bodies()]))

    def write(node: object) -> object:
        if id(node) in rewritten:
            return rewritten[id(node)]
        rebuilt = fd.deep_reconstruct(node, write)
        if isinstance(rebuilt, cat.Broadcasted):
            view_name = explained_view_name(rebuilt, explanations)
            reindexings = tuple(presented_and_explained(reindexing, view_name)
                                for reindexing in rebuilt.reindexings)
            if any(new is not old for new, old in zip(reindexings, rebuilt.reindexings)):
                rebuilt = rebuilt.reconstruct(reindexings=reindexings)
        rewritten[id(node)] = rebuilt
        return rebuilt

    return write(term)  # type: ignore[return-value]
