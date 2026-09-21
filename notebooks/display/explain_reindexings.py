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
'''
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import algebra.write_index_notation as write_index_notation
import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Term as fd


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


def drawn_in_place[A: cat.Axis](
    reindexing: cat.StrideMorphism[A], explanation: ReindexingExplanation,
) -> cat.Block[A, cat.StrideMorphism[A]]:
    return cat.Block.template(
        reindexing,
        title=reindexing.name.to_latex(),
        description=explanation.description,
        fill_color=None,
        references=explanation.references or None,
        formula=reindexing_formula(reindexing),
        drawing=cat.BlockDrawing.BODY_IN_PLACE)


def present[T: fd.GeneralTerm](
    term: T, explanations: ReindexingExplanations | None,
) -> T:
    '''`term` with every named `cat.StrideMorphism` that `explanations` explains, and
    that stands in the `reindexings` of a `cat.Broadcasted`, wrapped by
    `drawn_in_place`, inside the body of every box as well.

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

    def write(node: object) -> object:
        if id(node) in rewritten:
            return rewritten[id(node)]
        rebuilt = fd.deep_reconstruct(node, write)
        if isinstance(rebuilt, cat.Broadcasted):
            reindexings = tuple(map(explain, rebuilt.reindexings))
            if any(new is not old for new, old in zip(reindexings, rebuilt.reindexings)):
                rebuilt = rebuilt.reconstruct(reindexings=reindexings)
        rewritten[id(node)] = rebuilt
        return rebuilt

    return write(term)  # type: ignore[return-value]
