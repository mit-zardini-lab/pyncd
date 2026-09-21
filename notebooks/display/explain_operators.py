# Claude Fable 5.1, effort 80.
'''Wrapping an operator in a block that explains it and is drawn as the operator alone.

An inspection box opens over a block and shows the title, the formula, the description
and the code references the block's aesthetics hold. An operator such as a
`deepseek.data_structure.TopK` has no aesthetics, so it has nothing for a box to show.
The reviewer asked on 2026-09-16 for such an operator to be explained through the
mechanism the blocks already have: the operator is put in an `ops.BlockOperator` whose
block says `cat.BlockDrawing.BODY_IN_PLACE`, tsncd draws the body where the box would
stand, so the figure reads as it did without the block, and the block answers the
pointer. No operator needs a rule of its own in tsncd or in the packaging of the
auxiliary information.

The wrapping is a display pass. `present` is given the term as the transport sends it
and a table from operator class to `OperatorExplanation`, and returns the term with
every operator the table explains wrapped. The model is not edited, so every pass and
every validator that reads it finds the operators where the model wrote them. The
table belongs to the model that is drawn, because a formula and a sentence that suit
one model's use of an operator need not suit another's, and the references name that
model's released code. `notebooks/sota/DeepSeekV41Flash/operator_explanations.py` is
the table of DeepSeek-V4.1-Flash.

A row of the table is one explanation for every operator of a class, or a function
that writes the explanation from the `cat.Broadcasted` that carries the operator and
returns `None` for an operator that needs none. The reviewer asked on 2026-09-17 for the
box over a top-k selection to depend on the form the selection hands its results out
in, and for an elementwise map whose name hides its formula to open a box.
`explain_named_arithmetic` is the function for the second: an `ops.Arithmetic` named
after its formula says everything in the figure, and one given a shorter name, as the
square root of the softplus is named, is explained by the formula the name stands for.

An operator standing as the body of a `ParaWrap` is wrapped as every other is. The
explaining block has the operands and the results of the operator it holds, so the
entries of the wrap stand at the same ports, and tsncd draws the wrap over the operator
with the tapes where they were. The pass left such an operator alone until 2026-09-17,
when the reviewer found that the top-k selection of the Reindex layer, which reads the
candidate pool from the tape, opened no box.

An operator that is written out in its primitives is explained by its expansion
instead, through `algebra.registries.standard_expansions`, and
`notebooks/display/expand_with_parameters.py` draws that expansion with the operator's
parameters on the tape. `obsidian/05-backends/Advanced Display.md` states both routes.
'''
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import data_structure.Category as cat
import data_structure.Numeric as nm
import data_structure.Operators as ops
import data_structure.Term as fd
from notebooks.display.display_wording import TEXT as text


@dataclass(frozen=True)
class OperatorExplanation:
    '''What the inspection box over an operator shows. `title` and `formula` are
    LaTeX, and an explanation with no title takes the operator's own name.'''
    formula: str
    description: str
    title: str | None = None
    references: fd.Prod[cat.CodeReference] = ()


type ExplanationOfOperator = (
    OperatorExplanation | Callable[[cat.Broadcasted], OperatorExplanation | None])
type OperatorExplanations = Mapping[type[cat.Operator], ExplanationOfOperator]


def explanation_for(
    target: cat.Broadcasted, explanations: OperatorExplanations,
) -> OperatorExplanation | None:
    '''The explanation of `target` from the row of the nearest class of its operator
    the table holds, found along the MRO as
    `algebra.registries.standard_expansions.expansion_for` finds a rule, so a subclass
    takes its parent's row unless the table gives it one. A row that is a function is
    given `target`, and `None` from it leaves `target` unexplained.'''
    for kind in type(target.operator).__mro__:
        if kind in explanations:
            row = explanations[kind]
            return row if isinstance(row, OperatorExplanation) else row(target)
    return None


def shows_whole_formula(operator: ops.Arithmetic) -> bool:
    '''Whether the figure already shows everything `operator` computes: its name is
    the LaTeX of its formula, which is the name `ops.Arithmetic` gives itself where
    the model gives none, and the formula holds no function that the primitive
    numerics spell out, as a sigmoid is spelt.'''
    formula = operator.formula
    named_by_formula = (
        operator.name is None
        or operator.name.to_latex() == fd.DynamicName(formula.to_latex()).to_latex())
    return named_by_formula and nm.expand_every_expandable(formula) == formula


def written_formula(operator: ops.Arithmetic) -> str:
    '''The result `y` of `operator` at an input `x`, as the formula the model wrote.
    A function the primitive numerics spell out, such as a sigmoid, a sign or a
    square root, is left as its symbol. The reviewer ruled on 2026-09-19 that the
    spelling belongs in `nm.Expandable.expand_to_primitives` and in no inspection
    box, because every such function is one a reader knows.'''
    return f'y = {operator.formula.to_latex()}'


def explain_named_arithmetic(
    roles: Mapping[str, str],
    references: Mapping[str, fd.Prod[cat.CodeReference]] | None = None,
) -> Callable[[cat.Broadcasted], OperatorExplanation | None]:
    '''The row of the table for `ops.Arithmetic`. An elementwise map whose name is not
    its formula, or whose formula holds a function the primitive numerics spell out,
    is explained by the formula, after the sentence `roles` holds for the text of its
    name. A map the figure shows the whole of is left unexplained. `references` gives
    the released lines of a map by the same text.'''
    def explain(target: cat.Broadcasted) -> OperatorExplanation | None:
        operator = target.operator
        if not isinstance(operator, ops.Arithmetic) or shows_whole_formula(operator):
            return None
        name = operator.name.to_bodies()
        role = roles.get(name)
        applied = (
            text.ELEMENTWISE_MAP_SENTENCE)
        return OperatorExplanation(
            formula=written_formula(operator),
            description=applied if role is None else f'{role} {applied}',
            references=(references or {}).get(name, ()))
    return explain


def drawn_in_place[B: cat.Datatype, A: cat.Axis](
    target: cat.Broadcasted[B, A], explanation: OperatorExplanation,
) -> cat.Broadcasted[B, A]:
    '''`target` as the body of a block that carries `explanation` and is drawn as
    `target` alone. The block has no fill, because nothing of it is drawn.'''
    name = target.operator.name
    title = explanation.title
    if title is None and name is not None:
        title = name.to_latex()
    return ops.BlockOperator.template(cat.Block.template(
        target,
        title=title,
        description=explanation.description,
        fill_color=None,
        references=explanation.references or None,
        formula=explanation.formula,
        drawing=cat.BlockDrawing.BODY_IN_PLACE))


def present[T: fd.GeneralTerm](
    term: T, explanations: OperatorExplanations | None,
) -> T:
    '''`term` with every `cat.Broadcasted` whose operator `explanations` explains
    wrapped by `drawn_in_place`, inside the body of every box as well.

    Each node is rewritten once, by identity, so the sharing of the term is kept and
    an operator the term reaches along two paths is wrapped in one block. The body
    of a `ParaWrap` is wrapped as well, because the block has the ports of the
    operator it holds.
    '''
    if not explanations:
        return term
    rewritten: dict[int, object] = {}

    def write(node: object) -> object:
        if id(node) in rewritten:
            return rewritten[id(node)]
        rebuilt = fd.deep_reconstruct(node, write)
        if isinstance(rebuilt, cat.Broadcasted):
            explanation = explanation_for(rebuilt, explanations)
            if explanation is not None:
                rebuilt = drawn_in_place(rebuilt, explanation)
        rewritten[id(node)] = rebuilt
        return rebuilt

    return write(term)  # type: ignore[return-value]
