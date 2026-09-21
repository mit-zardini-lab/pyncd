# Claude Fable 5.1, effort 80.
'''What an inspection box shows over an operator of the integrated DeepSeek-V4.1-Flash.

The base model's tables in `notebooks.sota.DeepSeekV41Flash.operator_explanations`
explain the operators the base model holds. The integrated model adds the rotations,
the quantised caches, the score scales, the router temperature, the epsilons, Engram,
the image pathway and DSpark, and each part module exports the rows its operators
need. The five tables here join the base tables with the part tables, and
`join_tables` raises where two tables give one key two different rows, so a role
written in one module is never replaced by another's without notice.

`OPERATOR_EXPLANATIONS` differs from a plain union in four rows. The row for
`ops.GenericOperator` is a function that chooses an explanation by the text of the
operator's name, because one class stands for every operation the package cannot
state. Four generic operators remain in the whole model: the vision encoder, the draft
trunk of DSpark, the exponential draw of the sampler and the ceiling of the quantised
caches. Each row gives a one-line formula in the repository's index notation, a
description a practitioner can follow and links pinned to the released lines. The
n-gram hash of each Engram layer was a fifth until 2026-09-19, and is now written out
over the integer operators. The row for `ops.Embedding` chooses by name as well,
because the base row describes the token embedding and the model adds the token map,
the two n-gram tables and the Markov embedding. The row for
`aops.ConcatenateAxes` asks `rotary_embedding.explain_channel_join` first, for the
join that closes a rotation box, and takes the base row for every other
concatenation. The row for `ops.ConstantOp` chooses by the constant's value, which is
one for the arrays the image pathway injects and infinity for the pin.

`dst.Rotary` and `dst.YarnRotary` have no row. Their standard expansion in
`deepseek.registries.standard_expansions` supplies the box, with the table written
out in primitives, and an operator takes one route. `OPERATOR_ROLES` gives the two
tables a sentence and a link each, from `rotary_embedding.TABLE_ROLES`, which the
expansion box puts before the generated description.

`ARITHMETIC_ROLES` holds exactly the named elementwise maps of the whole model that
hide part of their formula, so the check that the wrapped maps equal its keys holds
as it does for the base model. A part module that replaces a mechanism of the base
model names the base rows its own mechanism leaves unused, and `without_rows` drops
them from the joined table. `list_names_missing_a_row` walks a term and returns
every name the tables give no row, and every role no map of the term uses, for the
validator to assert empty.

`RELEASED_CONSTANT_ROWS` is one markdown table row per named constant of the model,
with its released value, its meaning and the released line that sets it, for the
notebook to print under the figures that show the constants in their formulas.

Every line number was read from the released files at the commit
`reference_links.COMMIT` on 2026-09-17.
'''
from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import advanced_axis_dynamics.data_structure.Operators as aops
import data_structure.Category as cat
import data_structure.Operators as ops
import data_structure.Term as fd
import deepseek.data_structure as dst
import para.data_structure.inject as inject
import quantization.data_structure.Quantization as Quantization
import term_utilities.term_utilities as tutil
from websocket_transfer.auxiliary_information import OperatorRole

import notebooks.display.explain_operators as explain_operators
import notebooks.display.notebook_diagrams as notebook_diagrams
import notebooks.sota.DeepSeekV41Flash.omitted_mechanisms as omitted_mechanisms
import notebooks.sota.DeepSeekV41Flash.operator_explanations as operator_explanations
from notebooks.sota.DeepSeekV41Flash import (
    clamped_mixture_of_experts, dspark_draft_chain, gumbel_max_sampler,
    mhc_with_epsilons, pinned_candidate_pool, quantised_caches, rotary_embedding,
    rotated_indexer, scaled_attention_core, vision_pathway, write_at_token_positions)
from notebooks.sota.DeepSeekV41Flash import (
    engram_modules, integrated_whole_model, text_only_engram, text_only_model)
from notebooks.display.explain_operators import (
    ExplanationOfOperator, OperatorExplanation)
from notebooks.display.explain_reindexings import ReindexingExplanation
from notebooks.sota.DeepSeekV41Flash.reference_links import (
    engram_lines, inference_config_lines, model_lines)
from notebooks.sota.DeepSeekV41Flash.released_constants import ReleasedConstant
from notebooks.sota.DeepSeekV41Flash.block_titles_and_descriptions import TEXT as text

NORMALISATION_NAME = 'RMSNorm'
LOOKBACK_VIEW_NAME = omitted_mechanisms.LOOKBACK.name.to_bodies()


class TablesDisagreeOnAKey(ValueError):
    '''Two tables being joined give one key two different rows.'''


def join_tables[K, V](*tables: Mapping[K, V]) -> dict[K, V]:
    '''The union of `tables`. A key two tables share must carry the same row in both,
    so that a row written in one part module is never replaced by another's without
    notice.'''
    joined: dict[K, V] = {}
    for table in tables:
        for key, row in table.items():
            if key in joined and joined[key] != row:
                raise TablesDisagreeOnAKey(
                    f'{key!r} has the rows {joined[key]!r} and {row!r}')
            joined[key] = row
    return joined


def without_rows[K, V](table: Mapping[K, V], keys: Iterable[K]) -> dict[K, V]:
    '''`table` without `keys`. A part module of this model that replaces a mechanism of
    the base model names the rows of the base table its own mechanism leaves unused, and
    the joined table drops them, so the check that every row is used holds here as it
    does for the base model. A key the table does not hold is an error, because the row
    it named has moved or gone.'''
    missing = tuple(key for key in keys if key not in table)
    if missing:
        raise KeyError(f'{missing} is not a row of the table it is dropped from')
    dropped = set(keys)
    return {key: row for key, row in table.items() if key not in dropped}


table_key = operator_explanations.table_key


def embedding_key(subscript: str) -> str:
    '''The key of an `ops.Embedding` named `subscript`, whose `template` puts the name
    under the body `E`.'''
    return f'E{table_key(subscript)}'


explanation_from_row = operator_explanations.explanation_from_row


# The generic operators.

GENERIC_OPERATOR_EXPLANATIONS: dict[str, OperatorExplanation] = {
    quantised_caches.CEILING: quantised_caches.CEILING_EXPLANATION,
    **vision_pathway.GENERIC_OPERATOR_EXPLANATIONS,
    **dspark_draft_chain.GENERIC_OPERATOR_EXPLANATIONS,
    **gumbel_max_sampler.GENERIC_OPERATOR_EXPLANATIONS,
}


def explain_generic_operator(target: cat.Broadcasted) -> OperatorExplanation | None:
    '''The row for `ops.GenericOperator`, chosen by the text of the operator's name. A
    generic operator the table does not name is left unexplained.'''
    name = target.operator.name
    return None if name is None else GENERIC_OPERATOR_EXPLANATIONS.get(name.to_bodies())


# The lookups.

EMBEDDING_EXPLANATIONS: dict[str, ExplanationOfOperator] = {
    table_key('E'): operator_explanations.OPERATOR_EXPLANATIONS[ops.Embedding],
    **omitted_mechanisms.ENGRAM_EMBEDDING_EXPLANATIONS,
    **dspark_draft_chain.EMBEDDING_EXPLANATIONS,
}


def explain_embedding(target: cat.Broadcasted) -> OperatorExplanation | None:
    '''The row for `ops.Embedding`, chosen by the text of the operator's name, because
    the model holds the token embedding, the token map, two n-gram tables and the
    Markov embedding, and one sentence does not describe them all.'''
    name = target.operator.name
    if name is None:
        return None
    row = EMBEDDING_EXPLANATIONS.get(name.to_bodies())
    return None if row is None else explanation_from_row(row, target)


# The concatenation, the constants and the injection.

def explain_concatenation(target: cat.Broadcasted) -> OperatorExplanation | None:
    '''The row for `aops.ConcatenateAxes`. The join that closes a rotation box, onto
    the declared channel axis, takes `rotary_embedding.explain_channel_join`, and every
    other concatenation takes the base row.'''
    channel_join = rotary_embedding.explain_channel_join(target)
    if channel_join is not None:
        return channel_join
    return explanation_from_row(
        operator_explanations.OPERATOR_EXPLANATIONS[aops.ConcatenateAxes], target)


POSITIVE_INFINITY_EXPLANATION = operator_explanations.POSITIVE_INFINITY_EXPLANATION


# The elementwise maps that hide their formula.

ENGRAM_ARITHMETIC_ROLES: dict[str, str] = {
    omitted_mechanisms.GATE_NAME: (
        text.ENGRAM_GATE_ROLE),
}

ENGRAM_ARITHMETIC_REFERENCES: dict[str, fd.Prod[cat.CodeReference]] = {
    omitted_mechanisms.GATE_NAME: (model_lines(341), model_lines(360, 362)),
}

ARITHMETIC_ROLES: dict[str, str] = without_rows(join_tables(
    operator_explanations.ARITHMETIC_ROLES,
    scaled_attention_core.ARITHMETIC_ROLES,
    rotated_indexer.ARITHMETIC_ROLES,
    quantised_caches.ARITHMETIC_ROLES,
    clamped_mixture_of_experts.ARITHMETIC_ROLES,
    mhc_with_epsilons.ARITHMETIC_ROLES,
    vision_pathway.ARITHMETIC_ROLES,
    ENGRAM_ARITHMETIC_ROLES), mhc_with_epsilons.REPLACED_ARITHMETIC_ROLES)

ARITHMETIC_REFERENCES: dict[str, fd.Prod[cat.CodeReference]] = without_rows(join_tables(
    operator_explanations.ARITHMETIC_REFERENCES,
    scaled_attention_core.ARITHMETIC_REFERENCES,
    rotated_indexer.ARITHMETIC_REFERENCES,
    quantised_caches.ARITHMETIC_REFERENCES,
    clamped_mixture_of_experts.ARITHMETIC_REFERENCES,
    mhc_with_epsilons.ARITHMETIC_REFERENCES,
    vision_pathway.ARITHMETIC_REFERENCES,
    ENGRAM_ARITHMETIC_REFERENCES),
    mhc_with_epsilons.REPLACED_ARITHMETIC_ROLES)


# The five tables.

OPERATOR_EXPLANATIONS: dict[type[cat.Operator], ExplanationOfOperator] = {
    **operator_explanations.OPERATOR_EXPLANATIONS,
    **rotary_embedding.OPERATOR_EXPLANATIONS,
    ops.Arithmetic: explain_operators.explain_named_arithmetic(
        ARITHMETIC_ROLES, ARITHMETIC_REFERENCES),
    ops.Embedding: explain_embedding,
    aops.ConcatenateAxes: explain_concatenation,
    ops.GenericOperator: explain_generic_operator,
    ops.ConstantOp: operator_explanations.explain_omitted_constant,
    Quantization.TypeConvert: quantised_caches.explain_cast,
    inject.Inject: write_at_token_positions.INJECT_EXPLANATION,
}

OPERATOR_REFERENCES: dict[type[cat.Operator], fd.Prod[cat.CodeReference]] = join_tables(
    operator_explanations.OPERATOR_REFERENCES,
    {dst.Rotary: (model_lines(369, 389), model_lines(392, 406))})


MIXTURE_ROLES: dict[str, OperatorRole] = {
    key: OperatorRole(
        role=role,
        references=clamped_mixture_of_experts.WEIGHT_REFERENCES.get(key, ()))
    for key, role in clamped_mixture_of_experts.WEIGHT_ROLES.items()}

NORMALISATION_AND_ROTARY_ROLES: dict[str, OperatorRole] = {
    NORMALISATION_NAME: OperatorRole(
        role=(
            text.RMSNORM_ROLE),
        references=(inference_config_lines(23), model_lines(1114))),
    **rotary_embedding.TABLE_ROLES,
}

OPERATOR_ROLES: dict[str, OperatorRole] = join_tables(
    operator_explanations.OPERATOR_ROLES,
    dspark_draft_chain.OPERATOR_ROLES,
    MIXTURE_ROLES,
    omitted_mechanisms.ENGRAM_WEIGHT_ROLES,
    vision_pathway.OPERATOR_ROLES,
    NORMALISATION_AND_ROTARY_ROLES)

REINDEXING_EXPLANATIONS: dict[str, ReindexingExplanation] = join_tables(
    operator_explanations.REINDEXING_EXPLANATIONS,
    quantised_caches.REINDEXING_EXPLANATIONS,
    dspark_draft_chain.REINDEXING_EXPLANATIONS,
    gumbel_max_sampler.REINDEXING_EXPLANATIONS,
    omitted_mechanisms.ENGRAM_REINDEXING_EXPLANATIONS)


def with_explanation_tables(
    settings: notebook_diagrams.DiagramSettings,
) -> notebook_diagrams.DiagramSettings:
    '''`settings` with the four tables of this module in the fields the display
    reads.'''
    return dataclasses.replace(
        settings,
        operator_explanations=OPERATOR_EXPLANATIONS,
        operator_references=OPERATOR_REFERENCES,
        operator_roles=OPERATOR_ROLES,
        reindexing_explanations=REINDEXING_EXPLANATIONS)


# The tables of the text-only model.

TEXT_ONLY_GENERIC_OPERATOR_EXPLANATIONS: dict[str, OperatorExplanation] = {
    quantised_caches.CEILING: quantised_caches.CEILING_EXPLANATION,
}


def explain_text_only_generic_operator(
    target: cat.Broadcasted,
) -> OperatorExplanation | None:
    '''The row for `ops.GenericOperator` in the text-only model, which holds the
    ceiling of the quantised caches and no other generic operator.'''
    name = target.operator.name
    return (None if name is None
            else TEXT_ONLY_GENERIC_OPERATOR_EXPLANATIONS.get(name.to_bodies()))


TEXT_ONLY_OPERATOR_EXPLANATIONS: dict[type[cat.Operator], ExplanationOfOperator] = {
    **OPERATOR_EXPLANATIONS,
    ops.GenericOperator: explain_text_only_generic_operator,
}


def with_text_only_explanation_tables(
    settings: notebook_diagrams.DiagramSettings,
) -> notebook_diagrams.DiagramSettings:
    '''`settings` with the tables of the text-only model. Only the row for a generic
    operator differs, because the text-only model holds the ceiling of the quantised
    caches and no other generic operator.'''
    return dataclasses.replace(
        with_explanation_tables(settings),
        operator_explanations=TEXT_ONLY_OPERATOR_EXPLANATIONS)


# The coverage of the tables.

@dataclass(frozen=True)
class TableCoverage:
    '''The names of a term that the tables give no row, and the roles that no map of
    the term uses. Every field is empty when the tables cover the term.'''
    linears_without_a_role: tuple[str, ...]
    generic_operators_without_a_row: tuple[str, ...]
    embeddings_without_a_row: tuple[str, ...]
    hidden_arithmetics_without_a_role: tuple[str, ...]
    unused_arithmetic_roles: tuple[str, ...]
    reindexings_without_a_row: tuple[str, ...]

    def is_complete(self) -> bool:
        return not any(getattr(self, field.name) for field in dataclasses.fields(self))


def names_of_operators(
    broadcasts: tuple[cat.Broadcasted, ...], kind: type[cat.Operator],
) -> set[str]:
    return {node.operator.name.to_bodies() for node in broadcasts
            if isinstance(node.operator, kind) and node.operator.name is not None}


def names_of_hidden_arithmetics(broadcasts: tuple[cat.Broadcasted, ...]) -> set[str]:
    '''The names of the elementwise maps whose figure does not show the whole of
    their formula, which are the maps `explain_named_arithmetic` opens a box over.'''
    return {node.operator.name.to_bodies() for node in broadcasts
            if isinstance(node.operator, ops.Arithmetic)
            and node.operator.name is not None
            and not explain_operators.shows_whole_formula(node.operator)}


def names_of_reindexings(broadcasts: tuple[cat.Broadcasted, ...]) -> set[str]:
    return {stride.name.to_bodies()
            for node in broadcasts
            for reindexing in node.reindexings
            for stride in tutil.type_search(cat.StrideMorphism, reindexing)
            if stride.name is not None}


def sorted_difference(names: set[str], table: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(sorted(names - set(table)))


def list_names_missing_a_row(
    term: fd.GeneralTerm = integrated_whole_model.v41_flash_integrated,
) -> TableCoverage:
    '''Every name of `term` that the tables give no row, and every role of
    `ARITHMETIC_ROLES` that no map of `term` uses.'''
    broadcasts = tuple(tutil.type_search(cat.Broadcasted, term))
    hidden = names_of_hidden_arithmetics(broadcasts)
    return TableCoverage(
        linears_without_a_role=sorted_difference(
            names_of_operators(broadcasts, ops.Linear), OPERATOR_ROLES),
        generic_operators_without_a_row=sorted_difference(
            names_of_operators(broadcasts, ops.GenericOperator),
            GENERIC_OPERATOR_EXPLANATIONS),
        embeddings_without_a_row=sorted_difference(
            names_of_operators(broadcasts, ops.Embedding), EMBEDDING_EXPLANATIONS),
        hidden_arithmetics_without_a_role=sorted_difference(hidden, ARITHMETIC_ROLES),
        unused_arithmetic_roles=tuple(sorted(set(ARITHMETIC_ROLES) - hidden)),
        reindexings_without_a_row=sorted_difference(
            names_of_reindexings(broadcasts), REINDEXING_EXPLANATIONS))


def list_text_only_names_missing_a_row(
    term: fd.GeneralTerm = text_only_model.v41_flash_text_only,
) -> TableCoverage:
    '''Every name of the text-only model that its tables give no row. The roles no map
    of the model uses are not listed, because the tables hold the rows of both models
    and the image pathway and DSpark carry many maps this model does not.'''
    broadcasts = tuple(tutil.type_search(cat.Broadcasted, term))
    return TableCoverage(
        linears_without_a_role=sorted_difference(
            names_of_operators(broadcasts, ops.Linear), OPERATOR_ROLES),
        generic_operators_without_a_row=sorted_difference(
            names_of_operators(broadcasts, ops.GenericOperator),
            TEXT_ONLY_GENERIC_OPERATOR_EXPLANATIONS),
        embeddings_without_a_row=sorted_difference(
            names_of_operators(broadcasts, ops.Embedding), EMBEDDING_EXPLANATIONS),
        hidden_arithmetics_without_a_role=sorted_difference(
            names_of_hidden_arithmetics(broadcasts), ARITHMETIC_ROLES),
        unused_arithmetic_roles=(),
        reindexings_without_a_row=sorted_difference(
            names_of_reindexings(broadcasts), REINDEXING_EXPLANATIONS))


# The released constants.

RELEASED_CONSTANTS_TABLE_HEADER: tuple[str, str] = (
    '| symbol | released value | meaning | released line |',
    '|---|---|---|---|')


def released_constant_row(constant: ReleasedConstant) -> str:
    '''One markdown table row for `constant`, with its symbol and value set as
    mathematics and its released line as a link.'''
    reference = constant.reference
    link = (reference.label if reference.url is None
            else f'[{reference.label}]({reference.url})')
    return (f'| ${constant.symbol.to_latex()}$ | ${constant.released_value}$ | '
            f'{constant.meaning} | {link} |')


RELEASED_CONSTANT_ROWS: tuple[str, ...] = tuple(
    map(released_constant_row, integrated_whole_model.ALL_RELEASED_CONSTANTS))


def released_constants_table() -> str:
    '''The markdown table of every named constant of the model, for a notebook cell.'''
    return '\n'.join((*RELEASED_CONSTANTS_TABLE_HEADER, *RELEASED_CONSTANT_ROWS))
